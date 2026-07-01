"""Proximal latent-class deconfounder (PR-2b — the contribution).

Cell-7 deconfounding from the FIVE base keys only (never the realized U): a
stochastic-EM over a 2-component reward-mixing MDP, run THROUGH the OracleU
critic hook. The only proximal-specific machinery is here (``ProximalEM``); the
critic update is the base learner's existing ``learn`` + the ``Proximal``
strategy's ``critic_value = q_su(x, batch["confounder_u"])`` (identical to
OracleU), with ``confounder_u`` an INFERRED-and-sampled latent.

Model: ``r_t = r_clean(s_t,a_t) + c_r * U + noise``, U in {0,1} constant per
episode (``confounded.py``). The estimator:

  * E-step (per FULL episode tau, via ``SequenceReplayBuffer.iter_episodes``):
    posterior ``r_tau = P(U=1|tau) = sigmoid( sum_t [logN(r_t; r_hat+delta, s) -
    logN(r_t; r_hat, s)] + log(p/(1-p)) )`` — the residual-corrected per-step
    reward-sequence LLR under a fitted clean-reward model ``r_hat(s,a)`` + a
    learnable shift ``delta (~c_r)`` + noise ``s``. NOT ``rewards.sum()`` (the raw
    first moment conflates state-dependent r_clean with the U shift — insufficient).
    ``r_tau`` is written back into each transition dict, so the runner's
    ``sample_sequences`` carries it into the (B,T) windows.
  * M-step (per (B,T) window, in ``m_step``): sample ``u ~ Bernoulli(r_tau)`` once
    per episode-window, set ``batch["confounder_u"] = u`` (the INFERRED sample),
    flatten (B,T,*) -> (B*T,*), and call the base learner's ``learn`` — which
    routes ``q_su(x, u)`` through the Proximal hook. K=1 sample (K>1 MC averaging
    would be a config knob, not an architecture change).
  * Deploy: ``Q_adj = E_u[q_su]`` over the PRIOR p (``UMarginalizedQ.forward``) —
    never the posterior (the frozen IQL-leak-analog invariant).

At c_r=0 the strata coincide: ``delta -> 0``, ``r_tau -> p`` everywhere, so the
sampled u is prior noise, ``q_su(.,0) ~ q_su(.,1)``, ``Q_adj ~ plain Q ~`` the
Observational floor. That collapse is the safety proof (the c_r=0 gate).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from src.rl.models.backbone import select_backbone
from src.rl.nets.mlp import MLP
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.identification import Proximal
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.bcq import DiscreteBCQ
from src.rl.offline.cql import CQL
from src.rl.offline.iql import IQL
from src.rl.offline.oracle_u import UMarginalizedQ

_PRIOR_P = 0.5  # Bernoulli(p) prior on U (matches confounded.py)


class _RewardModel(nn.Module):
    """Per-stratum one-step reward model: clean reward ``r_hat(s,a)`` + a learnable
    shift ``delta`` for the U=1 stratum + a noise scale. ``r_hat`` is a per-action
    head (MLP obs -> action_dim), gathered at the data action."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.r_hat = MLP(obs_dim, action_dim)
        # delta = softplus(raw_delta) >= 0 is a LABELING CONVENTION: "U=1 is the
        # higher-reward stratum." It makes the label-swapped (-delta) EM basin
        # UNREACHABLE — that basin (r_tau anti-correlated with the true U, delta
        # dragged through 0) is what stalled sigma=1.0 (diagnosed corr(r_tau,U)
        # -> -1.0). softplus(raw)->0 is still reachable, so the c_r=0 collapse +
        # the delta_l2 shrinkage are untouched. raw init 0.5413 -> delta ~ 1.0.
        self.raw_delta = nn.Parameter(torch.tensor(0.5413))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    @property
    def delta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_delta)

    def reward_at(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.r_hat(obs).gather(1, actions.long().view(-1, 1)).squeeze(-1)


class ProximalEM:
    """The proximal-specific machinery: a reward model + the soft-EM E/M for the
    per-episode latent, plus the per-window u-sampler that feeds the OracleU hook.
    Attached to a base learner (any of DQN/CQL/IQL/BCQ) by ``install``; the base
    learner's ``learn`` and the ``Proximal`` strategy are otherwise unchanged."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        prior_p: float = _PRIOR_P,
        lr: float = 1e-3,
        reward_steps: int = 64,
        estep_interval: int = 20,
        delta_l2: float = 0.05,
    ) -> None:
        self.device = device
        self.prior_p = float(prior_p)
        self.rm = _RewardModel(obs_dim, action_dim).to(device)
        self.opt = torch.optim.Adam(self.rm.parameters(), lr=lr)
        self.reward_steps = int(reward_steps)
        self.estep_interval = max(1, int(estep_interval))
        # MAP shrinkage prior on the reward-shift: penalty lambda*delta^2 whose
        # gradient (2*lambda*delta) is INDEPENDENT of r_tau, so it pins delta->0
        # when a stratum empties and the likelihood gradient vanishes (the
        # degenerate one-component EM trap). Embodies "assume no confounding unless
        # the data evidences it" -> at c_r=0 delta->0, r_tau->prior, robust collapse;
        # at c_r>0 the data likelihood overcomes the small L2 and delta->c_r.
        self.delta_l2 = float(delta_l2)
        # Label-swap convention threshold: the E-step canonicalizes "high r_tau =
        # high reward-sum" ONLY when responsibilities are non-degenerate (delta and
        # r_tau-spread both above this eps). At c_r=0 (delta->0, r_tau->p uniform)
        # the guard never fires, so the collapse is untouched.
        self._canon_eps = 0.05
        self.seq_buffer = None
        self._updates = 0
        # Padded episode batch (n_ep, max_len, ·) + mask, cached on-device ONCE by
        # set_sequence_buffer — collapses the scalar path's 64*n_ep per-run_em
        # stack+transfer to a single build, and lets the E/M reductions run as
        # masked batched ops (GPU-fast) instead of a launch-bound per-episode loop.
        self._batch = None
        self._episodes = None

    # -- handoff (the runner's single set_sequence_buffer line) --
    def set_sequence_buffer(self, seq_buffer) -> None:
        """Receive the runner-owned SequenceReplayBuffer, build + cache the padded
        episode batch ON-DEVICE ONCE, WARM-START responsibilities from the canonical
        "higher-reward stratum = U=1" labeling, and run the first E/M pass. Cold
        r_tau=p is symmetric and lets the EM settle into the label-swapped basin
        (diagnosed at high sigma, corr(r_tau,U)->-1); the warm-start enters the
        correct basin."""
        self.seq_buffer = seq_buffer
        self._episodes = list(seq_buffer.iter_episodes())
        self._batch = self._build_batch(self._episodes)
        b = self._batch
        if b is not None:
            # MEAN reward per episode (length-invariant — reward-SUM = episode length
            # on a unit-reward env, which would spuriously label by length). NO-SPREAD
            # FALLBACK: if the mean reward has ~no cross-episode spread (c_r=0,
            # constant per-step reward), there is no separation to label -> symmetric
            # prior (restores r_tau -> p at sigma=0).
            means = (b["rew"] * b["mask"]).sum(1) / b["lengths"]
            if float(means.std()) < self._canon_eps:
                r_tau = torch.full_like(means, self.prior_p)
            else:
                med = means.median()
                r_tau = (means > med).float()
            b["r_tau"] = r_tau
            self._scatter_r_tau(r_tau)
        self.run_em()

    def _build_batch(self, episodes):
        """Pad episodes to (n_ep, max_len, ·) + a float length-mask, ONCE, cached on
        device. Padded positions are 0 and masked out of every reduction; sums divide
        by TRUE lengths, never max_len. Replaces the scalar path's per-episode
        stack+``.to(device)`` (64*n_ep tiny transfers per run_em) with one build."""
        if not episodes:
            return None
        dev = self.device
        n = len(episodes)
        lengths = torch.tensor(
            [len(e) for e in episodes], device=dev, dtype=torch.float32
        )
        max_len = int(max(len(e) for e in episodes))
        obs_dim = episodes[0][0]["obs"].shape[-1]
        obs = torch.zeros(n, max_len, obs_dim, device=dev)
        act = torch.zeros(n, max_len, dtype=torch.long, device=dev)
        rew = torch.zeros(n, max_len, device=dev)
        mask = torch.zeros(n, max_len, device=dev)
        for i, ep in enumerate(episodes):
            t = len(ep)
            obs[i, :t] = torch.stack([tr["obs"] for tr in ep]).to(dev).float()
            act[i, :t] = torch.stack([tr["actions"] for tr in ep]).to(dev)
            rew[i, :t] = torch.stack([tr["rewards"] for tr in ep]).to(dev).float()
            mask[i, :t] = 1.0
        return {
            "obs": obs,
            "act": act,
            "rew": rew,
            "mask": mask,
            "lengths": lengths,
            "n": n,
            "max_len": max_len,
            "obs_dim": obs_dim,
            "r_tau": torch.full((n,), self.prior_p, device=dev),
        }

    def _scatter_r_tau(self, r_tau) -> None:
        """Write the (n_ep,) responsibilities back into the stored transitions so
        ``sample_sequences`` carries r_tau into the (B,T) windows (n_ep-length loop,
        not the hot path)."""
        for ep, val in zip(self._episodes, r_tau.detach().cpu().tolist()):
            t = torch.tensor(val)
            for tr in ep:
                tr["r_tau"] = t

    def _reward_at_batch(self, b):
        """r_hat(s,a) over the padded batch -> (n_ep, max_len). One MLP forward on
        the flattened (n_ep*max_len, obs_dim) tensor, gathered at the data action."""
        rhat = self.rm.r_hat(b["obs"].reshape(-1, b["obs_dim"])).reshape(
            b["n"], b["max_len"], -1
        )
        return rhat.gather(-1, b["act"].unsqueeze(-1)).squeeze(-1)

    def run_em(self) -> None:
        """One EM pass over the cached padded batch: a masked-batched M-step (reward
        params, responsibility-weighted Gaussian NLL) then a masked-batched E-step
        (per-episode LLR posterior r_tau + mean residual), scattered back into the
        stored transitions. Vectorized form of the scalar per-episode loops."""
        b = self._batch
        if self.seq_buffer is None or b is None:
            return
        rew, mask = b["rew"], b["mask"]
        total = mask.sum()
        r_tau = b["r_tau"]  # (n_ep,) responsibilities from warm-start / prior E-step

        # M-step (reward params): masked responsibility-weighted NLL over both strata.
        for _ in range(self.reward_steps):
            self.opt.zero_grad(set_to_none=True)
            r_hat = self._reward_at_batch(b)
            sigma = self.rm.log_sigma.exp().clamp_min(1e-3)
            ll0 = -0.5 * ((rew - r_hat) / sigma) ** 2 - torch.log(sigma)
            ll1 = -0.5 * ((rew - r_hat - self.rm.delta) / sigma) ** 2 - torch.log(sigma)
            per = -((1.0 - r_tau[:, None]) * ll0 + r_tau[:, None] * ll1)
            # MAP shrinkage: r_tau-independent gradient pins delta->0 absent evidence.
            loss = (per * mask).sum() / total + self.delta_l2 * self.rm.delta**2
            loss.backward()
            self.opt.step()

        # E-step: masked per-episode reward-sequence LLR posterior + the per-episode
        # MEAN RESIDUAL mean_t(r_t - r_hat) — the state-baseline-removed shift signal
        # (the canonical observable). Sums divide by TRUE lengths, never max_len.
        logit_prior = math.log(self.prior_p / (1.0 - self.prior_p))
        with torch.no_grad():
            r_hat = self._reward_at_batch(b)
            sigma = self.rm.log_sigma.exp().clamp_min(1e-3)
            ll0 = -0.5 * ((rew - r_hat) / sigma) ** 2
            ll1 = -0.5 * ((rew - r_hat - self.rm.delta) / sigma) ** 2
            llr = ((ll1 - ll0) * mask).sum(1) + logit_prior
            r_taus = torch.sigmoid(llr)
            residuals = ((rew - r_hat) * mask).sum(1) / b["lengths"]

        # GUARDED label canonicalization: enforce "high r_tau = high mean-residual"
        # (the U=1=higher-reward convention) as an invariant of every E-step, so the
        # swapped basin is UNREACHABLE. NO-SPREAD FALLBACK: if the shift signal has
        # ~no cross-episode spread (c_r=0 -> residual ~ 0 for all), there is no
        # separation -> symmetric prior, so r_tau -> p exactly (the sigma=0 collapse).
        if float(residuals.std()) < self._canon_eps:
            r_taus = torch.full_like(r_taus, self.prior_p)
        else:
            non_degenerate = (
                float(self.rm.delta) > self._canon_eps
                and float(r_taus.std()) > self._canon_eps
            )
            if non_degenerate:
                corr = float(torch.corrcoef(torch.stack([r_taus, residuals]))[0, 1])
                if corr < 0.0:
                    r_taus = 1.0 - r_taus  # flip into the canonical basin

        b["r_tau"] = r_taus
        self._scatter_r_tau(r_taus)

    # -- the M-step entry point the wrapped agent.learn calls --
    def m_step(self, window: Dict[str, torch.Tensor], base_learn):
        """Per (B,T) window: refresh the EM on cadence, sample u from the episode
        posterior, set the inferred ``confounder_u``, flatten, run the base learn
        (which routes q_su(x, u) via the Proximal hook)."""
        self._updates += 1
        if self._updates % self.estep_interval == 0:
            self.run_em()
        r_tau = window["r_tau"]  # (B, T), constant over T within each episode-window
        # U is per-episode -> one Bernoulli draw per window row, broadcast over T.
        u_row = torch.bernoulli(r_tau[:, 0].clamp(0.0, 1.0))  # (B,)
        u = u_row.unsqueeze(1).expand_as(r_tau)  # (B, T)
        window = {**window, "confounder_u": u}
        flat = {
            k: (v.flatten(0, 1) if torch.is_tensor(v) and v.dim() >= 2 else v)
            for k, v in window.items()
        }
        return base_learn(flat)


def _install_proximal_em(agent, em: ProximalEM) -> None:
    """Wrap the base learner's ``learn`` with the proximal M-step and expose the
    ``set_sequence_buffer`` handoff. The base ``learn`` is unchanged; the Proximal
    strategy routes ``q_su`` — the agent simply provides the inferred U."""
    base_learn = agent.learn
    agent.learn = lambda batch: em.m_step(batch, base_learn)
    agent.set_sequence_buffer = em.set_sequence_buffer
    agent._proximal_em = em  # keep a ref alive + testable


# --------------------------------------------------------------------------
# Builders: base learner x Proximal strategy, UMarginalizedQ critics, + the EM.
# Discrete vector Cell-7 arm only; all four route through the OracleU hook.
# --------------------------------------------------------------------------
def _dims(kwargs):
    if kwargs.get("action_type", "discrete") != "discrete":
        raise NotImplementedError(
            "proximal: discrete Cell-7 arm only (offline_dqn/bcq/cql/iql)."
        )
    return kwargs["obs_dim"], kwargs["action_dim"], kwargs["device"]


def build_proximal_dqn(**kwargs):
    obs_dim, action_dim, device = _dims(kwargs)
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    agent = DQN(
        q, tgt, ReplayBuffer(1_000_000, device), device=device, strategy=Proximal()
    )
    _install_proximal_em(agent, ProximalEM(obs_dim, action_dim, device))
    return q, agent


def build_proximal_cql(**kwargs):
    obs_dim, action_dim, device = _dims(kwargs)
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    agent = CQL(
        q, tgt, ReplayBuffer(1_000_000, device), device=device, strategy=Proximal()
    )
    _install_proximal_em(agent, ProximalEM(obs_dim, action_dim, device))
    return q, agent


def build_proximal_iql(**kwargs):
    obs_dim, action_dim, device = _dims(kwargs)
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    policy_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    value_net = select_backbone(obs_shape, obs_dim, 1).to(device)
    agent = IQL(
        policy_net,
        q,
        tgt,
        value_net,
        ReplayBuffer(1_000_000, device),
        device=device,
        strategy=Proximal(),
    )
    _install_proximal_em(agent, ProximalEM(obs_dim, action_dim, device))
    return policy_net, agent


def build_proximal_bcq(**kwargs):
    obs_dim, action_dim, device = _dims(kwargs)
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    behavior_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    agent = DiscreteBCQ(
        q,
        tgt,
        behavior_net,
        ReplayBuffer(1_000_000, device),
        device=device,
        strategy=Proximal(),
    )
    _install_proximal_em(agent, ProximalEM(obs_dim, action_dim, device))
    return q, agent

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
from typing import Dict, List

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
        self.delta = nn.Parameter(torch.tensor(1.0))  # init >0 to bias the U=1 label
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

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
        self.seq_buffer = None
        self._updates = 0

    # -- handoff (the runner's single set_sequence_buffer line) --
    def set_sequence_buffer(self, seq_buffer) -> None:
        """Receive the runner-owned SequenceReplayBuffer, cold-start the
        responsibilities to the prior, and run the first E/M pass."""
        self.seq_buffer = seq_buffer
        for ep in seq_buffer.iter_episodes():
            for tr in ep:
                tr["r_tau"] = torch.tensor(self.prior_p)
        self.run_em()

    def _ep_tensors(self, ep: List[Dict[str, torch.Tensor]]):
        obs = torch.stack([tr["obs"] for tr in ep]).to(self.device).float()
        actions = torch.stack([tr["actions"] for tr in ep]).to(self.device)
        rewards = torch.stack([tr["rewards"] for tr in ep]).to(self.device).float()
        r_tau = float(ep[0].get("r_tau", torch.tensor(self.prior_p)))
        return obs, actions, rewards, r_tau

    def run_em(self) -> None:
        """One EM pass: M-step on the reward params (responsibility-weighted
        Gaussian NLL over both strata), then E-step recomputing ``r_tau`` per
        episode and writing it back into the stored transitions."""
        if self.seq_buffer is None:
            return
        episodes = list(self.seq_buffer.iter_episodes())
        if not episodes:
            return

        # M-step (reward params): minimize the responsibility-weighted NLL.
        for _ in range(self.reward_steps):
            self.opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=self.device)
            total = 0
            for ep in episodes:
                obs, actions, rewards, r_tau = self._ep_tensors(ep)
                r_hat = self.rm.reward_at(obs, actions)
                sigma = self.rm.log_sigma.exp().clamp_min(1e-3)
                ll0 = -0.5 * ((rewards - r_hat) / sigma) ** 2 - torch.log(sigma)
                ll1 = -0.5 * (
                    (rewards - r_hat - self.rm.delta) / sigma
                ) ** 2 - torch.log(sigma)
                loss = loss - ((1.0 - r_tau) * ll0 + r_tau * ll1).sum()
                total += len(ep)
            # MAP shrinkage: r_tau-independent gradient pins delta->0 absent evidence.
            loss = loss / max(total, 1) + self.delta_l2 * self.rm.delta**2
            loss.backward()
            self.opt.step()

        # E-step: posterior from the per-step reward-sequence LLR; write back.
        logit_prior = math.log(self.prior_p / (1.0 - self.prior_p))
        with torch.no_grad():
            sigma = self.rm.log_sigma.exp().clamp_min(1e-3)
            for ep in episodes:
                obs, actions, rewards, _ = self._ep_tensors(ep)
                r_hat = self.rm.reward_at(obs, actions)
                ll0 = -0.5 * ((rewards - r_hat) / sigma) ** 2
                ll1 = -0.5 * ((rewards - r_hat - self.rm.delta) / sigma) ** 2
                llr = (ll1 - ll0).sum() + logit_prior
                r_tau = torch.sigmoid(llr).detach().cpu()
                for tr in ep:
                    tr["r_tau"] = r_tau

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

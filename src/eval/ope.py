"""Off-policy evaluation estimators (§6.5).

Every estimator consumes ``(OfflineDatasetSource, target policy)`` and
returns an :class:`OPEResult` ``(value, ci)``. Correctness rule (§8, binding):
propensities come ONLY from logged ``source.behavior_logprob`` (cells that
declare pi_b known) or from a behavior policy CLONED on observed
``(obs, actions)`` — never from oracle quantities or variables the cell
declares unobserved.

Conventions: undiscounted episodic value ``J = E[sum_t r_t]`` (matching the
regret protocol); per-episode importance products for IPW; stepwise doubly
robust (Jiang & Li 2016) on top of FQE; 95% percentile bootstrap CIs over
episodes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.experience_source import OfflineDatasetSource
from src.rl.nets.mlp import MLP

# ---------------------------------------------------------------------------
# Target-policy adapters
# ---------------------------------------------------------------------------


class TargetPolicy(abc.ABC):
    """What OPE needs from a target policy."""

    @abc.abstractmethod
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log pi(a|s) of the TAKEN dataset actions."""

    @abc.abstractmethod
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action selection (for DM/DR continuation values)."""

    def action_probs(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """[B, A] action distribution for discrete policies; None otherwise."""
        return None


class StochasticPolicyAdapter(TargetPolicy):
    """Adapter over an ActorCriticMLP-style policy (distribution/log_prob)."""

    def __init__(self, policy) -> None:
        self.policy = policy

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dist = self.policy.distribution(obs)
            return self.policy.log_prob(dist, actions)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.act_deterministic(obs)

    def action_probs(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        with torch.no_grad():
            dist = self.policy.distribution(obs)
            if hasattr(dist, "logits"):
                return torch.softmax(dist.logits, dim=-1)
        return None


class DeterministicPolicyAdapter(TargetPolicy):
    """Deterministic (e.g. d3rlpy greedy) discrete target: pi(a|s) is an
    indicator, so IPW ratios become indicator/propensity."""

    def __init__(self, act_fn: Callable[[torch.Tensor], torch.Tensor], n_actions: int):
        self._act = act_fn
        self.n_actions = int(n_actions)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        chosen = self.act(obs)
        match = (chosen.long() == actions.long()).float()
        return torch.log(match.clamp_min(1e-12))

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._act(obs)

    def action_probs(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        chosen = self.act(obs).long()
        return F.one_hot(chosen, num_classes=self.n_actions).float()


# ---------------------------------------------------------------------------
# Results + bootstrap
# ---------------------------------------------------------------------------


@dataclass
class OPEResult:
    value: float
    ci_low: float
    ci_high: float
    estimator: str
    n_episodes: int


def _bootstrap_ci(
    per_episode: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_episode)
    if n < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = per_episode[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(
        np.quantile(means, 1 - alpha / 2)
    )


def _result(per_episode: torch.Tensor, name: str) -> OPEResult:
    vals = per_episode.detach().cpu().numpy().astype(np.float64)
    lo, hi = _bootstrap_ci(vals)
    return OPEResult(float(vals.mean()), lo, hi, name, len(vals))


# ---------------------------------------------------------------------------
# Naive
# ---------------------------------------------------------------------------


class NaiveEstimator:
    """Mean logged return — correct only when pi_b == pi_target."""

    name = "naive"

    def estimate(self, source: OfflineDatasetSource, target: TargetPolicy) -> OPEResult:
        _ = target
        return _result(source.episode_returns(), self.name)


# ---------------------------------------------------------------------------
# Direct method (FQE)
# ---------------------------------------------------------------------------


class FQE:
    """Fitted Q evaluation of a fixed target policy on logged data."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str,
        device: torch.device,
        gamma: float = 1.0,
        lr: float = 1e-3,
        hidden_dims=(64, 64),
        seed: int = 0,
    ) -> None:
        self.action_type = action_type
        self.device = device
        self.gamma = float(gamma)
        torch.manual_seed(seed)
        in_dim = obs_dim if action_type == "discrete" else obs_dim + action_dim
        out_dim = action_dim if action_type == "discrete" else 1
        self.q_net = MLP(in_dim, out_dim, hidden_dims=hidden_dims).to(device)
        self.q_target = MLP(in_dim, out_dim, hidden_dims=hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def _q(self, net, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.action_type == "discrete":
            return net(obs).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        act = actions.float()
        if act.ndim == 1:
            act = act.unsqueeze(-1)
        return net(torch.cat([obs, act], dim=-1)).squeeze(-1)

    def _continuation(
        self, target: TargetPolicy, next_obs: torch.Tensor
    ) -> torch.Tensor:
        probs = target.action_probs(next_obs)
        if self.action_type == "discrete" and probs is not None:
            return (self.q_target(next_obs) * probs).sum(dim=-1)
        next_actions = target.act(next_obs)
        return self._q(self.q_target, next_obs, next_actions)

    def fit(
        self,
        source: OfflineDatasetSource,
        target: TargetPolicy,
        n_iters: int = 400,
        batch_size: int = 512,
        sync_every: int = 50,
    ) -> "FQE":
        for it in range(int(n_iters)):
            batch = source.sample(batch_size)
            obs = batch["obs"].float()
            with torch.no_grad():
                cont = self._continuation(target, batch["next_obs"].float())
                tgt = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * cont
            pred = self._q(self.q_net, obs, batch["actions"])
            loss = F.mse_loss(pred, tgt)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if (it + 1) % sync_every == 0:
                self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.load_state_dict(self.q_net.state_dict())
        return self

    def q_values(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._q(self.q_net, obs.float(), actions)

    def state_value(self, target: TargetPolicy, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = target.action_probs(obs.float())
            if self.action_type == "discrete" and probs is not None:
                return (self.q_net(obs.float()) * probs).sum(dim=-1)
            actions = target.act(obs.float())
            return self._q(self.q_net, obs.float(), actions)


class DirectMethod:
    """DM/FQE: V̂ = mean_s0 V^π(s0) with the fitted Q."""

    name = "dm"

    def __init__(
        self,
        gamma: float = 1.0,
        n_iters: int = 400,
        sync_every: int = 50,
        seed: int = 0,
    ) -> None:
        self.gamma = gamma
        self.n_iters = n_iters
        self.sync_every = sync_every
        self.seed = seed
        self.fqe: Optional[FQE] = None

    def estimate(self, source: OfflineDatasetSource, target: TargetPolicy) -> OPEResult:
        obs_dim = source.obs.shape[-1]
        if source.actions.dtype in (torch.int64, torch.int32):
            action_type = "discrete"
            action_dim = int(source.actions.max().item()) + 1
        else:
            action_type = "continuous"
            action_dim = source.actions.shape[-1] if source.actions.ndim > 1 else 1
        self.fqe = FQE(
            obs_dim,
            action_dim,
            action_type,
            source.device,
            gamma=self.gamma,
            seed=self.seed,
        ).fit(source, target, n_iters=self.n_iters, sync_every=self.sync_every)
        starts = torch.stack([ep["obs"][0] for ep in source.episodes]).float()
        return _result(self.fqe.state_value(target, starts), self.name)


# ---------------------------------------------------------------------------
# Behavior cloning of pi_b (for IPW with unknown propensities)
# ---------------------------------------------------------------------------


def clone_behavior_policy(
    source: OfflineDatasetSource,
    n_iters: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """MLE-fit pi_b on observed (obs, actions); returns log pi_b(a|s).

    Discrete only for now (the CartPole tiers); continuous cloning arrives
    with the continuous confounded cells.
    """
    if source.actions.dtype not in (torch.int64, torch.int32):
        raise NotImplementedError("behavior cloning of continuous pi_b: Phase 6")
    torch.manual_seed(seed)
    n_actions = int(source.actions.max().item()) + 1
    net = MLP(source.obs.shape[-1], n_actions).to(source.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(int(n_iters)):
        batch = source.sample(batch_size)
        logits = net(batch["obs"].float())
        loss = F.cross_entropy(logits, batch["actions"].long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def logprob(obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logp = torch.log_softmax(net(obs.float()), dim=-1)
            return logp.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

    return logprob


# ---------------------------------------------------------------------------
# IPW
# ---------------------------------------------------------------------------


class IPWEstimator:
    """Per-episode importance weighting: V̂ = mean_ep [ (Π_t ρ_t) · G_ep ].

    ``behavior="known"`` uses LOGGED propensities and refuses to run without
    them; ``behavior="cloned"`` fits pi_b on observed (obs, actions).
    """

    def __init__(
        self,
        behavior: str = "known",
        clip: Optional[float] = 1e4,
        self_normalized: bool = False,
        seed: int = 0,
    ) -> None:
        if behavior not in ("known", "cloned"):
            raise ValueError("behavior must be 'known' or 'cloned'")
        self.behavior = behavior
        self.clip = clip
        self.self_normalized = self_normalized
        self.seed = seed

    @property
    def name(self) -> str:
        return f"ipw_{self.behavior}"

    def _episode_weights_returns(
        self, source: OfflineDatasetSource, target: TargetPolicy
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cloned = None
        if self.behavior == "cloned":
            cloned = clone_behavior_policy(source, seed=self.seed)
        elif source.behavior_logprob is None:
            raise ValueError(
                "IPW(known) requires logged propensities, but this source "
                "declares the behavior policy unknown (cells 5/6/8). "
                "Use behavior='cloned'."
            )
        # one batched pass over the flat tensors, then split per episode
        lengths = [int(ep["rewards"].shape[0]) for ep in source.episodes]
        logp_t = target.log_prob(source.obs.float(), source.actions)
        if self.behavior == "known":
            logp_b = source.behavior_logprob.float()
        else:
            logp_b = cloned(source.obs.float(), source.actions)
        log_ratio = logp_t - logp_b
        weights, returns = [], []
        for ep_ratio, ep in zip(torch.split(log_ratio, lengths), source.episodes):
            weights.append(torch.exp(ep_ratio.sum().clamp(max=30.0)))
            returns.append(ep["rewards"].sum())
        w = torch.stack(weights)
        if self.clip is not None:
            w = w.clamp(max=self.clip)
        return w, torch.stack(returns).float()

    def estimate(self, source: OfflineDatasetSource, target: TargetPolicy) -> OPEResult:
        w, g = self._episode_weights_returns(source, target)
        if self.self_normalized:
            per_ep = (w * g) * (len(w) / w.sum().clamp_min(1e-12))
        else:
            per_ep = w * g
        return _result(per_ep, self.name)


# ---------------------------------------------------------------------------
# Doubly robust (stepwise, Jiang & Li 2016)
# ---------------------------------------------------------------------------


class DoublyRobust:
    """Backward-recursive stepwise DR on top of FQE.

    DR_t = V̂(s_t) + ρ_t (r_t + γ DR_{t+1} − Q̂(s_t, a_t)), value = mean DR_0.
    """

    name = "dr"

    def __init__(
        self,
        behavior: str = "known",
        gamma: float = 1.0,
        n_fqe_iters: int = 400,
        fqe_sync_every: int = 50,
        rho_clip: float = 100.0,
        seed: int = 0,
    ) -> None:
        self.behavior = behavior
        self.gamma = gamma
        self.n_fqe_iters = n_fqe_iters
        self.fqe_sync_every = fqe_sync_every
        self.rho_clip = rho_clip
        self.seed = seed

    def estimate(self, source: OfflineDatasetSource, target: TargetPolicy) -> OPEResult:
        dm = DirectMethod(
            gamma=self.gamma,
            n_iters=self.n_fqe_iters,
            sync_every=self.fqe_sync_every,
            seed=self.seed,
        )
        dm.estimate(source, target)  # fits the FQE
        fqe = dm.fqe
        assert fqe is not None

        cloned = None
        if self.behavior == "cloned":
            cloned = clone_behavior_policy(source, seed=self.seed)
        elif source.behavior_logprob is None:
            raise ValueError(
                "DR(known) requires logged propensities; use behavior='cloned'."
            )

        # batched passes over flat tensors, then per-episode backward recursion
        lengths = [int(ep["rewards"].shape[0]) for ep in source.episodes]
        flat_obs = source.obs.float()
        logp_t = target.log_prob(flat_obs, source.actions)
        logp_b = (
            source.behavior_logprob.float()
            if self.behavior == "known"
            else cloned(flat_obs, source.actions)
        )
        rho_flat = torch.exp((logp_t - logp_b).clamp(max=30.0)).clamp(max=self.rho_clip)
        v_flat = fqe.state_value(target, flat_obs)
        q_flat = fqe.q_values(flat_obs, source.actions)
        r_flat = source.rewards.float()

        dr_values = []
        for rho, v, q, rewards in zip(
            torch.split(rho_flat, lengths),
            torch.split(v_flat, lengths),
            torch.split(q_flat, lengths),
            torch.split(r_flat, lengths),
        ):
            dr = torch.tensor(0.0, device=flat_obs.device)
            for t in reversed(range(len(rewards))):
                dr = v[t] + rho[t] * (rewards[t] + self.gamma * dr - q[t])
            dr_values.append(dr)
        return _result(torch.stack(dr_values), self.name)

from __future__ import annotations

import abc
from typing import Dict, Optional, Tuple

import torch

from src.rl.base import Algorithm
from src.rl.on_policy.base_actor_critic import RolloutBatch

# The canonical batch type for on-policy experience. Offline sources (Phase 3)
# will reuse the same container so algorithms stay source-agnostic.
Batch = RolloutBatch


class ExperienceSource(abc.ABC):
    """Where experience comes from: online interaction or a logged dataset.

    Online sources implement ``rollout`` (agent-controlled actions =
    interventions); offline dataset sources (Phase 3) implement ``sample`` and
    ``as_mdpdataset``. ``behavior_logprob`` exposes logged propensities when
    the dataset's behavior policy is known; ``None`` otherwise.
    """

    is_online: bool

    def rollout(self, policy, agent, *, n_steps: int, n_envs: int):
        raise NotImplementedError(
            f"{type(self).__name__} does not support online rollouts."
        )

    def sample(self, batch_size: int):
        raise NotImplementedError(
            f"{type(self).__name__} does not support dataset sampling."
        )

    def as_mdpdataset(self):
        """d3rlpy adapter; implemented by OfflineDatasetSource in Phase 3."""
        raise NotImplementedError(
            f"{type(self).__name__} has no MDPDataset representation."
        )

    @property
    def behavior_logprob(self) -> Optional[torch.Tensor]:
        return None


def validate_pairing(paradigm: str, source: ExperienceSource) -> None:
    """Config-time check that an algorithm paradigm matches its source.

    Raises ValueError for invalid combinations, e.g. an on-policy algorithm
    fed from an offline dataset.
    """
    if paradigm == "on_policy" and not source.is_online:
        raise ValueError(
            "on_policy algorithms require an online experience source; "
            f"got {type(source).__name__}."
        )
    if paradigm == "offline" and source.is_online:
        raise ValueError(
            "offline algorithms must consume an offline dataset source; "
            f"got {type(source).__name__}."
        )


class OfflineDatasetSource(ExperienceSource):
    """A fixed logged dataset (Cells 3–8).

    Episode structure is preserved (IPW needs per-episode importance
    products); flat tensors are derived for batch sampling. ``behavior_logprob``
    is exposed ONLY when the cell declares the behavior policy known —
    constructing with ``behavior_policy="unknown"`` discards logged
    propensities at load time, so downstream estimators cannot peek.
    """

    is_online = False

    def __init__(
        self,
        episodes: list,
        device: torch.device,
        behavior_policy: str = "known",
        rng_seed: int = 0,
    ) -> None:
        """``episodes``: list of dicts with keys ``obs [T+1, d]``,
        ``actions [T, ...]``, ``rewards [T]``, ``terminations [T]``,
        ``truncations [T]`` and optionally ``behavior_logprob [T]``
        (+ ``full_obs [T+1, D]`` when the learner view is masked)."""
        if behavior_policy not in ("known", "unknown"):
            raise ValueError("behavior_policy must be 'known' or 'unknown'")
        self.device = device
        self.behavior_policy = behavior_policy
        self._g = torch.Generator().manual_seed(int(rng_seed))

        self.episodes = []
        flat = {k: [] for k in ("obs", "actions", "rewards", "next_obs", "dones")}
        flat_logp = []
        for ep in episodes:
            ep = {
                k: (
                    v.to(device)
                    if isinstance(v, torch.Tensor)
                    else torch.as_tensor(v).to(device)
                )
                for k, v in ep.items()
            }
            if behavior_policy == "unknown":
                ep.pop("behavior_logprob", None)  # discard propensities
            self.episodes.append(ep)
            T = ep["rewards"].shape[0]
            flat["obs"].append(ep["obs"][:T].float())
            flat["next_obs"].append(ep["obs"][1 : T + 1].float())
            flat["actions"].append(ep["actions"])
            flat["rewards"].append(ep["rewards"].float())
            done = (ep["terminations"] | ep["truncations"]).float()
            flat["dones"].append(done)
            if "behavior_logprob" in ep:
                flat_logp.append(ep["behavior_logprob"].float())
        self.obs = torch.cat(flat["obs"])
        self.next_obs = torch.cat(flat["next_obs"])
        self.actions = torch.cat(flat["actions"])
        self.rewards = torch.cat(flat["rewards"])
        self.dones = torch.cat(flat["dones"])
        self._behavior_logprob = torch.cat(flat_logp) if flat_logp else None
        if (
            behavior_policy == "known"
            and self._behavior_logprob is not None
            and self._behavior_logprob.shape[0] != self.rewards.shape[0]
        ):
            raise ValueError("behavior_logprob must align with rewards.")

    def __len__(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def behavior_logprob(self) -> Optional[torch.Tensor]:
        if self.behavior_policy == "unknown":
            return None
        return self._behavior_logprob

    def episode_returns(self) -> torch.Tensor:
        return torch.stack([ep["rewards"].sum() for ep in self.episodes]).float()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, len(self), (batch_size,), generator=self._g).to(
            self.device
        )
        batch = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }
        if self.behavior_logprob is not None:
            batch["behavior_logprob"] = self.behavior_logprob[idx]
        return batch

    def as_mdpdataset(self):
        """d3rlpy MDPDataset adapter (built lazily; d3rlpy import deferred)."""
        import numpy as np
        from d3rlpy.dataset import MDPDataset

        obs, acts, rews, terms, tos = [], [], [], [], []
        for ep in self.episodes:
            T = ep["rewards"].shape[0]
            obs.append(ep["obs"][:T].cpu().numpy().astype(np.float32))
            a = ep["actions"].cpu().numpy()
            acts.append(a.reshape(T, -1) if a.ndim > 1 else a.reshape(T, 1))
            rews.append(ep["rewards"].cpu().numpy().reshape(T, 1).astype(np.float32))
            terms.append(ep["terminations"].cpu().numpy().astype(np.float32))
            tos.append(ep["truncations"].cpu().numpy().astype(np.float32))
        return MDPDataset(
            observations=np.concatenate(obs),
            actions=np.concatenate(acts),
            rewards=np.concatenate(rews),
            terminals=np.concatenate(terms),
            timeouts=np.concatenate(tos),
        )


class OnlineSource(ExperienceSource):
    """Online interaction with a (vectorized) environment.

    The two collection loops below are the pre-refactor
    ``BenchmarkRunner._collect_on_policy`` body and the inner loop of
    ``BenchmarkRunner._train_off_policy``, relocated VERBATIM (§3.2 of the
    refactor contract): same statement order, same seeding / ``env.step`` /
    sampling call order, so RNG consumption is bit-identical to master.
    Do not "clean up" these loops without regenerating the golden files.
    """

    is_online = True

    def __init__(self, env, device: torch.device) -> None:
        self.env = env
        self.device = device

    def rollout(
        self,
        policy,
        agent: Algorithm,
        *,
        n_steps: int,
        n_envs: int,
    ) -> Tuple[RolloutBatch, torch.Tensor]:
        """On-policy rollout; returns the GAE-annotated batch and per-env
        episode returns (aggregation stays with the caller)."""
        T = n_steps
        N = n_envs
        obs_buf = []
        next_obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = torch.zeros(T, N, device=self.device)
        done_buf = torch.zeros(T, N, device=self.device)
        val_buf = torch.zeros(T, N, device=self.device)
        next_val_buf = torch.zeros(T, N, device=self.device)

        obs, _ = self.env.reset()
        ep_returns = torch.zeros(N, device=self.device)
        for t in range(T):
            # Collect rollout with detached storage: learning steps will recompute fresh graphs.
            with torch.no_grad():
                val = policy.value(obs)
                action, logp = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = torch.logical_or(terminated, truncated).float()
            ep_returns += reward

            obs_buf.append(obs)
            next_obs_buf.append(next_obs)
            act_buf.append(action.detach())
            logp_buf.append(logp.detach())
            rew_buf[t] = reward
            done_buf[t] = done
            val_buf[t] = val.detach()
            # estimate next value
            with torch.no_grad():
                next_val_buf[t] = policy.value(next_obs).detach()

            obs = next_obs

        advantages, returns = agent.compute_gae(
            rew_buf, done_buf, val_buf, next_val_buf
        )
        batch = RolloutBatch(
            obs=torch.stack(obs_buf).reshape(T * N, -1),
            next_obs=torch.stack(next_obs_buf).reshape(T * N, -1),
            actions=(
                torch.stack(act_buf).reshape(T * N, -1)
                if act_buf[0].ndim > 1
                else torch.stack(act_buf).reshape(T * N)
            ),
            log_probs=torch.stack(logp_buf).reshape(T * N),
            rewards=rew_buf.reshape(T * N),
            dones=done_buf.reshape(T * N),
            values=val_buf.reshape(T * N),
            next_values=next_val_buf.reshape(T * N),
            advantages=advantages.reshape(T * N),
            returns=returns.reshape(T * N),
        )
        return batch, ep_returns

    def collect_off_policy(
        self,
        agent: Algorithm,
        replay_buffer,
        obs: torch.Tensor,
        *,
        behavior_policy,
        n_steps: int,
        n_envs: int,
        action_type: str,
        warmup: int,
        batch_size: int,
        metrics_cache: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """One episode worth of interleaved off-policy collection + updates;
        returns the carried observation and the latest update metrics.

        Actions come from ``behavior_policy`` (the collection seam). The default
        ``AgentBehaviorPolicy`` delegates to ``agent.act(obs)`` verbatim, so RNG
        consumption is identical to the pre-seam loop and golden stays bitwise.
        """
        for _ in range(n_steps):
            actions = behavior_policy.act(obs).action
            next_obs, reward, terminated, truncated, _ = self.env.step(actions)
            done = torch.logical_or(terminated, truncated).float()
            # store each env transition separately
            for i in range(n_envs):
                replay_buffer.add(
                    {
                        "obs": obs[i].detach(),
                        "actions": actions[i].detach(),
                        "rewards": reward[i].detach(),
                        "next_obs": next_obs[i].detach(),
                        "dones": done[i].detach(),
                    }
                )
            obs = next_obs

            if len(replay_buffer) > max(warmup, batch_size):
                batch = replay_buffer.sample(batch_size)
                metrics = agent.update(batch)
                metrics_cache = metrics
        return obs, metrics_cache

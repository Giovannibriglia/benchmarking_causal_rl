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
        n_steps: int,
        n_envs: int,
        action_type: str,
        warmup: int,
        batch_size: int,
        metrics_cache: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """One episode worth of interleaved off-policy collection + updates;
        returns the carried observation and the latest update metrics."""
        for _ in range(n_steps):
            if action_type == "discrete":
                actions = agent.act(obs).action
            else:
                actions = agent.act(obs).action
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

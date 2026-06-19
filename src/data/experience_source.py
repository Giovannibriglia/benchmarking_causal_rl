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


def validate_pairing(
    paradigm: str, source: ExperienceSource, *, data_regime: str = "online"
) -> None:
    """Config-time check that an algorithm paradigm matches its source.

    Raises ValueError for invalid combinations, e.g. an on-policy algorithm
    fed from an offline dataset. ``data_regime`` ("online"/"offline") is the
    spec's data-source axis: an offline regime requires an off-policy learner
    (DQN/BCQ/CQL/IQL), so ``data_regime="offline"`` with an ``on_policy``
    paradigm is rejected. Default leaves existing two-arg calls unchanged.
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
    if data_regime == "offline" and paradigm == "on_policy":
        raise ValueError(
            "data_regime='offline' requires an off_policy algorithm; "
            "on_policy is incompatible with offline (fixed-dataset) training."
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
            # flatten(0, 1) collapses (T, N) -> (T*N) while PRESERVING the obs
            # feature dims: for vector obs (T, N, D) -> (T*N, D), byte-identical
            # to the previous reshape(T*N, -1) (golden bitwise); for image obs
            # (T, N, C, H, W) -> (T*N, C, H, W), so the CNN encoder gets proper
            # (B, C, H, W) tensors instead of a flattened (B, C*H*W).
            obs=torch.stack(obs_buf).flatten(0, 1),
            next_obs=torch.stack(next_obs_buf).flatten(0, 1),
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

    def rollout_recurrent(
        self,
        policy,
        agent: Algorithm,
        *,
        n_steps: int,
        n_envs: int,
    ) -> Tuple[RolloutBatch, torch.Tensor]:
        """Recurrent on-policy rollout (separate code path from ``rollout`` so the
        verbatim, golden-pinned MLP loop above is untouched).

        Threads per-env hidden state through ``policy.act_step`` across steps,
        resets the state per-env on episode boundaries (``done``), and stores the
        flat (T*N) fields PLUS the additive ``recurrent_states`` (rollout-start
        state) and ``recurrent_seq_shape`` = (T, N) for the BPTT update. GAE's
        next-value bootstrap uses the post-step critic state (masked on done, so
        only the within-episode V(s') matters)."""
        T, N = n_steps, n_envs
        obs_buf, next_obs_buf, act_buf, logp_buf = [], [], [], []
        rew_buf = torch.zeros(T, N, device=self.device)
        done_buf = torch.zeros(T, N, device=self.device)
        val_buf = torch.zeros(T, N, device=self.device)
        next_val_buf = torch.zeros(T, N, device=self.device)

        obs, _ = self.env.reset()
        ep_returns = torch.zeros(N, device=self.device)
        # Rollout-start hidden state (zeros). Stored for the BPTT update's
        # sequence start; episode_starts also zero per-env at t=0 on the update.
        init_state = policy.initial_state(N, device=self.device)
        state = policy.initial_state(N, device=self.device)
        for t in range(T):
            with torch.no_grad():
                action, logp, val, new_state = policy.act_step(obs, state)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = torch.logical_or(terminated, truncated).float()
            ep_returns += reward
            with torch.no_grad():
                next_val, _ = policy.value_step(next_obs, new_state["critic"])

            obs_buf.append(obs)
            next_obs_buf.append(next_obs)
            act_buf.append(action.detach())
            logp_buf.append(logp.detach())
            rew_buf[t] = reward
            done_buf[t] = done
            val_buf[t] = val.detach()
            next_val_buf[t] = next_val.detach()

            # Reset per-env hidden state where the episode just ended, so the
            # next step starts the new (autoreset) episode from zeros.
            state = policy.reset_state_where(new_state, done.bool())
            obs = next_obs

        advantages, returns = agent.compute_gae(
            rew_buf, done_buf, val_buf, next_val_buf
        )
        batch = RolloutBatch(
            obs=torch.stack(obs_buf).flatten(0, 1),
            next_obs=torch.stack(next_obs_buf).flatten(0, 1),
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
            recurrent_states=init_state,
            recurrent_seq_shape=(T, N),
        )
        return batch, ep_returns

    def collect_off_policy(
        self,
        agent: Algorithm,
        replay_buffer,
        obs: torch.Tensor,
        *,
        collection_policy,
        n_steps: int,
        n_envs: int,
        warmup: int,
        batch_size: int,
        metrics_cache: Optional[Dict[str, float]] = None,
        aux_models=None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]], Optional[dict]]:
        """One episode worth of interleaved off-policy collection + updates;
        returns the carried observation, the latest update metrics, and the
        last sampled batch (for aux checkpoint logging without re-sampling).

        Actions come from ``collection_policy`` (the collection seam; named to
        avoid colliding with the ``OfflineDatasetSource.behavior_policy``
        known/unknown string flag in this module). The default
        ``AgentBehaviorPolicy`` delegates to ``agent.act(obs)`` verbatim, so RNG
        consumption is identical to the pre-seam loop and golden stays bitwise.
        """
        last_batch = None
        for _ in range(n_steps):
            actions = collection_policy.act(obs).action
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
                # aux models train on the SAME sampled batch (no new sampling,
                # no global-RNG draws) — keeps off-policy golden bitwise.
                if aux_models is not None:
                    aux_models.update(batch)
                last_batch = batch
        return obs, metrics_cache, last_batch

    def collect_off_policy_recurrent(
        self,
        agent: Algorithm,
        replay_buffer,
        obs: torch.Tensor,
        *,
        collection_policy=None,
        n_steps: int,
        n_envs: int,
        warmup: int,
        batch_size: int,
        seq_len: int = 8,
        metrics_cache: Optional[Dict[str, float]] = None,
        aux_models=None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]], Optional[dict]]:
        """Recurrent off-policy collection + update (parallel to the flat
        ``collect_off_policy``, which stays untouched/golden-pinned — same split
        as PR #49's ``rollout`` vs ``rollout_recurrent``).

        Per step: the agent's forward (``agent.act(obs, state)``) ADVANCES the
        per-env hidden state through the actual observations; the action stored/
        taken comes from the behavior ``collection_policy`` if one is configured
        (curiosity / anti_reward / bias_confounded override the agent's action),
        else the agent's own action. Transitions are stored episode-structured
        (``add(env_id, …)`` + ``mark_episode_end`` on done) and the hidden state
        is reset per-env at episode boundaries.

        Once warmed up and the buffer holds an episode >= ``seq_len``, each step
        samples a contiguous within-episode batch (``sample_sequences``) and runs
        ``agent.update`` — the recurrent sequence/BPTT update. aux_models train on
        the same sampled batch (no extra sampling)."""
        state = (
            agent.initial_state(n_envs, device=self.device)
            if hasattr(agent, "initial_state")
            else None
        )
        last_batch = None
        for _ in range(n_steps):
            # Agent forward advances hidden state through the observation.
            out = agent.act(obs, state)
            new_state = out.state
            # Behavior policy may override the emitted action (state still
            # evolved through the agent's forward on the real obs).
            if collection_policy is not None:
                actions = collection_policy.act(obs).action
            else:
                actions = out.action
            next_obs, reward, terminated, truncated, _ = self.env.step(actions)
            done = torch.logical_or(terminated, truncated).float()
            for i in range(n_envs):
                replay_buffer.add(
                    i,
                    {
                        "obs": obs[i].detach(),
                        "actions": actions[i].detach(),
                        "rewards": reward[i].detach(),
                        "next_obs": next_obs[i].detach(),
                        "dones": done[i].detach(),
                    },
                )
                if bool(done[i]):
                    replay_buffer.mark_episode_end(i)
            # Reset per-env hidden state at episode boundaries.
            if new_state is not None and hasattr(agent, "reset_state_where"):
                new_state = agent.reset_state_where(new_state, done.bool())
            state = new_state
            obs = next_obs

            # Sequence update once warmed up and a long-enough episode exists.
            if len(replay_buffer) > max(
                warmup, batch_size
            ) and replay_buffer.can_sample(seq_len):
                batch = replay_buffer.sample_sequences(batch_size, seq_len)
                metrics_cache = agent.update(batch)
                if aux_models is not None:
                    aux_models.update(batch)
                last_batch = batch
        return obs, metrics_cache, last_batch

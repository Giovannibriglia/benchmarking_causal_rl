"""Minari load/save/convert utilities (§6.3).

Conventions (load-bearing — Minari does not store propensities natively):

* Collection ALWAYS stores FULL observations; partial observability is
  applied at LOAD time via ``mask_indices`` (per-cell switches act on the
  learner view, never on the stored data).
* ``infos["behavior_logprob"]`` carries the logging policy's exact
  propensity for the taken action at every step (0.0 at the reset step to
  keep the info structure consistent — Minari requires identical info keys
  on every step).
* Dataset naming: ``causal/<env>/<tag>-v<version>``
  (e.g. ``causal/cartpole/medium-v0``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from minari import DataCollector, StepDataCallback

from src.data.behavior_policies import BehaviorPolicy
from src.data.experience_source import OfflineDatasetSource


class PropensityStepDataCallback(StepDataCallback):
    """Writes the logging policy's exact propensity into step infos.

    The collection loop stashes the current step's log-propensity in the
    class-level holder before calling ``env.step``; the callback moves it
    into ``step_data['info']``. Reset steps get 0.0 so the info structure
    is identical at every step.
    """

    current_logprob: float = 0.0

    def __call__(
        self, env, obs, info, action=None, rew=None, terminated=None, truncated=None
    ):
        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)
        info = dict(step_data["info"] or {})
        info["behavior_logprob"] = np.float64(
            0.0 if action is None else type(self).current_logprob
        )
        step_data["info"] = info
        return step_data


def collect_dataset(
    env_id: str,
    behavior_policy: BehaviorPolicy,
    dataset_id: str,
    n_episodes: int,
    seed: int,
    device: torch.device,
    max_steps_per_episode: Optional[int] = None,
    env_factory=None,
) -> None:
    """Roll ``behavior_policy`` in ``env_id`` (or ``env_factory()``, e.g. a
    ConfoundedEnv wrap) and save a Minari dataset with exact propensities in
    infos. Overwrites an existing local dataset id."""
    import minari

    if dataset_id in minari.list_local_datasets():
        minari.delete_dataset(dataset_id)

    base_env = env_factory() if env_factory is not None else gym.make(env_id)
    env = DataCollector(
        base_env,
        step_data_callback=PropensityStepDataCallback,
        record_infos=True,
    )
    for ep in range(int(n_episodes)):
        obs, info = env.reset(seed=seed + ep)
        done = False
        t = 0
        while not done:
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device
            )
            # Confounded envs expose the per-episode U in infos; the biased
            # logging policy needs it at action-selection time (U -> action).
            latent = None
            if isinstance(info, dict) and "confounder_u" in info:
                latent = torch.tensor([float(info["confounder_u"])], device=device)
            action, logp = behavior_policy.select_action(obs_t, latent=latent)
            PropensityStepDataCallback.current_logprob = float(logp.item())
            a = action.squeeze(0).detach().cpu().numpy()
            if a.ndim == 0:
                a = a.item()
            obs, _, terminated, truncated, info = env.step(a)
            t += 1
            done = terminated or truncated
            if max_steps_per_episode is not None and t >= max_steps_per_episode:
                break
    env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name=type(behavior_policy).__name__,
        description="Collected with exact behavior_logprob in infos "
        "(benchmarking_causal_rl, causal cells).",
    )
    env.close()


def to_offline_source(
    dataset_id: str,
    device: torch.device,
    behavior_policy: str = "known",
    mask_indices: Optional[Sequence[int]] = None,
    max_episodes: Optional[int] = None,
    rng_seed: int = 0,
) -> OfflineDatasetSource:
    """Load a Minari dataset into an :class:`OfflineDatasetSource`.

    ``behavior_policy="unknown"`` discards logged propensities (Cells 5/6/8).
    ``mask_indices`` applies learner-input masking (the source keeps the full
    observations under ``full_obs``).
    """
    import minari

    ds = minari.load_dataset(dataset_id)
    episodes = []
    for i, ep in enumerate(ds.iterate_episodes()):
        if max_episodes is not None and i >= max_episodes:
            break
        obs = torch.as_tensor(np.asarray(ep.observations, dtype=np.float32))
        entry = {
            "obs": obs,
            "actions": torch.as_tensor(np.asarray(ep.actions)),
            "rewards": torch.as_tensor(np.asarray(ep.rewards, dtype=np.float32)),
            "terminations": torch.as_tensor(np.asarray(ep.terminations, dtype=bool)),
            "truncations": torch.as_tensor(np.asarray(ep.truncations, dtype=bool)),
        }
        infos = ep.infos or {}
        if "behavior_logprob" in infos:
            # infos arrays have T+1 entries (reset + steps); step t's
            # propensity was recorded at the post-step index t+1.
            logp = np.asarray(infos["behavior_logprob"], dtype=np.float32)
            entry["behavior_logprob"] = torch.as_tensor(logp[1:])
        if "confounder_u" in infos:
            # per-episode U (constant across the episode); kept for the
            # assert_confounded gate ONLY - sample() never exposes it.
            u = np.asarray(infos["confounder_u"], dtype=np.float32)
            entry["confounder_u"] = torch.as_tensor(u[1:])
        if mask_indices is not None:
            keep = [j for j in range(obs.shape[-1]) if j not in set(mask_indices)]
            entry["full_obs"] = obs
            entry["obs"] = obs[:, keep]
        episodes.append(entry)
    return OfflineDatasetSource(
        episodes, device=device, behavior_policy=behavior_policy, rng_seed=rng_seed
    )

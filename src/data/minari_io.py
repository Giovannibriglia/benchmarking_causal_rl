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
    collection_config: Optional[dict] = None,
    force: bool = False,
) -> None:
    """Roll ``behavior_policy`` in ``env_id`` (or ``env_factory()``, e.g. a
    ConfoundedEnv wrap) and save a Minari dataset with exact propensities in
    infos.

    OVERWRITE GUARD (Phase-6A gate condition 2): an existing dataset id is a
    HARD ERROR unless ``force=True`` — any collection-parameter change must
    bump the version suffix instead. The full collection config is embedded
    in the Minari description as JSON and asserted at load time.
    """
    import json as _json

    import minari

    if dataset_id in minari.list_local_datasets():
        if not force:
            raise FileExistsError(
                f"Minari dataset '{dataset_id}' already exists. Datasets are "
                "content-versioned: bump the version suffix for changed "
                "collection parameters, or pass force=True to overwrite "
                "deliberately."
            )
        minari.delete_dataset(dataset_id)
    cfg = dict(collection_config or {})
    cfg.setdefault("env_id", env_id)
    cfg.setdefault("n_episodes", int(n_episodes))
    cfg.setdefault("seed", int(seed))
    cfg.setdefault("policy", type(behavior_policy).__name__)

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
        "(benchmarking_causal_rl, causal cells). "
        f"|collection_config={_json.dumps(cfg, sort_keys=True)}",
    )
    env.close()


def read_collection_config(dataset_id: str) -> Optional[dict]:
    """Parse the collection config embedded in the Minari description."""
    import json as _json

    import minari

    ds = minari.load_dataset(dataset_id)
    desc = ""
    storage = getattr(ds, "storage", None)
    if storage is not None:
        desc = str(storage.metadata.get("description", "") or "")
    if not desc:
        desc = str(getattr(ds.spec, "description", "") or "")
    marker = "|collection_config="
    if marker not in desc:
        return None
    return _json.loads(desc.split(marker, 1)[1])


def assert_collection_config(dataset_id: str, expect: dict) -> None:
    """Assert the dataset's embedded collection config matches the cell
    YAML's expectations (subset match). Hard error on mismatch or when the
    dataset predates config embedding."""
    cfg = read_collection_config(dataset_id)
    if cfg is None:
        raise AssertionError(
            f"dataset '{dataset_id}' has no embedded collection config; "
            "recollect it with the guarded collector before pinning "
            "expectations."
        )
    for key, val in expect.items():
        if key not in cfg or cfg[key] != val:
            raise AssertionError(
                f"dataset '{dataset_id}' collection config mismatch on "
                f"'{key}': expected {val!r}, recorded {cfg.get(key)!r}."
            )


def to_offline_source(
    dataset_id: str,
    device: torch.device,
    behavior_policy: str = "known",
    mask_indices: Optional[Sequence[int]] = None,
    max_episodes: Optional[int] = None,
    rng_seed: int = 0,
    expect: Optional[dict] = None,
) -> OfflineDatasetSource:
    """Load a Minari dataset into an :class:`OfflineDatasetSource`.

    ``behavior_policy="unknown"`` discards logged propensities (Cells 5/6/8).
    ``mask_indices`` applies learner-input masking (the source keeps the full
    observations under ``full_obs``).
    """
    import minari

    if expect:
        assert_collection_config(dataset_id, expect)
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

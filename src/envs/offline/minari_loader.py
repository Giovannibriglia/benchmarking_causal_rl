from __future__ import annotations

import numpy as np
import torch

from src.envs.wrappers.atari import normalize_image_obs


def load_minari_dataset(dataset_id: str):
    """Load a Minari dataset, downloading from the remote ONLY if it is not
    already cached locally.

    Checking ``list_local_datasets()`` first guarantees a local-only fixture
    (e.g. ``make_cartpole_offline``, which exists only in the local cache and is
    NOT on the Farama remote) never contacts the network — so the offline tests
    run fully offline. Hosted ids that aren't cached are downloaded once
    (``download=True``); a bad/unavailable id surfaces a clear error.
    """
    import minari

    if dataset_id in minari.list_local_datasets():
        return minari.load_dataset(dataset_id)
    try:
        return minari.load_dataset(dataset_id, download=True)
    except Exception as exc:  # pragma: no cover - network / availability
        raise RuntimeError(
            f"Could not load Minari dataset '{dataset_id}': it is not in the "
            f"local cache and the download failed ({type(exc).__name__}: {exc}). "
            "Check the id against `minari list remote`."
        ) from exc


def dataset_action_type(action_space) -> str:
    """``'discrete'`` for a ``Discrete`` action space (has ``.n``), else
    ``'continuous'`` — the dataset-side analogue of the runner's env-derived
    ``action_type``."""
    return "discrete" if hasattr(action_space, "n") else "continuous"


def assert_dataset_matches_algo(
    dataset, action_type: str, dataset_id: str, algorithm: str
) -> None:
    """Reject a discrete/continuous category error BEFORE training.

    Resolves the loaded dataset's action space (crux A: from the dataset's
    metadata, NOT a live env) and verifies it matches the consuming algorithm's
    action type. Catches e.g. a continuous hosted dataset paired with a discrete
    ``offline_dqn`` here, with a clear message, rather than as a tensor-shape
    crash mid-fill.
    """
    ds_type = dataset_action_type(dataset.action_space)
    if ds_type != action_type:
        raise ValueError(
            f"Offline dataset '{dataset_id}' has a {ds_type} action space "
            f"({dataset.action_space}), but algorithm '{algorithm}' is "
            f"{action_type}-only. Pair a {action_type} offline dataset with "
            f"'{algorithm}', or use a {ds_type} offline algorithm."
        )


def fill_replay_buffer_from_minari(
    dataset_id: str,
    buffer,
    device: torch.device,
) -> int:
    """Load a Minari dataset and fill ``buffer`` with its transitions.

    Reuses the existing ``ReplayBuffer`` API unchanged: each dataset transition
    is added as ``{obs, actions, rewards, next_obs, dones}`` via ``buffer.add``
    — the same dict shape the online off-policy collector produces, so the
    agent's batched ``update`` is identical offline. ``next_obs`` is the
    episode's next observation; ``dones = terminations | truncations``.

    This is a one-time setup loop (per-transition Python is fine here); the
    training hot path stays batched (``buffer.sample`` -> ``agent.update``).
    Returns the number of transitions added.
    """
    dataset = load_minari_dataset(dataset_id)
    n_added = 0
    for episode in dataset.iterate_episodes():
        raw_obs = np.asarray(episode.observations)
        # Image obs ((T+1, C, H, W) uint8) go through the SAME normalizer the
        # online wrapper uses (float CHW /255), so an offline frame is
        # byte-identical to the online representation. Vector obs ((T+1, D))
        # keep the original float path unchanged.
        if raw_obs.ndim == 4:
            obs = normalize_image_obs(raw_obs, device).cpu()
        else:
            obs = torch.as_tensor(raw_obs, dtype=torch.float32)
        actions = torch.as_tensor(episode.actions)
        rewards = torch.as_tensor(episode.rewards, dtype=torch.float32)
        terminations = torch.as_tensor(episode.terminations, dtype=torch.bool)
        truncations = torch.as_tensor(episode.truncations, dtype=torch.bool)
        dones = (terminations | truncations).float()
        steps = rewards.shape[0]
        for t in range(steps):
            buffer.add(
                {
                    "obs": obs[t],
                    "actions": actions[t],
                    "rewards": rewards[t],
                    "next_obs": obs[t + 1],
                    "dones": dones[t],
                }
            )
            n_added += 1
    return n_added

from __future__ import annotations

import numpy as np
import torch

from src.envs.wrappers.atari import normalize_image_obs


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
    import minari

    dataset = minari.load_dataset(dataset_id)
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

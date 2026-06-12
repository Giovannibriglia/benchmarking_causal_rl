#!/usr/bin/env python3
"""Generate a tiny Atari (ALE/Pong-v5) offline dataset (random policy) via Minari.

The acceptance / test fixture for offline IMAGE training. Frames are produced
through the SHARED ``make_atari_env`` chain (the same one the online wrapper
uses), so a stored frame is bit-for-bit the representation the online path
delivers — the offline loader's normalizer (``normalize_image_obs``) closes the
loop. Set ``MINARI_DATASETS_PATH`` to control where the dataset is written (the
test isolates it to a tmp dir so it stays hermetic and never pollutes ~/.minari).

    MINARI_DATASETS_PATH=/tmp/minari python tools/make_atari_offline.py

Builds ``EpisodeBuffer`` objects directly and calls
``create_dataset_from_buffers`` (rather than the ``DataCollector``, which pulls
in jax via per-step tree flattening) — so only ``minari[hdf5]`` is required.
Observations are stored as ``(4, 84, 84)`` uint8 (exactly what
``AtariPreprocessing(scale_obs=False)`` emits), keeping normalization a
load-time concern.
"""

from __future__ import annotations

import argparse

import minari
import numpy as np
from minari.data_collector.episode_buffer import EpisodeBuffer
from src.envs.wrappers.atari import make_atari_env


def make_atari_dataset(
    dataset_id: str = "atari/pong-random-v0",
    n_episodes: int = 3,
    seed: int = 0,
    env_id: str = "ALE/Pong-v5",
    max_steps: int = 50,
):
    """Collect ``n_episodes`` of an Atari game into a Minari dataset.

    Random policy; each episode is capped at ``max_steps`` (real Atari episodes
    run thousands of steps — the cap keeps the fixture tiny). Returns the created
    MinariDataset. Honors ``MINARI_DATASETS_PATH``.
    """
    env = make_atari_env(env_id)
    rng = np.random.default_rng(seed)
    n_actions = int(env.action_space.n)
    buffers = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        observations = [np.asarray(obs, dtype=np.uint8)]
        actions, rewards, terminations, truncations = [], [], [], []
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = int(rng.integers(0, n_actions))
            obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(np.asarray(obs, dtype=np.uint8))
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            steps += 1
            done = terminated or truncated
        # Cap-induced cutoff is a truncation, not a termination.
        if steps >= max_steps and not (terminations[-1] or truncations[-1]):
            truncations[-1] = True
        buffers.append(
            EpisodeBuffer(
                observations=np.asarray(observations, dtype=np.uint8),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminations=np.asarray(terminations, dtype=bool),
                truncations=np.asarray(truncations, dtype=bool),
            )
        )
    env.close()
    return minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        buffer=buffers,
        env=make_atari_env(env_id),
        algorithm_name="random",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-id", default="atari/pong-random-v0")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-id", default="ALE/Pong-v5")
    p.add_argument("--max-steps", type=int, default=50)
    args = p.parse_args()
    ds = make_atari_dataset(
        args.dataset_id, args.n_episodes, args.seed, args.env_id, args.max_steps
    )
    print(f"created {ds.id}: {ds.total_steps} steps, {ds.total_episodes} episodes")


if __name__ == "__main__":
    main()

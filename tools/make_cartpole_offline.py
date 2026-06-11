#!/usr/bin/env python3
"""Generate a tiny CartPole offline dataset (random policy) via Minari.

Used as the acceptance / test fixture for offline training. Set
``MINARI_DATASETS_PATH`` to control where the dataset is written (the test
isolates it to a tmp dir so it stays hermetic and does not pollute ~/.minari).

    MINARI_DATASETS_PATH=/tmp/minari python tools/make_cartpole_offline.py

Builds ``EpisodeBuffer`` objects directly and calls
``create_dataset_from_buffers`` (rather than the ``DataCollector``, which pulls
in jax via per-step tree flattening) — so only ``minari[hdf5]`` is required.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import minari
import numpy as np
from minari.data_collector.episode_buffer import EpisodeBuffer


def make_cartpole_dataset(
    dataset_id: str = "cartpole/random-v0",
    n_episodes: int = 20,
    seed: int = 0,
):
    """Collect ``n_episodes`` of random-policy CartPole-v1 into a Minari dataset.

    Returns the created MinariDataset. Honors ``MINARI_DATASETS_PATH``.
    """
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    buffers = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        observations = [obs]
        actions, rewards, terminations, truncations = [], [], [], []
        done = False
        while not done:
            action = int(rng.integers(0, 2))
            obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            done = terminated or truncated
        buffers.append(
            EpisodeBuffer(
                observations=np.asarray(observations, dtype=np.float32),
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
        env=gym.make("CartPole-v1"),
        algorithm_name="random",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-id", default="cartpole/random-v0")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ds = make_cartpole_dataset(args.dataset_id, args.n_episodes, args.seed)
    print(f"created {ds.id}: {ds.total_steps} steps, {ds.total_episodes} episodes")


if __name__ == "__main__":
    main()

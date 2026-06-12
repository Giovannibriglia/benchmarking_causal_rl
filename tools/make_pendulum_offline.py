#!/usr/bin/env python3
"""Generate a tiny Pendulum offline dataset via Minari (continuous actions).

The acceptance / test fixture for CONTINUOUS offline training (CQL-on-SAC,
IQL-Gaussian). Set ``MINARI_DATASETS_PATH`` to control where the dataset is
written (the test isolates it to a tmp dir so it stays hermetic).

    MINARI_DATASETS_PATH=/tmp/minari python tools/make_pendulum_offline.py

Builds ``EpisodeBuffer`` objects directly and calls
``create_dataset_from_buffers`` (NO jax/DataCollector) — only ``minari[hdf5]``
required. Mirrors ``tools/make_cartpole_offline.py`` for the continuous axis.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import minari
import numpy as np
from minari.data_collector.episode_buffer import EpisodeBuffer

_MAX_TORQUE = 2.0


def _heuristic_action(obs, rng) -> np.ndarray:
    """Energy-shaping swing-up + PD balance — a better-than-random policy.

    obs = [cos(theta), sin(theta), theta_dot], theta measured from upright.
    Near the top (cos > 0) apply a PD law that drives angle/rate to zero; in the
    lower half pump energy in the direction of motion. A little exploration
    noise keeps the dataset from collapsing onto one trajectory.
    """
    cos_th, sin_th, thdot = float(obs[0]), float(obs[1]), float(obs[2])
    theta = np.arctan2(sin_th, cos_th)
    if cos_th > 0.0:
        u = -(10.0 * theta + 2.0 * thdot)
    else:
        u = _MAX_TORQUE if thdot >= 0.0 else -_MAX_TORQUE
    u += rng.normal(0.0, 0.3)
    return np.clip([u], -_MAX_TORQUE, _MAX_TORQUE).astype(np.float32)


def make_pendulum_dataset(
    dataset_id: str = "pendulum/random-v0",
    n_episodes: int = 20,
    seed: int = 0,
    policy: str = "random",
):
    """Collect ``n_episodes`` of Pendulum-v1 into a Minari dataset.

    ``policy``: ``"random"`` (uniform torque) or ``"heuristic"`` (swing-up + PD,
    for learning-sanity tests). Returns the created MinariDataset. Honors
    ``MINARI_DATASETS_PATH``.
    """
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    buffers = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        observations = [np.asarray(obs, dtype=np.float32)]
        actions, rewards, terminations, truncations = [], [], [], []
        done = False
        while not done:
            if policy == "heuristic":
                action = _heuristic_action(obs, rng)
            else:
                action = rng.uniform(-_MAX_TORQUE, _MAX_TORQUE, size=(1,)).astype(
                    np.float32
                )
            obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            done = terminated or truncated
        buffers.append(
            EpisodeBuffer(
                observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminations=np.asarray(terminations, dtype=bool),
                truncations=np.asarray(truncations, dtype=bool),
            )
        )
    env.close()
    return minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        buffer=buffers,
        env=gym.make("Pendulum-v1"),
        algorithm_name=policy,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-id", default="pendulum/random-v0")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=["random", "heuristic"], default="random")
    args = p.parse_args()
    ds = make_pendulum_dataset(args.dataset_id, args.n_episodes, args.seed, args.policy)
    print(f"created {ds.id}: {ds.total_steps} steps, {ds.total_episodes} episodes")


if __name__ == "__main__":
    main()

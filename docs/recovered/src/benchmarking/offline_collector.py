"""Offline dataset collection utilities for causal/off-policy experiments.

This module centralizes biased behavior-policy rollouts and controls whether
latent factors and behavior log-probabilities are exposed in replay, according
to the eight-cell taxonomy used by the benchmark.
"""

from __future__ import annotations

from typing import Tuple

import torch
from src.envs.causal_base import CausalEnv
from src.envs.wrappers._causal_cell import get_cell_config
from src.rl.off_policy.biased_explorer import BiasedExplorer
from src.rl.off_policy.replay_buffer import ReplayBuffer


def cell_to_switches(cell: int) -> Tuple[bool, bool, bool]:
    """Return (expose_z_u, expose_pi_b, offline)."""
    cfg = get_cell_config(cell)
    expose_latent = bool(cfg.z_exposed and cfg.u_exposed)
    return expose_latent, bool(cfg.pi_b_known), bool(cfg.offline)


def collect_offline_dataset(
    env: CausalEnv,
    explorer: BiasedExplorer,
    n_episodes: int,
    expose_pi_b: bool,
    expose_latent: bool,
    buffer: ReplayBuffer,
) -> ReplayBuffer:
    """Collect biased transitions into a replay buffer."""

    horizon = int(getattr(env, "horizon", 20))
    obs, _ = env.reset()
    for _ in range(int(n_episodes)):
        obs, _ = env.reset()
        for _ in range(horizon):
            latent_for_behavior = env.latent_state()
            action, logp = explorer.select_action(obs, latent=latent_for_behavior)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = torch.logical_or(terminated, truncated).float()

            if expose_pi_b:
                logp_store = info.get("behavior_logprob", logp)
            else:
                logp_store = None
            latent_store = latent_for_behavior if expose_latent else None

            for i in range(env.n_envs):
                buffer.add(
                    {
                        "obs": obs[i].detach(),
                        "actions": action[i].detach(),
                        "rewards": reward[i].detach(),
                        "next_obs": next_obs[i].detach(),
                        "dones": done[i].detach(),
                        "behavior_logprob": (
                            None if logp_store is None else logp_store[i].detach()
                        ),
                        "latent": (
                            None if latent_store is None else latent_store[i].detach()
                        ),
                    }
                )
            obs = next_obs
    return buffer

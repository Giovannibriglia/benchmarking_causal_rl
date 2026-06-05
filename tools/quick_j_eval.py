#!/usr/bin/env python3
"""Per-episode J evaluation of a causal-cells run's final checkpoints.

Phase-2 acceptance helper (previews the Phase-3 RegretProtocol's J
accounting): loads the last checkpoint per environment from a run directory
and evaluates TRUE per-episode returns over N full episodes — unlike the
training-time eval, which accumulates rewards over a fixed step window and
saturates for CartPole-like tasks.

Usage:
    python tools/quick_j_eval.py <run_dir> [--episodes 100] [--algo ppo]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from src.benchmarking.checkpoints import load_checkpoint
from src.envs.registry import register_default_env_wrappers
from src.rl.on_policy.policy import ActorCriticMLP


def _final_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("ckpt_ep*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    return ckpts[-1]


def evaluate(
    env_id: str,
    ckpt_path: Path,
    n_episodes: int,
    deterministic: bool,
    seed_base: int = 10_000,
) -> np.ndarray:
    device = torch.device("cpu")
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    if hasattr(env.action_space, "n"):
        action_type, action_dim = "discrete", env.action_space.n
    else:
        action_type, action_dim = "continuous", env.action_space.shape[0]
    policy = ActorCriticMLP(obs_dim, action_dim, action_type, device)
    policy.load_state_dict(load_checkpoint(str(ckpt_path))["policy_state"])

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done, total = False, 0.0
        while not done:
            t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32).reshape(-1), device=device
            ).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    action = policy.act_deterministic(t)
                else:
                    action, _ = policy.act(t)
            a = action.squeeze(0).numpy()
            if action_type == "discrete":
                a = int(a)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return np.asarray(returns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory (with checkpoints/)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    register_default_env_wrappers()
    run_dir = Path(args.run_dir)
    pattern = re.compile(rf"^(.+)_{re.escape(args.algo)}_seed{args.seed}$")
    rows = []
    for ckpt_dir in sorted((run_dir / "checkpoints").iterdir()):
        m = pattern.match(ckpt_dir.name)
        if not m:
            continue
        env_id = m.group(1).replace("causal-", "causal/", 1)
        ckpt = _final_checkpoint(ckpt_dir)
        for det in (True, False):
            r = evaluate(env_id, ckpt, args.episodes, deterministic=det)
            rows.append((env_id, "det" if det else "stoch", r.mean(), r.std()))

    width = max(len(r[0]) for r in rows)
    print(f"J over {args.episodes} full episodes (final checkpoint):")
    for env_id, mode, mean, std in rows:
        print(f"  {env_id:<{width}}  {mode:5s}  J = {mean:8.1f} +- {std:.1f}")


if __name__ == "__main__":
    main()

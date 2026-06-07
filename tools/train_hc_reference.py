#!/usr/bin/env python3
"""Train + evaluate the continuous Cell-1 SAC reference (Phase 6B).

Usage: python -m tools.train_hc_reference [--episodes 125] [--seed 0] [--tag relu]
"""

from __future__ import annotations

import argparse

import torch
from src.benchmarking.causal_cells import _load_reference_act_fn
from src.envs.registry import register_default_env_wrappers
from src.eval.references import ensure_reference
from src.eval.regret import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="relu")
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    register_default_env_wrappers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = {
        "env": "causal/halfcheetah-cell1",
        "algo": "sac",
        "tag": args.tag,
        "n_episodes": args.episodes,
        "rollout_len": 1024,
        "n_train_envs": 8,
        "n_eval_envs": 8,
        "n_checkpoints": 12,
        "deterministic": True,
    }
    path = ensure_reference(spec, seed=args.seed, device=device)
    act = _load_reference_act_fn(path, "HalfCheetah-v5", device)
    rets = evaluate_policy(
        "HalfCheetah-v5", act, n_episodes=args.eval_episodes, seed_base=70_000
    )
    print(
        f"REFERENCE-J seed={args.seed} episodes={args.episodes} "
        f"J={rets.mean():.0f} +- {rets.std():.0f}"
    )


if __name__ == "__main__":
    main()

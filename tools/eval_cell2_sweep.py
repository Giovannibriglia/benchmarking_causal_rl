#!/usr/bin/env python3
"""Per-seed J + normalized regret for the Cell-2 online sweep (Phase 7).

Reads the five benchmark runs trained by tools/run_matrix_sweep.sh (one per
seed, each containing causal/cartpole-cell2 basic and -cell2fs variant PPO
policies), evaluates TRUE per-episode J, normalizes against the per-seed
Cell-1 references (reference rows of any offline-cell sweep CSV), and writes
``outputs/cell2_summary.csv`` in the matrix row format.

Usage: python -m tools.eval_cell2_sweep <cell3_run_dir> <bench_run_seed0> ...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.benchmarking.checkpoints import load_checkpoint
from src.envs.registry import register_default_env_wrappers
from src.eval.regret import compute_regret, evaluate_policy
from src.rl.on_policy.policy import ActorCriticMLP


def _policy_act_fn(ckpt_path: Path, obs_dim: int, device):
    policy = ActorCriticMLP(obs_dim, 2, "discrete", device)
    policy.load_state_dict(load_checkpoint(str(ckpt_path))["policy_state"])

    def act(obs: np.ndarray) -> int:
        t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            return int(policy.act_deterministic(t).item())

    return act


def main() -> None:
    register_default_env_wrappers()
    device = torch.device("cpu")
    ref_csv = Path(sys.argv[1]) / "causal_cells_metrics.csv"
    bench_dirs = [Path(p) for p in sys.argv[2:]]

    refs = pd.read_csv(ref_csv)
    j_ref = {
        int(r.seed): float(r.J) for _, r in refs[refs.role == "reference"].iterrows()
    }
    j_random = float(refs[refs.role == "random"].J.iloc[0])

    rows = []
    env_specs = {
        "basic": ("causal/cartpole-cell2", "causal-cartpole-cell2", 2, "ppo_ff"),
        "variant": ("causal/cartpole-cell2fs", "causal-cartpole-cell2fs", 8, "ppo_fs"),
    }
    for bench in bench_dirs:
        ckpt_root = bench / "checkpoints"
        seed_dirs = list(ckpt_root.glob("causal-cartpole-cell2_ppo_seed*"))
        m = re.search(r"seed(\d+)$", seed_dirs[0].name)
        seed = int(m.group(1))
        for role, (eval_env, tag, obs_dim, algo) in env_specs.items():
            ckpt = sorted((ckpt_root / f"{tag}_ppo_seed{seed}").glob("ckpt_ep*.pt"))[-1]
            returns = evaluate_policy(
                eval_env,
                _policy_act_fn(ckpt, obs_dim, device),
                n_episodes=100,
                seed_base=50_000 + 100 * seed,
            )
            reg = compute_regret(float(returns.mean()), j_ref[seed], j_random)
            rows.append(
                {
                    "cell": 2,
                    "task": "CartPole-v1",
                    "anchor": "discrete",
                    "tier": "online",
                    "algo": algo,
                    "role": role,
                    "seed": seed,
                    "J": reg.j,
                    "regret": reg.regret,
                    "normalized_regret": reg.normalized_regret,
                }
            )
            print(
                f"[cell 2|seed {seed}] {role}={algo}: J={reg.j:.1f} "
                f"(norm {reg.normalized_regret:.2f})"
            )
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "cell2_summary.csv", index=False)
    print("wrote outputs/cell2_summary.csv")


if __name__ == "__main__":
    main()

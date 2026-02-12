from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.device import detect_device


def parse_args():
    p = argparse.ArgumentParser(description="Benchmarking Causal RL")
    p.add_argument(
        "--envs", nargs="+", help="List of env ids to benchmark", default=None
    )
    p.add_argument(
        "--algos", nargs="+", help="List of algorithms to benchmark", default=None
    )
    p.add_argument("--n-train-envs", type=int, default=8)
    p.add_argument("--n-eval-envs", type=int, default=8)
    p.add_argument("--rollout-len", type=int, default=1024)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--n-checkpoints", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reproduce",
        type=str,
        help="Name of reproducibility YAML in reproducibility/ without extension",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable torch deterministic algorithms",
    )
    p.add_argument(
        "--aggregation",
        choices=["iqm", "mean"],
        default="iqm",
        help="Aggregation strategy for reported stats",
    )
    return p.parse_args()


def main():
    args = parse_args()
    register_default_algorithms()

    cfg_from_file = {}
    if args.reproduce:
        repro_path = Path("reproducibility") / f"{args.reproduce}.yaml"
        cfg_from_file = yaml.safe_load(repro_path.read_text())

    envs = cfg_from_file.get("envs") or cfg_from_file.get("env_list") or args.envs
    algos = cfg_from_file.get("algos") or cfg_from_file.get("algo_list") or args.algos
    seed = cfg_from_file.get("seed", args.seed)
    n_train_envs = cfg_from_file.get("n_train_envs", args.n_train_envs)
    n_eval_envs = cfg_from_file.get("n_eval_envs", args.n_eval_envs)
    rollout_len = cfg_from_file.get("rollout_len", args.rollout_len)
    n_episodes = cfg_from_file.get("n_episodes", args.n_episodes)
    n_checkpoints = cfg_from_file.get("n_checkpoints", args.n_checkpoints)
    # Checkpoint count validation: must be in [2, n_episodes]. Behavior: clamp to range.
    if n_checkpoints < 2:
        n_checkpoints = 2
    if n_checkpoints > n_episodes:
        n_checkpoints = n_episodes
    deterministic = cfg_from_file.get("deterministic", args.deterministic)
    aggregation = cfg_from_file.get("aggregation", args.aggregation)

    n_checkpoints = max(n_checkpoints, 2)
    n_checkpoints = min(n_checkpoints, n_episodes)

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/benchmark_{base_timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    device_str = str(detect_device())

    # Save config snapshot once
    config_snapshot = {
        "envs": envs,
        "algos": algos,
        "seed": seed,
        "n_train_envs": n_train_envs,
        "n_eval_envs": n_eval_envs,
        "rollout_len": rollout_len,
        "n_episodes": n_episodes,
        "n_checkpoints": n_checkpoints,
        "deterministic": deterministic,
        "device": device_str,
        "aggregation": aggregation,
    }
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config_snapshot, f)
    with (run_dir / "metadata.json").open("w") as f:
        import json

        json.dump({"timestamp": base_timestamp}, f, indent=2)

    run_cfg = RunConfig(run_dir=str(run_dir), timestamp=base_timestamp)

    for env_id in envs:
        for algo in algos:
            env_cfg = EnvConfig(
                env_id=env_id,
                n_train_envs=n_train_envs,
                n_eval_envs=n_eval_envs,
                rollout_len=rollout_len,
                seed=seed,
            )
            train_cfg = TrainingConfig(
                n_episodes=n_episodes,
                n_checkpoints=n_checkpoints,
                deterministic=deterministic,
                device=device_str,
                algorithm=algo,
                aggregation=aggregation,
            )
            spec = registry.get(algo)
            runner = BenchmarkRunner(
                env_cfg, train_cfg, run_cfg, spec, progress_label=f"{algo} - {env_id}"
            )
            runner.run()


if __name__ == "__main__":
    main()

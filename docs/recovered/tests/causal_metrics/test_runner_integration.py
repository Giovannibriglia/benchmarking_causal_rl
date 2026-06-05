from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.envs.registry import register_default_env_wrappers


def _run_once(tmp_dir: Path, env_id: str, algo: str = "dqn") -> Path:
    run_dir = tmp_dir / env_id.replace("/", "_")
    env_cfg = EnvConfig(
        env_id=env_id,
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=16,
        seed=123,
        env_kwargs={},
    )
    train_cfg = TrainingConfig(
        n_episodes=2,
        n_checkpoints=2,
        deterministic=False,
        device="cpu",
        algorithm=algo,
        aggregation="mean",
        divergences=("tv",),
    )
    run_cfg = RunConfig(run_dir=str(run_dir), timestamp="test")
    spec = registry.get(algo)
    runner = BenchmarkRunner(env_cfg, train_cfg, run_cfg, spec)
    runner.offpolicy_warmup = 0
    runner.offpolicy_batch_size = 8
    runner.run()
    return run_dir


def test_runner_logs_delta_columns_for_causal_and_nan_for_noncausal(
    tmp_path: Path,
) -> None:
    register_default_algorithms()
    register_default_env_wrappers()

    causal_run = _run_once(tmp_path, "causal-block-mdp-cell8")
    noncausal_run = _run_once(tmp_path, "CartPole-v1")

    causal_df = pd.read_csv(causal_run / "train_metrics.csv")
    assert "delta_tv" in causal_df.columns
    assert causal_df["delta_tv"].notna().any()

    noncausal_df = pd.read_csv(noncausal_run / "train_metrics.csv")
    for col in ("delta_tv", "delta_kl", "delta_chi2", "delta_sup"):
        assert col in noncausal_df.columns
        assert noncausal_df[col].isna().all()

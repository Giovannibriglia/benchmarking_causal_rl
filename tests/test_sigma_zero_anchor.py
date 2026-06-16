"""feat/sigma-zero-anchor-bypass (PR8.5) — σ=0.0 anchor bypass for the gate-check.

The σ=0.0 confounded dataset is unconfounded by construction (marginal
Corr(A,R) ≈ 0), so PR3's gate test fails (gate_test_passed=False) and the runner
would reject it — blocking the σ=0.0 anchor of the Cell 7 σ-sweep. This PR makes
the runner skip the gate ONLY when behavior_strength_sigma == 0.0; σ > 0 keeps
the PR3 rejection contract.
"""

from __future__ import annotations

import warnings

import minari
import pytest
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.device import detect_device
from src.envs.offline.generate import generate_offline_dataset
from src.envs.registry import register_default_env_wrappers

warnings.filterwarnings("ignore")


def _gen_confounded(dataset_id: str, sigma: float):
    try:
        minari.delete_dataset(dataset_id)
    except Exception:
        pass
    return generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="bias_confounded",
        behavior_strength=sigma,
        rollout_episodes=30,
        seed=0,
        dataset_id=dataset_id,
    )


def _run_cql(tmp_path, dataset_id: str, sigma: float):
    register_default_algorithms()
    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=2,
        seed=0,
        offline_dataset=dataset_id,
        behavior_policy="bias_confounded",
        behavior_strength=sigma,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,
        n_checkpoints=2,
        device=str(detect_device()),
        algorithm="cql",
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(tmp_path / "run"), timestamp="t"),
        registry.get("cql"),
    ).run()


def test_sigma_zero_anchor_bypasses_gate(tmp_path):
    did = "test/sigma0-anchor-v0"
    ds = _gen_confounded(did, 0.0)
    try:
        meta = ds.storage.metadata
        # By construction the σ=0.0 dataset is unconfounded -> gate FAILS.
        assert meta["behavior_strength_sigma"] == 0.0
        assert meta["gate_test_passed"] is False, "σ=0.0 should fail the gate"
        # The bypass fires: the runner does NOT raise; training proceeds.
        _run_cql(tmp_path, did, 0.0)
    finally:
        minari.delete_dataset(did)


def test_sigma_nonzero_still_enforces_gate(tmp_path):
    did = "test/sigma05-enforced-v0"
    ds = _gen_confounded(did, 0.5)
    try:
        # Force the gate to fail for a σ>0 dataset -> the bypass must NOT fire.
        ds.storage.update_metadata({"gate_test_passed": False})
        with pytest.raises(ValueError, match="gate_test_passed=False"):
            _run_cql(tmp_path, did, 0.5)
    finally:
        minari.delete_dataset(did)

"""feat/offline-value-trace (PR3) — Cells 7-8 apparent-value trace + signature.

offline_value_trace.csv is written ONLY for confounded offline runs (strict
opt-in, so a non-confounded run is byte-identical to a pre-PR3 run). Generation
stamps the per-dataset confounding signature into the Minari metadata, and the
runner rejects a confounded run whose dataset failed the gate. These tests cover
the gate-closed contract, the schema, the metadata, and the rejection path.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import minari
import pytest
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner, OFFLINE_VALUE_TRACE_COLUMNS
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.device import detect_device
from src.envs.offline.generate import generate_offline_dataset
from src.envs.registry import register_default_env_wrappers

warnings.filterwarnings("ignore")


def _gen(dataset_id: str, behavior_policy: str, sigma: float | None = None):
    """Fast random-tier CartPole dataset (no generator training)."""
    try:
        minari.delete_dataset(dataset_id)
    except Exception:
        pass
    return generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy=behavior_policy,
        behavior_strength=sigma,
        rollout_episodes=30,
        seed=0,
        dataset_id=dataset_id,
    )


def _run_cql(tmp_path: Path, dataset_id: str, behavior_policy: str) -> Path:
    register_default_algorithms()
    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=4,
        seed=0,
        offline_dataset=dataset_id,
        behavior_policy=behavior_policy,
        behavior_strength=0.5 if behavior_policy == "bias_confounded" else None,
    )
    train_cfg = TrainingConfig(
        n_episodes=3,
        n_checkpoints=2,
        device=str(detect_device()),
        algorithm="cql",
    )
    run_dir = tmp_path / "run"
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("cql"),
    ).run()
    return run_dir


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_offline_value_trace_not_written_without_confounding(tmp_path):
    """Gate closed (non-confounded dataset, agent behavior) -> no file."""
    did = "test/value-trace-clean-v0"
    _gen(did, behavior_policy="agent")
    try:
        run_dir = _run_cql(tmp_path, did, behavior_policy="agent")
        assert not (run_dir / "offline_value_trace.csv").exists()
    finally:
        minari.delete_dataset(did)


def test_offline_value_trace_schema_frozen(tmp_path):
    """Gate open -> file exists, header matches schema, one row per epoch."""
    did = "test/value-trace-conf-v0"
    _gen(did, behavior_policy="bias_confounded", sigma=0.5)
    try:
        run_dir = _run_cql(tmp_path, did, behavior_policy="bias_confounded")
        path = run_dir / "offline_value_trace.csv"
        assert path.exists()
        with path.open(newline="") as f:
            header = next(csv.reader(f))
        assert header == OFFLINE_VALUE_TRACE_COLUMNS
        rows = _rows(path)
        # one row per training epoch (n_episodes=3 above).
        assert len(rows) == 3
        assert [int(r["epoch"]) for r in rows] == [0, 1, 2]
    finally:
        minari.delete_dataset(did)


def test_confounding_metadata_written_at_generation(tmp_path):
    did = "test/value-trace-meta-v0"
    ds = _gen(did, behavior_policy="bias_confounded", sigma=0.5)
    try:
        meta = minari.load_dataset(did).storage.metadata
        for field in (
            "corr_a_r_marginal",
            "corr_a_r_partial_given_u",
            "gate_test_passed",
            "behavior_strength_sigma",
        ):
            assert field in meta, field
        assert meta["behavior_strength_sigma"] == 0.5
        assert isinstance(meta["gate_test_passed"], bool)
        assert abs(float(meta["corr_a_r_marginal"])) > 0.05  # genuine confounding
        assert abs(float(meta["corr_a_r_partial_given_u"])) < 0.05  # controllable by U
    finally:
        minari.delete_dataset(did)


def test_runner_rejects_ungate_tested_dataset(tmp_path):
    did = "test/value-trace-reject-v0"
    ds = _gen(did, behavior_policy="bias_confounded", sigma=0.5)
    try:
        # Force the gate to fail -> a confounded run must be rejected before training.
        ds.storage.update_metadata({"gate_test_passed": False})
        with pytest.raises(ValueError, match="gate_test_passed=False"):
            _run_cql(tmp_path / "fail", did, behavior_policy="bias_confounded")

        # Restore -> the run proceeds and writes the trace.
        ds.storage.update_metadata({"gate_test_passed": True})
        run_dir = _run_cql(tmp_path / "ok", did, behavior_policy="bias_confounded")
        assert (run_dir / "offline_value_trace.csv").exists()
    finally:
        minari.delete_dataset(did)

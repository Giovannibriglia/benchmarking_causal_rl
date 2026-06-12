"""B1 offline LOAD path — download-if-absent + the action-space guard.

Three layers:
  (i)  routine smoke: a tiny LOCAL CartPole fixture driven through the full
       load -> guard -> fill -> _train_offline -> CSV path (no network);
  (ii) guard, both directions: a continuous dataset against a discrete algo and
       vice versa is rejected with a ValueError BEFORE filling/training;
  (iii) one-time REAL hosted load+train, gated behind MINARI_NETWORK_TESTS so the
        default suite stays fast and offline.

MINARI_DATASETS_PATH is isolated to a tmp dir so the local fixtures neither
pollute nor depend on ~/.minari, and so download-if-absent never fires for them.
"""

from __future__ import annotations

import csv
import os

import pytest
from gymnasium.spaces import Box, Discrete  # noqa: E402
from src.envs.offline.minari_loader import (  # noqa: E402
    dataset_action_type,
    load_minari_dataset,
)

pytest.importorskip("minari")
pytest.importorskip("h5py")

from src.benchmarking.registry import (  # noqa: E402
    register_default_algorithms,
    registry,
)
from src.benchmarking.runner import (  # noqa: E402
    BenchmarkRunner,
    EVAL_COLUMNS,
    TRAIN_COLUMNS,
)
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig  # noqa: E402
from src.config.device import detect_device  # noqa: E402
from src.envs.registry import register_default_env_wrappers  # noqa: E402


# --------------------------------------------------------------------------
# Unit: action-space resolution + check-local-first loader
# --------------------------------------------------------------------------
def test_dataset_action_type_resolution():
    assert dataset_action_type(Discrete(4)) == "discrete"
    assert dataset_action_type(Box(low=-1.0, high=1.0, shape=(3,))) == "continuous"


def test_load_local_only_dataset_no_download(tmp_path, monkeypatch):
    """A local-only fixture loads from cache (no remote contact)."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(dataset_id="b1/local-only-v0", n_episodes=2, seed=0)
    ds = load_minari_dataset("b1/local-only-v0")  # must not raise / not download
    assert ds.total_episodes == 2
    assert dataset_action_type(ds.action_space) == "discrete"


# --------------------------------------------------------------------------
# (i) Routine smoke — full wired path on a local fixture
# --------------------------------------------------------------------------
def _make_runner(env_id, algo, dataset_id, run_dir):
    register_default_algorithms()
    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id=env_id,
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=5,
        seed=0,
        offline_dataset=dataset_id,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,
        n_checkpoints=1,
        device=str(detect_device()),
        algorithm=algo,
        aggregation="mean",
    )
    return BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get(algo),
    )


def test_offline_load_smoke_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(dataset_id="b1/cartpole-v0", n_episodes=8, seed=0)
    run_dir = tmp_path / "run"
    _make_runner("CartPole-v1", "offline_dqn", "b1/cartpole-v0", run_dir).run()

    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    assert train_rows[0]["q_loss"] != ""  # finite, non-blank
    assert train_rows[0]["train_return_mean"] == ""  # blank offline
    assert float(eval_rows[0]["eval_return_mean"]) >= 0.0  # numeric


# --------------------------------------------------------------------------
# (ii) Guard — both mismatch directions, rejected before training
# --------------------------------------------------------------------------
def test_guard_rejects_continuous_dataset_with_discrete_algo(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_pendulum_offline import make_pendulum_dataset

    make_pendulum_dataset(dataset_id="b1/pendulum-v0", n_episodes=2, seed=0)
    run_dir = tmp_path / "run"
    # Discrete env + discrete algo, but a CONTINUOUS dataset -> reject at guard.
    runner = _make_runner("CartPole-v1", "offline_dqn", "b1/pendulum-v0", run_dir)
    with pytest.raises(ValueError, match="continuous action space.*discrete-only"):
        runner.run()
    # The guard fired before fill/training -> no eval rows logged.
    assert not (run_dir / "eval_metrics.csv").exists() or _empty(
        run_dir / "eval_metrics.csv"
    )


def test_guard_rejects_discrete_dataset_with_continuous_algo(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(dataset_id="b1/cartpole-v0", n_episodes=2, seed=0)
    run_dir = tmp_path / "run"
    # Continuous env + continuous algo, but a DISCRETE dataset -> reject at guard.
    runner = _make_runner("Pendulum-v1", "cql_continuous", "b1/cartpole-v0", run_dir)
    with pytest.raises(ValueError, match="discrete action space.*continuous-only"):
        runner.run()


def _empty(path):
    with path.open() as f:
        return len(list(csv.DictReader(f))) == 0


# --------------------------------------------------------------------------
# (iii) One-time REAL hosted load+train — opt-in (network/slow)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("MINARI_NETWORK_TESTS"),
    reason="real hosted-dataset test; set MINARI_NETWORK_TESTS=1 to run",
)
def test_real_hosted_continuous_dataset_load_and_train(tmp_path):
    import minari

    # Verified continuous hosted dataset (100 eps / 100k steps, InvertedPendulum
    # -v5; ~6 MB). Executed once during B1 Phase 2: iql_continuous trained to
    # completion, eval 20.0 (perfect over the 20-step eval window).
    dataset_id = os.environ.get(
        "MINARI_TEST_DATASET", "mujoco/invertedpendulum/expert-v0"
    )
    ds = load_minari_dataset(dataset_id)
    assert dataset_action_type(ds.action_space) == "continuous"
    run_dir = tmp_path / "run"
    env_spec_id = ds.spec.env_spec.id  # the env this dataset evaluates in
    _make_runner(env_spec_id, "iql_continuous", dataset_id, run_dir).run()
    with (run_dir / "train_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == TRAIN_COLUMNS and rows[0]["q_loss"] != ""
    _ = minari  # keep import referenced

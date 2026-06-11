"""Offline (fixed-dataset) training: end-to-end on a tiny Minari CartPole set.

Hermetic: builds the dataset in a tmp dir with MINARI_DATASETS_PATH isolated,
so it never touches ~/.minari. Skips cleanly if the offline extra
(minari[hdf5]) is not installed.
"""

from __future__ import annotations

import csv

import pytest

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
from src.data.experience_source import OnlineSource, validate_pairing  # noqa: E402
from src.envs.registry import register_default_env_wrappers  # noqa: E402

DATASET_ID = "cartpole/random-test-v0"


@pytest.fixture
def cartpole_dataset(tmp_path, monkeypatch):
    """Build a tiny CartPole Minari dataset in an isolated tmp dir."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(dataset_id=DATASET_ID, n_episodes=10, seed=0)
    return DATASET_ID


def test_offline_dqn_end_to_end(cartpole_dataset, tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()
    device = str(detect_device())
    run_dir = tmp_path / "run"

    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=5,  # offline: gradient steps per epoch
        seed=0,
        offline_dataset=cartpole_dataset,
    )
    train_cfg = TrainingConfig(
        n_episodes=2,  # offline: epochs
        n_checkpoints=2,
        device=device,
        algorithm="offline_dqn",
        aggregation="iqm",
    )
    run_cfg = RunConfig(run_dir=str(run_dir), timestamp="t")
    spec = registry.get("offline_dqn")
    assert spec.data_regime == "offline" and spec.kind == "off_policy"

    BenchmarkRunner(env_cfg, train_cfg, run_cfg, spec, progress_label="offline").run()

    # CSVs exist and match the frozen schema exactly.
    train_csv = run_dir / "train_metrics.csv"
    eval_csv = run_dir / "eval_metrics.csv"
    assert train_csv.exists() and eval_csv.exists()

    with train_csv.open() as f:
        train_rows = list(csv.DictReader(f))
    with eval_csv.open() as f:
        eval_rows = list(csv.DictReader(f))

    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    # offline populates loss columns and leaves train_return_* blank (as online
    # off-policy does); eval (live env) yields a numeric return.
    assert train_rows[0]["q_loss"] != ""
    assert train_rows[0]["train_return_mean"] == ""
    assert float(eval_rows[0]["eval_return_mean"]) > 0.0


def test_validate_pairing_rejects_offline_on_policy():
    online = OnlineSource(env=None, device=detect_device())
    # offline data_regime requires an off_policy learner.
    with pytest.raises(ValueError):
        validate_pairing("on_policy", online, data_regime="offline")
    # off_policy + offline is accepted.
    validate_pairing("off_policy", online, data_regime="offline")


def test_offline_dqn_spec_registered():
    register_default_algorithms()
    spec = registry.get("offline_dqn")
    assert spec.kind == "off_policy"
    assert spec.data_regime == "offline"
    # online dqn left untouched (still online).
    assert registry.get("dqn").data_regime == "online"

"""Offline RL on Atari image datasets (PR7).

Three things proven here:
  * ``offline_dqn`` runs end-to-end on a tiny Atari Minari fixture through the
    Nature-CNN -> frozen-schema CSVs (eval in the live env).
  * BYTE-IDENTITY: a frame stored in the offline fixture, loaded via the real
    ``fill_replay_buffer_from_minari`` path, equals ``GymnasiumEnv._image_obs``
    of the same raw uint8 frame — both go through the shared
    ``normalize_image_obs``, so this pins the single-source equality against
    regression.
  * bcq / cql / iql are CNN-capable (construction + forward on image obs); the
    full training loop is exercised only for offline_dqn to keep CI cheap.

Gated behind ``ale_py`` (image env) AND the offline extra (minari/h5py); the
Minari dataset is isolated to a tmp dir so it never touches ~/.minari.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

pytest.importorskip("ale_py")
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
from src.rl.offline.bcq import build_bcq  # noqa: E402
from src.rl.offline.cql import build_cql  # noqa: E402
from src.rl.offline.dqn import build_offline_dqn  # noqa: E402
from src.rl.offline.iql import build_iql  # noqa: E402

DATASET_ID = "atari/pong-random-test-v0"
OBS_SHAPE = (4, 84, 84)


@pytest.fixture
def atari_dataset(tmp_path, monkeypatch):
    """Build a tiny ALE/Pong Minari dataset in an isolated tmp dir."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_atari_offline import make_atari_dataset

    make_atari_dataset(dataset_id=DATASET_ID, n_episodes=2, seed=0, max_steps=30)
    return DATASET_ID


def test_offline_dqn_atari_end_to_end(atari_dataset, tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()
    device = str(detect_device())
    run_dir = tmp_path / "run"

    env_cfg = EnvConfig(
        env_id="ALE/Pong-v5",
        n_train_envs=1,
        n_eval_envs=1,
        rollout_len=3,  # offline: gradient steps per epoch
        seed=0,
        offline_dataset=atari_dataset,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,  # offline: epochs
        n_checkpoints=1,
        device=device,
        algorithm="offline_dqn",
        aggregation="mean",
    )
    run_cfg = RunConfig(run_dir=str(run_dir), timestamp="t")
    spec = registry.get("offline_dqn")

    BenchmarkRunner(
        env_cfg, train_cfg, run_cfg, spec, progress_label="atari-offline"
    ).run()

    train_csv = run_dir / "train_metrics.csv"
    eval_csv = run_dir / "eval_metrics.csv"
    assert train_csv.exists() and eval_csv.exists()
    with train_csv.open() as f:
        train_rows = list(csv.DictReader(f))
    with eval_csv.open() as f:
        eval_rows = list(csv.DictReader(f))

    # Frozen schema, byte-for-byte.
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    # offline populates loss columns, leaves train_return_* blank.
    assert train_rows[0]["q_loss"] != ""
    assert train_rows[0]["train_return_mean"] == ""
    # eval ran in the live Atari env (numeric, not blank).
    assert eval_rows[0]["eval_return_mean"] != ""


def test_offline_frame_matches_online_representation(atari_dataset):
    """A stored fixture frame, loaded through the real offline path, is
    byte-identical to the online wrapper's representation of the same frame."""
    import minari
    from src.envs.offline.minari_loader import fill_replay_buffer_from_minari
    from src.envs.wrappers.gymnasium_env import GymnasiumEnv
    from src.rl.off_policy.replay_buffer import ReplayBuffer

    device = torch.device("cpu")
    buffer = ReplayBuffer(capacity=10_000, device=device)
    n_added = fill_replay_buffer_from_minari(atari_dataset, buffer, device)
    assert n_added > 0

    dataset = minari.load_dataset(atari_dataset)
    episode = next(dataset.iterate_episodes())
    raw_frame = np.asarray(episode.observations[0])
    assert raw_frame.shape == OBS_SHAPE and raw_frame.dtype == np.uint8

    # Online path: _image_obs only reads self.device, so a tiny stub suffices
    # (avoids spinning up a live vector env). Both sides route through the shared
    # normalize_image_obs, so equality here proves the delegation is intact.
    class _Stub:
        device = torch.device("cpu")

    expected = GymnasiumEnv._image_obs(_Stub(), raw_frame)
    stored = buffer.storage[0]["obs"]

    assert torch.equal(stored, expected)
    assert stored.shape == OBS_SHAPE
    assert stored.dtype == torch.float32
    assert float(stored.max()) <= 1.0  # normalized, not raw [0, 255]


@pytest.mark.parametrize("build", [build_offline_dqn, build_bcq, build_cql, build_iql])
def test_offline_builder_cnn_capable(build):
    """Each offline builder routes image obs through the Nature-CNN: construct on
    (4, 84, 84) and forward a batch to a finite (B, action_dim) head."""
    device = torch.device("cpu")
    action_dim = 6
    net, agent = build(
        obs_dim=int(np.prod(OBS_SHAPE)),
        action_dim=action_dim,
        action_type="discrete",
        device=device,
        action_space=None,
        obs_shape=OBS_SHAPE,
    )
    obs = torch.zeros(2, *OBS_SHAPE)
    out = net(obs)
    assert out.shape == (2, action_dim)
    assert torch.isfinite(out).all()

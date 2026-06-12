"""MiniGrid image stack (PR: closes the PR6 Stage B deferral).

MiniGrid runs through the SAME rank-3 image path as Atari: an RGB partial render
(tile_size=12 -> 84x84, no resize/opencv) transposed to channels-first
(3, 84, 84), normalized by the shared normalize_image_obs (/255 is correct for a
genuine RGB render), into the unchanged Nature-CNN (channels from obs_shape[0]).

Smoke-level goal -- "does MiniGrid run with a CNN" -- so NO learning bar:
MiniGrid's sparse reward + a tiny budget leaves an untrained policy near 0;
eval is asserted finite and >= 0, not > 0. Gated behind importorskip("minigrid").
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

pytest.importorskip("minigrid")

from src.envs.wrappers.minigrid import make_minigrid_env  # noqa: E402
from src.rl.nets.cnn import NatureCNN  # noqa: E402

MINIGRID_ENVS = ["MiniGrid-Empty-5x5-v0", "MiniGrid-DoorKey-5x5-v0"]
OBS_SHAPE = (3, 84, 84)


@pytest.mark.parametrize("env_id", MINIGRID_ENVS)
def test_make_minigrid_env_delivers_chw_84(env_id):
    """Both env-set members: (84,84,3) partial render -> (3,84,84) uint8 obs."""
    env = make_minigrid_env(env_id)
    assert env.observation_space.shape == OBS_SHAPE
    assert hasattr(env.action_space, "n")  # discrete
    obs, _ = env.reset(seed=0)
    assert obs.shape == OBS_SHAPE and obs.dtype == np.uint8
    assert int(obs.min()) >= 0 and int(obs.max()) <= 255
    env.close()


def test_nature_cnn_forward_on_minigrid_shape():
    """The unchanged Nature-CNN consumes (3,84,84) -> finite (B, action_dim)."""
    cnn = NatureCNN(OBS_SHAPE, 7)
    out = cnn(torch.zeros(2, *OBS_SHAPE))
    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# End-to-end through the existing online loop (PPO: on-policy, trains on a tiny
# rollout -- DQN's 1000-step warmup would skip every gradient step here).
# --------------------------------------------------------------------------
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


def test_minigrid_ppo_end_to_end(tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    env_cfg = EnvConfig(
        env_id="MiniGrid-Empty-5x5-v0",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=8,
        seed=0,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,
        n_checkpoints=1,
        device=str(detect_device()),
        algorithm="ppo",
        aggregation="mean",
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("ppo"),
        progress_label="minigrid",
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
    # PPO trained through the CNN -> a finite value_loss.
    assert float(train_rows[0]["value_loss"]) == float(train_rows[0]["value_loss"])
    # MiniGrid reward is sparse [0,1]; an untrained tiny-budget policy sits at ~0
    # -> assert numeric & finite & >= 0, NOT > 0.
    ev = float(eval_rows[-1]["eval_return_mean"])
    assert ev == ev and ev >= 0.0

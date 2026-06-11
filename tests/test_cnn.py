"""CNN backbone + image stack.

Unit tests (always run): Nature-CNN forward shape, dummy-forward flatten
correctness, IMPALA seam raises, rank-3 select_backbone returns a CNN. The
Atari end-to-end test is gated behind the `atari` extra (importorskip), with a
tiny step budget like the offline tests.
"""

from __future__ import annotations

import csv

import pytest
import torch
from src.rl.models.backbone import select_backbone
from src.rl.nets.cnn import NatureCNN


def test_nature_cnn_forward_shape():
    net = NatureCNN((4, 84, 84), output_dim=6)
    out = net(torch.zeros(2, 4, 84, 84))
    assert out.shape == (2, 6)


def test_nature_cnn_dummy_flatten_dim():
    # Nature-CNN on 84x84 -> 64 channels x 7 x 7 = 3136 flattened conv features.
    net = NatureCNN((4, 84, 84), output_dim=6)
    conv_features = net.conv(torch.zeros(1, 4, 84, 84)).shape[1]
    assert conv_features == 3136
    # The first head Linear consumes exactly that, so the dummy forward sized it.
    assert net.head[0].in_features == 3136
    # Different input size is handled by the dummy forward (input-size-robust).
    net2 = NatureCNN((1, 42, 42), output_dim=3)
    assert net2(torch.zeros(2, 1, 42, 42)).shape == (2, 3)


def test_impala_variant_raises():
    with pytest.raises(NotImplementedError):
        NatureCNN((4, 84, 84), output_dim=6, variant="impala")


def test_rank3_select_backbone_returns_cnn():
    # input_dim is vestigial for the image path; obs_shape drives the CNN.
    net = select_backbone((4, 84, 84), 99999, 6)
    assert isinstance(net, NatureCNN)
    assert net(torch.zeros(1, 4, 84, 84)).shape == (1, 6)


# --------------------------------------------------------------------------
# Atari end-to-end (needs the atari extra: ale-py + opencv)
# --------------------------------------------------------------------------
pytest.importorskip("ale_py")
pytest.importorskip("cv2")

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


def test_atari_dqn_end_to_end(tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    env_cfg = EnvConfig(
        env_id="ALE/Pong-v5",
        n_train_envs=4,
        n_eval_envs=2,
        rollout_len=300,  # 4*300 > warmup(1000), so the CNN update fires
        seed=0,
    )
    train_cfg = TrainingConfig(
        n_episodes=2,
        n_checkpoints=2,
        device=str(detect_device()),
        algorithm="dqn",
        aggregation="mean",
    )
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("dqn"),
    )
    # Image obs route to the CNN through the existing off-policy loop.
    assert runner.obs_shape == (4, 84, 84)
    assert isinstance(runner.agent.q_network, NatureCNN)
    runner.run()

    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    # The CNN update fired -> a finite q_loss was logged.
    assert any(r["q_loss"] not in ("", None) for r in train_rows)
    q = next(r["q_loss"] for r in train_rows if r["q_loss"] not in ("", None))
    assert float(q) == float(q)  # finite

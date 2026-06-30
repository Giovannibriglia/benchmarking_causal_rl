"""PR-2a — runner OFFLINE loop honors episode grouping (proximal path).

A *_proximal offline run routes THROUGH THE RUNNER to a SequenceReplayBuffer
(filled episode-grouped via the PR-0 sequence loader; same-episode (B,T) windows
-> agent.update, the stub flattening + degrading to the floor). Non-grouped
offline stays on the flat ReplayBuffer path (the hard byte-identity gate is the
golden suite; here we assert the buffer type the runner ends up with).
"""

from __future__ import annotations

import csv
import warnings

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

warnings.filterwarnings("ignore")

_CPU = torch.device("cpu")


# NOTE: the PR-2a "stub consumes (B,T)" test was removed — PR-2b replaced the
# flatten+floor stub with the real ProximalEM (it needs the E-step's per-episode
# r_tau in the window + a handed-off sequence buffer, so a raw synthetic (B,T) is
# no longer a valid standalone input). The real EM consumption is exercised
# end-to-end below (proximal-through-runner) and in test_proximal_estimator.py.


# --------------------------------------------------------------------------
# Through-the-runner routing (needs a tiny Minari dataset).
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")


def _gen(tmp_path, monkeypatch, dataset_id):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from src.envs.offline.generate import generate_offline_dataset
    from src.envs.registry import register_default_env_wrappers

    torch.manual_seed(0)
    register_default_algorithms()
    register_default_env_wrappers()
    generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="agent",
        rollout_episodes=16,
        seed=0,
        dataset_id=dataset_id,
        device="cpu",
    )
    return dataset_id


def _run(tmp_path, dataset_id, algo, run_name):
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    register_default_algorithms()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=4,
        rollout_len=20,
        seed=0,
        offline_dataset=dataset_id,
    )
    train_cfg = TrainingConfig(
        n_episodes=3,
        n_checkpoints=2,
        device="cpu",
        algorithm=algo,
        aggregation="mean",
    )
    run_dir = tmp_path / run_name
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get(algo),
    )
    runner.run()
    return runner, run_dir


def test_proximal_offline_routes_to_sequence_buffer_through_runner(
    tmp_path, monkeypatch
):
    did = _gen(tmp_path, monkeypatch, "grouping/prox-v0")
    runner, run_dir = _run(tmp_path, did, "cql_proximal", "prox")
    # Routed to the grouped path: the runner OWNS a SequenceReplayBuffer.
    assert isinstance(runner.replay_buffer, SequenceReplayBuffer)
    # Run completed end to end.
    with (run_dir / "eval_metrics.csv").open() as f:
        assert len(list(csv.DictReader(f))) > 0


def test_base_offline_stays_on_flat_buffer(tmp_path, monkeypatch):
    did = _gen(tmp_path, monkeypatch, "grouping/base-v0")
    runner, _ = _run(tmp_path, did, "cql", "base")
    # Non-grouped: the existing flat ReplayBuffer path, untouched.
    assert isinstance(runner.replay_buffer, ReplayBuffer)
    assert not isinstance(runner.replay_buffer, SequenceReplayBuffer)

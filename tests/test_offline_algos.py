"""Offline algorithms BCQ / CQL / IQL — correctness + end-to-end + learning.

Beyond "runs to completion", these tests exercise each algorithm's DISTINGUISHING
term directly (CQL's conservative penalty, IQL's asymmetric expectile, BCQ's
behavior-constrained action set), so a silently-wrong-but-finite implementation
fails. The learning-sanity tests additionally prove each algo learns (eval beats
the random baseline) on a better-than-random CartPole dataset.
"""

from __future__ import annotations

import csv

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner, EVAL_COLUMNS, TRAIN_COLUMNS
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.device import detect_device
from src.rl.offline.bcq import build_bcq
from src.rl.offline.cql import build_cql, CQL
from src.rl.offline.iql import expectile_loss

ALGOS = ["bcq", "cql", "iql"]


# --------------------------------------------------------------------------
# Registration / dispatch
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALGOS)
def test_registered_offline_off_policy(name):
    register_default_algorithms()
    spec = registry.get(name)
    assert spec.kind == "off_policy"
    assert spec.data_regime == "offline"


# --------------------------------------------------------------------------
# Distinguishing-term correctness (fast, CPU, no dataset)
# --------------------------------------------------------------------------
def test_cql_penalty_present_and_nonnegative():
    torch.manual_seed(0)
    # Static form: logsumexp_a Q >= Q(a_data) always, and strictly > when the
    # data action is not the per-row argmax.
    q = torch.randn(128, 4)
    actions = torch.randint(0, 4, (128,))
    penalty = CQL.conservative_penalty(q, actions)
    assert penalty.item() >= 0.0
    # A batch where the data action is never the max -> strictly positive.
    q2 = torch.zeros(8, 3)
    q2[:, 0] = 5.0  # action 0 dominates
    worst = torch.full((8,), 2)  # data action is never the argmax
    assert CQL.conservative_penalty(q2, worst).item() > 0.0
    # And the penalty actually flows through the real update path.
    _, agent = build_cql(
        obs_dim=4,
        action_dim=3,
        action_type="discrete",
        device=torch.device("cpu"),
        action_space=None,
    )
    batch = {
        "obs": torch.randn(64, 4),
        "actions": torch.randint(0, 3, (64,)),
        "rewards": torch.randn(64),
        "next_obs": torch.randn(64, 4),
        "dones": torch.zeros(64),
    }
    metrics = agent.update(batch)
    assert "cql_penalty" in metrics and metrics["cql_penalty"] >= 0.0
    # alpha=0 must drop the penalty's contribution: loss == td_loss.
    agent.alpha = 0.0
    m0 = agent.update(batch)
    assert abs(m0["loss"] - m0["td_loss"]) < 1e-6


def test_iql_expectile_is_asymmetric():
    u = torch.tensor([1.0])
    # tau != 0.5 weights positive vs negative residuals differently.
    pos = expectile_loss(u, 0.7).item()
    neg = expectile_loss(-u, 0.7).item()
    assert pos != neg
    assert pos == pytest.approx(0.7) and neg == pytest.approx(0.3)
    # tau = 0.5 is symmetric (ordinary L2 weighting).
    assert expectile_loss(u, 0.5).item() == pytest.approx(
        expectile_loss(-u, 0.5).item()
    )


def test_bcq_allowed_set_collapses_at_threshold_extremes():
    torch.manual_seed(0)
    _, agent = build_bcq(
        obs_dim=4,
        action_dim=4,
        action_type="discrete",
        device=torch.device("cpu"),
        action_space=None,
    )
    obs = torch.randn(32, 4)
    # tau -> 0: every action allowed (== unconstrained DQN).
    assert bool(agent.allowed_mask(obs, threshold=0.0).all())
    # tau -> 1: exactly the behavior argmax per row (BC-greedy).
    mask1 = agent.allowed_mask(obs, threshold=1.0)
    assert bool((mask1.sum(dim=1) == 1).all())
    bc_argmax = torch.argmax(agent.behavior_net(obs), dim=1)
    assert torch.equal(mask1.float().argmax(dim=1), bc_argmax)


# --------------------------------------------------------------------------
# End-to-end + learning (need the offline extra)
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")


def _build_dataset(tmp_path, monkeypatch, dataset_id, policy, n_episodes, seed):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(
        dataset_id=dataset_id, n_episodes=n_episodes, seed=seed, policy=policy
    )
    return dataset_id


def _run(dataset_id, algo, epochs, grad_steps, run_dir, seed=0):
    register_default_algorithms()
    from src.envs.registry import register_default_env_wrappers

    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=10,
        rollout_len=grad_steps,
        seed=seed,
        offline_dataset=dataset_id,
    )
    train_cfg = TrainingConfig(
        n_episodes=epochs,
        n_checkpoints=2,
        device=str(detect_device()),
        algorithm=algo,
        aggregation="mean",
    )
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get(algo),
    )
    runner.run()
    return run_dir


@pytest.mark.parametrize("algo", ALGOS)
def test_offline_algo_end_to_end_schema(algo, tmp_path, monkeypatch):
    ds = _build_dataset(tmp_path, monkeypatch, "cartpole/rand-v0", "random", 10, 0)
    run_dir = tmp_path / f"run_{algo}"
    _run(ds, algo, epochs=2, grad_steps=10, run_dir=run_dir)

    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    assert float(train_rows[0]["q_loss"]) == float(train_rows[0]["q_loss"])  # finite


@pytest.mark.parametrize("algo", ALGOS)
def test_offline_algo_learns_beats_random(algo, tmp_path, monkeypatch):
    """Trained on better-than-random data, eval return clears the random
    baseline (~22) by a clear margin."""
    ds = _build_dataset(tmp_path, monkeypatch, "cartpole/heur-v0", "heuristic", 60, 0)
    run_dir = tmp_path / f"learn_{algo}"
    _run(ds, algo, epochs=40, grad_steps=100, run_dir=run_dir)

    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    final_return = float(eval_rows[-1]["eval_return_mean"])
    assert final_return > 50.0, f"{algo} eval={final_return:.1f} did not beat random"

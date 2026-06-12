"""Continuous offline algorithms — CQL-on-SAC + IQL-Gaussian.

Mirrors the discrete offline test discipline (PR5): each algo's DISTINGUISHING
term is exercised directly (CQL importance-sampled penalty + cql_alpha gating;
IQL squashed_log_prob clamp + expectile asymmetry + AWR weighting), plus an
end-to-end schema run and a learning-sanity check on a better-than-random
Pendulum dataset. Also pins the discrete-builder guard that rejects continuous
action spaces.
"""

from __future__ import annotations

import csv

import pytest
import torch
from src.rl.offline.bcq import build_bcq
from src.rl.offline.cql import build_cql
from src.rl.offline.cql_continuous import build_cql_continuous
from src.rl.offline.dqn import build_offline_dqn
from src.rl.offline.iql import build_iql, expectile_loss
from src.rl.offline.iql_continuous import build_iql_continuous, squashed_log_prob

CPU = torch.device("cpu")


def _batch(b=64, obs_dim=3, act_dim=1, scale=2.0):
    return {
        "obs": torch.randn(b, obs_dim),
        "actions": torch.empty(b, act_dim).uniform_(-scale, scale),
        "rewards": torch.randn(b),
        "next_obs": torch.randn(b, obs_dim),
        "dones": torch.zeros(b),
    }


# --------------------------------------------------------------------------
# Registration / dispatch + the discrete-builder guard
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["cql_continuous", "iql_continuous"])
def test_registered_continuous_offline(name):
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    spec = registry.get(name)
    assert spec.kind == "off_policy"
    assert spec.data_regime == "offline"


@pytest.mark.parametrize("build", [build_bcq, build_cql, build_iql, build_offline_dqn])
def test_discrete_builder_rejects_continuous(build):
    with pytest.raises(ValueError, match="continuous"):
        build(
            obs_dim=3,
            action_dim=1,
            action_type="continuous",
            device=CPU,
            action_space=None,
        )


# --------------------------------------------------------------------------
# CQL-on-SAC: penalty present/finite + cql_alpha gating
# --------------------------------------------------------------------------
def test_cql_continuous_penalty_present_and_gated():
    torch.manual_seed(0)
    _, agent = build_cql_continuous(
        obs_dim=3, action_dim=1, action_type="continuous", device=CPU, action_space=None
    )
    metrics = agent.update(_batch())
    assert "cql_penalty" in metrics
    # Importance-sampled estimator is finite (NOT sign-guaranteed, unlike the
    # discrete closed-form logsumexp).
    assert metrics["cql_penalty"] == metrics["cql_penalty"]  # not NaN
    assert abs(metrics["cql_penalty"]) < float("inf")

    # cql_alpha = 0 must drop the conservative term: critic_loss == td_loss.
    agent.cql_alpha = 0.0
    m0 = agent.update(_batch())
    assert abs(m0["critic_loss"] - m0["td_loss"]) < 1e-5


# --------------------------------------------------------------------------
# IQL-continuous: the atanh clamp, expectile asymmetry, AWR weighting
# --------------------------------------------------------------------------
def test_squashed_log_prob_finite_at_saturated_actions():
    """Saturated dataset actions scale to exactly +-1; atanh(+-1) = +-inf would
    poison the AWR weight. The clamp must keep log pi finite."""
    mean = torch.zeros(4, 1)
    log_std = torch.zeros(4, 1)
    action = torch.tensor([[1.0], [-1.0], [0.999999], [0.0]])
    logp = squashed_log_prob(mean, log_std, action)
    assert torch.isfinite(logp).all()


def test_iql_expectile_asymmetry_reused():
    u = torch.tensor([1.0])
    assert expectile_loss(u, 0.7).item() == pytest.approx(0.7)
    assert expectile_loss(-u, 0.7).item() == pytest.approx(0.3)


def test_iql_awr_weight_positive_and_clipped():
    beta, clip = 3.0, 100.0
    adv = torch.tensor([-5.0, 0.0, 50.0])
    weight = torch.clamp(torch.exp(beta * adv), max=clip)
    assert (weight > 0).all()
    assert weight.max().item() <= clip + 1e-6
    assert weight[2].item() == pytest.approx(clip)  # large positive adv -> clipped


def test_iql_continuous_update_finite():
    torch.manual_seed(0)
    _, agent = build_iql_continuous(
        obs_dim=3, action_dim=1, action_type="continuous", device=CPU, action_space=None
    )
    m = agent.update(_batch())
    for k in ("q_loss", "value_loss", "actor_loss"):
        assert m[k] == m[k] and abs(m[k]) < float("inf")


# --------------------------------------------------------------------------
# End-to-end + learning (need the offline extra)
# --------------------------------------------------------------------------
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

ALGOS = ["cql_continuous", "iql_continuous"]


def _build_dataset(tmp_path, monkeypatch, dataset_id, policy, n_episodes, seed):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_pendulum_offline import make_pendulum_dataset

    make_pendulum_dataset(
        dataset_id=dataset_id, n_episodes=n_episodes, seed=seed, policy=policy
    )
    return dataset_id


def _run(dataset_id, algo, epochs, grad_steps, run_dir, seed=0, n_eval=5):
    register_default_algorithms()
    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id="Pendulum-v1",
        n_train_envs=1,
        n_eval_envs=n_eval,
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
def test_continuous_offline_end_to_end_schema(algo, tmp_path, monkeypatch):
    ds = _build_dataset(tmp_path, monkeypatch, "pendulum/rand-v0", "random", 6, 0)
    run_dir = tmp_path / f"run_{algo}"
    _run(ds, algo, epochs=2, grad_steps=10, run_dir=run_dir)

    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    assert float(train_rows[0]["q_loss"]) == float(train_rows[0]["q_loss"])  # finite
    # Pendulum return is negative; assert numeric+finite, NOT > 0.
    ev = float(eval_rows[-1]["eval_return_mean"])
    assert ev == ev and ev < 0.0


@pytest.mark.parametrize("algo", ALGOS)
def test_continuous_offline_learns_beats_random(algo, tmp_path, monkeypatch):
    """Trained on better-than-random (swing-up) data, eval clears the random
    Pendulum baseline (full-episode return ~= -1200) by a clear margin."""
    ds = _build_dataset(tmp_path, monkeypatch, "pendulum/heur-v0", "heuristic", 40, 0)
    run_dir = tmp_path / f"learn_{algo}"
    # 20 eval envs to tame Pendulum's high init-state variance. Grounded margin:
    # random baseline ~= -1224, heuristic data ceiling ~= -289; both algos land
    # in [-700, -537] across seeds (cql_alpha=5.0), so -850 is a clear margin
    # above random with comfortable headroom below the worst observed run.
    _run(ds, algo, epochs=40, grad_steps=200, run_dir=run_dir, n_eval=20)

    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    final_return = float(eval_rows[-1]["eval_return_mean"])
    assert final_return > -850.0, f"{algo} eval={final_return:.0f} did not beat random"

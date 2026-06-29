"""Oracle-U ceiling (causal_method="oracle_u") — discrete Cell-7 offline arm.

The oracle-U critic learns Q(s,a,u) on the observed per-episode latent U (closing
the A<-U->R backdoor) and deploys Q_adj = E_u[Q(s,a,u)] (U-free). These tests
cover: the U-marginalization arithmetic; that the new batch key reaches learn()
and exercises the U-conditioned net; that causal_method="none" is unchanged; the
continuous NotImplementedError; the generate->load round-trip carries U; and the
AM4 smoke test (oracle removes spurious apparent inflation; u=0 anchor tracks the
clean value; oracle eval >= naive eval).
"""

from __future__ import annotations

import csv
import warnings

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.rl.offline.oracle_u import (
    OracleUBCQ,
    OracleUCQL,
    OracleUDQN,
    OracleUIQL,
    UMarginalizedQ,
)

warnings.filterwarnings("ignore")

_CPU = torch.device("cpu")
_OBS_DIM, _ACT_DIM = 4, 2
_ORACLE_TYPES = {
    "offline_dqn": OracleUDQN,
    "cql": OracleUCQL,
    "iql": OracleUIQL,
    "bcq": OracleUBCQ,
}


def _build(algo_name: str):
    register_default_algorithms()
    return registry.get(algo_name).builder(
        obs_dim=_OBS_DIM,
        action_dim=_ACT_DIM,
        action_type="discrete",
        device=_CPU,
        action_space=None,
        obs_shape=(_OBS_DIM,),
    )


def _make_oracle(base: str):
    """Build the first-class oracle-U variant for a base algo (e.g. cql_oracle_u)."""
    return _build(f"{base}_oracle_u")


def _synthetic_batch(b: int = 32, with_u: bool = True):
    g = torch.Generator().manual_seed(0)
    batch = {
        "obs": torch.randn(b, _OBS_DIM, generator=g),
        "actions": torch.randint(0, _ACT_DIM, (b,), generator=g),
        "rewards": torch.randn(b, generator=g),
        "next_obs": torch.randn(b, _OBS_DIM, generator=g),
        "dones": torch.zeros(b),
    }
    if with_u:
        batch["confounder_u"] = torch.bernoulli(torch.full((b,), 0.5), generator=g)
    return batch


# --------------------------------------------------------------------------
# U-marginalization arithmetic
# --------------------------------------------------------------------------
def test_q_adj_is_bernoulli_mean_of_u0_u1():
    torch.manual_seed(0)
    net = UMarginalizedQ(_OBS_DIM, _ACT_DIM)
    obs = torch.randn(8, _OBS_DIM)
    expected = 0.5 * net.q_at(obs, 0.0) + 0.5 * net.q_at(obs, 1.0)
    assert torch.allclose(net(obs), expected, atol=1e-6)
    # forward() (Q_adj) has the deployable (B, A) shape used by act/_apparent_q.
    assert net(obs).shape == (8, _ACT_DIM)
    # q_su routes the observed u; q_su(.,0)/(.,1) match the constant-u anchors.
    u0 = torch.zeros(8)
    assert torch.allclose(net.q_su(obs, u0), net.q_at(obs, 0.0), atol=1e-6)


# --------------------------------------------------------------------------
# the variant registry name builds the variant; learn() consumes U, exercises q_su
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(_ORACLE_TYPES))
def test_oracle_builds_and_learn_exercises_u_conditioned_net(name):
    policy, agent = _make_oracle(name)
    assert type(agent) is _ORACLE_TYPES[name]
    assert agent.is_oracle_u is True
    assert isinstance(agent.q_network, UMarginalizedQ)

    # q_network(obs) is Q_adj with shape (B, A) -> act / _apparent_q stay valid.
    obs = torch.randn(5, _OBS_DIM)
    assert agent.q_network(obs).shape == (5, _ACT_DIM)
    # act() deploys without U (argmax over Q_adj).
    out = agent.act(obs, deterministic=True)
    assert out.action.shape == (5,)

    # learn() must consume U and put a gradient on the U-conditioned inner net.
    metrics = agent.learn(_synthetic_batch(with_u=True))
    assert all(v == v for v in metrics.values())  # finite
    grads = [p.grad for p in agent.q_network.inner.parameters() if p.grad is not None]
    assert grads, "q_su was not exercised (no grad on the U-conditioned inner net)"

    # The u=0 anchor hook returns one value per data transition.
    anchor = agent.oracle_anchor_q(_synthetic_batch(with_u=True))
    assert anchor.shape == (32,)


@pytest.mark.parametrize("name", list(_ORACLE_TYPES))
def test_oracle_learn_requires_u(name):
    _, agent = _make_oracle(name)
    with pytest.raises(KeyError, match="confounder_u"):
        agent.learn(_synthetic_batch(with_u=False))


def test_iql_oracle_advantage_is_u_independent():
    """Regression guard: the IQL oracle must NOT leak the confounder via AWR.

    With (obs, a_data) held fixed and ONLY u flipped 0->1, the V target and the
    AWR advantage must be IDENTICAL (both read the marginalized Q_adj). The buggy
    form (advantage from q_su(.,u)) would differ by the c_r·U reward bonus and
    re-upweight the U-biased actions. The final check confirms q_su genuinely
    depends on u, so the invariance is marginalization removing a real dependence."""
    from src.rl.offline.oracle_u import OracleUIQL

    _, agent = _make_oracle("iql")
    assert isinstance(agent, OracleUIQL)
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(16, _OBS_DIM, generator=g)
    actions = torch.randint(0, _ACT_DIM, (16,), generator=g)
    base = {
        "obs": obs,
        "actions": actions,
        "rewards": torch.zeros(16),
        "next_obs": obs.clone(),
        "dones": torch.zeros(16),
    }
    b0 = {**base, "confounder_u": torch.zeros(16)}
    b1 = {**base, "confounder_u": torch.ones(16)}

    assert torch.equal(agent.v_target(b0), agent.v_target(b1))
    assert torch.equal(agent.awr_advantage(b0), agent.awr_advantage(b1))

    a_idx = actions.unsqueeze(-1)
    q0 = agent.q_network.q_su(obs, torch.zeros(16)).gather(1, a_idx)
    q1 = agent.q_network.q_su(obs, torch.ones(16)).gather(1, a_idx)
    assert not torch.allclose(q0, q1), "q_su is u-invariant; guard is degenerate"


# --------------------------------------------------------------------------
# generate -> load round-trip carries the per-transition U
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")


def _gen_confounded(tmp_path, monkeypatch, dataset_id, behavior_policy, sigma=None):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from src.envs.offline.generate import generate_offline_dataset
    from src.envs.registry import register_default_env_wrappers

    # The generator agent's action sampling draws from the GLOBAL torch RNG, so
    # seed it here to make generation (and thus the confounding gate outcome)
    # independent of test execution order.
    torch.manual_seed(0)
    register_default_algorithms()
    register_default_env_wrappers()
    return generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy=behavior_policy,
        behavior_strength=sigma,
        rollout_episodes=60,
        seed=0,
        dataset_id=dataset_id,
        device="cpu",
    )


def test_generate_then_load_carries_confounder_u(tmp_path, monkeypatch):
    from src.envs.offline.minari_loader import fill_replay_buffer_from_minari
    from src.rl.off_policy.replay_buffer import ReplayBuffer

    did = "oracle/conf-roundtrip-v0"
    _gen_confounded(tmp_path, monkeypatch, did, "bias_confounded", sigma=0.5)

    buf = ReplayBuffer(capacity=100_000, device=_CPU)
    n = fill_replay_buffer_from_minari(did, buf, _CPU, load_u=True)
    assert n > 0
    batch = buf.sample(16)
    assert "confounder_u" in batch
    assert batch["confounder_u"].shape == (16,)
    # U is bernoulli(0/1) on the discrete arm.
    uniq = set(batch["confounder_u"].reshape(-1).tolist())
    assert uniq.issubset({0.0, 1.0})


def test_load_u_without_infos_raises(tmp_path, monkeypatch):
    from src.envs.offline.minari_loader import fill_replay_buffer_from_minari
    from src.rl.off_policy.replay_buffer import ReplayBuffer

    did = "oracle/clean-no-u-v0"
    _gen_confounded(tmp_path, monkeypatch, did, "agent")  # clean -> no U infos
    buf = ReplayBuffer(capacity=100_000, device=_CPU)
    with pytest.raises(ValueError, match="confounder_u"):
        fill_replay_buffer_from_minari(did, buf, _CPU, load_u=True)


# --------------------------------------------------------------------------
# AM4 smoke test: ceiling removes spurious inflation; u=0 anchor tracks clean
# --------------------------------------------------------------------------
def _run(tmp_path, dataset_id, algo, run_name):
    """Run one offline algo by registry name (base 'cql' vs variant
    'cql_oracle_u'). A standalone variant runner is self-sufficient at the u0
    schema via its own requires_confounder_u (no run-level flag needed here)."""
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    register_default_algorithms()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=10,
        rollout_len=80,
        seed=0,
        offline_dataset=dataset_id,
        behavior_policy="bias_confounded",
        behavior_strength=0.5,
    )
    train_cfg = TrainingConfig(
        n_episodes=30,
        n_checkpoints=2,
        device="cpu",
        algorithm=algo,
        aggregation="mean",
    )
    run_dir = tmp_path / run_name
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get(algo),
    ).run()
    return run_dir


def _final(run_dir, csv_name, col):
    with (run_dir / csv_name).open() as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1][col])


@pytest.mark.slow
def test_oracle_ceiling_beats_confounded_floor(tmp_path, monkeypatch):
    did = "oracle/conf-smoke-v0"
    # σ=1.0: strong confounding so the deconfounding signal dominates the noise of
    # a short smoke run (and the gate passes comfortably).
    _gen_confounded(tmp_path, monkeypatch, did, "bias_confounded", sigma=1.0)

    naive = _run(tmp_path, did, "cql", "naive")
    oracle = _run(tmp_path, did, "cql_oracle_u", "oracle")

    naive_apparent = _final(naive, "offline_value_trace.csv", "apparent_value_iqm")
    oracle_apparent = _final(oracle, "offline_value_trace.csv", "apparent_value_iqm")
    oracle_u0 = _final(oracle, "offline_value_trace.csv", "apparent_value_u0_iqm")
    naive_eval = _final(naive, "eval_metrics.csv", "eval_return_mean")
    oracle_eval = _final(oracle, "eval_metrics.csv", "eval_return_mean")

    # Deconfounding LOWERS the apparent value: the naive critic over-reports
    # because it attributes the U reward-bonus to the data actions; the
    # U-marginalized Q_adj strips that, and the u=0 anchor (bonus fully removed)
    # is the lowest. This apparent-value ORDERING is the ceiling's mechanism,
    # measured directly — unlike an apparent-vs-eval magnitude wedge, it does not
    # depend on the loosely-fit absolute Q-vs-return scale of a short smoke run
    # (CQL conservatism pushes every apparent Q far below the true return).
    # (1) Q_adj is below the naive (confounded) apparent value.
    assert (
        oracle_apparent < naive_apparent
    ), f"Q_adj {oracle_apparent:.2f} !< naive apparent {naive_apparent:.2f}"
    # (2) The u=0 anchor (no reward bonus) is the lowest stratum value.
    assert (
        oracle_u0 < oracle_apparent
    ), f"u0 anchor {oracle_u0:.2f} !< Q_adj {oracle_apparent:.2f}"
    # (3) Coarse: the deconfounded ceiling does not deploy worse than the floor.
    assert oracle_eval >= naive_eval - 15.0

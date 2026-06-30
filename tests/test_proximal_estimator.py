"""PR-2b — proximal deconfounding estimator (stochastic-EM through the OracleU hook).

Gates: the FIVE-KEYS invariant (proximal never loads the realized U — it INFERS
it), c_r=0 collapse-to-floor (the safety proof), and c_r=0.5 movement off the
naive floor toward the oracle-U ceiling. No byte-identity net for proximal.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry

warnings.filterwarnings("ignore")
_CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# Gate 1 — five-keys invariant (fast, no dataset)
# --------------------------------------------------------------------------
def test_proximal_spec_does_not_require_u():
    register_default_algorithms()
    for name in (
        "offline_dqn_proximal",
        "bcq_proximal",
        "cql_proximal",
        "iql_proximal",
    ):
        spec = registry.get(name)
        assert spec.requires_confounder_u is False  # never loads the realized U
        assert spec.needs_episode_grouping is True
        assert spec.kind == "off_policy" and spec.data_regime == "offline"


def test_proximal_source_never_reads_realized_u():
    """confounder_u on the proximal path is WRITTEN by the E-step sampler, never
    READ from the dataset. Grep the real read-patterns (not prose): the source must
    not pull the dataset's stored U (`infos["confounder_u"]`) nor ask the loader for
    it (`load_u=True`)."""
    src = Path("src/rl/offline/proximal.py").read_text()
    assert 'infos["confounder_u"]' not in src  # never reads the dataset's stored U
    assert 'infos.get("confounder_u")' not in src
    assert "load_u=True" not in src  # never asks the loader to load the realized U


# --------------------------------------------------------------------------
# Training-based gates (need tiny Minari datasets).
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")


def _gen(tmp_path, monkeypatch, dataset_id, sigma):
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
        behavior_policy="bias_confounded",
        behavior_strength=sigma,
        rollout_episodes=40,
        seed=0,
        dataset_id=dataset_id,
        device="cpu",
    )
    return dataset_id


def _run(tmp_path, dataset_id, algo, run_name):
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    torch.manual_seed(0)
    register_default_algorithms()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=4,
        rollout_len=60,
        seed=0,
        offline_dataset=dataset_id,
        behavior_policy="bias_confounded",
        behavior_strength=0.5,
    )
    train_cfg = TrainingConfig(
        n_episodes=25,
        n_checkpoints=2,
        device="cpu",
        algorithm=algo,
        aggregation="mean",
    )
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(tmp_path / run_name), timestamp="t"),
        registry.get(algo),
    )
    runner.run()
    return runner.agent


def _apparent_q_at_data(agent, dataset_id):
    """Mean critic value at the dataset's data actions: Q for the plain (floor)
    critic, Q_adj=forward for the U-marginalized (proximal/oracle) critic."""
    from src.envs.offline.minari_loader import fill_replay_buffer_from_minari
    from src.rl.off_policy.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=100_000, device=_CPU)
    fill_replay_buffer_from_minari(dataset_id, buf, _CPU, load_u=False)
    batch = buf.sample(256)
    with torch.no_grad():
        q = agent.q_network(batch["obs"])  # plain Q or Q_adj (forward)
        a = batch["actions"].long().view(-1, 1)
        return float(q.gather(1, a).squeeze(-1).mean().item())


def test_proximal_loader_supplies_no_realized_u(tmp_path, monkeypatch):
    """Five-keys behavioral proof: on the proximal path the dataset is loaded
    load_u=False, so the loaded batch carries NONE of the realized U — proximal
    must INFER it. (That it then deconfounds is the c_r=0.5 movement gate.)"""
    from src.envs.offline.minari_loader import fill_replay_buffer_from_minari
    from src.rl.off_policy.replay_buffer import ReplayBuffer

    did = _gen(tmp_path, monkeypatch, "prox/no-u-v0", sigma=0.5)  # confounded data
    buf = ReplayBuffer(capacity=100_000, device=_CPU)
    fill_replay_buffer_from_minari(did, buf, _CPU, load_u=False)  # the proximal path
    batch = buf.sample(32)
    assert "confounder_u" not in batch  # the realized U is never available


def test_proximal_infers_u_and_collapses_at_cr0(tmp_path, monkeypatch):
    """Gate 2 — c_r=0 collapse at the MECHANISM level (the L2-on-delta safety proof).

    Strata coincide, so the shrinkage prior pins delta->0, the posterior reverts to
    the prior (r_tau->~0.5 for all episodes), both q_su heads train on the same
    ~50/50 data -> q_su(.,0) ~ q_su(.,1) (init-INDEPENDENT) -> Q_adj ~ the floor."""
    import statistics

    did = _gen(tmp_path, monkeypatch, "prox/cr0-v0", sigma=0.0)
    floor = _run(tmp_path, did, "cql", "floor0")
    prox = _run(tmp_path, did, "cql_proximal", "prox0")

    delta = abs(float(prox._proximal_em.rm.delta))
    r_taus = [
        float(ep[0]["r_tau"]) for ep in prox._proximal_em.seq_buffer.iter_episodes()
    ]
    floor_q = _apparent_q_at_data(floor, did)
    prox_q = _apparent_q_at_data(prox, did)
    obs = torch.randn(128, 4)
    with torch.no_grad():
        q_su_gap = float(
            (
                prox.q_network.q_su(obs, torch.ones(128))
                - prox.q_network.q_su(obs, torch.zeros(128))
            )
            .abs()
            .mean()
            .item()
        )

    assert delta < 0.1, f"delta did not collapse: {delta}"  # shift -> 0
    assert 0.35 <= statistics.median(r_taus) <= 0.65, r_taus  # posterior -> prior
    assert q_su_gap < 0.5, q_su_gap  # heads ~ U-invariant
    assert abs(prox_q - floor_q) < 2.0, (prox_q, floor_q)  # value ~ floor


def test_proximal_beats_floor_toward_oracle_at_cr05(tmp_path, monkeypatch):
    """Gate 3 — c_r=0.5: proximal removes spurious inflation (apparent below the
    naive floor) and moves toward the oracle-U ceiling. Inequalities, not a match."""
    did = _gen(tmp_path, monkeypatch, "prox/cr05-v0", sigma=0.5)
    floor = _run(tmp_path, did, "cql", "floorc")
    prox = _run(tmp_path, did, "cql_proximal", "proxc")
    oracle = _run(
        tmp_path, did, "cql_oracle_u", "oraclec"
    )  # ceiling reference (reads U)

    floor_q = _apparent_q_at_data(floor, did)
    prox_q = _apparent_q_at_data(prox, did)
    oracle_q = _apparent_q_at_data(oracle, did)

    # The EM separated the strata (it inferred U from the 5 keys).
    assert abs(float(prox._proximal_em.rm.delta)) > 0.1
    # Deconfounding lowers the apparent value from the naive floor...
    assert prox_q < floor_q, (prox_q, floor_q)
    # ...toward the oracle ceiling (closes the floor->ceiling gap).
    assert abs(prox_q - oracle_q) < abs(floor_q - oracle_q), (floor_q, prox_q, oracle_q)

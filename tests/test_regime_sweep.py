"""PR 5 — layout migration + shared-checkpoint sweep. Tests M1-M5.

M1  a cell's 7 datasets carry the SAME generator-checkpoint hash; the driver REFUSES
    a cell whose arms differ (on master every arm gets a fresh generator, so the
    hashes differ and the guard fires).
M2  a full offline_mdp cell runs end-to-end and produces the 7 result dirs at the
    expected parameter paths, each leaf holding the same file set a run dir holds.
M3  the reporting layer DERIVES {basic, biased, confounded} from (beta, sigma) and
    reslices without re-running — labels are never stored in a path.
M4  basic runs the full critic set and emits the RAW value_mse_to_oracle signal for
    the adaptive critics (the per-run null_calibrated column was removed in PR 6).
M5  _legacy/ is inert: no live code path globs the legacy cell_N taxonomy.
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import minari
import pytest
from src.benchmarking.regime_sweep import (
    arm_behavior,
    arm_label,
    assert_shared_generator,
    critics_for_arm,
    load_sweep_spec,
    param_dirname,
    reslice_results,
    results_leaf,
    run_cell,
    sweep_points,
)
from src.config.device import detect_device
from src.envs.offline.generate import build_generator_agent, generate_offline_dataset

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
_OFFLINE_MDP = _REPO / "reproducibility" / "rl_regimes" / "offline_mdp" / "sweep.yaml"
_DEV = str(detect_device())
_TINY = {
    "n_episodes": 1,
    "n_checkpoints": 2,
    "n_train_envs": 2,
    "n_eval_envs": 2,
    "rollout_len": 2,
    "rollout_episodes": 40,
}
_LEAF_FILES = {
    "config.yaml",
    "metadata.json",
    "train_metrics.csv",
    "eval_metrics.csv",
    "arm_diagnostics.csv",
    "critic_ablation_metrics.csv",
}


def _purge(prefix: str) -> None:
    for d in list(minari.list_local_datasets()):
        if str(d).startswith(prefix):
            try:
                minari.delete_dataset(d)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Layout: every cell's sweep.yaml declares the SAME canonical L + critic sets.  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "regime", ["offline_mdp", "offline_pomdp", "online_mdp", "online_pomdp"]
)
def test_sweep_yamls_declare_canonical_L(regime):
    import yaml as _yaml

    p = _REPO / "reproducibility" / "rl_regimes" / regime / "sweep.yaml"
    assert p.exists(), p
    cfg = _yaml.safe_load(p.read_text())
    assert cfg["regime"] == regime
    assert cfg.get("discrete_only") is True  # continuous arms hard-gated (PR 1/3)
    sw = cfg["sweep"]
    # the declared L must equal the canonical 7-point L (two arms sharing an origin).
    declared = {(0.0, 0.0)}
    declared |= {(float(b), 0.0) for b in sw["biased"]["beta"]}
    declared |= {(0.0, float(s)) for s in sw["confounded"]["sigma"]}
    assert declared == set(sweep_points())
    assert float(sw["basic"]["beta"]) == 0.0 and float(sw["basic"]["sigma"]) == 0.0
    # if critic sets are declared, they must match the canonical per-arm sets.
    if "critics" in cfg:
        for arm in ("basic", "biased", "confounded"):
            assert cfg["critics"][arm] == critics_for_arm(arm), (regime, arm)


# --------------------------------------------------------------------------- #
# offline_pomdp must use a recurrent-capable base (accepts lstm), and each        #
# runnable offline cell ships a tiny-budget sweep_smoke.yaml.                     #
# --------------------------------------------------------------------------- #
def test_offline_pomdp_uses_recurrent_capable_base():
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    spec = load_sweep_spec(
        _REPO / "reproducibility" / "rl_regimes" / "offline_pomdp" / "sweep.yaml"
    )
    # the pomdp arm runs with critic_network=lstm; the base MUST accept that. Plain
    # offline_dqn carries the reject-guard, so the cell has to declare the recurrent
    # variant (registered WITHOUT the guard). Regression guard for the base fix.
    assert spec.algos == ["offline_dqn_recurrent"]
    base = registry.get("offline_dqn_recurrent")  # registered (else KeyError)
    assert base.data_regime == "offline"


@pytest.mark.parametrize("regime", ["offline_mdp", "offline_pomdp"])
def test_sweep_smoke_yaml_is_a_tiny_runnable_spec(regime):
    p = _REPO / "reproducibility" / "rl_regimes" / regime / "sweep_smoke.yaml"
    assert p.exists(), p
    spec = load_sweep_spec(p)
    assert spec.regime == regime and spec.data_regime == "offline"
    assert spec.budget("n_episodes", 999) == 1  # tiny budget baked in
    assert len(spec.envs) == 1 and len(spec.algos) == 1 and len(spec.seeds) == 1


# --------------------------------------------------------------------------- #
# M1 — shared generator checkpoint hash + refusal                              #
# --------------------------------------------------------------------------- #
def test_m1_shared_generator_hash_and_refusal():
    _purge("test/m1")
    agent, shared_hash = build_generator_agent("CartPole-v1", "dqn", "random", seed=0)
    hashes = {}
    for beta, sigma in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)]:
        bp, strength = arm_behavior(beta, sigma)
        did = f"test/m1-{param_dirname(beta, sigma)}-v0"
        try:
            minari.delete_dataset(did)
        except Exception:
            pass
        ds = generate_offline_dataset(
            env_id="CartPole-v1",
            generator_algo="dqn",
            tier="random",
            behavior_policy=bp,
            behavior_strength=strength,
            pi_basic_epsilon=0.5,
            confounder_c_r=(1.0 if bp == "bias_confounded_action" else None),
            rollout_episodes=8,
            seed=0,
            dataset_id=did,
            agent=agent,  # the ONE shared generator
        )
        hashes[(beta, sigma)] = ds.storage.metadata["generator_checkpoint_hash"]

    # all arms collected under ONE pi_basic -> ONE hash; the driver ACCEPTS.
    assert len(set(hashes.values())) == 1
    assert assert_shared_generator(hashes) == shared_hash

    # the master behavior (a FRESH generator per arm) yields a DIFFERENT hash for the
    # arm built without the shared agent -> the guard REFUSES the cell.
    _purge("test/m1-fresh")
    fresh = generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="bias_confounded_action",
        behavior_strength=0.5,
        pi_basic_epsilon=0.5,
        confounder_c_r=1.0,
        rollout_episodes=8,
        seed=0,
        dataset_id="test/m1-fresh-v0",
    )
    fresh_hash = fresh.storage.metadata["generator_checkpoint_hash"]
    assert fresh_hash != shared_hash  # fresh agent -> different pi_basic
    mismatched = dict(hashes)
    mismatched[(0.0, 0.5)] = fresh_hash
    with pytest.raises(ValueError, match="shared-generator violation"):
        assert_shared_generator(mismatched)
    _purge("test/m1")


# --------------------------------------------------------------------------- #
# M2 — a full offline_mdp cell runs end-to-end -> 7 parameter leaves           #
# --------------------------------------------------------------------------- #
def test_m2_offline_mdp_cell_end_to_end(tmp_path):
    _purge("m2test/")
    root = tmp_path / "results"
    written = run_cell(
        _OFFLINE_MDP,
        results_root=str(root),
        dataset_prefix="m2test",
        envs=["CartPole-v1"],
        algos=["cql"],
        seeds=[0],
        budget_overrides=_TINY,
        device=_DEV,
    )
    # the 7 L-points, parameter-addressed (no label segments).
    pdirs = sorted(p.name for p in (root / "offline_mdp").iterdir() if p.is_dir())
    assert pdirs == [
        "beta_000_sigma_000",  # basic
        "beta_000_sigma_025",  # confounded
        "beta_000_sigma_050",
        "beta_000_sigma_100",
        "beta_025_sigma_000",  # biased
        "beta_050_sigma_000",
        "beta_075_sigma_000",
    ]
    # 19 leaves = basic(4) + biased(1×3) + confounded(4×3); each a full run dir.
    assert len(written) == 19
    for leaf in written:
        assert _LEAF_FILES <= {f.name for f in Path(leaf).iterdir()}, leaf
    _purge("m2test/")


# --------------------------------------------------------------------------- #
# M3 — derive {basic, biased, confounded} from (beta, sigma); reslice, no rerun #
# --------------------------------------------------------------------------- #
def test_m3_reslice_derives_arms_from_params(tmp_path):
    # the derivation is the ONLY source of the label.
    assert arm_label(0.0, 0.0) == "basic"
    assert arm_label(0.5, 0.0) == "biased"
    assert arm_label(0.0, 0.5) == "confounded"
    with pytest.raises(ValueError, match="off the L"):
        arm_label(0.5, 0.5)  # no cross-product

    # a synthetic parameter tree with NO label stored anywhere in a path.
    root = tmp_path / "results"
    for beta, sigma in sweep_points():
        for critic in critics_for_arm(arm_label(beta, sigma)):
            leaf = results_leaf(
                root, "offline_mdp", beta, sigma, "CartPole-v1", "cql", critic, 0
            )
            leaf.mkdir(parents=True, exist_ok=True)
            (leaf / "config.yaml").write_text("x: 1\n")

    recs = reslice_results(str(root), "offline_mdp")
    by_arm: dict[str, set] = {}
    for r in recs:
        by_arm.setdefault(r["arm"], set()).add((r["beta"], r["sigma"]))
    assert by_arm["basic"] == {(0.0, 0.0)}
    assert by_arm["biased"] == {(0.25, 0.0), (0.5, 0.0), (0.75, 0.0)}
    assert by_arm["confounded"] == {(0.0, 0.25), (0.0, 0.5), (0.0, 1.0)}
    # the label is DERIVED, never a path segment -> reslice is possible.
    for r in recs:
        assert not re.search(r"basic|biased|confounded", r["path"])


# --------------------------------------------------------------------------- #
# M4 — basic runs the FULL critic set; emits the RAW value_mse_to_oracle signal  #
#      the reporting layer's relative gate consumes (oracle_u = exact anchor)     #
# --------------------------------------------------------------------------- #
def test_m4_basic_runs_full_critic_set_and_emits_raw_signal(tmp_path):
    # basic's critic set is the FULL set (not optional — it is the null-calibration
    # run that makes the gate meaningful).
    assert critics_for_arm("basic") == [
        "observational",
        "proximal",
        "oracle_u",
        "sensitivity",
    ]

    _purge("m4test/")
    root = tmp_path / "results"
    run_cell(
        _OFFLINE_MDP,
        results_root=str(root),
        dataset_prefix="m4test",
        envs=["CartPole-v1"],
        algos=["cql"],  # conservative base — the right null-calibration learner
        seeds=[0],
        budget_overrides=_TINY,
        device=_DEV,
    )
    basic = root / "offline_mdp" / "beta_000_sigma_000" / "CartPole-v1" / "cql"
    # the full set actually ran -> all four per-critic leaves exist.
    assert sorted(p.name for p in basic.iterdir()) == [
        "observational",
        "oracle_u",
        "proximal",
        "sensitivity",
    ]

    def _last(critic):
        rows = list(
            csv.DictReader(
                (basic / critic / "0" / "critic_ablation_metrics.csv").open()
            )
        )
        top = max(int(r["episode"]) for r in rows)
        return [r for r in rows if int(r["episode"]) == top][0]

    # PR 6 (N1): the broken per-run null_calibrated column is GONE. The basic run
    # emits the RAW value_mse_to_oracle for the adaptive critics — the signal the
    # reporting layer's relative, seed-based, cell-level gate consumes (see
    # test_regime_report P5/P6). oracle_u is the exact anchor (scores against itself
    # -> MSE 0). The non-adaptive sensitivity critic reports pessimism_cost + gamma.
    for critic in ("observational", "proximal", "oracle_u"):
        row = _last(critic)
        assert "null_calibrated" not in row  # removed per-run column
        assert row["value_mse_to_oracle"] != ""  # RAW signal is logged
        assert row["pessimism_cost"] == ""  # adaptive: no pessimism column
    assert float(_last("oracle_u")["value_mse_to_oracle"]) == 0.0  # exact anchor
    sens = _last("sensitivity")
    assert sens["gamma"] == "2.0"  # the active MSM default, logged (PR 4)
    assert sens["pessimism_cost"] != ""  # sensitivity reports its cost
    _purge("m4test/")


# --------------------------------------------------------------------------- #
# M5 — _legacy/ is inert: no live code path globs the cell_N taxonomy          #
# --------------------------------------------------------------------------- #
def test_m5_legacy_is_inert_no_live_cell_glob():
    # the NEW driver never names the legacy cell taxonomy.
    assert (
        "cell_" not in (_REPO / "src" / "benchmarking" / "regime_sweep.py").read_text()
    )
    # the NEW sweep tool does not glob cell_N dirs. Only the historical comment
    # naming the replaced script mentions cells; no EXECUTABLE (non-comment) line does.
    tool = (_REPO / "tools" / "run_regime_sweep.sh").read_text()
    code_lines = [
        ln for ln in tool.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert not any("cell_" in ln for ln in code_lines)
    # no live src code ENUMERATES the reproducibility cell dirs. (The reporting layer
    # reconstructs cell_N from run NAMES — PR 6 scope — it does not glob cell_* dirs.)
    for py in (_REPO / "src").rglob("*.py"):
        txt = py.read_text()
        assert not re.search(r"glob\([^)]*cell_", txt), py
        assert "reproducibility/rl_regimes/cell_" not in txt, py

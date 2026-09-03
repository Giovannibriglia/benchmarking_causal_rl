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
_OFFLINE_MDP = (
    _REPO / "reproducibility" / "rl_regimes" / "offline_mdp" / "critic_ablation.yaml"
)
_DEV = str(detect_device())
_TINY = {
    "n_episodes": 1,
    "n_checkpoints": 2,
    "n_train_envs": 2,
    "n_eval_envs": 2,
    "rollout_len": 2,
    "rollout_episodes": 40,
    # small offline budget (else the merge with _base inherits 50_000 offline_grad_steps).
    "offline_grad_steps": 4,
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
# Layout: every cell's critic_ablation.yaml declares the canonical L + critic sets. #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "regime", ["offline_mdp", "offline_pomdp", "online_mdp", "online_pomdp"]
)
def test_sweep_yamls_declare_canonical_L(regime):
    import yaml as _yaml

    p = _REPO / "reproducibility" / "rl_regimes" / regime / "critic_ablation.yaml"
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
    # if critic sets are declared, they must match the canonical per-arm sets
    # for the cell's data regime (online sets exclude the offline-only
    # oracle_u/sensitivity strategies — no online algo variant exists).
    # EXCEPTION online_pomdp: observational only (online_dqn_proximal has no
    # recurrent trunk; an mlp-proximal-vs-lstm-observational comparison would
    # confound the encoder axis).
    if "critics" in cfg:
        data_regime = cfg.get("data_regime", "offline")
        for arm in ("basic", "biased", "confounded"):
            expected = (
                ["observational"]
                if regime == "online_pomdp"
                else critics_for_arm(arm, data_regime)
            )
            assert cfg["critics"][arm] == expected, (regime, arm)


# --------------------------------------------------------------------------- #
# offline_pomdp must use a recurrent-capable base (accepts lstm), and each        #
# runnable offline cell ships a tiny-budget critic_ablation_smoke.yaml.           #
# --------------------------------------------------------------------------- #
def test_offline_pomdp_uses_recurrent_capable_base():
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    spec = load_sweep_spec(
        _REPO
        / "reproducibility"
        / "rl_regimes"
        / "offline_pomdp"
        / "critic_ablation.yaml"
    )
    # the pomdp arm runs with critic_network=lstm; the base MUST accept that. Plain
    # offline_dqn carries the reject-guard, so the cell has to declare the recurrent
    # variant (registered WITHOUT the guard). Regression guard for the base fix.
    assert spec.algos == ["offline_dqn_recurrent"]
    base = registry.get("offline_dqn_recurrent")  # registered (else KeyError)
    assert base.data_regime == "offline"


@pytest.mark.parametrize("regime", ["offline_mdp", "offline_pomdp"])
def test_sweep_smoke_yaml_is_a_tiny_runnable_spec(regime):
    p = _REPO / "reproducibility" / "rl_regimes" / regime / "critic_ablation_smoke.yaml"
    assert p.exists(), p
    spec = load_sweep_spec(p)
    assert spec.regime == regime and spec.data_regime == "offline"
    # tiny budget baked in (20 since the offline-budget recalibration @d7f137e —
    # the file changed without this pin; production is 250)
    assert spec.budget("n_episodes", 999) == 20
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


def test_e1_yamls_resolve_from_declaration_alone():
    """Every e1_*.yaml parses under STRICT mode and carries the facts the
    driver's CELLS tuple used to encode — the two-construction-sites fix
    (2026-09-03). A key the loader does not parse now RAISES instead of
    silently documenting."""
    from pathlib import Path

    from src.benchmarking.regime_sweep import load_sweep_spec

    d = Path("reproducibility/rl_regimes/diagrams")
    pairs = 0
    for f in sorted(d.glob("e1_*.yaml")):
        spec = load_sweep_spec(f)
        assert spec.source_cell, f"{f.name}: source_cell missing"
        assert spec.e1_cell, f"{f.name}: e1_cell missing"
        assert spec.eval_confounded_reward and spec.eval_confounded_mode == "analytic"
        is_grace = f.name.endswith("_grace.yaml")
        assert spec.grace_reward_transform is is_grace, f.name
        if is_grace and spec.e1_cell != "danull":
            assert spec.grace_proxy_names == ("Z", "W", "V"), f.name
        pairs += 1
    assert pairs == 10  # five cells x two arms, d100s0 included


def test_e1_strict_mode_refuses_unknown_keys(tmp_path):
    import pytest as _pytest
    from src.benchmarking.regime_sweep import load_sweep_spec

    bad = tmp_path / "e1_bogus.yaml"
    bad.write_text("regime: offline_mdp\ndata_regime: offline\nnot_a_key: 1\n")
    with _pytest.raises(ValueError, match="not_a_key"):
        load_sweep_spec(bad)


def test_d100s0_declares_the_substituted_seed():
    from src.benchmarking.regime_sweep import load_sweep_spec

    spec = load_sweep_spec("reproducibility/rl_regimes/diagrams/e1_d100s0.yaml")
    assert spec.seeds == [0, 2, 3]  # s1's declared-in-advance substitution
    # sigma=0 is declared through the BASIC origin (arm_behavior: basic IS
    # the confounded mechanism at sigma=0), never a sigma=0 confounded entry.
    assert spec.include_basic and spec.sigma_arm == ()


def test_e1_yamls_resolve_to_certified_ids():
    """THE closing test (required on review, 2026-09-03): every declared e1
    cell resolves — from its YAML alone — to dataset ids that exist in a
    generation report and carry their certification stamp; base and grace
    pairs share ids (paired arms), distinct cells never do. This is the loop
    between 'the YAML defines the campaign' and 'the campaign runs on
    certified data' closed as one fact."""
    from pathlib import Path

    from src.benchmarking.regime_sweep import load_sweep_spec
    from tools.run_e1 import resolve_certified_ids_for_spec

    by_cell = {}
    for f in sorted(Path("reproducibility/rl_regimes/diagrams").glob("e1_*.yaml")):
        spec = load_sweep_spec(f)
        ids = resolve_certified_ids_for_spec(spec)
        assert ids, f.name
        for (env, sd, sg), (did, stamp) in ids.items():
            assert stamp and all(v is not False for v in stamp.values()), (
                f.name,
                did,
                stamp,
            )
        got = tuple(sorted(d for d, _ in ids.values()))
        prior = by_cell.setdefault(spec.e1_cell, (f.name, got))
        assert (
            prior[1] == got
        ), f"paired arms of {spec.e1_cell} resolve differently: {prior[0]} vs {f.name}"
    all_sets = {cell: set(v[1]) for cell, v in by_cell.items()}
    for a in all_sets:
        for b in all_sets:
            if a < b:
                assert not (all_sets[a] & all_sets[b]), (a, b)
    # the reviewer's named case, verbatim: d100s0 == the sigma-0 report's ids
    import json

    report_ids = {
        r["dataset_id"]
        for r in json.loads(
            Path("results/dd_sweep_sigma0_generation/report.json").read_text()
        )
    }
    assert set(all_sets["d100s0"]) == report_ids

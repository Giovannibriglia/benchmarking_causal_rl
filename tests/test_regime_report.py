"""PR 6 — the reporting reader for the (regime × L-sweep) results/ tree. Tests P1-P6.

P1  segment parser: a results/ leaf -> (regime,beta,sigma,env,algo,critic,seed); a σ
    sweep is collected across DIFFERENT beta_* parents (the sibling-glob fix).
P2  legacy runs/ name-parsing STILL works (dispatch didn't break it).
P3  subcell labels DERIVED from (beta,sigma) match reslice_results; no label read.
P4  seed aggregation: mean + across-seed sd over 5 seeds at one point.
Q1  the gate uses the FIXED reference denominator, not the judged cell's own noise:
    inflating cell noise 4x with the gap unchanged keeps the verdict (the pin-k fix).
Q2  a missing (env,algo) reference -> uncalibrated (blank), never True.
Q3  correct endpoint (gap ~ noise_ref) -> True at k=2.4; broken (gap ~ 5.75*noise_ref)
    -> False. Both from the measured numbers.
P6  reads a real offline_mdp cell end-to-end (5 seeds, tiny budget) and emits the
    aggregated table with derived labels + the fixed-denominator null_calibrated verdict.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import minari
import pytest
from src.benchmarking import regime_report as rr
from src.benchmarking.plotting import _sigma_from_run
from src.benchmarking.regime_sweep import (
    arm_label,
    critics_for_arm,
    reslice_results,
    results_leaf,
    run_cell,
    sweep_points,
)
from src.config.device import detect_device

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
_OFFLINE_MDP = _REPO / "reproducibility" / "rl_regimes" / "offline_mdp" / "sweep.yaml"
_SEEDS = (0, 1, 2, 3, 4)


def _write_leaf(root, regime, beta, sigma, env, algo, critic, seed, mse):
    leaf = results_leaf(root, regime, beta, sigma, env, algo, critic, seed)
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "config.yaml").write_text("x: 1\n")
    with (leaf / "critic_ablation_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["episode", "critic", "value_mse_to_oracle", "apparent_q_mean"],
        )
        w.writeheader()
        w.writerow(
            {
                "episode": 1,
                "critic": critic,
                "value_mse_to_oracle": mse,
                "apparent_q_mean": 1.0,
            }
        )


def _synth_tree(
    root, *, obs_offset=0.30, prox_offset=0.28, jitter=0.02, sigma_slope=0.5
):
    """A synthetic results/ tree: obs and prox carry (near-)matched architectural
    offsets from the oracle at ALL σ; the confounding grows the MSE with σ."""
    for beta, sigma in sweep_points():
        for critic in critics_for_arm(arm_label(beta, sigma)):
            base = {
                "observational": obs_offset,
                "proximal": prox_offset,
                "oracle_u": 0.0,
            }
            for s in _SEEDS:
                mse = (
                    base.get(critic, obs_offset)
                    + jitter * ((s % 5) - 2)
                    + sigma_slope * sigma
                )
                _write_leaf(
                    root,
                    "offline_mdp",
                    beta,
                    sigma,
                    "CartPole-v1",
                    "cql",
                    critic,
                    s,
                    round(mse, 6),
                )


# --------------------------------------------------------------------------- #
# P1 — segment parser + σ-sibling collection across beta_* parents             #
# --------------------------------------------------------------------------- #
def test_p1_segment_parser_and_sigma_siblings(tmp_path):
    root = tmp_path / "results"
    _synth_tree(root)

    leaf = results_leaf(
        root, "offline_mdp", 0.0, 0.5, "CartPole-v1", "cql", "observational", 3
    )
    got = rr.parse_results_leaf(leaf)
    assert got["regime"] == "offline_mdp"
    assert (got["beta"], got["sigma"]) == (0.0, 0.5)
    assert got["arm"] == "confounded"  # DERIVED
    assert (got["env"], got["algo"], got["critic"], got["seed"]) == (
        "CartPole-v1",
        "cql",
        "observational",
        3,
    )

    # the σ-sweep siblings live under DIFFERENT beta_* parents (beta_000_sigma_000,
    # _025, _050, _100) yet are collected into one group keyed by (env,algo,critic,seed).
    sib = rr.collect_sigma_siblings(str(root), "offline_mdp")
    key = ("CartPole-v1", "cql", "observational", 0)
    assert sorted(sib[key].keys()) == [0.0, 0.25, 0.5, 1.0]
    # each collected sibling really lives under its own beta_000_sigma_* parent.
    for sigma, path in sib[key].items():
        assert f"sigma_{int(sigma * 100):03d}" in Path(path).parts[-5]


# --------------------------------------------------------------------------- #
# P2 — legacy runs/ name-parsing still works (dispatch didn't break it)        #
# --------------------------------------------------------------------------- #
def test_p2_legacy_name_parsing_still_works(tmp_path):
    # legacy runs/ folder: σ in the NAME.
    legacy = (
        tmp_path
        / "runs"
        / "rl_regimes"
        / "cell_7"
        / "confounded_sigma_050_discrete_20260101_000000"
    )
    legacy.mkdir(parents=True)
    assert _sigma_from_run(legacy) == 0.50
    # new results/ leaf: σ in the SEGMENT.
    seg = results_leaf(
        tmp_path / "results",
        "offline_mdp",
        0.0,
        0.25,
        "CartPole-v1",
        "cql",
        "proximal",
        0,
    )
    seg.mkdir(parents=True)
    assert _sigma_from_run(seg) == 0.25
    # σ=1.00 segment (three digits, not truncated).
    seg100 = results_leaf(
        tmp_path / "results",
        "offline_mdp",
        0.0,
        1.0,
        "CartPole-v1",
        "cql",
        "proximal",
        0,
    )
    seg100.mkdir(parents=True)
    assert _sigma_from_run(seg100) == 1.00


# --------------------------------------------------------------------------- #
# P3 — labels DERIVED from (beta,sigma) match reslice_results; none in a path  #
# --------------------------------------------------------------------------- #
def test_p3_labels_derived_match_reslice(tmp_path):
    root = tmp_path / "results"
    _synth_tree(root)
    recs = rr.aggregate_over_seeds(str(root), "offline_mdp", ("value_mse_to_oracle",))
    # arms present + derived-from-params, matching reslice_results' derivation.
    by_arm = {}
    for r in recs:
        by_arm.setdefault(r["arm"], set()).add((r["beta"], r["sigma"]))
    assert by_arm["basic"] == {(0.0, 0.0)}
    assert by_arm["biased"] == {(0.25, 0.0), (0.5, 0.0), (0.75, 0.0)}
    assert by_arm["confounded"] == {(0.0, 0.25), (0.0, 0.5), (0.0, 1.0)}
    # cross-check against PR 5's reslice_results (same derivation, no stored labels).
    resliced = {
        (r["beta"], r["sigma"]): r["arm"]
        for r in reslice_results(str(root), "offline_mdp")
    }
    for r in recs:
        assert resliced[(r["beta"], r["sigma"])] == r["arm"]
    # no subcell label is stored in any leaf path (labels are derived, not read).
    for rec in reslice_results(str(root), "offline_mdp"):
        assert not any(lbl in rec["path"] for lbl in ("basic", "biased", "confounded"))


# --------------------------------------------------------------------------- #
# P4 — seed aggregation: mean + across-seed sd over 5 seeds                     #
# --------------------------------------------------------------------------- #
def test_p4_seed_aggregation(tmp_path):
    root = tmp_path / "results"
    _synth_tree(root, obs_offset=0.30, jitter=0.02)
    recs = rr.aggregate_over_seeds(str(root), "offline_mdp", ("value_mse_to_oracle",))
    basic_obs = next(
        r for r in recs if r["arm"] == "basic" and r["critic"] == "observational"
    )
    assert basic_obs["n_seeds"] == 5
    # mean over the symmetric jitter {-2,-1,0,1,2}*0.02 -> the offset itself.
    assert abs(basic_obs["value_mse_to_oracle_mean"] - 0.30) < 1e-6
    # across-seed sd is a real positive number (needs n>=2).
    assert basic_obs["value_mse_to_oracle_sd"] > 0.0


# --------------------------------------------------------------------------- #
# Q1 — the gate uses the FIXED reference denominator, NOT the judged cell's own  #
#      noise: inflating cell noise 4x with the gap unchanged keeps the verdict.  #
#      (The regression the pin-k finding demands.)                               #
# --------------------------------------------------------------------------- #
_REF = {("CartPole-v1", "cql"): 0.04257}  # the measured correct-pipeline noise_ref


def test_q1_gate_uses_fixed_reference_not_cell_noise(tmp_path):
    # the synthetic tree's jitter is symmetric (mean 0), so the cell MEAN — and thus
    # the gap — is the offset regardless of jitter; only the cell noise scales.
    root_a = tmp_path / "a"
    _synth_tree(root_a, obs_offset=0.069, prox_offset=0.026, jitter=0.005)
    a = rr.compute_null_calibration(str(root_a), "offline_mdp", reference=_REF)[0]
    root_b = tmp_path / "b"
    _synth_tree(root_b, obs_offset=0.069, prox_offset=0.026, jitter=0.020)  # 4x jitter
    b = rr.compute_null_calibration(str(root_b), "offline_mdp", reference=_REF)[0]

    assert b["cell_noise"] > 3 * a["cell_noise"]  # cell noise really inflated ~4x
    assert b["gap"] == pytest.approx(a["gap"], abs=1e-9)  # gap unchanged
    assert b["noise_ref"] == a["noise_ref"] == 0.04257  # FIXED denominator
    assert b["ratio"] == pytest.approx(a["ratio"])  # denominator did NOT move
    assert b["null_calibrated"] is a["null_calibrated"]  # SAME verdict


# --------------------------------------------------------------------------- #
# Q2 — a missing (env,algo) reference -> UNCALIBRATED (blank), never True       #
# --------------------------------------------------------------------------- #
def test_q2_missing_reference_is_uncalibrated(tmp_path):
    root = tmp_path / "r"
    _synth_tree(root, obs_offset=0.069, prox_offset=0.026, jitter=0.01)
    # reference has NO (CartPole-v1, cql) key -> the gate cannot verdict.
    nc = rr.compute_null_calibration(
        str(root), "offline_mdp", reference={("Other-v0", "xyz"): 0.05}
    )[0]
    assert nc["noise_ref"] is None
    assert nc["ratio"] is None
    assert nc["null_calibrated"] is None  # never a silent True without a reference


# --------------------------------------------------------------------------- #
# Q3 — correct endpoint calibrated at k=2.4; broken NOT (from the measured gaps) #
# --------------------------------------------------------------------------- #
def test_q3_correct_calibrated_broken_not_at_pinned_k(tmp_path):
    assert rr.NULL_CALIBRATION_K == 2.4  # the documented pinned constant

    # correct: gap ~ noise_ref (measured gap 0.043) -> ratio ~1.0 < k=2.4 -> True.
    root_ok = tmp_path / "ok"
    _synth_tree(root_ok, obs_offset=0.069, prox_offset=0.026, jitter=0.01)
    ok = rr.compute_null_calibration(str(root_ok), "offline_mdp", reference=_REF)[0]
    assert ok["ratio"] == pytest.approx(0.043 / 0.04257, abs=0.05)
    assert ok["null_calibrated"] is True

    # broken (bare-DQN obs): gap ~ 5.75 * noise_ref (measured 0.245) -> ratio ~5.75
    # > k=2.4 -> False. The confound the gap/(cell-noise) gate greenlit now FAILS.
    root_bad = tmp_path / "bad"
    _synth_tree(root_bad, obs_offset=0.271, prox_offset=0.026, jitter=0.01)
    bad = rr.compute_null_calibration(str(root_bad), "offline_mdp", reference=_REF)[0]
    assert bad["ratio"] > 5.0
    assert bad["null_calibrated"] is False


# --------------------------------------------------------------------------- #
# P6 — reads a REAL offline_mdp cell end-to-end (5 seeds, tiny budget)          #
# --------------------------------------------------------------------------- #
def _purge(prefix):
    for d in list(minari.list_local_datasets()):
        if str(d).startswith(prefix):
            try:
                minari.delete_dataset(d)
            except Exception:
                pass


def test_p6_reads_real_offline_mdp_cell_end_to_end(tmp_path):
    _purge("p6test/")
    root = tmp_path / "results"
    run_cell(
        _OFFLINE_MDP,
        results_root=str(root),
        dataset_prefix="p6test",
        envs=["CartPole-v1"],
        algos=["cql"],
        seeds=list(_SEEDS),  # 5 seeds — the axis the relative gate needs
        budget_overrides={
            "n_episodes": 1,
            "n_checkpoints": 2,
            "n_train_envs": 2,
            "n_eval_envs": 2,
            "rollout_len": 2,
            "rollout_episodes": 40,
            # small offline budget (else the _base merge inherits 50_000 steps).
            "offline_grad_steps": 4,
        },
        device=str(detect_device()),
    )
    agg, nc = rr.build_report(str(root), "offline_mdp", k=2.0)

    # aggregated table: derived labels + seed aggregation over the real cell.
    assert {r["arm"] for r in agg} == {"basic", "biased", "confounded"}
    basic_obs = next(
        r for r in agg if r["arm"] == "basic" and r["critic"] == "observational"
    )
    assert basic_obs["n_seeds"] == 5
    assert (
        basic_obs["value_mse_to_oracle_mean"] == basic_obs["value_mse_to_oracle_mean"]
    )  # not NaN

    # null-calibration verdict computed at the cell level on the FIXED reference
    # denominator (loaded from null_cal_reference.yaml for CartPole-v1/cql).
    assert len(nc) == 1
    row = nc[0]
    assert row["env"] == "CartPole-v1" and row["algo"] == "cql"
    assert row["n_seeds"] == 5
    for col in ("gap", "noise_ref", "cell_noise", "ratio", "null_calibrated"):
        assert col in row
    # noise_ref re-measured 2026-07-21 (feat/noise-ref-v3) at the NEW offline budget
    # (offline_grad_steps=50_000, rollout_episodes=3000): cql 132.26 (was 574.37 at the
    # old 256k/RE40 budget). Still straddles k=2.4 but the cql margin is now THIN
    # (broken gap/noise_ref 2.95) — see null_cal_reference.yaml.
    assert (
        row["noise_ref"] == 132.26
    )  # the stored production reference, not the cell's noise
    assert row["null_calibrated"] is not None  # reference present -> a real verdict
    _purge("p6test/")

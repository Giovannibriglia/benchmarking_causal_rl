"""PR 2 — declarative action-dependent confounding gate.

The gate is DISPATCHED on the declarative gate['type'] (never string-matched on the
behavior-policy name). The additive path (cells 7/8) is byte-frozen; the
action-dependent path is a POINT check (see _action_dependent_signature).

  G1  a cell_9-style action-gated dataset now PASSES and trains end-to-end
      (non-empty eval_metrics.csv -- exactly what cell_9 failed to produce).
  G2  the additive path is byte-frozen: 4-key signature, no gate_type.
  G3  a DEGENERATE pi_basic (epsilon->0) FAILS the gate on A3 (inert confounder).
  G4  sigma=0 with c_r=1.0 PASSES (no U->A edge, reward shift present).
  G5  a tampered dataset (shuffle U vs A) FAILS A2.
"""

from __future__ import annotations

import csv
import warnings

import minari
import numpy as np
import pytest
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.device import detect_device
from src.envs.offline.generate import (
    ACTION_DEPENDENT_GATE,
    compute_confounding_signature,
    enforce_confounding_gate,
    generate_offline_dataset,
)
from src.envs.registry import register_default_env_wrappers

warnings.filterwarnings("ignore")


def _gen(dataset_id, behavior_policy, sigma, *, n_eps=200, pi_basic_epsilon=None):
    try:
        minari.delete_dataset(dataset_id)
    except Exception:
        pass
    return generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy=behavior_policy,
        behavior_strength=sigma,
        pi_basic_epsilon=pi_basic_epsilon,
        rollout_episodes=n_eps,
        seed=0,
        dataset_id=dataset_id,
    )


def _train_cql(tmp_path, dataset_id, behavior_policy, sigma):
    register_default_algorithms()
    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=2,
        seed=0,
        offline_dataset=dataset_id,
        behavior_policy=behavior_policy,
        behavior_strength=sigma,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,
        n_checkpoints=2,
        device=str(detect_device()),
        algorithm="cql",
    )
    run_dir = tmp_path / "run"
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("cql"),
    ).run()
    return run_dir


# ---------------------------------------------------------------------------
# G1 — action-gated dataset PASSES and trains end-to-end (the cell_9 fix).
# ---------------------------------------------------------------------------
def test_g1_action_gated_passes_and_trains(tmp_path):
    did = "test/pr2-g1-action-v0"
    ds = _gen(did, "bias_confounded_action", 1.0, n_eps=250)
    try:
        meta = ds.storage.metadata
        assert meta["gate_type"] == "action_dependent"
        assert meta["gate_test_passed"] is True, meta
        run_dir = _train_cql(tmp_path, did, "bias_confounded_action", 1.0)
        with open(run_dir / "eval_metrics.csv") as f:
            rows = list(csv.DictReader(f))
        # The acceptance bar: cell_9 produced a header-only eval_metrics.csv.
        assert len(rows) > 0, "eval_metrics.csv must be non-empty"
    finally:
        minari.delete_dataset(did)


# ---------------------------------------------------------------------------
# G2 — additive path byte-frozen: exact 4-key signature, no gate_type.
# ---------------------------------------------------------------------------
def test_g2_additive_signature_byte_frozen():
    did = "test/pr2-g2-additive-v0"
    ds = _gen(did, "bias_confounded", 0.5, n_eps=60)
    try:
        meta = ds.storage.metadata
        assert set(meta.keys()) >= {
            "corr_a_r_marginal",
            "corr_a_r_partial_given_u",
            "gate_test_passed",
            "behavior_strength_sigma",
        }
        assert "gate_type" not in meta  # additive stamps no gate_type (byte-frozen)
        assert "p_hat" not in meta and "pi_basic_entropy" not in meta
        assert meta["behavior_strength_sigma"] == 0.5
    finally:
        minari.delete_dataset(did)


def test_g2_additive_computation_unchanged():
    """The additive branch is bit-for-bit the pre-PR-2 computation."""
    rng = np.random.default_rng(0)
    n = 5000
    u = (rng.random(n) < 0.5).astype(float)
    a = (rng.random(n) < 0.5 + 0.3 * (u - 0.5)).astype(float)
    r = a + 1.0 * u + 0.1 * rng.standard_normal(n)
    samples = {"a": a, "r": r, "u": u, "intervened": np.array([]), "p_s": np.array([])}
    sig = compute_confounding_signature(samples, 0.5)  # default gate = additive
    assert set(sig.keys()) == {
        "corr_a_r_marginal",
        "corr_a_r_partial_given_u",
        "gate_test_passed",
        "behavior_strength_sigma",
    }

    # recompute the exact legacy formula
    def pear(x, y):
        return 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    r_ar, r_au, r_ru = pear(a, r), pear(a, u), pear(r, u)
    partial = (r_ar - r_au * r_ru) / np.sqrt((1 - r_au**2) * (1 - r_ru**2))
    assert sig["corr_a_r_marginal"] == float(r_ar)
    assert abs(sig["corr_a_r_partial_given_u"] - partial) < 1e-12
    assert sig["gate_test_passed"] == bool(abs(r_ar) > 0.2 and abs(partial) < 0.05)


# ---------------------------------------------------------------------------
# G3 — a DEGENERATE pi_basic (epsilon -> 0 = greedy, p_s in {0,1}) FAILS on A3.
# The single most important guarantee: a silently-inert confounder cannot ship.
# ---------------------------------------------------------------------------
def test_g3_degenerate_pi_basic_fails_a3(tmp_path):
    did = "test/pr2-g3-degenerate-v0"
    ds = _gen(did, "bias_confounded_action", 1.0, n_eps=80, pi_basic_epsilon=0.0)
    try:
        meta = ds.storage.metadata
        assert meta["pi_basic_entropy"] < 1e-9  # greedy -> p_s in {0,1} -> inert
        assert meta["check_a3_p_nondegenerate"] is False
        assert meta["gate_test_passed"] is False
        # the runner must REFUSE to train on it.
        with pytest.raises(ValueError, match="action-dependent confounding gate"):
            _train_cql(tmp_path, did, "bias_confounded_action", 1.0)
    finally:
        minari.delete_dataset(did)


# ---------------------------------------------------------------------------
# G4 — sigma=0 with c_r=1.0 PASSES: no U->A edge (positive assertion), reward
# shift present. NOT a "skip"; the action-dependent gate is authoritative at sigma=0.
# ---------------------------------------------------------------------------
def test_g4_sigma_zero_passes(tmp_path):
    did = "test/pr2-g4-sigma0-v0"
    ds = _gen(did, "bias_confounded_action", 0.0, n_eps=150)
    try:
        meta = ds.storage.metadata
        assert meta["gate_type"] == "action_dependent"
        assert abs(meta["edge_statistic_observed"]) < 0.03  # NO U->A edge at sigma=0
        assert meta["check_a4_gated_reward"] is True  # c_r=1.0 reward shift present
        assert meta["gate_test_passed"] is True, meta
        _train_cql(tmp_path, did, "bias_confounded_action", 0.0)  # must not raise
    finally:
        minari.delete_dataset(did)


# ---------------------------------------------------------------------------
# G5 — a tampered dataset (shuffle U against A) FAILS A2.
# ---------------------------------------------------------------------------
def _swap_samples(sigma, p=0.5, n=30000, c_r=1.0, seed=0):
    rng = np.random.default_rng(seed)
    u = (rng.random(n) < 0.5).astype(float)
    ps = np.full(n, p)
    p1, p0 = p * (1 + sigma * (1 - p)), p * (1 - sigma * (1 - p))
    ab = (rng.random(n) < np.where(u > 0.5, p1, p0)).astype(float)
    r = 0.3 + 0.7 * (1 - ab) + c_r * u * ab + 0.1 * rng.standard_normal(n)
    return {"a": ab, "r": r, "u": u, "p_s": ps, "intervened": np.zeros(n)}


def test_g5_tampered_dataset_fails_a2():
    s = _swap_samples(1.0)
    ok = compute_confounding_signature(s, 1.0, gate=ACTION_DEPENDENT_GATE)
    assert ok["gate_test_passed"] is True  # the honest dataset passes

    tampered = dict(s)
    tampered["u"] = np.random.default_rng(1).permutation(s["u"])  # break U vs A
    bad = compute_confounding_signature(tampered, 1.0, gate=ACTION_DEPENDENT_GATE)
    assert bad["check_a2_point_corr"] is False
    assert bad["gate_test_passed"] is False
    with pytest.raises(ValueError, match="check_a2_point_corr"):
        enforce_confounding_gate({**bad, "gate_type": "action_dependent"}, "tampered")


# ---------------------------------------------------------------------------
# enforce_confounding_gate — the deduped enforcement (declarative dispatch).
# ---------------------------------------------------------------------------
def test_enforce_missing_signature_raises():
    with pytest.raises(ValueError, match="requires the confounding-signature"):
        enforce_confounding_gate({}, "no-sig")


def test_enforce_additive_sigma0_skips_but_action_dependent_authoritative():
    # additive (no gate_type) sigma=0 baseline: skipped even with gate_test_passed False.
    enforce_confounding_gate(
        {"gate_test_passed": False, "behavior_strength_sigma": 0.0}, "add-s0"
    )
    # action_dependent at sigma=0 is NOT skipped: a failed gate still raises.
    with pytest.raises(ValueError, match="action-dependent"):
        enforce_confounding_gate(
            {
                "gate_test_passed": False,
                "gate_type": "action_dependent",
                "behavior_strength_sigma": 0.0,
                "check_a3_p_nondegenerate": False,
            },
            "act-s0",
        )

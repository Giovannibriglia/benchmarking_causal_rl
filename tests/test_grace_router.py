"""§D.5 — router calibration + the R5 leakage assertions + A9.3 gating.

The router's thresholds are null-calibrated (mean + k*sd on basic-run stats,
k = NULL_CALIBRATION_K); on null data the confounding gate fires at a rate
consistent with that threshold; on synthetic strong-U data it fires. The
router consumes ONLY data-derivable statistics: a poisoned realized-U never
changes any output (R5), and a declared graph without U -> A blocks the
confounded serving mode regardless of the statistics (A9.3)."""

from __future__ import annotations

import torch
from src.rl.offline.grace import cell_graph, GraceMachinery, GraceOptions
from src.rl.offline.grace.router import RegimeRouter
from tests._grace_test_utils import DEV, FakeSeqBuffer, make_confounded_episodes


def _null_stats(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        out.append(
            {
                "delta_a": float(torch.randn(1, generator=g)) * 0.01,
                "delta_r": 0.2 + float(torch.randn(1, generator=g)) * 0.02,
                "coverage": 0.4 + float(torch.randn(1, generator=g)) * 0.03,
                "width": 0.3 + float(torch.randn(1, generator=g).abs()) * 0.05,
            }
        )
    return out


def test_null_fire_rate_consistent_with_threshold():
    thr = RegimeRouter.calibrate(_null_stats(40, seed=1))
    router = RegimeRouter(thr)
    fresh = _null_stats(200, seed=2)
    fired = sum(1 for s in fresh if router.verdict(s).label == "confounded")
    # k = 1.5 one-sided on ~normal null stats -> ~7% tail; allow slack.
    assert fired / len(fresh) < 0.15, fired
    # And the null verdicts overwhelmingly read "basic".
    basics = sum(1 for s in fresh if router.verdict(s).label == "basic")
    assert basics / len(fresh) > 0.6


def test_full_path_confounded_fires_and_basic_does_not():
    def _components(confounded: bool, seed: int):
        eps, _ = make_confounded_episodes(
            n_ep=250, t_len=12, confounded=confounded, seed=seed
        )
        m = GraceMachinery(
            cell_graph("mdp", "template"),
            GraceOptions(n_bins=3, em_iters=8, ensemble_k=2),
            n_actions=2,
            device=DEV,
            gamma=0.9,
        )
        m.fit_from_buffer(FakeSeqBuffer(eps))
        return m.verdict.stats

    null_runs = [_components(False, s) for s in (11, 12, 13)]
    thr = RegimeRouter.calibrate(null_runs)
    router = RegimeRouter(thr)
    strong = _components(True, 21)
    v = router.verdict(strong)
    assert v.label == "confounded" and v.serve in ("do", "lower"), v
    v_null = router.verdict(_components(False, 22))
    assert v_null.label != "confounded", v_null


def test_coverage_defect_routes_to_lower():
    thr = RegimeRouter.calibrate(_null_stats(40))
    router = RegimeRouter(thr)
    v = router.verdict({"delta_a": 0.0, "delta_r": 0.2, "coverage": 0.01, "width": 0.3})
    assert v.label == "biased" and v.serve == "lower"


def test_unhealthy_identification_downgrades_to_lower():
    thr = RegimeRouter.calibrate(_null_stats(40))
    router = RegimeRouter(thr)
    v = router.verdict({"delta_a": 5.0, "delta_r": 1.0, "coverage": 0.4, "width": 99.0})
    assert v.label == "confounded" and v.serve == "lower"


def test_uncalibrated_never_routes_causally():
    v = RegimeRouter(None).verdict({"delta_a": 99.0, "delta_r": 99.0})
    assert v.label == "uncalibrated" and v.serve == "obs" and not v.calibrated


def test_a9_graph_gate_blocks_confounded_serving():
    thr = RegimeRouter.calibrate(_null_stats(40))
    router = RegimeRouter(thr, graph_ok=False)
    v = router.verdict({"delta_a": 99.0, "delta_r": 0.2, "coverage": 0.4, "width": 0.3})
    assert v.label != "confounded"
    assert any("U->A" in r for r in v.reasons)


def test_r5_realized_u_never_changes_any_output():
    """Identical machinery on the same data with (a) poisoned NaN U, (b) the
    true U, (c) no U key at all — every diagnostic and the served tables must
    be bitwise identical (the estimator/router never read the key)."""

    def _run(mode: str):
        eps, u_true = make_confounded_episodes(n_ep=120, t_len=10, seed=4)
        for ep, u in zip(eps, u_true):
            for tr in ep:
                if mode == "poison":
                    tr["confounder_u"] = torch.tensor(float("nan"))
                elif mode == "true":
                    tr["confounder_u"] = torch.tensor(float(u))
                else:
                    tr.pop("confounder_u", None)
        m = GraceMachinery(
            cell_graph("mdp", "template"),
            GraceOptions(n_bins=3, em_iters=6, ensemble_k=2),
            n_actions=2,
            device=DEV,
            gamma=0.9,
        )
        m.fit_from_buffer(FakeSeqBuffer(eps))
        return m

    runs = {mode: _run(mode) for mode in ("poison", "true", "absent")}
    ref = runs["poison"]
    # The poisoned-U run is finite everywhere: a single estimator-side read of
    # the NaN key would have propagated.
    assert torch.isfinite(ref.q_do_table).all() and torch.isfinite(ref.q_lo).all()
    for mode in ("true", "absent"):
        other = runs[mode]
        # allclose, not bitwise: CUDA index_add_/index_put_(accumulate) are
        # atomically non-deterministic across runs; any REAL read of the U key
        # would move results by O(1), not O(float-accumulation-order).
        assert torch.allclose(
            ref.q_do_table, other.q_do_table, rtol=1e-3, atol=1e-3
        ), mode
        assert torch.allclose(ref.q_lo, other.q_lo, rtol=1e-3, atol=1e-3), mode
        for k, v in ref._diag.items():
            ov = other._diag[k]
            if v != v:  # NaN-safe
                assert ov != ov, (mode, k)
            else:
                assert abs(float(v) - float(ov)) < 1e-3, (mode, k, v, ov)

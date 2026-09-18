"""Regressions for the L4 -> seam -> artifact diagnostics path.

Three defects, all in the REPORTING layer and all running in the flattering
direction, which is why none of them produced an error:

1. ``procedural_share`` returned ``inf`` on 0/0 -- maximal procedural
   instability reported for a statistic that cannot vary (d_a_null).
2. ``optimiser_var`` returned ``0.0`` when NO init fit survived -- "the
   optimiser contributes no variance" reported on no measurement (S8/S9).
3. The seam dropped ``res.meta``, so the bootstrap failure REASONS never
   reached a run artifact, leaving a bare rate -- the shape L4's own founding
   rule (2026-08-23) forbids.
"""

import json
import math

import torch
from src.rl.offline.grace.estimator import EpisodeData
from src.rl.offline.grace.l4 import point_id_interval
from src.rl.offline.grace.serving import _l4_diagnostics


class _Fit:
    """A clean fit, by the flags ``_dirty`` reads."""

    finished = True
    tau1_budget_bound = False
    backtrack_exhausted = False
    degenerate_mechanism = False
    reached_tau_one = True
    mechanism_degeneracy = {}
    converged = True
    stationary = True
    monotone = True
    backtracks = 0
    saturated_at_init = False
    initial_saturation = 0.0
    n_anneal = 1

    def estimate(self, _x):
        return self

    def label(self):
        return "stub"


class _Est:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, data, **_kw):
        return _Fit()

    def pin_reward_resolution(self, _other):
        pass


def _data(n_ep=8, per_ep=5):
    n = n_ep * per_ep
    return EpisodeData(
        state=torch.zeros(n, 2),
        action=torch.zeros(n, dtype=torch.long),
        reward=torch.ones(n),
        episode_ids=torch.arange(n_ep).repeat_interleave(per_ep),
        proxy={},
    )


def _run(target, **kw):
    return point_id_interval(
        make_estimator=_Est,
        data=_data(),
        target=target,
        alpha=0.1,
        b=19,
        fit_seed=0,
        init_seeds=(1, 2),
        **kw,
    )


def test_constant_statistic_reports_undefined_share_not_inf():
    """d_a_null's shape: every replicate returns the SAME value.

    Both variances are zero, the ratio is undefined, and the old code called
    that ``inf`` -- "100% procedural instability" on a statistic with no
    variance at all.
    """
    res = _run(lambda est, fit: 0.0)
    assert res.kind == "interval"
    assert res.lo == 0.0 and res.hi == 0.0
    assert math.isnan(res.procedural_share), res.procedural_share
    assert not math.isinf(res.procedural_share)


def test_real_spread_still_reports_a_finite_share():
    """The working case must keep working: both arms vary, ratio is a number."""
    seen = {"n": 0}

    def target(est, fit):
        seen["n"] += 1
        return float(est.seed) + 0.01 * seen["n"]

    res = _run(target)
    assert res.kind == "interval"
    assert math.isfinite(res.procedural_share)
    assert res.procedural_share >= 0.0


def test_optimiser_var_is_nan_when_no_init_fit_survives():
    """Unmeasurable must not read as measured-zero (S8/S9)."""

    class _DirtyOnInit(_Est):
        def fit(self, data, **kw):
            f = _Fit()
            if self.seed != 0:  # every init-perturbation fit comes back dirty
                f.finished = False
            return f

    res = point_id_interval(
        make_estimator=_DirtyOnInit,
        data=_data(),
        target=lambda est, fit: 1.0,
        alpha=0.1,
        b=19,
        fit_seed=0,
        init_seeds=(1, 2),
    )
    assert res.meta["n_init_fits"] == 1
    assert math.isnan(res.meta["optimiser_var"])
    assert math.isnan(res.procedural_share)


def test_seam_carries_the_bootstrap_reasons_not_just_the_rate():
    """L4's founding rule: a rate without reasons is uninterpretable."""
    res = _run(lambda est, fit: 0.0)
    flat = _l4_diagnostics(res)
    assert "boot_reasons" in flat
    assert flat["boot_n_requested"] == 19
    assert "boot_failure_rate" in flat
    assert flat["n_init_fits"] >= 1
    # every propagated value must survive the artifact writers' scalar filter
    for k, v in flat.items():
        assert isinstance(v, (int, float, str, bool)), (k, type(v))


def test_provenance_meta_is_valid_json(monkeypatch, tmp_path):
    """Non-finite floats must not reach the artifact as bare Infinity/NaN.

    ``monkeypatch`` is load-bearing, not decoration: importing the driver runs
    its module-level ``os.environ.setdefault("MINARI_DATASETS_PATH", ...)``,
    which is UNSCOPED. Without this the import silently repoints the whole
    pytest process at the real 16 GB store for every test that follows -- every
    other test in this suite isolates that variable per-test for exactly that
    reason.
    """
    import importlib.util
    import pathlib

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    spec = importlib.util.spec_from_file_location(
        "_run_e1", pathlib.Path("tools/run_e1.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    meta = {
        "procedural_share": float("inf"),
        "optimiser_var": float("nan"),
        "n_transitions": 49125,
        "boot_reasons": "",
        "ok": True,
    }
    out = json.dumps(mod._scalar_meta(meta))
    assert "Infinity" not in out and "NaN" not in out
    back = json.loads(out)
    assert back["procedural_share"] is None
    assert back["optimiser_var"] is None
    assert back["n_transitions"] == 49125
    assert back["ok"] is True


def test_transform_records_that_it_actually_fired():
    """The sixth-catch rule: record what the run DID, not only what it produced.

    A partial or absent substitution is invisible in the output CSVs -- the
    arm still trains, still evaluates, still writes a plausible number. So the
    count of rewards actually overwritten has to be on the artifact.
    """
    from src.rl.offline.grace.serving import apply_reward_transform, GraceServing

    class _Buf:
        def __init__(self, n):
            self._data = {"rewards": torch.zeros(n)}

    serving = GraceServing(
        mode="Q-minus", rewards=torch.full((20,), 7.0), l4_kind="interval"
    )
    buf = _Buf(20)
    assert apply_reward_transform(buf, serving) is True
    assert serving.meta["transform_applied"] is True
    assert serving.meta["n_rewards_written"] == 20
    assert serving.meta["rewards_coverage"] == 1.0


def test_partial_prefix_write_is_visible_as_coverage():
    """A buffer longer than the fitted rows keeps OBSERVATIONAL rewards on the
    tail -- a silent half-no-op unless coverage is recorded."""
    from src.rl.offline.grace.serving import apply_reward_transform, GraceServing

    class _Buf:
        def __init__(self, n):
            self._data = {"rewards": torch.zeros(n)}

    serving = GraceServing(
        mode="Q-minus", rewards=torch.full((20,), 7.0), l4_kind="interval"
    )
    buf = _Buf(50)  # 30 rows the transform never touches
    apply_reward_transform(buf, serving)
    assert serving.meta["n_rewards_written"] == 20
    assert serving.meta["n_buffer_rows"] == 50
    assert serving.meta["rewards_coverage"] == 0.4


def test_abstention_records_that_nothing_fired():
    from src.rl.offline.grace.serving import apply_reward_transform, GraceServing

    class _Buf:
        def __init__(self, n):
            self._data = {"rewards": torch.zeros(n)}

    serving = GraceServing(reason="fit was dirty")
    buf = _Buf(20)
    assert apply_reward_transform(buf, serving) is False
    assert serving.meta["transform_applied"] is False
    assert serving.meta["n_rewards_written"] == 0
    assert torch.equal(buf._data["rewards"], torch.zeros(20))


def test_coverage_is_against_the_buffers_fill_not_its_capacity():
    """A ReplayBuffer allocates its reward column at CAPACITY. Measuring
    coverage against that reported 300/310 on a complete transform — a false
    partial-write flag on every healthy run."""
    from src.rl.off_policy.replay_buffer import ReplayBuffer
    from src.rl.offline.grace.serving import apply_reward_transform, GraceServing

    buf = ReplayBuffer(capacity=310, device=torch.device("cpu"))
    for i in range(300):
        buf.add(
            {
                "obs": torch.zeros(4),
                "actions": torch.tensor(i % 2),
                "rewards": torch.tensor(1.0),
                "next_obs": torch.zeros(4),
                "dones": torch.tensor(float((i + 1) % 5 == 0)),
            }
        )
    serving = GraceServing(
        mode="Q-minus", rewards=torch.full((300,), 7.0), l4_kind="interval"
    )
    apply_reward_transform(buf, serving)
    assert serving.meta["n_buffer_rows"] == 300, serving.meta["n_buffer_rows"]
    assert serving.meta["rewards_coverage"] == 1.0


def test_selfcheck_is_conditioned_when_the_contrast_is_near_zero():
    """On d_a_null the contrast is float noise around 0. Normalising by it made
    the determinism sentinel read 1.3 — five orders past its 1e-5 flag line —
    on a healthy fit. The reward scale is the honest floor."""
    obs, point, r_scale = 2.384185791015625e-07, -7.112821265309321e-08, 1.0
    rel = abs(point - obs) / max(abs(obs), r_scale)
    assert rel < 1e-5, rel
    # and the real d100 numbers still land where they did
    obs_d, point_d = 0.49433350563049316, 0.49433374404907227
    assert abs(point_d - obs_d) / max(abs(obs_d), 1.0) < 1e-5

"""The shared calibration mechanism — L4's threshold and L5's nulls, one module.

Unit scale throughout: these use cheap synthetic statistics, not EM fits, so the
contract is tested without benchmark-scale compute.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.rl.offline.grace.bootstrap import (
    bootstrap_null,
    BootstrapNull,
    lr_statistic,
    resample_episodes,
)


# --------------------------------------------------------------------------- #
# S1: episode-level resampling                                                  #
# --------------------------------------------------------------------------- #
def test_resampling_moves_whole_episodes_never_transitions():
    """The rule the whole module rests on. U, the proxies and the instrument are
    episode-constant, so a transition-level resample shatters the blocks and
    yields a null far tighter than the statistic's own sampling law."""
    ids = np.repeat(np.arange(20), 5)
    rng = np.random.default_rng(0)
    rows = resample_episodes(ids, rng)
    assert rows.size == ids.size  # same number of transitions

    # Every episode present in the resample must appear with ALL of its rows.
    drawn = ids[rows]
    counts = {int(e): int((drawn == e).sum()) for e in np.unique(drawn)}
    assert all(c % 5 == 0 for c in counts.values()), counts
    # ...and it must actually resample (with replacement), not just permute.
    assert len(np.unique(drawn)) < 20 or max(counts.values()) > 5


def test_resampling_is_deterministic_under_a_seed():
    ids = np.repeat(np.arange(15), 4)
    a = resample_episodes(ids, np.random.default_rng(7))
    b = resample_episodes(ids, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_resampling_handles_ragged_episode_lengths():
    """Real episodes differ in length; the resample must carry each episode's own
    rows, not assume a common block size."""
    ids = np.concatenate([np.full(n, i) for i, n in enumerate([3, 7, 2, 9])])
    rows = resample_episodes(ids, np.random.default_rng(1))
    drawn = ids[rows]
    lengths = {0: 3, 1: 7, 2: 2, 3: 9}
    for e in np.unique(drawn):
        assert int((drawn == e).sum()) % lengths[int(e)] == 0


# --------------------------------------------------------------------------- #
# The hazard: failed replicates                                                 #
# --------------------------------------------------------------------------- #
def test_failed_replicates_are_counted_and_reported_never_dropped():
    """Dropping failures is a BIAS, not a convenience: convergence failure
    plausibly correlates with the statistic being bootstrapped, so excluding
    them conditions the null on convergence and moves the threshold by an
    unknown amount in an unknown direction."""

    def stat(seed):
        if seed % 3 == 0:
            raise RuntimeError("EM diverged")
        return float(seed)

    null = bootstrap_null(stat, b=30, seed=0)
    assert null.n_failed > 0
    assert null.n_requested == 30
    assert null.successes.size == 30 - null.n_failed
    assert null.failure_rate == pytest.approx(null.n_failed / 30)
    # The count must be impossible to miss on the label.
    assert "failed=" in null.label() and "NOT-TRUSTWORTHY" in null.label()
    assert not null.trustworthy
    assert any("EM diverged" in r.reason for r in null.replicates)


def test_a_clean_run_is_trustworthy_and_says_B_honestly():
    null = bootstrap_null(lambda s: float(s), b=20, seed=0)
    assert null.n_failed == 0 and null.trustworthy
    assert null.successes.size == 20
    assert "used=20" in null.label() and "NOT-TRUSTWORTHY" not in null.label()


def test_on_failure_raise_refuses_to_produce_a_conditioned_null():
    """Where a threshold conditioned on convergence would be worse than none."""

    def stat(seed):
        if seed == 3:
            return None
        return float(seed)

    with pytest.raises(RuntimeError, match="condition the null on convergence"):
        bootstrap_null(stat, b=10, seed=0, on_failure="raise")


def test_an_exhausted_backtrack_budget_counts_as_a_FAILED_replicate():
    """EM that stopped on a decrease it could not repair returns a number, but
    not one from a finished model. Counting it as a success would launder a
    known-bad fit into the null.

    The fixture originally set ``converged=True`` alongside
    ``backtrack_exhausted=True``. The estimator cannot produce that pair -- the
    EM loop breaks on one or the other -- and once the failure test became
    ``finished`` rather than the raw flag, the impossible combination read as a
    finished fit and the test stopped exercising anything. Corrected to the
    realistic case; the intent is unchanged, and the point it makes is now
    sharper: what fails a replicate is exhausting the budget WHILE STILL
    IMPROVING, not exhausting it at a stationary point.
    """

    class Out:
        def __init__(self, v, exhausted):
            self.value, self.backtrack_exhausted = v, exhausted
            self.monotone, self.backtracks = True, 0
            # stopped on a decrease => not converged, and not stationary either
            self.converged = not exhausted
            self.stationary = False
            self.finished = not exhausted

    null = bootstrap_null(lambda s: Out(float(s), exhausted=(s % 4 == 0)), b=16, seed=0)
    assert null.n_failed > 0
    assert any("still improving" in r.reason for r in null.replicates)


def test_non_finite_statistics_are_failures_not_data():
    null = bootstrap_null(lambda s: float("nan") if s % 2 else float(s), b=10, seed=0)
    assert null.n_failed == 5
    assert np.isfinite(null.successes).all()


def test_a_null_with_no_successes_refuses_a_quantile():
    null = bootstrap_null(lambda s: None, b=5, seed=0)
    with pytest.raises(ValueError, match="every one of"):
        null.quantile(0.95)


# --------------------------------------------------------------------------- #
# B as a reported precision parameter, not a calibration constant               #
# --------------------------------------------------------------------------- #
def test_the_threshold_carries_its_monte_carlo_error():
    """A threshold quoted without its own uncertainty invites over-reading."""
    rng = np.random.default_rng(0)
    draws = rng.normal(size=400)
    null = bootstrap_null(lambda s, d=draws: float(d[s % 400]), b=99, seed=0)
    err = null.mc_error(0.95)
    assert np.isfinite(err) and err > 0
    assert "±" in null.label()


def test_monte_carlo_error_shrinks_as_B_grows():
    """B is a precision knob: more replicates, a tighter threshold estimate --
    which is exactly what makes it a reported parameter rather than a tuned one."""
    rng = np.random.default_rng(1)
    pool = rng.normal(size=5000)
    errs = [
        bootstrap_null(lambda s, d=pool: float(d[s % 5000]), b=bb, seed=0).mc_error(0.9)
        for bb in (40, 400)
    ]
    assert errs[1] < errs[0], errs


def test_the_null_is_reproducible_under_a_seed():
    """Including the per-replicate seeds, which are passed IN so the caller's
    refit is seeded too -- a threshold must be reproducible end to end."""
    f = lambda s: float((s * 37) % 11)  # noqa: E731
    a = bootstrap_null(f, b=25, seed=5)
    b = bootstrap_null(f, b=25, seed=5)
    assert np.array_equal(a.successes, b.successes)
    assert [r.seed for r in a.replicates] == [r.seed for r in b.replicates]
    c = bootstrap_null(f, b=25, seed=6)
    assert not np.array_equal(a.successes, c.successes)


# --------------------------------------------------------------------------- #
# One module, two consumers                                                     #
# --------------------------------------------------------------------------- #
def test_L4_and_L5_share_one_calibration_call():
    """Two code paths would be two places for the calibration to drift, and the
    drift would be invisible in the outputs. L4's likelihood-ratio threshold and
    L5's falsification null are THE SAME CALL with a different statistic."""

    class Fit:
        def __init__(self, ll):
            self.final_ll = ll

    # L4: a likelihood-ratio statistic built from two fits.
    l4 = lr_statistic(
        fit_full=lambda s: Fit(-100.0 + 0.01 * s),
        fit_restricted=lambda s: Fit(-101.0),
    )
    null_l4 = bootstrap_null(l4, b=20, seed=0, statistic_name="lr")

    # L5: a falsification statistic -- different quantity, identical machinery.
    null_l5 = bootstrap_null(
        lambda s: abs(np.sin(s)), b=20, seed=0, statistic_name="cond_indep"
    )

    for null in (null_l4, null_l5):
        assert isinstance(null, BootstrapNull)
        assert null.n_requested == 20 and null.trustworthy
        assert np.isfinite(null.quantile(0.95))
    assert null_l4.statistic_name != null_l5.statistic_name


def test_per_replicate_diagnostics_survive_to_the_aggregate():
    """A suspicious null is diagnosable only if each fit's convergence status,
    backtracks and monotonicity are retained rather than aggregated away."""

    class Out:
        def __init__(self, v, mono, bt):
            self.value, self.monotone, self.backtracks = v, mono, bt
            self.converged, self.backtrack_exhausted = True, False

    null = bootstrap_null(
        lambda s: Out(float(s), mono=(s % 5 != 0), bt=s % 3), b=15, seed=0
    )
    d = null.diagnostics()
    assert d["n_non_monotone"] > 0
    assert d["total_backtracks"] == sum(r.backtracks for r in null.replicates)
    assert d["n_used"] + d["n_failed"] == d["n_requested"]


def test_parallel_replicates_give_the_identical_null():
    """Parallelism across replicates is statistically NEUTRAL -- independent
    refits -- so it is the first lever to reach for, ahead of anything that
    touches the procedure. Determinism must survive it: seeds are assigned from
    the INDEX before dispatch and results collected back by index, so the null
    cannot depend on which replicate finished first."""
    f = lambda s: float((s * 17) % 23)  # noqa: E731
    serial = bootstrap_null(f, b=32, seed=3, n_jobs=1)
    parallel = bootstrap_null(f, b=32, seed=3, n_jobs=8)
    assert np.array_equal(serial.successes, parallel.successes)
    assert [r.seed for r in serial.replicates] == [r.seed for r in parallel.replicates]
    assert [r.index for r in parallel.replicates] == list(range(32))


def test_parallelism_preserves_failure_accounting():
    """A failure in a worker must be recorded, not swallowed by the pool."""

    def stat(seed):
        if seed % 4 == 0:
            raise RuntimeError("EM diverged")
        return float(seed)

    a = bootstrap_null(stat, b=20, seed=0, n_jobs=1)
    b = bootstrap_null(stat, b=20, seed=0, n_jobs=6)
    assert a.n_failed == b.n_failed > 0
    assert a.diagnostics() == b.diagnostics()


def test_saturation_alone_does_not_fail_a_replicate():
    """CORRECTED RULE. Saturation is a property of the DATA, not of the fit.

    Measured on the T = 500 arm: ``initial_saturation`` is 0.95-1.00 in EVERY
    fit, including the ones recovering at 0.99. The first version of this
    contract failed any saturated replicate, which would have failed every
    replicate on exactly the long-episode environments L4 and L5 most need --
    and with ``max_failure_rate`` defaulting to zero, rejected every null there
    while looking principled.

    A diagnostic that fires on healthy fits is a RISK flag, not a FAILURE flag.
    """

    class _Fit:
        value = 1.0
        converged = True
        monotone = True
        backtracks = 0
        backtrack_exhausted = False
        initial_saturation = 0.98
        saturated_at_init = True
        n_anneal = 9  # the anneal DID run: saturation was handled
        reached_tau_one = True
        degenerate_mechanism = False

    null = bootstrap_null(lambda seed: _Fit(), b=8, seed=0)
    assert null.n_failed == 0, [r.reason for r in null.replicates if r.failed]
    assert null.trustworthy
    # ...but it is still REPORTED, because it is why annealing was needed.
    assert null.diagnostics()["n_saturated_at_init"] == 8
    assert null.diagnostics()["n_init_determined"] == 0


def test_a_saturated_replicate_is_a_failed_replicate():
    """§3 — the consequence that reaches L4 and L5.

    Every bootstrap replicate is an EM fit. If a replicate's E-step saturates it
    is frozen at its initialisation, so its statistic is a draw from the
    INITIALISER rather than from the sampling distribution. A null built out of
    those measures initialisation variance -- and it does so while looking
    impeccable: narrow, smooth, and about the wrong thing. L4's compatible set
    and L5's thresholds are both read off these nulls, so the failure would
    propagate into every calibrated number without ever presenting as an error.

    Saturated belongs in the same category as an exhausted backtrack budget:
    a number came back, and it is not a number about the target.
    """

    class _Fit:
        def __init__(self, value, **kw):
            self.value = value
            self.converged = True
            self.monotone = True
            self.backtracks = 0
            self.backtrack_exhausted = False
            self.initial_saturation = 0.0
            self.saturated_at_init = False
            self.n_anneal = 9
            self.reached_tau_one = True
            self.degenerate_mechanism = False
            for k, v in kw.items():
                setattr(self, k, v)

    def statistic(seed: int):
        # every third replicate is frozen at its init
        if seed % 3 == 0:
            return _Fit(
                1.0, initial_saturation=0.97, saturated_at_init=True, n_anneal=0
            )
        return _Fit(float(seed % 5))

    null = bootstrap_null(statistic, b=12, seed=0, statistic_name="t")
    assert null.n_failed > 0
    assert not null.trustworthy, "a null with frozen replicates must not pass"
    d = null.diagnostics()
    assert d["n_init_determined"] == null.n_failed, d
    assert d["max_initial_saturation"] > 0.9, d
    assert any("NO annealing" in r for r in d["reasons"]), d["reasons"]


def test_a_replicate_that_stopped_while_tempered_is_a_failed_replicate():
    """Its parameters maximise a smoothed surrogate, not the likelihood, so the
    statistic is not an estimate of the target at all."""

    class _Fit:
        value = 1.0
        converged = True
        monotone = True
        backtracks = 0
        backtrack_exhausted = False
        initial_saturation = 0.0
        saturated_at_init = False
        n_anneal = 9
        reached_tau_one = False
        degenerate_mechanism = False

    null = bootstrap_null(lambda seed: _Fit(), b=4, seed=0)
    assert null.n_failed == 4
    assert null.diagnostics()["n_stopped_while_tempered"] == 4


def test_a_stationary_replicate_is_not_a_failure():
    """The failure test is ``finished``, not ``converged`` — and the two now
    diverge routinely.

    A fit has two legitimate end states: the tolerance window, and STATIONARITY
    (no improving step at any step size tried, with the last improvement already
    sub-tolerance). A stationary fit sets ``backtrack_exhausted``, so a contract
    testing that flag alone would fail it. Measured: the production-scale
    CartPole fit at 55k transitions finishes by stationarity, NOT by the
    tolerance window, so a converged-based rule would reject the typical
    production replicate.

    Same class as the first saturation rule: a condition that used to fire
    almost never and now fires often.
    """

    class _Fit:
        value = 1.0
        converged = False  # the tolerance window never fired...
        stationary = True  # ...but the objective is stationary
        finished = True
        monotone = True
        backtracks = 12
        backtrack_exhausted = True  # which a stationary fit always sets
        initial_saturation = 0.98
        saturated_at_init = True
        n_anneal = 5
        reached_tau_one = True
        degenerate_mechanism = False

    null = bootstrap_null(lambda seed: _Fit(), b=6, seed=0)
    assert null.n_failed == 0, [r.reason for r in null.replicates if r.failed]
    assert null.trustworthy
    assert null.diagnostics()["n_stationary"] == 6
    assert null.diagnostics()["n_not_finished"] == 0

    # ...while exhausting the budget WHILE STILL IMPROVING remains a failure.
    class _Stuck(_Fit):
        stationary = False
        finished = False

    stuck = bootstrap_null(lambda seed: _Stuck(), b=6, seed=0)
    assert stuck.n_failed == 6
    assert any("still improving" in r for r in stuck.diagnostics()["reasons"])


def _null_from(values, seed=0):
    class _Out:
        def __init__(self, v):
            self.value = v
            self.converged = True
            self.stationary = False
            self.finished = True
            self.monotone = True
            self.backtracks = 0
            self.backtrack_exhausted = False
            self.initial_saturation = 0.0
            self.saturated_at_init = False
            self.n_anneal = 5
            self.reached_tau_one = True
            self.degenerate_mechanism = False

    it = iter(values)
    return bootstrap_null(lambda s: _Out(next(it)), b=len(values), seed=seed)


def test_pooling_across_seeds_is_licensed_by_a_TESTED_exchangeability_check():
    """LEVER A. Seeds within a configuration are i.i.d. draws from the same
    generator, so under H0 their nulls are the SAME distribution and computing
    one per seed estimates a single object five times. Pooling B/5 from each
    gives the same configuration-level precision for a fifth of the fits.

    Exchangeability here is STRUCTURAL -- seeds are i.i.d. draws from the same
    generator by construction -- so the KS check is a SMOKE TEST against
    implementation error (a seed that silently used different parameters, a
    leaked RNG, a dataset misfiled into the wrong configuration), not the licence
    for the design. At ~20 replicates over 10 pairs it detects only gross
    differences, which is all it is asked to do. A refusal means something is
    wrong with the generator or the bookkeeping, which is when a loud stop beats
    a quiet fallback.
    """
    from src.rl.offline.grace.bootstrap import pooled_null

    rng = np.random.default_rng(0)
    same = {s: _null_from(list(rng.normal(0, 1, 40)), seed=s) for s in range(5)}
    pooled = pooled_null(same, statistic_name="t")
    assert pooled.successes.size == 200
    assert pooled.trustworthy

    # A seed drawn from a DIFFERENT distribution must refuse pooling, loudly.
    shifted = dict(same)
    shifted[2] = _null_from(list(rng.normal(6.0, 1, 40)), seed=2)
    with pytest.raises(ValueError, match="NOT exchangeable"):
        pooled_null(shifted)


def test_pooling_refuses_to_launder_a_failed_seed():
    """A seed whose replicates failed contributes no successes; pooling it in
    would hide its failure rate inside a pool that looks healthy."""
    from src.rl.offline.grace.bootstrap import pooled_null

    rng = np.random.default_rng(1)
    d = {s: _null_from(list(rng.normal(0, 1, 30)), seed=s) for s in range(3)}

    class _Dead:
        value = float("nan")

    d[1] = bootstrap_null(lambda s: _Dead(), b=4, seed=1)
    with pytest.raises(ValueError, match="pooling a null over a failed one"):
        pooled_null(d)

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
    not one from a converged model. Counting it as a success would launder a
    known-bad fit into the null."""

    class Out:
        def __init__(self, v, exhausted):
            self.value, self.backtrack_exhausted = v, exhausted
            self.converged, self.monotone, self.backtracks = True, True, 0

    null = bootstrap_null(lambda s: Out(float(s), exhausted=(s % 4 == 0)), b=16, seed=0)
    assert null.n_failed > 0
    assert any("backtrack budget exhausted" in r.reason for r in null.replicates)


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

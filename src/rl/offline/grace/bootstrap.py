"""Episode-level bootstrap calibration — ONE mechanism, two consumers.

L4's likelihood-ratio threshold (which models count as "compatible with the
observational distribution") and L5's falsification nulls are the same call with
a different statistic. They are deliberately not two code paths: two
separately-tuned calibrations would be two places for a constant to hide, and
any drift between them would be invisible in the outputs.

Why a bootstrap at all, rather than an asymptotic reference: a finite mixture
with an unidentified latent puts the null on a **boundary of the parameter
space**, where the regularity conditions behind `chi^2_df` fail. The reference
distribution is not chi-2 and is not known in closed form, so estimating it from
the data at hand is the only sound route — not a fallback.

**AN INIT-DETERMINED REPLICATE IS NOT A REPLICATE.** Every replicate is an EM
fit, and the L3 E-step sums per-row log-likelihoods over the episode -- so on
long episodes the responsibilities saturate to 0/1 immediately and an untempered
fit is frozen at whatever its initialisation picked. Such a fit does not sample
the statistic's sampling distribution; it samples the INITIALISER. A null built
from them measures initialisation variance while looking impeccable -- narrow,
smooth, and about the wrong thing -- and since L4's compatible set and L5's
falsification thresholds are both read off these nulls, it would propagate into
every calibrated number without presenting as an error.

**CORRECTED RULE, and the wrong version is preserved because the reasoning that
produced it is the general lesson.** The first version failed any replicate whose
E-step saturated. That was wrong, and measurably so: on the T = 500 arm
``initial_saturation`` is 0.95-1.00 in **every** fit, including the ones that
recover at 0.99 -- so the rule would have failed EVERY replicate on exactly the
long-episode environments this layer most needs, and with ``max_failure_rate``
defaulting to zero it would have rejected every null there while looking
principled. Saturation is a property of the DATA (episode length x per-step
separation), not of the fit's health: it says an untempered EM *would be*
init-determined here, which is a statement about what the optimiser must do, not
about whether it did it.

    **A diagnostic that fires on healthy fits is a RISK flag, not a FAILURE
    flag.** Failing on it conflates "this was hard" with "this went wrong".

So saturation fails a replicate only when nothing was done about it -- saturated
**and** no annealing. Saturated with the anneal active is the diagnostic doing
its job and is reported, not failed. The genuine failure conditions are:
an exhausted backtrack budget, a fit that stopped mid-anneal (its parameters
maximise a tempered surrogate), a degenerate mechanism (its likelihood is
measuring ``min_scale``), and non-convergence.

**The hazard this module is designed around: silently dropped replicates.**
Every replicate is an EM fit, and EM fits can fail — non-convergence, or an
exhausted backtrack budget under the monotone guard. Dropping the failures and
computing the null from what survives is *a bias, not a convenience*:
convergence failure plausibly correlates with the very statistic being
bootstrapped (a replicate drawn near a degenerate configuration is both harder
to fit and unusual in the statistic), so dropping them conditions the null on
convergence and shifts the threshold by an unknown amount in an unknown
direction. Failures are therefore counted, reported alongside every threshold,
and handled by an explicit choice — never by default. A bootstrap reporting
"B = 99" while having used 71 is the same species of error as a run reporting 27
rows into 8 datasets.

**THE SYMMETRY RULE — the invariant a reviewer probes first.** Whatever fitting
procedure produces the OBSERVED statistic must produce the REPLICATE statistics:
same lr schedule, same epoch budget, same initialisation policy, same
convergence tolerance, same guard settings. *Any* procedural asymmetry biases
the threshold, in a direction that is not knowable from the outputs.

Two concrete traps this rules out:

* **Warm-starting replicates from the null-generating parameters.** Tempting and
  very effective — and wrong: the replicate is generated *from* those
  parameters, so warm-starting hands it a head start the observed fit never got.
  The replicate statistics come out systematically better-optimised than the
  observed one and the null shifts.
* **A cheaper fit budget for replicates than for the observed fit.** Legitimate
  only if applied to BOTH sides; applied to one, it is the same bias wearing
  different clothes.

Both are special cases of the same rule, and both look like pure speed-ups.

``B`` is a **reported Monte-Carlo precision parameter**, not a calibration
constant — the same distinction as ``alpha`` in L4. Every threshold is returned
with its own Monte-Carlo error, because a threshold quoted without its
uncertainty invites over-reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

__all__ = [
    "ReplicateResult",
    "BootstrapNull",
    "resample_episodes",
    "bootstrap_null",
]


@dataclass
class ReplicateResult:
    """One replicate, including WHY it failed if it did.

    Per-replicate diagnostics are surfaced rather than aggregated away: a
    suspicious null is diagnosable only if the convergence status, backtrack
    count and monotonicity of each fit survive.
    """

    index: int
    seed: int
    statistic: Optional[float] = None
    converged: bool = False
    monotone: bool = True
    backtracks: int = 0
    backtrack_exhausted: bool = False
    initial_saturation: float = 0.0
    saturated_at_init: bool = False
    n_anneal: int = 0
    reached_tau_one: bool = True
    degenerate_mechanism: bool = False
    failed: bool = False
    reason: str = ""


@dataclass
class BootstrapNull:
    """A calibrated reference distribution, with the evidence for its own quality."""

    replicates: List[ReplicateResult]
    n_requested: int
    seed: int
    statistic_name: str = ""
    max_failure_rate: float = 0.0

    @property
    def successes(self) -> np.ndarray:
        return np.asarray(
            [
                r.statistic
                for r in self.replicates
                if not r.failed and r.statistic is not None
            ],
            dtype=np.float64,
        )

    @property
    def n_failed(self) -> int:
        return int(sum(1 for r in self.replicates if r.failed))

    @property
    def failure_rate(self) -> float:
        return self.n_failed / max(1, self.n_requested)

    @property
    def trustworthy(self) -> bool:
        """Whether the threshold may be used without qualification.

        The default tolerance is ZERO: any failed replicate flags the null,
        because the bias from excluding it is of unknown sign and magnitude.
        Relaxing that is an explicit caller decision (``max_failure_rate``) and
        is reported in ``label()`` — it is a disclosure, not a threshold the
        module picks.
        """
        return self.failure_rate <= self.max_failure_rate and self.successes.size > 1

    def quantile(self, q: float) -> float:
        """The threshold at level ``q``, computed from the SUCCESSFUL replicates.

        Read this together with ``failure_rate``: if replicates failed, this is a
        null conditioned on convergence, not the null that was asked for.
        """
        s = self.successes
        if s.size == 0:
            raise ValueError(
                f"every one of {self.n_requested} replicates failed; there is no "
                "null to take a quantile of"
            )
        return float(np.quantile(s, q))

    def mc_error(self, q: float, n_resample: int = 500) -> float:
        """Monte-Carlo standard error of ``quantile(q)`` — the B-induced noise.

        Estimated by resampling the null statistics themselves, which needs no
        density estimate at the quantile (the asymptotic formula does, and that
        density is exactly what is unknown here). Cheap: it resamples numbers,
        not fits.
        """
        s = self.successes
        if s.size < 2:
            return float("nan")
        rng = np.random.default_rng(self.seed + 99_991)
        draws = [
            np.quantile(rng.choice(s, size=s.size, replace=True), q)
            for _ in range(n_resample)
        ]
        return float(np.std(draws))

    def label(self, q: float = 0.95) -> str:
        """What must travel with any number derived from this null."""
        bits = [
            f"B={self.n_requested}",
            f"used={self.successes.size}",
            f"failed={self.n_failed} ({100 * self.failure_rate:.1f}%)",
        ]
        if self.successes.size > 1:
            bits.append(f"thr@{q}={self.quantile(q):.4f}±{self.mc_error(q):.4f}")
        if not self.trustworthy:
            bits.append("NOT-TRUSTWORTHY")
        return " ".join(bits)

    def diagnostics(self) -> dict:
        """Aggregate per-replicate health, for reporting next to a threshold."""
        return {
            "n_requested": self.n_requested,
            "n_used": int(self.successes.size),
            "n_failed": self.n_failed,
            "failure_rate": self.failure_rate,
            "n_non_monotone": sum(1 for r in self.replicates if not r.monotone),
            "n_saturated_at_init": sum(
                1 for r in self.replicates if r.saturated_at_init
            ),
            "n_init_determined": sum(
                1 for r in self.replicates if r.saturated_at_init and r.n_anneal == 0
            ),
            "n_degenerate_mechanism": sum(
                1 for r in self.replicates if r.degenerate_mechanism
            ),
            "n_stopped_while_tempered": sum(
                1 for r in self.replicates if not r.reached_tau_one
            ),
            "max_initial_saturation": max(
                (r.initial_saturation for r in self.replicates), default=0.0
            ),
            "n_backtrack_exhausted": sum(
                1 for r in self.replicates if r.backtrack_exhausted
            ),
            "total_backtracks": sum(r.backtracks for r in self.replicates),
            "reasons": sorted({r.reason for r in self.replicates if r.reason}),
        }


def resample_episodes(episode_ids, rng: np.random.Generator) -> np.ndarray:
    """Row indices of an EPISODE-level resample with replacement (S1).

    Whole episodes are drawn, never transitions. `U`, the proxies and the
    instrument are episode-constant, so the effective sample size is the episode
    count; resampling transitions shatters those blocks and produces a null far
    tighter than the statistic's own sampling law. Same rule as the k-rank
    permutation, the C1 splitter and every preflight check.

    Returns row indices so the caller can slice whatever representation it holds.
    """
    ids = np.asarray(episode_ids).reshape(-1)
    uniq = np.unique(ids)
    rows_by_ep = {int(e): np.flatnonzero(ids == e) for e in uniq}
    picked = rng.choice(uniq, size=uniq.size, replace=True)
    return np.concatenate([rows_by_ep[int(e)] for e in picked])


def bootstrap_null(
    statistic: Callable[[int], object],
    *,
    b: int = 99,
    seed: int = 0,
    statistic_name: str = "",
    max_failure_rate: float = 0.0,
    on_failure: str = "report",
    n_jobs: int = 1,
) -> BootstrapNull:
    """Calibrate a reference distribution by running ``statistic`` ``b`` times.

    ``statistic(replicate_seed)`` performs ONE replicate — draw, refit, compute —
    and returns either a float or an object exposing ``.value`` plus the fit
    diagnostics (``converged``, ``monotone``, ``backtracks``,
    ``backtrack_exhausted``). Raising, or returning ``None``, marks the replicate
    FAILED; it is recorded with its reason, never silently dropped.

    Determinism: replicate ``i`` gets ``seed + 1 + i``, and that seed is passed
    in rather than drawn here, so the caller's refit is seeded too and a
    threshold is reproducible end to end.

    ``n_jobs`` runs replicates concurrently. Replicates are independent refits, so
    this is **statistically neutral** — the first lever to reach for, ahead of
    anything that touches the procedure. Determinism is preserved regardless of
    completion order because each replicate's seed is assigned from its INDEX
    before dispatch and results are collected back by index. Threads rather than
    processes: the fitting releases the GIL, and the statistic closes over live
    model objects that would not pickle. Cap each fit's own thread count
    (``torch.set_num_threads``) so the pool does not oversubscribe.

    ``on_failure``:
      * ``"report"`` (default) — record failures, compute the threshold from the
        successes, and flag the null unless ``max_failure_rate`` permits it. The
        caller must look at ``failure_rate``.
      * ``"raise"`` — refuse to produce a null at all if any replicate failed.
        Correct where a threshold conditioned on convergence would be worse than
        no threshold.
    """
    if on_failure not in ("report", "raise"):
        raise ValueError(f"on_failure must be 'report' or 'raise', got {on_failure!r}")
    if b < 2:
        raise ValueError(f"b must be at least 2 to form a null, got {b}")

    def _one(i: int) -> ReplicateResult:
        rep_seed = seed + 1 + i
        rec = ReplicateResult(index=i, seed=rep_seed)
        try:
            out = statistic(rep_seed)
        except Exception as exc:  # a failed fit is DATA about the null, not noise
            rec.failed = True
            rec.reason = f"{type(exc).__name__}: {exc}"[:200]
            return rec
        if out is None:
            rec.failed = True
            rec.reason = "statistic returned None"
            return rec
        value = getattr(out, "value", out)
        try:
            rec.statistic = float(value)
        except (TypeError, ValueError):
            rec.failed = True
            rec.reason = f"statistic returned non-numeric {type(out).__name__}"
            return rec
        if not np.isfinite(rec.statistic):
            rec.failed = True
            rec.reason = f"statistic was {rec.statistic}"
            rec.statistic = None
            return rec
        rec.converged = bool(getattr(out, "converged", True))
        rec.monotone = bool(getattr(out, "monotone", True))
        rec.backtracks = int(getattr(out, "backtracks", 0))
        rec.backtrack_exhausted = bool(getattr(out, "backtrack_exhausted", False))
        rec.initial_saturation = float(getattr(out, "initial_saturation", 0.0))
        rec.saturated_at_init = bool(getattr(out, "saturated_at_init", False))
        rec.n_anneal = int(getattr(out, "n_anneal", 0))
        rec.reached_tau_one = bool(getattr(out, "reached_tau_one", True))
        rec.degenerate_mechanism = bool(getattr(out, "degenerate_mechanism", False))
        # An exhausted backtrack budget means EM stopped on a decrease it could
        # not repair. That is a FAILED fit for calibration purposes even though a
        # number came back, because the number is not from a converged model.
        if rec.backtrack_exhausted:
            rec.failed = True
            rec.reason = "backtrack budget exhausted (EM stopped on a decrease)"
        # SATURATION IS A RISK FLAG, NOT A FAILURE FLAG -- corrected, see the
        # module docstring. It fails a replicate only when NOTHING WAS DONE
        # ABOUT IT: saturated with no annealing means the fit is
        # init-determined and its statistic is initialisation noise. Saturated
        # WITH the anneal active is the diagnostic doing its job.
        elif rec.saturated_at_init and rec.n_anneal == 0:
            rec.failed = True
            rec.reason = (
                f"E-step saturated at initialisation ({rec.initial_saturation:.2f} of "
                "episodes hard-assigned before the first M-step) with NO annealing; "
                "this replicate is init-determined and its statistic is "
                "initialisation noise, not a resample"
            )
        # A degenerate mechanism IS a failure regardless: a scale on its floor
        # makes the likelihood a function of min_scale, and every statistic here
        # is likelihood-based.
        elif rec.degenerate_mechanism:
            rec.failed = True
            rec.reason = (
                "a mechanism's fitted scale is on its min_scale floor; the "
                "likelihood is measuring the floor, not the data"
            )
        # Stopped mid-anneal: the parameters maximise a tempered surrogate, so
        # the statistic is not an estimate of the target at all.
        elif not rec.reached_tau_one:
            rec.failed = True
            rec.reason = "fit stopped while still tempered (anneal did not reach tau=1)"
        return rec

    if n_jobs == 1:
        results: List[ReplicateResult] = [_one(i) for i in range(b)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            # Collected BY INDEX, so the null does not depend on which replicate
            # finished first -- the same discipline as the vectorized rollout's
            # deterministic output order.
            results = list(pool.map(_one, range(b)))

    null = BootstrapNull(
        replicates=results,
        n_requested=b,
        seed=seed,
        statistic_name=statistic_name,
        max_failure_rate=max_failure_rate,
    )
    if on_failure == "raise" and null.n_failed:
        raise RuntimeError(
            f"{null.n_failed}/{b} bootstrap replicates failed "
            f"({100 * null.failure_rate:.1f}%). Excluding them would condition the "
            "null on convergence and shift the threshold by an unknown amount in "
            "an unknown direction. Reasons: "
            f"{sorted({r.reason for r in results if r.reason})[:3]}"
        )
    return null


def lr_statistic(
    fit_full: Callable[[int], object], fit_restricted: Callable[[int], object]
) -> Callable[[int], float]:
    """A likelihood-ratio replicate — L4's consumer of ``bootstrap_null``.

    ``2 * (ll_full - ll_restricted)``, whose reference distribution is what
    defines L4's compatible set. Returned as a closure over one seed so it is
    exactly the shape ``bootstrap_null`` expects, which is what lets L4 and L5
    share the calibration rather than each growing their own.
    """

    def _one(replicate_seed: int) -> float:
        full = fit_full(replicate_seed)
        restricted = fit_restricted(replicate_seed)
        return 2.0 * (float(full.final_ll) - float(restricted.final_ll))

    return _one

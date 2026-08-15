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

    results: List[ReplicateResult] = []
    for i in range(b):
        rep_seed = seed + 1 + i
        rec = ReplicateResult(index=i, seed=rep_seed)
        try:
            out = statistic(rep_seed)
        except Exception as exc:  # a failed fit is DATA about the null, not noise
            rec.failed = True
            rec.reason = f"{type(exc).__name__}: {exc}"[:200]
            results.append(rec)
            continue
        if out is None:
            rec.failed = True
            rec.reason = "statistic returned None"
            results.append(rec)
            continue
        value = getattr(out, "value", out)
        try:
            rec.statistic = float(value)
        except (TypeError, ValueError):
            rec.failed = True
            rec.reason = f"statistic returned non-numeric {type(out).__name__}"
            results.append(rec)
            continue
        if not np.isfinite(rec.statistic):
            rec.failed = True
            rec.reason = f"statistic was {rec.statistic}"
            rec.statistic = None
            results.append(rec)
            continue
        rec.converged = bool(getattr(out, "converged", True))
        rec.monotone = bool(getattr(out, "monotone", True))
        rec.backtracks = int(getattr(out, "backtracks", 0))
        rec.backtrack_exhausted = bool(getattr(out, "backtrack_exhausted", False))
        # An exhausted backtrack budget means EM stopped on a decrease it could
        # not repair. That is a FAILED fit for calibration purposes even though a
        # number came back, because the number is not from a converged model.
        if rec.backtrack_exhausted:
            rec.failed = True
            rec.reason = "backtrack budget exhausted (EM stopped on a decrease)"
        results.append(rec)

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

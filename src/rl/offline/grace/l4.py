"""L4 — uncertainty. Intervals for point-ID cells, bounds for bounds-only,
abstention as a first-class outcome.

Design: docs/grace_v2_l4_design.md (read-through rulings 2026-08-23 folded
in). The load-bearing decisions, restated where the code enacts them:

* **Three kinds — ``interval``, ``bounds``, ``abstain``. No bare point.** An
  identified cell returns an interval whose width collapses toward zero; the
  collapse property IS the point-serving semantics ("no number without its
  uncertainty"). ``abstain`` is NO statement and carries its reason;
  ``bounds`` is a valid non-point statement; conflating them would read
  silence as an infinitely wide bound.
* **The interval is replicate spread over the episode-level bootstrap** (S1,
  via ``bootstrap.py`` — one calibration device shared with L5, failures
  counted never dropped, symmetry rule enforced by using the SAME fit
  procedure for observed and replicate fits).
* **The variance share, not a ratio against a cut**: with determinism on and
  a fixed fit seed, replicate variation comes only from resampled data. The
  share ``optimiser_var / replicate_var`` separates estimand uncertainty
  (more data narrows it) from procedural instability (it does not — the
  falsified 1/sqrt(n) check), reported always, thresholded never.
* **Balke–Pearl closed form** anchors q1 on D-E via the WITHIN-PAIR
  restriction, valid on both environments because the instrument dictates
  only the binary a_bad-vs-a_good choice on in-pair steps (verified in
  behavior_policy.py, 2026-08-23); on Acrobot the bonus indicator must be
  read with the terminal flags (value collision at 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from .bootstrap import bootstrap_null
from .estimator import EpisodeData, LatentClassEstimator


# --------------------------------------------------------------------- result
@dataclass
class L4Result:
    """The only shapes L4 returns. ``abstain`` carries its reason first-class."""

    kind: str  # "interval" | "bounds" | "abstain"
    lo: float = float("nan")
    hi: float = float("nan")
    alpha: float = 0.1
    b: int = 0
    reason: str = ""
    # Share of the interval's variance reproduced by init-perturbation on
    # IDENTICAL data: "this interval is X% procedural instability". Constant-
    # free; a consumer needing a binary picks its own line, explicitly.
    procedural_share: float = float("nan")
    failure_rate: float = 0.0
    observed: float = float("nan")
    label: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def summary(self) -> str:
        if self.kind == "abstain":
            return f"ABSTAIN: {self.reason} [{self.label}]"
        ps = (
            f" procedural_share={self.procedural_share:.0%}"
            if np.isfinite(self.procedural_share)
            else ""
        )
        return (
            f"{self.kind} [{self.lo:+.4f}, {self.hi:+.4f}] width={self.width:.4f} "
            f"alpha={self.alpha} B={self.b} fail={self.failure_rate:.0%}{ps} "
            f"[{self.label}]"
        )


# ----------------------------------------------------------------- resampling
def _resample_episode_data(data: EpisodeData, rng: np.random.Generator) -> EpisodeData:
    """Episode-level resample with replacement (S1), ids REMAPPED per copy.

    Reusing a drawn episode's original id would let the estimator's
    episode-level machinery MERGE the copies into one block — silently halving
    the resample's effective size. Each drawn copy gets a fresh consecutive id.
    """
    ep = data.episode_ids.detach().cpu().numpy()
    uniq = np.unique(ep)
    rows_by_ep = {int(e): np.flatnonzero(ep == e) for e in uniq}
    picked = rng.choice(uniq, size=uniq.size, replace=True)
    idx_parts, new_ids = [], []
    for k, e in enumerate(picked):
        r = rows_by_ep[int(e)]
        idx_parts.append(r)
        new_ids.append(np.full(r.size, k, dtype=np.int64))
    idx = torch.as_tensor(np.concatenate(idx_parts), device=data.state.device)
    new_ep = torch.as_tensor(np.concatenate(new_ids), device=data.state.device)
    return EpisodeData(
        state=data.state[idx],
        action=data.action[idx],
        reward=data.reward[idx],
        episode_ids=new_ep,
        proxy={k: v[idx] for k, v in (data.proxy or {}).items()},
    )


# ------------------------------------------------------------------- interval
def _dirty(fit) -> str:
    """The abstention conditions, read off the C3 flags. Empty string = clean."""
    if not fit.finished:
        if fit.tau1_budget_bound:
            return "fit ended budget-bound mid-ascent (tau1_budget_bound)"
        if fit.backtrack_exhausted:
            return "fit stuck: line search exhausted while still improving"
        return "fit unfinished (neither converged nor stationary)"
    if fit.degenerate_mechanism:
        worst = max(fit.mechanism_degeneracy.items(), key=lambda kv: kv[1])
        return f"degenerate mechanism ({worst[0]}: {worst[1]:.2f} on the scale floor)"
    if not fit.reached_tau_one:
        return "stopped while tempered: parameters maximise a surrogate"
    return ""


def point_id_interval(
    *,
    make_estimator: Callable[[int], LatentClassEstimator],
    data: EpisodeData,
    target: Callable[[LatentClassEstimator, object], float],
    fit_kwargs: Optional[Dict] = None,
    alpha: float = 0.1,
    b: int = 99,
    fit_seed: int = 0,
    init_seeds: Sequence[int] = (1, 2, 3, 4),
    n_jobs: int = 1,
) -> L4Result:
    """Replicate-spread interval for a point-ID cell, or an abstention.

    ``make_estimator(seed)`` builds a fresh estimator; ``target(est, fit)``
    reads the estimand (e.g. the gate do-contrast). The observed fit and every
    replicate use the SAME procedure (symmetry rule); determinism means each
    replicate's spread is pure data variation at fixed fit seed, and the
    init-perturbation arm (same data, ``init_seeds``) is pure optimiser
    variation — the two arms of the variance share.
    """
    fk = dict(fit_kwargs or {})

    est0 = make_estimator(fit_seed)
    fit0 = est0.fit(data, **fk)
    reason = _dirty(fit0)
    label0 = fit0.estimate(torch.tensor(0.0)).label()
    if reason:
        return L4Result(kind="abstain", reason=reason, alpha=alpha, label=label0)
    observed = float(target(est0, fit0))

    # --- init-perturbation arm: identical data, fit-seed varied -------------
    init_vals = [observed]
    for s in init_seeds:
        est_i = make_estimator(int(s))
        fit_i = est_i.fit(data, **fk)
        if not _dirty(fit_i):
            init_vals.append(float(target(est_i, fit_i)))
    optimiser_var = float(np.var(init_vals, ddof=1)) if len(init_vals) > 1 else 0.0

    # --- replicate arm: resampled data, fit seed FIXED ----------------------
    def statistic(rep_seed: int):
        rng = np.random.default_rng(rep_seed)
        rdata = _resample_episode_data(data, rng)
        est_r = make_estimator(fit_seed)
        fit_r = est_r.fit(rdata, **fk)

        class _Rep:
            value = float(target(est_r, fit_r))
            converged = bool(fit_r.converged)
            stationary = bool(fit_r.stationary)
            finished = bool(fit_r.finished)
            monotone = bool(fit_r.monotone)
            backtracks = int(fit_r.backtracks)
            backtrack_exhausted = bool(fit_r.backtrack_exhausted)
            reached_tau_one = bool(fit_r.reached_tau_one)
            degenerate_mechanism = bool(fit_r.degenerate_mechanism)

        return _Rep()

    null = bootstrap_null(
        statistic, b=b, seed=fit_seed, statistic_name="l4_interval", n_jobs=n_jobs
    )
    vals = np.asarray(null.successes, dtype=float)
    if vals.size < 2:
        return L4Result(
            kind="abstain",
            reason=f"bootstrap produced {vals.size} usable replicates "
            f"(failure rate {null.failure_rate:.0%}) — no interval is honest",
            alpha=alpha,
            label=label0,
            failure_rate=null.failure_rate,
            meta={"bootstrap_diagnostics": null.diagnostics()},
        )
    lo, hi = (
        float(np.quantile(vals, alpha / 2)),
        float(np.quantile(vals, 1 - alpha / 2)),
    )
    replicate_var = float(np.var(vals, ddof=1))
    share = optimiser_var / replicate_var if replicate_var > 0 else float("inf")
    return L4Result(
        kind="interval",
        lo=lo,
        hi=hi,
        alpha=alpha,
        b=b,
        observed=observed,
        procedural_share=share,
        failure_rate=null.failure_rate,
        label=label0,
        meta={
            "optimiser_var": optimiser_var,
            "replicate_var": replicate_var,
            "n_init_fits": len(init_vals),
            # The module's founding rule: failures may correlate with the
            # statistic, so their REASONS travel with every interval -- a rate
            # without reasons is uninterpretable (ruled 2026-08-23).
            "bootstrap_diagnostics": null.diagnostics(),
        },
    )


# --------------------------------------------------------------- Balke–Pearl
def balke_pearl_contrast_bounds(
    *,
    bonus: np.ndarray,  # binary outcome per in-pair step (terminal-safe, caller-built)
    x: np.ndarray,  # binary treatment: 1[a == a_bad] per in-pair step
    z: np.ndarray,  # binary instrument per in-pair step
) -> tuple[float, float]:
    """Closed-form bounds on E[Y|do(X=1)] − E[Y|do(X=0)], binary (Z, X, Y).

    The within-pair restriction is what licenses this on BOTH environments:
    the instrument dictates only the a_bad-vs-a_good choice on in-pair steps
    (behavior_policy.py), so treatment is binary there by construction. The
    caller supplies the bonus indicator built with the terminal flags — on
    Acrobot the bonus value collides with the terminal reward at 0, so a
    value-only read would be wrong (see the naive-bias tool's a_bad lesson).

    Natural (Manski–instrument / Balke–Pearl for the contrast without
    response-monotonicity): for each x, E[Y|do(x)] is bounded by
        max_z [ P(Y=1, X=x | z) ]  <=  E[Y|do(x)]
        E[Y|do(x)]  <=  min_z [ P(Y=1, X=x | z) + P(X != x | z) ]
    and the contrast bounds are [L1 − U0, U1 − L0].
    """
    bonus = np.asarray(bonus, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=int).reshape(-1)
    z = np.asarray(z, dtype=int).reshape(-1)
    L, U = {}, {}
    for xv in (0, 1):
        lows, highs = [], []
        for zv in (0, 1):
            m = z == zv
            if not m.any():
                continue
            p_y1_x = float(((bonus == 1) & (x == xv))[m].mean())
            p_not_x = float((x != xv)[m].mean())
            lows.append(p_y1_x)
            highs.append(p_y1_x + p_not_x)
        L[xv], U[xv] = max(lows), min(highs)
    return L[1] - U[0], U[1] - L[0]

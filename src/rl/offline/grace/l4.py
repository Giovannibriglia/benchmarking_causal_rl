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
    # LR-walk bounds are INNER approximations (ruled 2026-08-24): only
    # feasible iterates report, so every bound is achieved by SOME compatible
    # model -- "the identified set contains at least this interval". The
    # precise claim, not a hedge; production regions over network weights are
    # smooth but not guaranteed convex and no oracle exists there, so the
    # limitation is disclosed rather than tested away. False for closed-form
    # bounds (Balke-Pearl is exact given the data).
    inner_approximation: bool = False
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
        inner = " INNER-APPROX" if self.inner_approximation else ""
        return (
            f"{self.kind}{inner} [{self.lo:+.4f}, {self.hi:+.4f}] width={self.width:.4f} "
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
    # S8/S9: "not measurable" is NOT "measured as zero". If every init fit came
    # back dirty only ``observed`` survives, there is no perturbation arm, and
    # reporting 0.0 would be the most REASSURING possible reading of no evidence
    # -- the exact shape those rules exist to stop. nan says "no reading", and
    # ``summary()`` already gates the label on np.isfinite.
    optimiser_var = (
        float(np.var(init_vals, ddof=1)) if len(init_vals) > 1 else float("nan")
    )

    # --- replicate arm: resampled data, fit seed FIXED ----------------------
    def statistic(rep_seed: int):
        rng = np.random.default_rng(rep_seed)
        rdata = _resample_episode_data(data, rng)
        est_r = make_estimator(fit_seed)
        # Symmetry rule, applied to the model CLASS: re-resolving the reward
        # type on the resample let replicates fit a different mechanism than
        # the observed fit (V4's Acrobot-s1 failure cluster). Pin it.
        est_r.pin_reward_resolution(est0)
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
    # 0/0 is UNDEFINED, not "100% procedural". When every replicate returns the
    # same value there is no spread for either arm to explain -- d_a_null is
    # exactly this: R is constant, so the contrast is identically zero for the
    # data and every resample, and inf there reported maximal procedural
    # instability on a statistic that CANNOT VARY (S9, again). inf is kept for
    # the case it truly describes: real optimiser variance over zero replicate
    # variance. A nan numerator propagates, so unmeasurable stays unmeasurable.
    if replicate_var > 0:
        share = optimiser_var / replicate_var
    elif optimiser_var > 0:
        share = float("inf")
    else:
        share = float("nan")
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


# ------------------------------------------------------------- LR-region bounds
def _observed_ll_differentiable(model, prior_logits, est, data: EpisodeData):
    """The observed-data mixture log-likelihood, DIFFERENTIABLE in the model's
    parameters and the prior logits.

    DEDUPLICATED (2026-08-27, the V4 walk root cause): this used to be a
    second implementation "mirroring" ``e_step`` and did not mirror it — it
    called full ``model.log_prob``, which also sums the S marginal and ``U``'s
    mechanism log-prob PER ROW (``T·log p_U(k)`` per episode instead of once),
    putting ``LR(θ̂) = 70,686`` against ``c = 821`` where the invariant says
    exactly 0. Every V4 bounds row was invalid — the 8 frozen rows via the
    never-feasible fallback, the 4 "moving" rows via an offset region. It now
    DELEGATES to ``_episode_log_liks`` — the one construction site — so
    agreement with ``fit.final_ll`` is structural; the invariant is pinned by
    ``test_lr_at_theta_hat_is_zero``. The prior enters once per episode
    through ``prior_logits`` (the walk's free parameter).
    """
    ll = est._episode_log_liks(data, model=model, differentiable=True)
    ll = ll + torch.log_softmax(prior_logits, dim=0).reshape(1, -1)
    return torch.logsumexp(ll, dim=1).sum()


def lr_region_bounds(
    *,
    estimator: LatentClassEstimator,
    fit,
    data: EpisodeData,
    target_of_model: Callable,  # (model, prior: Tensor) -> scalar Tensor, differentiable
    make_estimator: Callable[[int], LatentClassEstimator],
    fit_kwargs: Optional[Dict] = None,
    alpha: float = 0.1,
    b: int = 39,
    fit_seed: int = 0,
    steps: int = 300,  # SAFETY LIMIT, not a target: the walk stops on the
    # plateau test below. If it binds, the row is BUDGET-TRUNCATED and says so.
    plateau_window: int = 50,  # measurement interval for the plateau test
    opt_lr: float = 1e-3,
    n_starts: int = 4,
    penalty: float = 100.0,  # retained in the signature; the projected walk
    # replaced the penalty pattern (exactness test, first failure) and the
    # knob is currently unread -- removing it would break callers silently
    n_jobs: int = 1,
) -> L4Result:
    """Min/max of the target over the LR confidence region — bounds-only cells.

    ``C(α) = {θ : 2(ℓ(θ̂) − ℓ(θ)) ≤ c(α)}`` with ``c(α)`` calibrated by the
    within-dataset parametric bootstrap (episode-level, refit each replicate,
    the SAME fit procedure — the one calibration device shared with L5; χ²
    asymptotics are unavailable on the mixture boundary by design-doc
    argument). The optimiser is penalised Adam over a CLONE of the fitted
    model's parameters plus the prior logits, through the two differentiable
    paths R1 pins: ``model.log_prob`` for the constraint, the caller's
    ``target_of_model`` (built on ``sample(do=)``, N1) for the objective.
    Only FEASIBLE iterates (LR ≤ c) update the bound, so a penalty violation
    can never widen the answer.

    ``penalty`` and ``steps``/``opt_lr`` are optimisation-quality knobs, not
    calibration constants: they can only make the bounds NARROWER than the
    true min/max over C(α) (a weak optimiser under-explores), never wider,
    and the Balke–Pearl reproduction is the check that they explore enough.
    """
    import copy

    fk = dict(fit_kwargs or {})
    ll_hat = float(fit.final_ll)

    # ---- c(alpha) by parametric bootstrap of the LR statistic --------------
    def statistic(rep_seed: int):
        rng = np.random.default_rng(rep_seed)
        rdata = _resample_episode_data(data, rng)
        est_r = make_estimator(fit_seed)
        # Symmetry rule, applied to the model CLASS (see point_id_interval).
        est_r.pin_reward_resolution(estimator)
        fit_r = est_r.fit(rdata, **fk)
        # LR_r = 2(l_r(theta_hat_r) - l_r(theta_hat)): the replicate's own
        # optimum against the OBSERVED parameters, both on replicate data.
        _, ll_obs_on_r, _, _ = estimator.e_step(rdata, fit.prior)

        class _Rep:
            value = max(0.0, 2.0 * (float(fit_r.final_ll) - float(ll_obs_on_r)))
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
        statistic, b=b, seed=fit_seed, statistic_name="l4_lr_calibration", n_jobs=n_jobs
    )
    vals = np.asarray(null.successes, dtype=float)
    if vals.size < 2:
        return L4Result(
            kind="abstain",
            reason=f"LR calibration produced {vals.size} usable replicates",
            alpha=alpha,
            failure_rate=null.failure_rate,
            meta={"bootstrap_diagnostics": null.diagnostics()},
        )
    c = float(np.quantile(vals, 1 - alpha))

    # ---- two-sided PROJECTED-WALK optimisation over a clone -----------------
    # The stiff-penalty pattern is retired by measurement (the exactness
    # test's first recorded failure): it cannot traverse the constraint
    # surface and collapses at theta-hat, which for a BOUNDS claim is
    # anti-conservative -- too-narrow bounds. The walk moves ALONG the LR
    # surface: the target gradient is projected onto the constraint's tangent
    # space, feasibility is restored by constraint-ascent steps when the walk
    # drifts, and only feasible iterates update the bound.
    # ---- the walk's STOP, derived from the bound's own Monte-Carlo error ----
    # A fixed step count is a constant, and a relative-improvement threshold is
    # a constant wearing a different hat; neither survives A2. The bound
    # already carries a Monte-Carlo error, because ``c(alpha)`` is a QUANTILE
    # of B replicates: ``mc_error`` is its resampling standard error, in LR
    # units. Converting it into TARGET units is free, because the walk records
    # every feasible iterate -- the bound under a threshold ``c'`` is just the
    # best target among iterates that satisfy it. So the tolerance is the
    # spread of the bound over the threshold's own MC interval. Below that,
    # further movement is unmeasurable: it is smaller than the noise in the
    # region the bound is taken over.
    se_c = null.mc_error(1 - alpha)
    if not np.isfinite(se_c):
        se_c = 0.0

    def extremum(sign: float, perturb: float = 0.0):
        model_c = copy.deepcopy(estimator.model)
        for p_ in model_c.parameters():
            p_.requires_grad_(True)
            if perturb:
                with torch.no_grad():
                    p_ += perturb * torch.randn_like(p_) * p_.abs().clamp_min(1e-3)
        prior_logits = torch.nn.Parameter(
            torch.log(fit.prior.detach().clamp_min(1e-8)).clone()
        )
        params = [p_ for p_ in model_c.parameters() if p_.requires_grad]
        params.append(prior_logits)

        def flat_grad(scalar, retain=False):
            gs = torch.autograd.grad(
                scalar, params, retain_graph=retain, allow_unused=True
            )
            return [torch.zeros_like(q) if g is None else g for g, q in zip(gs, params)]

        def dot(a, b):
            return sum((x * y).sum() for x, y in zip(a, b))

        best = None
        seen: list[tuple[float, float]] = []  # (LR, target) of feasible iterates
        stopped_by = "budget"

        def bound_at(c_limit: float):
            """The bound this walk would report at threshold ``c_limit``."""
            vals = [t for lr, t in seen if lr <= c_limit]
            return (max(vals) if sign > 0 else min(vals)) if vals else None

        def mc_tolerance():
            """The bound's own MC spread, in TARGET units (0 if unresolved)."""
            hi_b, lo_b = bound_at(c + se_c), bound_at(max(c - se_c, 0.0))
            if hi_b is None or lo_b is None:
                return 0.0
            return abs(hi_b - lo_b)

        window_start_best = None
        for step in range(steps):
            prior = torch.softmax(prior_logits, dim=0)
            tgt = target_of_model(model_c, prior)
            ll = _observed_ll_differentiable(model_c, prior_logits, estimator, data)
            g_t = flat_grad(tgt, retain=True)
            g_l = flat_grad(ll)
            denom = dot(g_l, g_l).clamp_min(1e-12)
            coef = dot(g_t, g_l) / denom
            with torch.no_grad():
                norm = torch.sqrt(
                    sum(((a - coef * b) ** 2).sum() for a, b in zip(g_t, g_l))
                ).clamp_min(1e-12)
                for q, a, b_ in zip(params, g_t, g_l):
                    q += sign * opt_lr * (a - coef * b_) / norm
            # feasibility restoration: constraint-ascent until inside (capped)
            for _ in range(20):
                ll = _observed_ll_differentiable(model_c, prior_logits, estimator, data)
                # ``.detach()`` before the float read, and free the per-row
                # graph explicitly on BOTH exits: converting a live graph
                # tensor warns, and on the break path it kept the whole
                # full-dataset graph alive into the next step's forward --
                # two resident graphs is what OOM'd the V4 run mid-flight
                # (results/v4/driver.log). ``flat_grad`` frees the graph's
                # buffers via backward; the ``del`` releases the break path
                # and drops the reference promptly on the ascent path too.
                if float(2.0 * (ll_hat - ll.detach())) <= c:
                    del ll
                    break
                g_l = flat_grad(ll)
                del ll
                with torch.no_grad():
                    n2 = torch.sqrt(dot(g_l, g_l)).clamp_min(1e-12)
                    for q, b_ in zip(params, g_l):
                        q += opt_lr * b_ / n2
            with torch.no_grad():
                prior = torch.softmax(prior_logits, dim=0)
                tgt_v = float(target_of_model(model_c, prior))
                ll_v = _observed_ll_differentiable(
                    model_c, prior_logits, estimator, data
                )
                lr_v = float(2.0 * (ll_hat - ll_v))
                if lr_v <= c:  # FEASIBLE iterate only
                    seen.append((lr_v, tgt_v))
                    if best is None or sign * tgt_v > sign * best:
                        best = tgt_v
            # PLATEAU TEST, once per window: has the bound improved by more
            # than its own MC error? The window is a measurement interval, not
            # a tolerance -- the quantity compared against is derived.
            if (step + 1) % plateau_window == 0:
                if window_start_best is not None and best is not None:
                    gained = sign * (best - window_start_best)
                    tol = mc_tolerance()
                    if tol > 0.0 and gained < tol:
                        stopped_by = "plateau"
                        break
                window_start_best = best
        if best is None:
            # never feasible -- fall back to theta_hat's own target (always
            # feasible: LR(theta_hat) = 0)
            with torch.no_grad():
                best = float(target_of_model(estimator.model, fit.prior))
            stopped_by = "never-feasible"
        return best, stopped_by

    # MULTI-START (ruled 2026-08-24): reduces under-exploration risk on a
    # region with no convexity guarantee; the per-start spread is reported --
    # material disagreement across starts means the region is non-convex in
    # practice, which is worth knowing. Start k perturbs the clone's
    # parameters deterministically by seed before the walk.
    def extremum_multi(sign: float):
        out = [extremum(sign)]
        for k in range(1, n_starts):
            torch.manual_seed(fit_seed * 1000 + k)  # perturbation stream
            out.append(extremum(sign, perturb=0.01 * k))
        return [v for v, _ in out], [why for _, why in out]

    his, why_hi = extremum_multi(+1.0)
    los, why_lo = extremum_multi(-1.0)
    hi = max(his)
    lo = min(los)
    # A budget that never binds is a safety limit; a budget that BINDS is a
    # knob and must be disclosed (the binding audit's own rule). A walk that
    # ran out of steps is still a valid INNER bound -- every reported value is
    # achieved by some compatible model -- but it is not a converged one, so
    # it carries a sub-condition and is never mistaken for one.
    truncated = [w for w in (why_hi + why_lo) if w == "budget"]
    label = fit.estimate(torch.tensor(0.0)).label()
    if truncated:
        label = f"{label} BUDGET-TRUNCATED({len(truncated)}/{len(why_hi + why_lo)})"
    return L4Result(
        kind="bounds",
        inner_approximation=True,
        lo=min(lo, hi),
        hi=max(lo, hi),
        alpha=alpha,
        b=b,
        failure_rate=null.failure_rate,
        label=label,
        meta={
            "c_alpha": c,
            "c_alpha_mc_error": se_c,
            "walk_termination": {"hi": why_hi, "lo": why_lo},
            "budget_truncated": bool(truncated),
            "max_steps": steps,
            "plateau_window": plateau_window,
            "start_spread": {
                "lo": [round(v, 4) for v in sorted(los)],
                "hi": [round(v, 4) for v in sorted(his)],
            },
            "lr_calibration_quantiles": {
                "q50": float(np.quantile(vals, 0.5)),
                "q90": float(np.quantile(vals, 0.9)),
            },
            "bootstrap_diagnostics": null.diagnostics(),
        },
    )

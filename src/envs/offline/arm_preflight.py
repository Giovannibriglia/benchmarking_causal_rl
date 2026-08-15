"""Preflight verification for the GRACE v2 arms (D-D, D-E, D-B').

**Direction of validation, stated explicitly because it is easy to get
backwards.** Every check here validates the GENERATOR against GROUND TRUTH —
the logged `U`, the declared parameters — exactly as the existing confounding
signature gates do. None of it uses GRACE's own estimator or L5's
conditional-independence tests. L5 is validated against the generator
afterwards, never the reverse: if each validated the other, a misconception
shared by both would pass in silence.

These are *measurements*, not merely assertions. The k-rank check in
particular reports the estimated ranks and their conditioning, because a
generated proxy that comes out with k-rank 1 means D-D is **not identified**
either — and that is worth discovering here in one line rather than in V-C as
a confusing estimation failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

__all__ = [
    "ProxyReport",
    "InstrumentReport",
    "DriftReport",
    "check_proxies",
    "check_instrument",
    "check_drift",
]

# Correlation below this counts as "independent" for the exclusion checks. It
# is a measurement tolerance on a finite sample, not a calibration constant:
# with N ~ 4e4 the standard error of a correlation is ~5e-3, so 0.05 is ~10 SE.
_INDEP_TOL = 0.05


def _residualise(x: np.ndarray, *by: np.ndarray) -> np.ndarray:
    """Centre ``x`` within each stratum of ``by`` — a conditional-correlation
    device. Used because the exclusion restrictions are CONDITIONAL statements
    and testing their marginal shadows gives confidently wrong verdicts (see
    the module docstring's note on collider-induced dependence)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    key = np.stack([np.asarray(b, dtype=np.float64).reshape(-1) for b in by], axis=1)
    for row in np.unique(key, axis=0):
        m = (key == row).all(axis=1)
        if m.sum() > 1:
            x[m] -= x[m].mean()
    return x


def _k_rank_permutation(
    values: np.ndarray,
    u: np.ndarray,
    episode_ids: np.ndarray | None = None,
    n_perm: int = 40,
    seed: int = 0,
) -> tuple[int, float, float]:
    """k-rank of a view, with the null ratio ESTIMATED rather than assumed.

    A fixed relative tolerance on singular values cannot distinguish a genuinely
    informative view from sampling noise: with finite samples the two rows of an
    empirical histogram are never exactly proportional, so a rank-1 view reports
    numerical rank 2. Measured on a zero-signal view the ratio s2/s1 was 0.010,
    against 0.888 for a real one — but 0.010 is not zero, and no fixed threshold
    is defensible across sample sizes.

    So: permute the U labels (destroying any U-view relationship while keeping
    both marginals), recompute s2/s1, and call the view informative only if the
    observed ratio exceeds the permutation null's maximum. Returns
    ``(k_rank, observed_ratio, null_max_ratio)``.

    **The permutation must move WHOLE EPISODES.** U is episode-static and the
    proxies are drawn once per episode, so both are block-constant and the
    effective sample size is the number of EPISODES, not steps. A step-level
    permutation shatters those blocks and produces a null far tighter than the
    statistic's own sampling law — measured on a zero-signal view: observed
    0.0310 against a step-level null max of 0.0347 (margin 0.9x, but with the
    null so tight that noise crosses it) versus an episode-level null max of
    0.1049 (margin 0.3x, correctly rank 1). Same lesson as the C1 splitter and
    the L5 bootstrap: episode granularity, never transition.
    """
    rng = np.random.default_rng(seed)
    obs_ratio = _sv_ratio(_view_matrix(values, u))

    def _permuted_u() -> np.ndarray:
        if episode_ids is None:
            return rng.permutation(u)
        ids = np.asarray(episode_ids).reshape(-1)
        uniq, inv = np.unique(ids, return_inverse=True)
        per_ep = np.array([u[ids == e][0] for e in uniq])
        return rng.permutation(per_ep)[inv]

    null = [_sv_ratio(_view_matrix(values, _permuted_u())) for _ in range(n_perm)]
    null_max = float(max(null)) if null else 0.0
    n_classes = int(np.unique(u).size)
    return (n_classes if obs_ratio > null_max else 1), obs_ratio, null_max


def _sv_ratio(matrix: np.ndarray) -> float:
    sv = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if sv.size < 2 or sv[0] <= 0:
        return 0.0
    return float(sv[1] / sv[0])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class ProxyReport:
    """D-D: are the generated proxies covariate-free, excluded, and
    INFORMATIVE ENOUGH for Kruskal's condition?"""

    n: int
    corr_z_u: float
    corr_w_u: float
    max_abs_corr_proxy_state: float
    corr_z_w_given_u: float
    max_abs_corr_proxy_action: float
    k_ranks: Dict[str, int] = field(default_factory=dict)
    singular_values: Dict[str, Sequence[float]] = field(default_factory=dict)
    condition_numbers: Dict[str, float] = field(default_factory=dict)
    covariate_free: bool = False
    exclusions_hold: bool = False
    kruskal_ok: bool = False
    reasons: tuple = ()

    def _rounded_margins(self) -> dict:
        return {k: round(v, 2) for k, v in self.condition_numbers.items()}

    def summary(self) -> str:
        return (
            f"n={self.n} corr(Z,U)={self.corr_z_u:+.3f} corr(W,U)={self.corr_w_u:+.3f} "
            f"max|corr(proxy,S)|={self.max_abs_corr_proxy_state:.4f} "
            f"corr(Z,W|U)={self.corr_z_w_given_u:+.4f} "
            f"max|corr(proxy,A)|={self.max_abs_corr_proxy_action:.4f} "
            f"k_ranks={self.k_ranks} margin={self._rounded_margins()} "
            f"covariate_free={self.covariate_free} exclusions={self.exclusions_hold} "
            f"kruskal={self.kruskal_ok}"
        )


def _k_rank(
    matrix: np.ndarray, tol_ratio: float = 0.05
) -> tuple[int, np.ndarray, float]:
    """Kruskal rank of a small measurement matrix, via its singular values.

    For an R-row matrix the k-rank is at most R, and equals R only when every
    row is linearly independent of the others — i.e. every latent class is
    distinguishable through this view. We take the numerical rank at a relative
    tolerance, which for these 2- and 4-row matrices coincides with the k-rank.
    """
    sv = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if sv.size == 0 or sv[0] <= 0:
        return 0, sv, float("inf")
    keep = int((sv > tol_ratio * sv[0]).sum())
    cond = float(sv[0] / sv[keep - 1]) if keep > 0 else float("inf")
    return keep, sv, cond


def _view_matrix(values: np.ndarray, u: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Empirical P(view | U = k) as an (R x n_bins) matrix.

    Binning here is a MEASUREMENT device for estimating the rank of the
    conditional law, not a modelling choice imposed on the estimator — GRACE
    itself never discretises.
    """
    classes = np.unique(u)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    mat = np.zeros((classes.size, n_bins), dtype=np.float64)
    for i, c in enumerate(classes):
        idx = np.digitize(values[u == c], edges)
        counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
        mat[i] = counts / max(counts.sum(), 1.0)
    return mat


def check_proxies(
    *,
    z: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray | None = None,
    episode_ids: np.ndarray | None = None,
) -> ProxyReport:
    """Validate D-D's proxies against ground truth (the logged U).

    Checks, in the order they would bite:

    1. **Covariate-free** — ``corr(proxy, S) ~ 0`` for every state dimension.
       This is the property that makes the measurement matrices GLOBAL and so
       pins the latent's labelling globally. If proxy noise were allowed to
       scale with the state, D-D would quietly acquire covariate-conditional
       proxies and lose exactly that, with no error raised anywhere.
    2. **Exclusions** — ``Z indep W | U`` (checked as the residual correlation
       after removing the U-conditional means) and neither proxy correlated
       with A.
    3. **Kruskal** — the empirical k-ranks of the three views. For binary U the
       condition ``sum(k-rank) >= 2R + 2 = 6`` with each view capped at 2 means
       ALL THREE must have k-rank 2; there is no slack.
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state[:, None]
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    reasons: list[str] = []

    corr_state = [
        max(abs(_corr(z, state[:, j])), abs(_corr(w, state[:, j])))
        for j in range(state.shape[1])
    ]
    max_state = max(corr_state) if corr_state else 0.0
    covariate_free = max_state < _INDEP_TOL
    if not covariate_free:
        reasons.append(
            f"proxy correlates with a state dimension at {max_state:.3f} — the "
            "proxies are NOT covariate-free, so their measurement matrices are "
            "not global and the labelling is not pinned across configurations"
        )

    # Z indep W | U: residualise on the U-conditional mean, then correlate.
    zr, wr = z.copy(), w.copy()
    for c in np.unique(u):
        m = u == c
        zr[m] -= z[m].mean()
        wr[m] -= w[m].mean()
    corr_zw_u = _corr(zr, wr)
    # "Neither proxy is affected by A" is the CONDITIONAL statement
    # `proxy indep A | U`. The marginal correlation is nonzero BY DESIGN --
    # a proxy measures U and A depends on U -- so testing it marginally
    # reports a violation that is not there (measured +0.50 marginal vs
    # +0.003 given U).
    a_resid = _residualise(action, u)
    max_action = max(
        abs(_corr(_residualise(z, u), a_resid)),
        abs(_corr(_residualise(w, u), a_resid)),
    )
    exclusions = abs(corr_zw_u) < _INDEP_TOL and max_action < _INDEP_TOL
    if abs(corr_zw_u) >= _INDEP_TOL:
        reasons.append(
            f"Z and W are dependent given U (residual corr {corr_zw_u:+.3f})"
        )
    if max_action >= _INDEP_TOL:
        reasons.append(f"a proxy correlates with A at {max_action:.3f}")

    views = {"Z": z, "W": w}
    if reward is not None:
        views["R"] = np.asarray(reward, dtype=np.float64).reshape(-1)
    k_ranks, svs, conds = {}, {}, {}
    for name, v in views.items():
        kr, ratio, null_max = _k_rank_permutation(v, u, episode_ids)
        k_ranks[name] = kr
        svs[name] = [round(ratio, 4), round(null_max, 4)]  # (observed, null max)
        conds[name] = ratio / null_max if null_max > 0 else float("inf")

    n_classes = int(np.unique(u).size)
    required = 2 * n_classes + 2
    achieved = sum(k_ranks.get(n, 0) for n in ("Z", "W", "R"))
    kruskal_ok = achieved >= required
    if not kruskal_ok:
        reasons.append(
            f"Kruskal k-rank sum {achieved} < required {required} for R={n_classes}: "
            "the latent structure is NOT identified from these views"
        )

    return ProxyReport(
        n=int(z.size),
        corr_z_u=_corr(z, u),
        corr_w_u=_corr(w, u),
        max_abs_corr_proxy_state=max_state,
        corr_z_w_given_u=corr_zw_u,
        max_abs_corr_proxy_action=max_action,
        k_ranks=k_ranks,
        singular_values=svs,
        condition_numbers=conds,
        covariate_free=covariate_free,
        exclusions_hold=exclusions,
        kruskal_ok=kruskal_ok,
        reasons=tuple(reasons),
    )


@dataclass
class InstrumentReport:
    n: int
    corr_i_u: float
    corr_i_action: float
    corr_i_reward_given_action_and_u: float
    independent_of_u: bool
    relevant: bool
    exclusion_holds: bool
    reasons: tuple = ()

    def summary(self) -> str:
        return (
            f"n={self.n} corr(I,U)={self.corr_i_u:+.4f} corr(I,A)={self.corr_i_action:+.3f} "
            f"corr(I,R|A,U)={self.corr_i_reward_given_action_and_u:+.4f} "
            f"indep_U={self.independent_of_u} relevant={self.relevant} "
            f"exclusion={self.exclusion_holds}"
        )


def check_instrument(
    *, i: np.ndarray, u: np.ndarray, action: np.ndarray, reward: np.ndarray
) -> InstrumentReport:
    """Validate D-E's instrument against ground truth.

    An instrument that leaks into the reward is invalid, and D-E is L4's ONLY
    exact reference (closed-form Balke-Pearl bounds) — a leaking instrument
    would make that anchor meaningless while still looking plausible.
    """
    i = np.asarray(i, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    reasons: list[str] = []

    c_iu = _corr(i, u)
    c_ia = _corr(i, a)
    # The exclusion restriction is `I indep R | (A, U)`, NOT `I indep R | A`.
    # A is a COLLIDER on I -> A <- U, so conditioning on A alone opens a path
    # from I to U and hence to R: measured -0.048 given A, versus -0.003 given
    # (A, U). Testing the wrong one condemns a perfectly valid instrument.
    c_ir_a = _corr(_residualise(i, a, u), _residualise(r, a, u))

    indep_u = abs(c_iu) < _INDEP_TOL
    relevant = abs(c_ia) > _INDEP_TOL
    exclusion = abs(c_ir_a) < _INDEP_TOL
    if not indep_u:
        reasons.append(f"I correlates with U at {c_iu:+.3f} — not exogenous")
    if not relevant:
        reasons.append(f"I barely moves A ({c_ia:+.3f}) — a weak/irrelevant instrument")
    if not exclusion:
        reasons.append(
            f"I correlates with R given (A,U) at {c_ir_a:+.3f} — the exclusion "
            "restriction fails, so the Balke-Pearl anchor would be invalid"
        )
    return InstrumentReport(
        n=int(i.size),
        corr_i_u=c_iu,
        corr_i_action=c_ia,
        corr_i_reward_given_action_and_u=c_ir_a,
        independent_of_u=indep_u,
        relevant=relevant,
        exclusion_holds=exclusion,
        reasons=tuple(reasons),
    )


@dataclass
class DriftReport:
    declared_rho: float
    predicted_autocorr: float
    realised_autocorr: float
    n_pairs: int
    matches: bool

    def summary(self) -> str:
        return (
            f"rho={self.declared_rho:.3f} predicted autocorr={self.predicted_autocorr:+.3f} "
            f"realised={self.realised_autocorr:+.3f} n={self.n_pairs} matches={self.matches}"
        )


def check_drift(
    *, u_by_episode: Sequence[Sequence[float]], rho: float, tol: float = 0.05
) -> DriftReport:
    """D-B': does the realised within-episode autocorrelation of U match rho?

    For a symmetric binary chain with per-step flip probability ``rho``,
    ``corr(U_t, U_{t+1}) = 1 - 2*rho``. rho = 0 must give autocorrelation 1
    (episode-static, i.e. D-B).
    """
    lhs: list[float] = []
    rhs: list[float] = []
    for ep in u_by_episode:
        arr = np.asarray(ep, dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            lhs.extend(arr[:-1])
            rhs.extend(arr[1:])
    realised = _corr(np.asarray(lhs), np.asarray(rhs)) if lhs else 1.0
    predicted = 1.0 - 2.0 * float(rho)
    return DriftReport(
        declared_rho=float(rho),
        predicted_autocorr=predicted,
        realised_autocorr=realised,
        n_pairs=len(lhs),
        matches=abs(realised - predicted) < tol,
    )

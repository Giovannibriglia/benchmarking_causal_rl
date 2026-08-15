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
    "NullArmReport",
    "ProxyReport",
    "InstrumentReport",
    "DriftReport",
    "check_proxies",
    "check_instrument",
    "check_drift",
    "check_null_arm",
]

# How far outside its own permutation null a statistic has to sit before we call
# it a real association. This is a DISTRIBUTIONAL cutoff -- "outside the null" --
# not a calibrated magnitude: the null is re-estimated from the data at hand for
# every check, so nothing here has to be tuned per environment or sample size.
#
# It replaces a fixed correlation tolerance of 0.05, which was wrong in a way
# only the real generator exposed. That number was justified as "~10 SE at
# N ~ 4e4 transitions" -- but U, the proxies and the instrument are all
# EPISODE-CONSTANT, so their effective sample size is the number of EPISODES.
# At 600 episodes the true SE of these correlations is ~0.04, not 0.005, and the
# fixed tolerance was condemning correct generators: an instrument drawn
# independently of U by construction measured corr(I, U) = +0.086 and was
# reported "not exogenous", with the sign flipping across strengths exactly as
# noise would. Same granularity lesson as the k-rank null, the C1 splitter and
# the L5 bootstrap.
_NULL_SDS = 3.0
_N_PERM = 200


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


def _episode_view(x: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """One row per episode for a quantity that is constant within the episode."""
    ids = np.asarray(episode_ids).reshape(-1)
    uniq = np.unique(ids)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.array([x[ids == e][0] for e in uniq])


def _episode_permutation_z(
    statistic,
    block: np.ndarray,
    episode_ids: np.ndarray,
    strata: np.ndarray | None = None,
    n_perm: int = _N_PERM,
    seed: int = 0,
) -> tuple[float, float]:
    """``(observed, z)`` for a statistic of an EPISODE-CONSTANT variable.

    ``block`` is the episode-constant series (broadcast over transitions);
    ``statistic`` takes a broadcast series and returns a scalar. The null is
    built by permuting whole episodes' values -- optionally only WITHIN strata
    of ``strata`` (also episode-constant), which preserves ``P(block | stratum)``
    while destroying every other association. That is what makes it a test of a
    CONDITIONAL independence rather than a marginal one.

    Returning a z against a re-estimated null, instead of comparing a raw
    correlation to a fixed number, is what lets the same check work at any
    episode count without a per-environment constant.
    """
    ids = np.asarray(episode_ids).reshape(-1)
    uniq, inv = np.unique(ids, return_inverse=True)
    per_ep = _episode_view(block, ids)
    strat_ep = (
        _episode_view(strata, ids) if strata is not None else np.zeros_like(per_ep)
    )

    rng = np.random.default_rng(seed)
    observed = float(statistic(np.asarray(block, dtype=np.float64).reshape(-1)))
    null = []
    for _ in range(n_perm):
        shuffled = per_ep.copy()
        for s in np.unique(strat_ep):  # permute only within a stratum
            m = strat_ep == s
            shuffled[m] = rng.permutation(per_ep[m])
        null.append(float(statistic(shuffled[inv])))
    sd = float(np.std(null))
    z = 0.0 if sd < 1e-12 else abs(observed - float(np.mean(null))) / sd
    return observed, z


def _max_family_z(
    family, episode_ids, strata=None, n_perm: int = _N_PERM, seed: int = 0
) -> float:
    """``z`` of the MAXIMUM statistic over a family, against the max's own null.

    Each entry is ``(episode_constant_series, statistic)``. In every permutation
    the whole family is recomputed and its maximum taken, so the reference
    distribution is that of the max — which is what makes a single cutoff valid
    for a family test. Without this, a per-test cutoff applied to a max reports
    roughly ``len(family)`` times the intended false-alarm rate.
    """
    ids = np.asarray(episode_ids).reshape(-1)
    uniq, inv = np.unique(ids, return_inverse=True)
    strat_ep = _episode_view(strata, ids) if strata is not None else np.zeros(uniq.size)
    rng = np.random.default_rng(seed)
    per_ep = [_episode_view(b, ids) for b, _ in family]
    observed = max(
        abs(float(stat(np.asarray(b, dtype=np.float64).reshape(-1))))
        for b, stat in family
    )
    null = []
    for _ in range(n_perm):
        vals = []
        for arr, (_, stat) in zip(per_ep, family):
            shuffled = arr.copy()
            for s in np.unique(strat_ep):
                m = strat_ep == s
                shuffled[m] = rng.permutation(arr[m])
            vals.append(abs(float(stat(shuffled[inv]))))
        null.append(max(vals))
    sd = float(np.std(null))
    return 0.0 if sd < 1e-12 else abs(observed - float(np.mean(null))) / sd


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
    null_sds: Dict[str, float] = field(default_factory=dict)
    covariate_free: bool = False
    exclusions_hold: bool = False
    kruskal_ok: bool = False
    reasons: tuple = ()

    def _rounded_null_sds(self) -> dict:
        return {k: round(v, 1) for k, v in self.null_sds.items()}

    def _rounded_margins(self) -> dict:
        return {k: round(v, 2) for k, v in self.condition_numbers.items()}

    def summary(self) -> str:
        return (
            f"n={self.n} corr(Z,U)={self.corr_z_u:+.3f} corr(W,U)={self.corr_w_u:+.3f} "
            f"max|corr(proxy,S)|={self.max_abs_corr_proxy_state:.4f} "
            f"nullSDs={self._rounded_null_sds()} "
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
    # A view with too few distinct values -- Acrobot's reward is -1 almost
    # everywhere -- collapses the quantile edges, and the histogram degenerates
    # to a single occupied bin. That is NOT evidence of an uninformative view,
    # it is the binning failing, and the two must not be reported alike.
    if np.unique(edges).size < 2:
        return np.zeros((classes.size, n_bins), dtype=np.float64)
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

    if episode_ids is None:
        raise ValueError(
            "check_proxies needs episode_ids: Z, W and U are episode-constant, so "
            "every independence check here has an EPISODE-level null. Without the "
            "blocks the null is computed at transition granularity and is far too "
            "tight -- it condemns correct generators."
        )
    episode_ids = np.asarray(episode_ids).reshape(-1)

    # 1. COVARIATE-FREE: Z indep S | U. This is a CONDITIONAL statement and the
    #    marginal shadow is nonzero BY DESIGN -- U drives the action, the action
    #    drives the next state, so the state carries information about U and a
    #    proxy that measures U is marginally correlated with it. Measured on the
    #    real generator at strength 1.5: max|corr(proxy, S)| = 0.226 marginally,
    #    which the earlier marginal check reported as "the proxies are NOT
    #    covariate-free" -- a confident false alarm about a generator whose proxy
    #    noise provably never reads obs. Permuting proxies BETWEEN EPISODES OF
    #    THE SAME U stratum holds P(Z|U) fixed and destroys only the state link,
    #    which is exactly the null this claim needs.
    state_z = []
    for j in range(state.shape[1]):
        col = state[:, j]
        for proxy in (z, w):
            _, zz = _episode_permutation_z(
                lambda b, c=col: _corr(_residualise(b, u), _residualise(c, u)),
                proxy,
                episode_ids,
                strata=u,
            )
            state_z.append(zz)
    max_state_z = max(state_z) if state_z else 0.0
    max_state = max(
        (
            max(abs(_corr(z, state[:, j])), abs(_corr(w, state[:, j])))
            for j in range(state.shape[1])
        ),
        default=0.0,
    )
    covariate_free = max_state_z < _NULL_SDS
    if not covariate_free:
        reasons.append(
            f"a proxy is associated with a state dimension GIVEN U at "
            f"{max_state_z:.1f} null SDs -- the proxies are not covariate-free, so "
            "their measurement matrices are not global and the labelling is not "
            "pinned across configurations"
        )

    # 2. Z indep W | U, and proxy indep A | U. Both are conditional, and both
    #    involve episode-constant proxies, so both get the same episode-level
    #    within-stratum null. The transition-level version reported a residual
    #    corr(Z, W | U) of +0.054 at EVERY proxy strength including 0.0, where Z
    #    and W are independent noise by construction -- the constancy across the
    #    sweep was the tell that it was measuring episode-count noise.
    corr_zw_u, zw_z = _episode_permutation_z(
        lambda b: _corr(_residualise(b, u), _residualise(w, u)),
        z,
        episode_ids,
        strata=u,
    )
    a_resid = _residualise(action, u)
    max_action_z = _max_family_z(
        [(proxy, lambda b: _corr(_residualise(b, u), a_resid)) for proxy in (z, w)],
        episode_ids,
        strata=u,
    )
    max_action = max(
        abs(_corr(_residualise(z, u), a_resid)),
        abs(_corr(_residualise(w, u), a_resid)),
    )
    exclusions = zw_z < _NULL_SDS and max_action_z < _NULL_SDS
    if zw_z >= _NULL_SDS:
        reasons.append(
            f"Z and W are dependent given U ({zw_z:.1f} null SDs, residual corr "
            f"{corr_zw_u:+.3f})"
        )
    if max_action_z >= _NULL_SDS:
        reasons.append(
            f"a proxy is associated with A given U at {max_action_z:.1f} null SDs"
        )

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
        null_sds={
            "proxy_vs_state_given_u": max_state_z,
            "z_vs_w_given_u": zw_z,
            "proxy_vs_action_given_u": max_action_z,
        },
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
    null_sds: Dict[str, float] = field(default_factory=dict)
    exclusion_testable: bool = False
    independent_of_u: bool = False
    relevant: bool = False
    exclusion_holds: bool = False
    reasons: tuple = ()

    def _rounded_null_sds(self) -> dict:
        return {k: round(v, 1) for k, v in self.null_sds.items()}

    def summary(self) -> str:
        return (
            f"n={self.n} corr(I,U)={self.corr_i_u:+.4f} corr(I,A)={self.corr_i_action:+.3f} "
            f"corr(I,R|A,U)={self.corr_i_reward_given_action_and_u:+.4f} "
            f"nullSDs={self._rounded_null_sds()} "
            f"indep_U={self.independent_of_u} relevant={self.relevant} "
            f"exclusion={self.exclusion_holds} "
            f"exclusion_testable={self.exclusion_testable}"
        )


def check_instrument(
    *,
    i: np.ndarray,
    u: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    episode_ids: np.ndarray,
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

    # I is drawn ONCE PER EPISODE, so every claim about it has an episode-level
    # null. Judged against a fixed correlation tolerance instead, a provably
    # exogenous instrument measured corr(I, U) = +0.086 at lambda = 0.1 and was
    # reported "not exogenous"; the sign flipped across strengths (+0.086,
    # +0.018, -0.036), which is what episode-count noise looks like and what a
    # real dependence never does.
    episode_ids = np.asarray(episode_ids).reshape(-1)
    c_iu, z_iu = _episode_permutation_z(lambda b: _corr(b, u), i, episode_ids)
    c_ia, z_ia = _episode_permutation_z(lambda b: _corr(b, a), i, episode_ids)
    # The exclusion restriction is `I indep R | (A, U)`, NOT `I indep R | A`.
    # A is a COLLIDER on I -> A <- U, so conditioning on A alone opens a path
    # from I to U and hence to R: measured -0.048 given A, versus -0.003 given
    # (A, U). Testing the wrong one condemns a perfectly valid instrument.
    # Permuted WITHIN U strata, so the null keeps P(I | U) and destroys only the
    # reward link -- the conditional statement, not its marginal shadow.
    c_ir_a, z_ir = _episode_permutation_z(
        lambda b: _corr(_residualise(b, a, u), _residualise(r, a, u)),
        i,
        episode_ids,
        strata=u,
    )

    # DEGENERACY GUARD. On CartPole the reward is r = 1 + c_r*U*1[a = a_bad], a
    # DETERMINISTIC function of (A, U) -- so residualising on (A, U) annihilates
    # it and the exclusion statistic is identically zero for the observed data
    # AND for every permutation. That reads as a clean pass and is actually a
    # measurement of nothing. The structural argument still holds (the wrapper
    # never reads I when perturbing the reward), but this check must not be
    # allowed to claim it verified it.
    r_resid_var = float(np.var(_residualise(r, a, u)))
    exclusion_testable = r_resid_var > 1e-12
    if not exclusion_testable:
        reasons.append(
            "exclusion NOT TESTABLE on this env: R is a deterministic function of "
            "(A, U), so conditioning on them leaves no residual variance for I to "
            "correlate with. The restriction holds by construction, but no "
            "evidence for it is available here -- it needs an env with stochastic "
            "reward given (A, U)"
        )

    # Note the asymmetry, which is deliberate: independence and exclusion must
    # sit INSIDE the null (nothing to detect), relevance must sit OUTSIDE it (an
    # instrument that does not move A is useless, and "no detectable effect" is
    # exactly the failure).
    indep_u = z_iu < _NULL_SDS
    relevant = z_ia >= _NULL_SDS
    exclusion = exclusion_testable and z_ir < _NULL_SDS
    if not indep_u:
        reasons.append(
            f"I is associated with U at {z_iu:.1f} null SDs (corr {c_iu:+.3f}) — "
            "not exogenous"
        )
    if not relevant:
        reasons.append(
            f"I barely moves A ({c_ia:+.3f}, {z_ia:.1f} null SDs) — a weak or "
            "irrelevant instrument"
        )
    if not exclusion:
        reasons.append(
            f"I is associated with R given (A,U) at {z_ir:.1f} null SDs "
            f"(corr {c_ir_a:+.3f}) — the exclusion "
            "restriction fails, so the Balke-Pearl anchor would be invalid"
        )
    return InstrumentReport(
        n=int(i.size),
        corr_i_u=c_iu,
        corr_i_action=c_ia,
        corr_i_reward_given_action_and_u=c_ir_a,
        null_sds={"i_vs_u": z_iu, "i_vs_a": z_ia, "i_vs_r_given_a_u": z_ir},
        exclusion_testable=exclusion_testable,
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


@dataclass
class NullArmReport:
    """D-A-null: is there genuinely NOTHING for L5 to find?"""

    n: int
    n_episodes: int
    null_sds: Dict[str, float] = field(default_factory=dict)
    u_inert: bool = False
    reasons: tuple = ()

    def _rounded(self) -> dict:
        return {k: round(v, 1) for k, v in self.null_sds.items()}

    def summary(self) -> str:
        return (
            f"n={self.n} episodes={self.n_episodes} "
            f"nullSDs={self._rounded()} "
            f"inert={self.u_inert}"
        )


def check_null_arm(
    *,
    u: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    episode_ids: np.ndarray,
) -> NullArmReport:
    """Certify that the logged U is INERT — it touches neither A nor R.

    This is the arm L5's FALSE-POSITIVE RATE is read from, so its validity is
    the thing that makes a refutation there interpretable as a false alarm. If
    U were not actually inert, every "false positive" measured here would be
    partly a true detection and the rate would be silently understated — in the
    flattering direction, which is the one that needs guarding.

    It matters that this is a distinct check rather than the confounded gate
    with the dial at zero. The gate asks "is the declared confounding present at
    the declared strength"; this asks "is there any association at all", which
    is a different question and the only one that licenses the null arm's use.

    Episode-level null throughout (S1): U is drawn once per episode.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    episode_ids = np.asarray(episode_ids).reshape(-1)
    reasons: list[str] = []

    _, z_ua = _episode_permutation_z(lambda b: _corr(b, a), u, episode_ids)
    _, z_ur = _episode_permutation_z(lambda b: _corr(b, r), u, episode_ids)
    # Also the gated contrast the confounded arms rely on: within a == a_bad,
    # is the reward associated with U? That is the exact channel c_r opens, so
    # it is the one that must be dead here.
    gated = a == 1.0
    if int(gated.sum()) > 1:
        _, z_gated = _episode_permutation_z(
            lambda b: _corr(b[gated], r[gated]), u, episode_ids
        )
    else:
        z_gated = 0.0

    inert = max(z_ua, z_ur, z_gated) < _NULL_SDS
    if z_ua >= _NULL_SDS:
        reasons.append(
            f"U is associated with A at {z_ua:.1f} null SDs — not a null arm"
        )
    if z_ur >= _NULL_SDS:
        reasons.append(
            f"U is associated with R at {z_ur:.1f} null SDs — not a null arm"
        )
    if z_gated >= _NULL_SDS:
        reasons.append(
            f"U is associated with R within a = a_bad at {z_gated:.1f} null SDs — "
            "the gated reward channel is live, so c_r did not reach zero"
        )
    return NullArmReport(
        n=int(u.size),
        n_episodes=int(np.unique(episode_ids).size),
        null_sds={"u_vs_a": z_ua, "u_vs_r": z_ur, "u_vs_r_gated": z_gated},
        u_inert=inert,
        reasons=tuple(reasons),
    )

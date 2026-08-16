"""Preflight verification for the GRACE v2 arms (D-D, D-E, D-B').

**Direction of validation, stated explicitly because it is easy to get
backwards.** Every check here validates the GENERATOR against GROUND TRUTH --
the logged `U`, the declared parameters -- exactly as the existing confounding
signature gates do. None of it uses GRACE's own estimator or L5's
conditional-independence tests. L5 is validated against the generator
afterwards, never the reverse: if each validated the other, a misconception
shared by both would pass in silence.

These are *measurements*, not merely assertions. The k-rank check in
particular reports the estimated ranks and their conditioning, because a
generated proxy that comes out with k-rank 1 means D-D is **not identified**
either -- and that is worth discovering here in one line rather than in V-C as
a confusing estimation failure.

GRANULARITY (rule S1b), which governs every statistic below
-----------------------------------------------------------
**In RL, episode length is an OUTCOME.** `U`, the proxies `Z`/`W` and the
instrument `I` are drawn ONCE PER EPISODE. Pooling such a quantity over
transitions replicates each draw once per step, so every episode enters the
statistic with weight proportional to its own length -- and length is driven by
the behaviour policy, which `U` drives. The statistic therefore acquires a
dependence that has nothing to do with the claim under test.

This is not repaired by an episode-level *null*. Permuting whole episodes
destroys the value/length pairing in the null while the observed statistic
keeps it, so the artefact surfaces as a large deviation rather than as noise.
Measured on an instrument drawn from its own Bernoulli that provably never
reads `U`: corr(I, U) = **-0.590** pooled over transitions against **-0.034**
with one row per episode.

So the rule enforced throughout this module:

* an episode-constant quantity enters as **one row per episode**
  (`_episode_constant`, which RAISES if the quantity is not in fact constant
  within the episode -- the mistake must not be silently absorbable);
* a genuinely per-step companion (state, action, reward) is reduced to an
  **episode statistic** (`_episode_mean`) before being paired with one;
* the permutation null then shuffles episode rows, optionally within strata.

The only quantity left at transition level is `check_drift`'s within-episode
autocorrelation, which is per-step by construction -- and even there the
length-weighting exemption is now *measured* rather than asserted.
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
# it a real association. This is a DISTRIBUTIONAL level -- "outside the null" --
# not a calibrated magnitude: the null is re-estimated from the data at hand for
# every check, so nothing here has to be tuned per environment or sample size.
#
# It replaces a fixed correlation tolerance of 0.05, which was wrong in a way
# only the real generator exposed. That number was justified as "~10 SE at
# N ~ 4e4 transitions" -- but U, the proxies and the instrument are all
# EPISODE-CONSTANT, so their effective sample size is the number of EPISODES.
# At 600 episodes the true SE of these correlations is ~0.04, not 0.005, and the
# fixed tolerance was condemning correct generators.
#
# WHY A QUANTILE AND NOT A z-SCORE. The previous spelling built the null
# correctly and then read it with `|obs - mean| / sd > 3`. That is only a
# ~0.1% cutoff for a roughly symmetric null, and several of these statistics
# are MAXIMA over a family, whose distribution is strongly right-skewed: the
# mean sits well below the upper tail's mode and the sd is inflated by that same
# tail, so a 3-sd rule is not the level it looks like. The permutation p-value
# below needs no distributional assumption at all -- it is read off the draws
# that were actually taken. The `z` is still reported alongside, purely as a
# human-readable effect size; nothing decides on it any more.
_N_PERM = 200
_NULL_ALPHA = 0.01  # smallest attainable p at _N_PERM draws is 1/201 = 0.005
_NULL_SDS = 3.0  # RETAINED FOR REPORTING ONLY -- no verdict reads this


def _residualise(x: np.ndarray, *by: np.ndarray) -> np.ndarray:
    """Centre ``x`` within each stratum of ``by`` -- a conditional-correlation
    device. Used because the exclusion restrictions are CONDITIONAL statements
    and testing their marginal shadows gives confidently wrong verdicts (see
    the module docstring's note on collider-induced dependence).

    Pointwise: the value at row ``i`` depends only on row ``i``'s stratum. That
    is what makes it safe to apply at transition level and aggregate to episode
    level afterwards -- see ``check_instrument``'s exclusion statistic.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    key = np.stack([np.asarray(b, dtype=np.float64).reshape(-1) for b in by], axis=1)
    for row in np.unique(key, axis=0):
        m = (key == row).all(axis=1)
        if m.sum() > 1:
            x[m] -= x[m].mean()
    return x


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------
# Episode granularity (S1b). Every statistic in this module is built on these.
# --------------------------------------------------------------------------


def _episode_index(episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(unique_ids, inverse)`` -- the transition -> episode-row map."""
    ids = np.asarray(episode_ids).reshape(-1)
    uniq, inv = np.unique(ids, return_inverse=True)
    return uniq, inv.reshape(-1)


def _episode_mean(x: np.ndarray, inv: np.ndarray, n_ep: int) -> np.ndarray:
    """Episode MEAN of a genuinely per-step quantity -- the length-free summary.

    The episode SUM (a return, a step count) is deliberately not offered: it is
    proportional to length by construction, which is the exact quantity S1b
    exists to keep out of these statistics.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    total = np.bincount(inv, weights=x, minlength=n_ep)
    count = np.bincount(inv, minlength=n_ep).astype(np.float64)
    return total / np.maximum(count, 1.0)


def _episode_constant(x: np.ndarray, inv: np.ndarray, n_ep: int, name: str):
    """One row per episode for a quantity that must be constant within it.

    RAISES if it is not. The whole class of S1b bugs is "an episode-constant
    quantity was handled at transition level"; the mirror-image mistake --
    handing a per-step quantity to an episode-constant reduction -- would
    silently keep one arbitrary step's value and read as a clean measurement. It
    has to be loud. ``D-B'``'s drifting `U` is the one legitimate per-step
    latent and it never reaches this function: that arm runs ``check_drift``,
    which is transition-level by construction.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    lo = np.full(n_ep, np.inf)
    hi = np.full(n_ep, -np.inf)
    np.minimum.at(lo, inv, x)
    np.maximum.at(hi, inv, x)
    spread = float(np.max(hi - lo)) if n_ep else 0.0
    if spread > 1e-9:
        raise ValueError(
            f"{name} varies WITHIN an episode (max within-episode spread "
            f"{spread:.6g}) but is being reduced to one row per episode. Either "
            f"{name} is genuinely per-step -- in which case it needs an episode "
            "STATISTIC (_episode_mean), not a constant reduction -- or the "
            "episode ids are wrong. Do not silently take the first step's value."
        )
    return hi


def _within_strata_permutation(strata_ep: np.ndarray, rng) -> np.ndarray:
    """A permutation of EPISODE ROWS that stays inside each stratum.

    Holding ``P(block | stratum)`` fixed while destroying every other
    association is what makes the resulting test a test of a CONDITIONAL
    independence rather than of its marginal shadow (S2).
    """
    idx = np.arange(strata_ep.size)
    out = idx.copy()
    for s in np.unique(strata_ep):
        m = strata_ep == s
        out[m] = rng.permutation(idx[m])
    return out


def _permutation_family_test(
    entries,
    strata_ep: np.ndarray | None = None,
    n_perm: int = _N_PERM,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(observed, p_value, z)`` for the MAXIMUM over a family of statistics.

    ``entries`` is a list of ``(episode_level_series, statistic)``; the
    statistic receives the (possibly permuted) series and returns a scalar,
    whose absolute value is taken. A single test is just a family of one.

    Two things this gets right that the previous spelling did not:

    * **The null is the null OF THE MAX** (S3). A maximum over 8 tests judged
      by a per-test cutoff runs ~8x the intended false-alarm rate -- it fired on
      provably covariate-free proxies. Every permutation recomputes the whole
      family and takes its max.
    * **One permutation per draw, shared across the family.** Drawing an
      independent permutation per entry decouples statistics that are dependent
      in reality, widening the null. That errs safe, but it is not the null of
      the statistic actually computed, which is what S3 asks for.

    The verdict is the permutation p-value, not a z: several of these families
    have strongly right-skewed nulls and a z-score misreads their tails.
    """
    n_ep = entries[0][0].size
    strata_ep = np.zeros(n_ep) if strata_ep is None else np.asarray(strata_ep)
    rng = np.random.default_rng(seed)
    observed = max(abs(float(stat(series))) for series, stat in entries)
    null = []
    for _ in range(n_perm):
        perm = _within_strata_permutation(strata_ep, rng)
        null.append(max(abs(float(stat(series[perm]))) for series, stat in entries))
    null_arr = np.asarray(null, dtype=np.float64)
    # The +1 in both places is what makes the permutation p-value exact rather
    # than merely approximate: under the null the observed value is itself one
    # of the draws.
    p = float((1 + int((null_arr >= observed).sum())) / (1 + n_perm))
    sd = float(null_arr.std())
    z = 0.0 if sd < 1e-12 else abs(observed - float(null_arr.mean())) / sd
    return observed, p, z


def _signed_permutation_test(
    series_ep: np.ndarray,
    statistic,
    strata_ep: np.ndarray | None = None,
    n_perm: int = _N_PERM,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(signed observed, p, z)`` for a single statistic.

    Identical machinery to ``_permutation_family_test``; it exists only so the
    reports can carry the SIGNED correlation, which is diagnostic -- a sign that
    flips across a strength sweep is what episode-count noise looks like, and
    what a real dependence never does.
    """
    signed = float(statistic(series_ep))
    _, p, z = _permutation_family_test(
        [(series_ep, statistic)], strata_ep, n_perm=n_perm, seed=seed
    )
    return signed, p, z


# --------------------------------------------------------------------------
# Kruskal rank of a measurement view
# --------------------------------------------------------------------------


def _sv_ratio(matrix: np.ndarray) -> float:
    sv = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if sv.size < 2 or sv[0] <= 0:
        return 0.0
    return float(sv[1] / sv[0])


def _view_matrix(values: np.ndarray, u: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Empirical P(view | U = k) as an (R x n_bins) matrix, ONE ROW PER EPISODE.

    Binning here is a MEASUREMENT device for estimating the rank of the
    conditional law, not a modelling choice imposed on the estimator -- GRACE
    itself never discretises.

    ``values`` and ``u`` are already at episode granularity when this is
    reached. Pooling over transitions instead would build each row from a
    length-weighted mixture of episodes, and since length depends on U the two
    rows would differ for that reason alone -- manufacturing rank 2 out of an
    uninformative view (S1b).
    """
    classes = np.unique(u)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    # A view with too few distinct values -- Acrobot's reward is -1 almost
    # everywhere -- collapses the quantile edges, and the histogram degenerates
    # to a single occupied bin. That is NOT evidence of an uninformative view,
    # it is the binning failing, and the two must not be reported alike: the
    # caller receives the degeneracy separately and says so in its reasons.
    if np.unique(edges).size < 2:
        return np.zeros((classes.size, n_bins), dtype=np.float64)
    mat = np.zeros((classes.size, n_bins), dtype=np.float64)
    for i, c in enumerate(classes):
        idx = np.digitize(values[u == c], edges)
        counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
        mat[i] = counts / max(counts.sum(), 1.0)
    return mat


def _binning_degenerate(values: np.ndarray, n_bins: int = 8) -> bool:
    """Did the quantile grid collapse? Reported separately from the verdict so a
    failed MEASUREMENT is never filed as an uninformative VIEW (S3, S8)."""
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return bool(np.unique(edges).size < 2)


def _k_rank_permutation(
    values_ep: np.ndarray,
    u_ep: np.ndarray,
    n_perm: int = _N_PERM,
    seed: int = 0,
) -> tuple[int, float, float, float]:
    """k-rank of a view, with the null ratio ESTIMATED rather than assumed.

    A fixed relative tolerance on singular values cannot distinguish a genuinely
    informative view from sampling noise: with finite samples the two rows of an
    empirical histogram are never exactly proportional, so a rank-1 view reports
    numerical rank 2. Measured on a zero-signal view the ratio s2/s1 was 0.010,
    against 0.888 for a real one -- but 0.010 is not zero, and no fixed threshold
    is defensible across sample sizes.

    So: permute the U labels (destroying any U-view relationship while keeping
    both marginals), recompute s2/s1, and call the view informative only if the
    observed ratio lands outside the permutation null. Returns
    ``(k_rank, observed_ratio, null_cutoff, p_value)``.

    **Both arguments arrive at EPISODE granularity.** U is episode-static and
    the proxies are drawn once per episode, so a transition-level histogram
    weights each episode by its own length (S1b). A step-level *permutation* was
    the first version of this bug -- it shattered the blocks and gave a null far
    tighter than the statistic's sampling law -- but permuting episodes while
    still *pooling* transitions leaves the observed ratio length-weighted and
    the null not, which is the same defect wearing a different hat.

    The cutoff is the ``1 - alpha`` QUANTILE of the draws, not their maximum and
    not a z-score: s2/s1 is a bounded, right-skewed statistic and neither of the
    other two reads its tail correctly.
    """
    rng = np.random.default_rng(seed)
    obs_ratio = _sv_ratio(_view_matrix(values_ep, u_ep))
    null = np.asarray(
        [
            _sv_ratio(_view_matrix(values_ep, rng.permutation(u_ep)))
            for _ in range(n_perm)
        ]
    )
    cutoff = float(np.quantile(null, 1.0 - _NULL_ALPHA)) if null.size else 0.0
    p = float((1 + int((null >= obs_ratio).sum())) / (1 + n_perm))
    n_classes = int(np.unique(u_ep).size)
    return (n_classes if p < _NULL_ALPHA else 1), obs_ratio, cutoff, p


# --------------------------------------------------------------------------
# D-D: the proxies
# --------------------------------------------------------------------------


@dataclass
class ProxyReport:
    """D-D: are the generated proxies covariate-free, excluded, and
    INFORMATIVE ENOUGH for Kruskal's condition?"""

    n: int
    n_episodes: int
    corr_z_u: float
    corr_w_u: float
    max_abs_corr_proxy_state: float
    corr_z_w_given_u: float
    max_abs_corr_proxy_action: float
    k_ranks: Dict[str, int] = field(default_factory=dict)
    singular_values: Dict[str, Sequence[float]] = field(default_factory=dict)
    condition_numbers: Dict[str, float] = field(default_factory=dict)
    null_p: Dict[str, float] = field(default_factory=dict)
    null_sds: Dict[str, float] = field(default_factory=dict)
    binning_degenerate: Dict[str, bool] = field(default_factory=dict)
    covariate_free: bool = False
    exclusions_hold: bool = False
    kruskal_ok: bool = False
    reasons: tuple = ()

    def _rounded_p(self) -> dict:
        return {k: round(v, 4) for k, v in self.null_p.items()}

    def _rounded_margins(self) -> dict:
        return {k: round(v, 2) for k, v in self.condition_numbers.items()}

    def summary(self) -> str:
        return (
            f"n={self.n} episodes={self.n_episodes} "
            f"corr(Z,U)={self.corr_z_u:+.3f} corr(W,U)={self.corr_w_u:+.3f} "
            f"max|corr(proxy,S)|={self.max_abs_corr_proxy_state:.4f} "
            f"nullP={self._rounded_p()} "
            f"corr(Z,W|U)={self.corr_z_w_given_u:+.4f} "
            f"max|corr(proxy,A)|={self.max_abs_corr_proxy_action:.4f} "
            f"k_ranks={self.k_ranks} margin={self._rounded_margins()} "
            f"covariate_free={self.covariate_free} exclusions={self.exclusions_hold} "
            f"kruskal={self.kruskal_ok}"
        )


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

    1. **Covariate-free** -- ``Z indep S | U`` for every state dimension. This is
       the property that makes the measurement matrices GLOBAL and so pins the
       latent's labelling globally. If proxy noise were allowed to scale with
       the state, D-D would quietly acquire covariate-conditional proxies and
       lose exactly that, with no error raised anywhere.
    2. **Exclusions** -- ``Z indep W | U`` (checked as the residual correlation
       after removing the U-conditional means) and neither proxy correlated
       with A.
    3. **Kruskal** -- the empirical k-ranks of the three views. For binary U the
       condition ``sum(k-rank) >= 2R + 2 = 6`` with each view capped at 2 means
       ALL THREE must have k-rank 2; there is no slack.

    GRANULARITY (S1b). ``Z``, ``W`` and ``U`` are episode-constant and enter as
    one row per episode. Their per-step companions are reduced to an episode
    statistic first: the state to its per-dimension episode mean, the action to
    its episode mean (for a binary action that is the episode's P(a = a_bad)),
    the reward to its per-step episode mean. The mean and not the sum -- an
    episode return is proportional to length by construction and would smuggle
    back the very weighting this rule removes.
    """
    if episode_ids is None:
        raise ValueError(
            "check_proxies needs episode_ids: Z, W and U are episode-constant, so "
            "every statistic here is computed at EPISODE granularity and its null "
            "permutes whole episodes. Without the blocks the statistic is weighted "
            "by episode length -- an OUTCOME -- and it condemns correct generators."
        )
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state[:, None]
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    reasons: list[str] = []

    uniq, inv = _episode_index(episode_ids)
    n_ep = int(uniq.size)
    z_ep = _episode_constant(z, inv, n_ep, "Z")
    w_ep = _episode_constant(w, inv, n_ep, "W")
    u_ep = _episode_constant(u, inv, n_ep, "U")
    a_ep = _episode_mean(action, inv, n_ep)
    s_ep = np.stack(
        [_episode_mean(state[:, j], inv, n_ep) for j in range(state.shape[1])], axis=1
    )

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
    #
    #    ONE FAMILY, ONE NULL (S3). This is |state dims| x 2 statistics and the
    #    verdict is their maximum, so it is judged against the null OF THE
    #    MAXIMUM. Judged per-test at 3 sd it ran roughly 8x the intended
    #    false-alarm rate on CartPole -- four state dimensions times two proxies
    #    -- which is where four of V-B's D-D failures came from.
    s_resid = [_residualise(s_ep[:, j], u_ep) for j in range(s_ep.shape[1])]
    state_family = [
        (proxy, (lambda b, c=col: _corr(_residualise(b, u_ep), c)))
        for col in s_resid
        for proxy in (z_ep, w_ep)
    ]
    max_state, p_state, z_state = _permutation_family_test(state_family, strata_ep=u_ep)
    max_state_marginal = max(
        (
            max(abs(_corr(z_ep, s_ep[:, j])), abs(_corr(w_ep, s_ep[:, j])))
            for j in range(s_ep.shape[1])
        ),
        default=0.0,
    )
    covariate_free = p_state >= _NULL_ALPHA
    if not covariate_free:
        reasons.append(
            f"a proxy is associated with a state dimension GIVEN U at "
            f"permutation p={p_state:.4f} (max |residual corr| {max_state:.3f}, "
            f"{z_state:.1f} null SDs) -- the proxies are not covariate-free, so "
            "their measurement matrices are not global and the labelling is not "
            "pinned across configurations"
        )

    # 2. Z indep W | U, and proxy indep A | U. Both are conditional, and both
    #    involve episode-constant proxies, so both get the same episode-level
    #    within-stratum null. The transition-level version reported a residual
    #    corr(Z, W | U) of +0.054 at EVERY proxy strength including 0.0, where Z
    #    and W are independent noise by construction -- the constancy across the
    #    sweep was the tell that it was measuring episode-count noise.
    w_resid = _residualise(w_ep, u_ep)
    corr_zw_u, p_zw, z_zw = _signed_permutation_test(
        z_ep,
        lambda b: _corr(_residualise(b, u_ep), w_resid),
        strata_ep=u_ep,
    )
    a_resid = _residualise(a_ep, u_ep)
    max_action, p_action, z_action = _permutation_family_test(
        [
            (proxy, lambda b: _corr(_residualise(b, u_ep), a_resid))
            for proxy in (z_ep, w_ep)
        ],
        strata_ep=u_ep,
    )
    exclusions = p_zw >= _NULL_ALPHA and p_action >= _NULL_ALPHA
    if p_zw < _NULL_ALPHA:
        reasons.append(
            f"Z and W are dependent given U (permutation p={p_zw:.4f}, residual "
            f"corr {corr_zw_u:+.3f})"
        )
    if p_action < _NULL_ALPHA:
        reasons.append(
            f"a proxy is associated with A given U at permutation p={p_action:.4f} "
            f"(max |residual corr| {max_action:.3f})"
        )

    # 3. Kruskal. The reward view is the one genuinely per-step quantity among
    #    the three, and it enters as its EPISODE MEAN -- a lossy but valid view:
    #    a function of the episode's own data, still conditionally independent of
    #    Z and W given U. Lossy in the safe direction, since a summary that
    #    clears k-rank 2 implies the full sequence does.
    views = {"Z": z_ep, "W": w_ep}
    if reward is not None:
        views["R"] = _episode_mean(
            np.asarray(reward, dtype=np.float64).reshape(-1), inv, n_ep
        )
    k_ranks, svs, conds, degenerate, view_p = {}, {}, {}, {}, {}
    for name, v in views.items():
        degenerate[name] = _binning_degenerate(v)
        kr, ratio, cutoff, p = _k_rank_permutation(v, u_ep)
        k_ranks[name] = kr
        svs[name] = [round(ratio, 4), round(cutoff, 4)]  # (observed, null cutoff)
        conds[name] = ratio / cutoff if cutoff > 0 else float("inf")
        view_p[f"k_rank_{name}"] = p

    n_classes = int(np.unique(u_ep).size)
    required = 2 * n_classes + 2
    achieved = sum(k_ranks.get(n, 0) for n in ("Z", "W", "R"))
    kruskal_ok = achieved >= required
    if not kruskal_ok:
        collapsed = [n for n, bad in degenerate.items() if bad]
        detail = (
            f"; the {'/'.join(collapsed)} view's quantile grid COLLAPSED, so that "
            "view is a FAILED MEASUREMENT rather than evidence of an uninformative "
            "view -- it must not be read as a structural verdict"
            if collapsed
            else ""
        )
        reasons.append(
            f"Kruskal k-rank sum {achieved} < required {required} for R={n_classes}: "
            f"the latent structure is NOT identified from these views{detail}"
        )

    return ProxyReport(
        n=int(np.asarray(z).size),
        n_episodes=n_ep,
        corr_z_u=_corr(z_ep, u_ep),
        corr_w_u=_corr(w_ep, u_ep),
        max_abs_corr_proxy_state=max_state_marginal,
        corr_z_w_given_u=corr_zw_u,
        max_abs_corr_proxy_action=max_action,
        null_p={
            "proxy_vs_state_given_u": p_state,
            "z_vs_w_given_u": p_zw,
            "proxy_vs_action_given_u": p_action,
            **view_p,
        },
        null_sds={
            "proxy_vs_state_given_u": z_state,
            "z_vs_w_given_u": z_zw,
            "proxy_vs_action_given_u": z_action,
        },
        k_ranks=k_ranks,
        singular_values=svs,
        condition_numbers=conds,
        binning_degenerate=degenerate,
        covariate_free=covariate_free,
        exclusions_hold=exclusions,
        kruskal_ok=kruskal_ok,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------
# D-E: the instrument
# --------------------------------------------------------------------------


@dataclass
class InstrumentReport:
    n: int
    n_episodes: int
    corr_i_u: float
    corr_i_action: float
    corr_i_reward_given_action_and_u: float
    null_p: Dict[str, float] = field(default_factory=dict)
    null_sds: Dict[str, float] = field(default_factory=dict)
    exclusion_testable: bool = False
    independent_of_u: bool = False
    relevant: bool = False
    exclusion_holds: bool = False
    reasons: tuple = ()

    def _rounded_p(self) -> dict:
        return {k: round(v, 4) for k, v in self.null_p.items()}

    def summary(self) -> str:
        return (
            f"n={self.n} episodes={self.n_episodes} "
            f"corr(I,U)={self.corr_i_u:+.4f} corr(I,A)={self.corr_i_action:+.3f} "
            f"corr(I,R|A,U)={self.corr_i_reward_given_action_and_u:+.4f} "
            f"nullP={self._rounded_p()} "
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
    exact reference (closed-form Balke-Pearl bounds) -- a leaking instrument
    would make that anchor meaningless while still looking plausible.

    **THE TWO CHECKS CONDITION ON DIFFERENT SETS, AND THE ASYMMETRY IS THE
    POINT.** They sit side by side, they look like the same kind of claim, and
    the correct treatment is opposite:

    * **Exogeneity is ``I indep U``, unconditionally -- it must NOT condition on
      ``A``.** ``A`` is a COLLIDER on ``I -> A <- U``. Conditioning on a
      collider *creates* dependence between its parents, so an exogenous
      instrument tested given ``A`` would be found associated with ``U``
      precisely *because* it is a valid instrument. The check would fire hardest
      where the generator is most correct.
    * **Exclusion is ``I indep R | (A, U)`` -- it MUST condition on ``A``, and on
      ``U`` as well.** Here ``A`` is a mediator on the legitimate path
      ``I -> A -> R``, which is the instrument doing its job; leave it open and
      every valid instrument fails. But blocking it opens the collider path
      ``I -> A <- U -> R``, so ``U`` must join the conditioning set to close what
      conditioning on ``A`` just opened. Measured: -0.048 given ``A`` alone
      against -0.003 given ``(A, U)``.

    So conditioning on *more* is not uniformly safer and conditioning on *less*
    is not uniformly safer; the graph decides, separately for each claim.

    GRANULARITY (S1b). ``I`` and ``U`` are episode-constant and enter as one row
    per episode; the action enters as its episode mean. The exclusion statistic
    residualises the reward on ``(A, U)`` at TRANSITION level first -- that is a
    pointwise map, so it is granularity-neutral -- and only then averages the
    residual within each episode. The conditioning therefore happens where the
    action actually varies, while the correlation is still one row per episode.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    reasons: list[str] = []

    uniq, inv = _episode_index(episode_ids)
    n_ep = int(uniq.size)
    i_ep = _episode_constant(i, inv, n_ep, "I")
    u_ep = _episode_constant(u, inv, n_ep, "U")
    a_ep = _episode_mean(a, inv, n_ep)

    # I is drawn ONCE PER EPISODE. Judged against a fixed correlation tolerance
    # instead, a provably exogenous instrument measured corr(I, U) = +0.086 at
    # lambda = 0.1 and was reported "not exogenous"; the sign flipped across
    # strengths (+0.086, +0.018, -0.036), which is what episode-count noise looks
    # like and what a real dependence never does. Pooled over TRANSITIONS the
    # same quantity reached -0.590 -- pure length-weighting.
    c_iu, p_iu, z_iu = _signed_permutation_test(i_ep, lambda b: _corr(b, u_ep))
    c_ia, p_ia, z_ia = _signed_permutation_test(i_ep, lambda b: _corr(b, a_ep))

    r_resid_ep = _episode_mean(_residualise(r, a, u), inv, n_ep)
    c_ir_a, p_ir, z_ir = _signed_permutation_test(
        i_ep,
        lambda b: _corr(_residualise(b, u_ep), r_resid_ep),
        strata_ep=u_ep,
    )

    # DEGENERACY GUARD. On CartPole the reward is r = 1 + c_r*U*1[a = a_bad], a
    # DETERMINISTIC function of (A, U) -- so residualising on (A, U) annihilates
    # it and the exclusion statistic is identically zero for the observed data
    # AND for every permutation. That reads as a clean pass and is actually a
    # measurement of nothing. The structural argument still holds (the wrapper
    # never reads I when perturbing the reward), but this check must not be
    # allowed to claim it verified it. The variance tested is that of the
    # EPISODE-MEAN residual, because that is the series the statistic consumes.
    exclusion_testable = float(np.var(r_resid_ep)) > 1e-12
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
    indep_u = p_iu >= _NULL_ALPHA
    relevant = p_ia < _NULL_ALPHA
    exclusion = exclusion_testable and p_ir >= _NULL_ALPHA
    if not indep_u:
        reasons.append(
            f"I is associated with U at permutation p={p_iu:.4f} (corr {c_iu:+.3f}) "
            "-- not exogenous"
        )
    if not relevant:
        reasons.append(
            f"I barely moves A ({c_ia:+.3f}, permutation p={p_ia:.4f}) -- a weak or "
            "irrelevant instrument"
        )
    if exclusion_testable and p_ir < _NULL_ALPHA:
        reasons.append(
            f"I is associated with R given (A,U) at permutation p={p_ir:.4f} "
            f"(corr {c_ir_a:+.3f}) -- the exclusion restriction fails, so the "
            "Balke-Pearl anchor would be invalid"
        )
    return InstrumentReport(
        n=int(np.asarray(i).size),
        n_episodes=n_ep,
        corr_i_u=c_iu,
        corr_i_action=c_ia,
        corr_i_reward_given_action_and_u=c_ir_a,
        null_p={"i_vs_u": p_iu, "i_vs_a": p_ia, "i_vs_r_given_a_u": p_ir},
        null_sds={"i_vs_u": z_iu, "i_vs_a": z_ia, "i_vs_r_given_a_u": z_ir},
        exclusion_testable=exclusion_testable,
        independent_of_u=indep_u,
        relevant=relevant,
        exclusion_holds=exclusion,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------
# D-B': the drifting latent
# --------------------------------------------------------------------------


@dataclass
class DriftReport:
    declared_rho: float
    predicted_autocorr: float
    realised_autocorr: float
    n_pairs: int
    matches: bool
    autocorr_short_episodes: float = 0.0
    autocorr_long_episodes: float = 0.0
    length_weighting_gap: float = 0.0
    length_weighting_inert: bool = True

    def summary(self) -> str:
        return (
            f"rho={self.declared_rho:.3f} "
            f"predicted autocorr={self.predicted_autocorr:+.3f} "
            f"realised={self.realised_autocorr:+.3f} n={self.n_pairs} "
            f"matches={self.matches} "
            f"short={self.autocorr_short_episodes:+.3f} "
            f"long={self.autocorr_long_episodes:+.3f} "
            f"length_gap={self.length_weighting_gap:.3f} "
            f"length_weighting_inert={self.length_weighting_inert}"
        )


def check_drift(
    *, u_by_episode: Sequence[Sequence[float]], rho: float, tol: float = 0.05
) -> DriftReport:
    """D-B': does the realised within-episode autocorrelation of U match rho?

    For a symmetric binary chain with per-step flip probability ``rho``,
    ``corr(U_t, U_{t+1}) = 1 - 2*rho``. rho = 0 must give autocorrelation 1
    (episode-static, i.e. D-B).

    **THIS IS THE ONE STATISTIC THAT STAYS AT TRANSITION LEVEL, AND THE
    EXEMPTION IS NARROWER THAN IT LOOKS.** The reason is *not* simply "U is
    per-step here". Pooling within-episode lag-1 pairs across episodes DOES
    weight longer episodes more, exactly as S1b describes. What makes it sound
    is that the autocorrelation is HOMOGENEOUS ACROSS EPISODES BY CONSTRUCTION
    -- a single declared flip probability rho, identical in every episode -- so
    the length-weighting merely reweights an already-unbiased quantity.

    That safety would evaporate silently under a state- or policy-dependent
    drift variant: rho would vary across episodes, longer episodes would carry
    more weight, and the pooled autocorrelation would drift toward whatever rho
    prevails in long episodes with nothing raising an error. So the exemption is
    MEASURED here rather than asserted: episodes are split at the median length
    and the pooled autocorrelation is recomputed within each half. Homogeneous
    rho implies the halves agree; a gap is the signature of exactly the variant
    that voids the exemption.

    The split is pooled-within-half rather than a per-episode autocorrelation
    averaged over episodes, because at rho = 0 every episode is constant, its
    own autocorrelation is undefined, and an average of undefined quantities
    would read as a clean zero.

    ``length_weighting_inert`` is REPORTED, not folded into ``matches``: the
    declared-rho comparison is the assertion this check exists to make, and the
    length diagnostic is evidence about whether the S1b exemption still applies.
    Conflating them would hide which of the two failed.
    """
    eps = [np.asarray(ep, dtype=np.float64).reshape(-1) for ep in u_by_episode]
    eps = [e for e in eps if e.size >= 2]

    def _pooled(chunk) -> tuple[float, int]:
        lhs: list[float] = []
        rhs: list[float] = []
        for arr in chunk:
            lhs.extend(arr[:-1])
            rhs.extend(arr[1:])
        if not lhs:
            return 1.0, 0
        return _corr(np.asarray(lhs), np.asarray(rhs)), len(lhs)

    realised, n_pairs = _pooled(eps)
    predicted = 1.0 - 2.0 * float(rho)

    lengths = np.array([e.size for e in eps]) if eps else np.zeros(0)
    if lengths.size >= 2:
        cut = float(np.median(lengths))
        short = [e for e in eps if e.size <= cut]
        long_ = [e for e in eps if e.size > cut]
        # A degenerate split (every episode the same length -- the synthetic
        # harnesses do this) is not evidence either way; report it inert.
        if short and long_:
            ac_short, _ = _pooled(short)
            ac_long, _ = _pooled(long_)
        else:
            ac_short = ac_long = realised
    else:
        ac_short = ac_long = realised
    gap = abs(ac_short - ac_long)

    return DriftReport(
        declared_rho=float(rho),
        predicted_autocorr=predicted,
        realised_autocorr=realised,
        n_pairs=n_pairs,
        matches=abs(realised - predicted) < tol,
        autocorr_short_episodes=ac_short,
        autocorr_long_episodes=ac_long,
        length_weighting_gap=gap,
        length_weighting_inert=bool(gap < tol),
    )


# --------------------------------------------------------------------------
# D-A-null: the reference null arm
# --------------------------------------------------------------------------


@dataclass
class NullArmReport:
    """D-A-null: is there genuinely NOTHING for L5 to find?"""

    n: int
    n_episodes: int
    null_p: Dict[str, float] = field(default_factory=dict)
    null_sds: Dict[str, float] = field(default_factory=dict)
    gated_episodes: int = 0
    reward_testable: bool = False
    gated_testable: bool = False
    u_inert: bool = False
    reasons: tuple = ()

    def _rounded(self) -> dict:
        return {k: round(v, 4) for k, v in self.null_p.items()}

    def summary(self) -> str:
        return (
            f"n={self.n} episodes={self.n_episodes} "
            f"nullP={self._rounded()} gated_episodes={self.gated_episodes} "
            f"reward_testable={self.reward_testable} "
            f"gated_testable={self.gated_testable} "
            f"inert={self.u_inert}"
        )


def check_null_arm(
    *,
    u: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    episode_ids: np.ndarray,
    a_bad: float = 1.0,
) -> NullArmReport:
    """Certify that the logged U is INERT -- it touches neither A nor R.

    This is the arm L5's FALSE-POSITIVE RATE is read from, so its validity is
    the thing that makes a refutation there interpretable as a false alarm. If
    U were not actually inert, every "false positive" measured here would be
    partly a true detection and the rate would be silently understated -- in the
    flattering direction, which is the one that needs guarding.

    It matters that this is a distinct check rather than the confounded gate
    with the dial at zero. The gate asks "is the declared confounding present at
    the declared strength"; this asks "is there any association at all", which
    is a different question and the only one that licenses the null arm's use.

    Episode granularity throughout (S1b): U is drawn once per episode, so it is
    one row per episode, and the action and reward enter as episode means. The
    gated contrast restricts to the episodes that actually contain an ``a_bad``
    step and averages the reward over those steps only -- an episode with no
    such step carries no information about that channel and must be dropped,
    not handed a row full of zeros.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    reasons: list[str] = []

    uniq, inv = _episode_index(episode_ids)
    n_ep = int(uniq.size)
    u_ep = _episode_constant(u, inv, n_ep, "U")
    a_ep = _episode_mean(a, inv, n_ep)
    r_ep = _episode_mean(r, inv, n_ep)

    _, p_ua, z_ua = _signed_permutation_test(u_ep, lambda b: _corr(b, a_ep))
    _, p_ur, z_ur = _signed_permutation_test(u_ep, lambda b: _corr(b, r_ep))

    # DEGENERACY GUARD on the reward channel -- the same S8 trap check_instrument
    # already guards, in the arm where it matters most. On CartPole with c_r = 0
    # the reward is 1.0 at EVERY step, so var(R) is exactly zero: the statistic
    # is identically zero for the observed data and for all 200 permutations,
    # the p-value comes back at 1.000, and "U is inert in the reward channel"
    # reads as the cleanest pass in the table while resting on no evidence at
    # all. Measured across the 10 D-A-null datasets: var(episode-mean R) = 0.0e0
    # on all five CartPole seeds and ~1e-8 on two of the five Acrobot seeds.
    #
    # Inertness still HOLDS there -- a constant reward cannot depend on anything
    # -- and it is credited, exactly as an untestable exclusion is. What must not
    # happen is crediting it as a measured result: this arm is where L5's
    # FALSE-POSITIVE RATE is read from, so an unverified inertness claim
    # understates that rate in the flattering direction.
    reward_testable = float(np.var(r_ep)) > 1e-12

    # Also the gated contrast the confounded arms rely on: within a == a_bad,
    # is the reward associated with U? That is the exact channel c_r opens, so
    # it is the one that must be dead here.
    gated = a == float(a_bad)
    gated_counts = np.bincount(inv[gated], minlength=n_ep)
    has_gated = gated_counts > 0
    n_gated_ep = int(has_gated.sum())
    gated_testable = False
    if n_gated_ep > 2:
        gated_sum = np.bincount(inv[gated], weights=r[gated], minlength=n_ep)
        r_gated_ep = gated_sum[has_gated] / gated_counts[has_gated]
        u_gated_ep = u_ep[has_gated]
        gated_testable = float(np.var(r_gated_ep)) > 1e-12
        _, p_gated, z_gated = _signed_permutation_test(
            u_gated_ep, lambda b, c=r_gated_ep: _corr(b, c)
        )
    else:
        p_gated, z_gated = 1.0, 0.0

    # An untestable channel is CREDITED but recorded as such, never conflated
    # with a verified one (S8) -- the same treatment D-E's exclusion gets.
    inert = (
        p_ua >= _NULL_ALPHA
        and (p_ur >= _NULL_ALPHA or not reward_testable)
        and (p_gated >= _NULL_ALPHA or not gated_testable)
    )
    if p_ua < _NULL_ALPHA:
        reasons.append(
            f"U is associated with A at permutation p={p_ua:.4f} -- not a null arm"
        )
    if reward_testable and p_ur < _NULL_ALPHA:
        reasons.append(
            f"U is associated with R at permutation p={p_ur:.4f} -- not a null arm"
        )
    if gated_testable and p_gated < _NULL_ALPHA:
        reasons.append(
            f"U is associated with R within a = a_bad at permutation "
            f"p={p_gated:.4f} -- the gated reward channel is live, so c_r did not "
            "reach zero"
        )
    if not reward_testable:
        reasons.append(
            "U-vs-R NOT TESTABLE on this env: the reward has no variance across "
            "episodes (CartPole with c_r = 0 pays exactly 1.0 every step), so the "
            "statistic is identically zero for the data AND for every "
            "permutation. Inertness holds by construction -- a constant reward "
            "cannot depend on U -- but NO evidence for it was gathered here, and "
            "L5's false-positive rate rests on this arm. It needs an env whose "
            "reward varies under c_r = 0"
        )
    if n_gated_ep > 2 and not gated_testable:
        reasons.append(
            "the gated U-vs-R contrast is NOT TESTABLE on this env for the same "
            "reason: no reward variance within a = a_bad"
        )
    return NullArmReport(
        n=int(np.asarray(u).size),
        n_episodes=n_ep,
        null_p={"u_vs_a": p_ua, "u_vs_r": p_ur, "u_vs_r_gated": p_gated},
        null_sds={"u_vs_a": z_ua, "u_vs_r": z_ur, "u_vs_r_gated": z_gated},
        gated_episodes=n_gated_ep,
        reward_testable=reward_testable,
        gated_testable=gated_testable,
        u_inert=inert,
        reasons=tuple(reasons),
    )

"""L5 — falsification of the declared observability (the Markov test).

The contract row this serves: a user who declares MDP on a POMDP environment
must be TOLD, not silently mis-served. The declared-MDP diagram's testable
shadow is Markovianity of the observed process:

    (O_{t+1}, R_t)  independent of  history_{<t}  given  (O_t, A_t)

reported at the SERVED lag k (declared-MDP is k = 0) as a C3 record on the
served value. It is a diagnostic, not a decision: the POMDP branch's window
is chosen by MATERIALITY-BY-REFIT against L4's own interval
(``pomdp_branch``), not by this statistic — ruled 2026-09-03, below.

**Statistic.** Episode-blocked K-fold cross-validated one-step predictive
improvement from adding the lag-(k+1) history block to the lag-k model, per
target dimension, ON THE DELTA-R-SQUARED SCALE — ``(sse0 - sse1) / SST`` —
reduced to the FAMILY maximum over testable next-observation dimensions (L5
tests many things at once, so the null is built for the max, never read
against a per-dimension cutoff; S3). The variance denominator is the point:
normalising by the base RESIDUAL amplifies floor-level residuals into large
"improvements" (failure (e) below), while variance-normalised improvement
scores a halved 1e-8-relative residual as ~1e-8 — zero information reads as
zero. No absolute threshold appears anywhere: the placebo distribution
supplies the scale.

**Null: the history block itself, SHIFT-SCRAMBLED within episodes.** Each
null draw runs the IDENTICAL cross-validated procedure with the history
block's rows circularly shifted by a random nonzero per-episode offset — the
SAME features (columns, marginals, smoothness, episode identity, overfitting
character) with only the lag ALIGNMENT destroyed: the null of the statistic
actually computed (S3). p = (1 + #{draw >= observed}) / (b + 1), a quantile
read, never a z-score. Episode structure governs the folds and the
scrambling (S1). Observed and placebo runs share one code path (the
two-block Frisch-Waugh CV below), so "identical procedure" is structural,
not disciplinary.

**On the error direction (ruled): conservatism is the DANGEROUS direction
here, not the safe one.** A non-rejection LICENSES the reward-transform
reduction — it is the verdict that tells GRACE "proceed as declared" — so the
harmful error is the false negative, and a null inflated to buy FPR (the
withdrawn max-over-placebos construction, which made the reference the
distribution of a maximum over competitors and the p-value uninterpretable)
optimises the wrong tail. The design goal is a CALIBRATED simple null over a
statistic that measures information rather than capacity; power is reported
because non-rejection is the consequential verdict (approval condition 1).

**The family is the NEXT-OBSERVATION dims only; the reward channel is a
separate diagnostic, deliberately.** On a confounded arm (sigma > 0) the
reward is GENUINELY history-dependent through the episode-constant U (the
lagged action carries U-information) — that is D-D's DECLARED structure, not
an observability violation, and an R-inclusive family would falsify the MDP
declaration on exactly the cells the MDP branch exists to serve. The R
channel's improvement and its placebo quantile are still computed and
reported on the verdict (``reward_channel``) — history-dependence of the
reward given (O, A) is evidence about CONFOUNDING, useful, and not this
test's question. Note the shift placebo also cancels the U channel by itself
(a shifted lagged action carries the SAME episode-constant U), which is
defence in depth, not a licence to re-widen the family. **SCOPE, stated for
the paper:** the declared-MDP falsifier therefore tests the OBSERVATION
channel only — partial observability that manifests solely in the reward is
not detected by this test (it surfaces in ``reward_channel`` as a diagnostic,
entangled with declared confounding, and is not adjudicated).

**The selector's features EQUAL the served state's features (ruled
2026-09-03, standing rule S19).** The family's history blocks carry lagged
(O, A) ONLY — column for column what ``pomdp_branch._augmented_cols`` hands
the estimator — so the k the selector certifies is certified on exactly the
feature set the estimator receives. Lagged R is NOT in those blocks: at
deployment U -> R is intact, so an R-carrying state is U-informative and
the served estimand's warrant (the exogenous-marginal reason, catalogue
Q2 Step 1: under do(pi) the trajectory is independent of U because no
U -> S' edge exists and pi does not read U) fails on it — R-in-state is a
belief-state critic, a different served object (recorded as the D-F/D-G
shape in the handoff). The reward-channel DIAGNOSTIC keeps an R-inclusive
history block (a hidden state visible only through past rewards is
otherwise invisible; found by the reward-relevant-hidden-state unit test):
**the selector certifies observation-channel sufficiency for the exact
features served; reward-channel dependence is reported, not selected on.**
The blind spot this fixes in place: a reward-only-visible hidden state is
a reward-channel phenomenon by construction and lands in
``reward_channel`` / ``serving_material``, not in k.

**Why not simpler statistics/nulls — five measured failures, the design record.**
(a) Linear basis + parametric (iid-Gaussian target redraw) null: EVERY true-MDP
CartPole chunk rejected (frac<=alpha = 1.00, KS = 0.99;
``results/l5_calibration``, first pass). (b) RFF-64 basis + the same
parametric null: still p = 0.005 at improvement 0.50 on true-MDP data.
Mechanism, both times: these environments' dynamics are DETERMINISTIC, so the
lag-k residual is pure approximation error, and ANY state-correlated feature
(history included) buys improvement by enlarging the function class — which a
redrawn-target null can never reproduce, at any basis size. (c) A fresh-RFF
placebo of matched column count fixed the deterministic case (history 0.499
BELOW the placebo cloud 0.60-0.87; masked view: history 1.0000, every placebo
negative) and then failed the opposite regime: on a WELL-SPECIFIED noisy
linear system, random-projection placebo columns overfit like noise (negative
improvement) while history columns are smooth and merely useless (~0), so
history "won" spuriously — anti-conservative exactly where (a)/(b) were not.
(d) The shift-scrambled placebo alone fixed (c) and then under-covered the
deterministic regime: real full-obs CartPole read p = 0.010 at ratio
improvement 0.50 — a shifted row is a worse basis for mopping up LOCAL
approximation error than the truly adjacent state. A max-over-placebos
reference was tried and WITHDRAWN on review: it makes the reference the
distribution of a maximum over competitors, so the p-value stops being a
p-value, and its conservatism sits in the dangerous tail (see the
error-direction ruling above). (e) The root cause of (a)-(d), found by
measuring the BASE model rather than re-engineering the null: the base was
never underfit — held-out base R^2 on full-obs CartPole is 0.99999944-1.0 at
EVERY capacity and bandwidth probed — and the 0.50 "improvement" was history
halving a residual of 1.5e-8 in variance units (Delta-R^2 = 7.7e-9). The
RESIDUAL-normalised statistic amplified a numerical floor into a headline
effect; on the variance scale the same quantity is zero and the masked view's
genuine signal (Delta-R^2 ~ 1e-2, ~1e6 x the H0 floor) stands clear. The
statistic was the problem, not the null.

**THE DECLARATION IS AN INPUT, NEVER A HYPOTHESIS GRACE MAY OVERRULE
(ruled 2026-09-03).** When the falsifier's record contradicts the
declaration, GRACE KEEPS SERVING under the declared branch and reports the contradiction — it never
silently switches branches and never refuses to run. Three binding reasons:
(a) the declaration must remain the interface — auto-switching makes the run
irreproducible from the config; (b) the user may have reasons (deployment,
interpretability, compute) — "your assumption looks wrong, here is the
evidence, here is what I served anyway" respects that; (c) auto-repair would
erase contract row 3 — the degradation under (declared MDP, true POMDP) must
be OBSERVABLE alongside the warning. Two mechanisms, kept strictly separate:

    L4 abstention        the FIT is unhealthy          do not serve; base fallback, labelled
    L5 falsification     the DECLARATION is contradicted   SERVE AS DECLARED; warn; C3 condition

``MarkovVerdict.record`` travels as a reported condition on the served value
(C3), never as an abstention trigger; ``serving_material`` GRADES the warning
("contradicted and the correction moves" is louder than "contradicted,
correction unaffected"), never gates serving.

**FALSIFICATION IS REPORT-ONLY — NO THRESHOLD (ruled 2026-09-03, superseding
the 2026-09-02 two-verdict design).** Since the verdict changes no behaviour,
it needs no binary, and the Delta-R^2 cut it once carried (a "stated
convention in the calibration-measured gap") was a PER-ENVIRONMENT CONSTANT
— exactly what A2 forbids. What travels instead is the record: the effect
size Delta-R^2 and its p (the statistical tier's read at the stated alpha —
on deterministic systems a point null of exact Markovianity rejects at floor
effect sizes, S18, reported once as a FINDING), the CAPACITY-SHRINK ratio
(the mechanistic, dimensionless separator: approximation error shrinks with
base capacity, measured ~56x on a true MDP; information does not, ~1x), base
R^2 per dimension with ``scale_invalid`` where negative, and the
reward-channel diagnostic. The user reads the magnitude and its evidence.

**SELECTION IS MATERIALITY-BY-REFIT, not this statistic.** The predictive
test asks "is the process exactly Markov?", whose answer is always no (S18).
The estimator's question is whether another lag changes WHAT GRACE SERVES by
more than the uncertainty GRACE already reports:
``k* = min { k : |contrast(k+1) - contrast(k)| <= w_k }``, contrast = the
served action contrast on the lag-k augmented state, w_k = L4's half-width
there — every term measured per fit, no constant, no environment dependence
(the same family as the walk's derived stop, the bootstrap MC-error
criterion and ``tau_R``). Lives in ``pomdp_branch``. ``serving_material``
(below) stays as the DERIVED grading of the reward-channel diagnostic.

**What calibration is still for — and it is not per-environment.** (i) The
POWER of the materiality criterion: how large a violation must be before it
is caught — a property of the method, measured on SYNTHETIC fixtures where
the truth is dialable, never on real environments. (ii) The S18 result,
reported once from the rows already collected (`results/l5_calibration`).
Nothing measured on one environment transfers to another, because nothing is
measured to be transferred: the parameter taxonomy is declarations /
budgets / derived / calibration constants = NONE (handoff, 2026-09-03).

Procedural constants (disclosed, budget-class, not tuned thresholds): K=5
episode folds, 99 placebo draws, 64 RFF features per block. Alpha is STATED
by the caller.

Numpy-only on purpose: closed-form fits, deterministic given the seed, cheap
enough to run per dataset at calibration scale (the statistic consumes
datasets, not RL runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Procedural budgets (disclosed above). Not thresholds: changing them trades
# compute for resolution, and the p-value's meaning is unchanged.
_K_FOLDS = 5
_B_DRAWS = 99  # placebo draws; min attainable p = 1/(B+1) = 0.01

# Function-class size per block: linear terms + this many random Fourier
# features, the SAME basis construction for the conditioning blocks, the
# history block, and every placebo block — capacity is matched by
# construction, which is what makes the placebo a placebo.
_N_RFF = 64
_RFF_SCALE = 1.0  # bandwidth on STANDARDISED inputs (unit variance), not raw

# Ridge-free by design (A2): rank deficiency is handled by SVD truncation at
# numpy's lstsq-default rcond — eps * max(n, p), DERIVED from the floating
# representation (the same family as the min_scale ceiling), never tuned.
# The solve factors X ITSELF, not the gram X'X: forming the gram SQUARES the
# condition number, and that was a measured correctness bug — at RFF-256 on
# the 2-dim masked view the gram-pinv fit exploded on held-out data
# (residual fraction 14-19x SST, Delta-R^2 = -58.7) while the SVD path is
# stable. Any calibration computed through the gram path is void.


def _lstsq_map(x: np.ndarray) -> np.ndarray:
    """The truncated-SVD least-squares solve as a reusable map: ``M [p, n]``
    with ``M @ b`` the solution for any RHS ``b``.

    Truncation at ``s_i < s_0 * sqrt(eps)`` — DERIVED, not tuned: the LS
    solution's sensitivity along direction i scales as ``(s0/si)^2`` times the
    input's own float noise, so directions with ``(s0/si)^2 > 1/eps`` admit
    O(1) noise into the coefficients and are numerically meaningless. Measured
    both failure modes this guards: numpy's eps-level rcond KEPT such
    directions (held-out base R^2 = -1052 on the synthetic random-walk
    fixture), and the gram-pinv path implemented roughly this cut by accident
    on its SQUARED spectrum while computing that spectrum inaccurately
    (the RFF-256 masked explosion, residual fraction 14-19x SST)."""
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    cut = np.sqrt(np.finfo(x.dtype).eps) * (s[0] if s.size else 0.0)
    s_inv = np.where(s > cut, 1.0 / np.where(s > 0, s, 1.0), 0.0)
    return (vt.T * s_inv) @ u.T


@dataclass
class MarkovVerdict:
    """The test's full record — travels with any decision made on it (C3)."""

    lag: int  # H0: lag-`lag` features suffice
    p_value: float
    statistic: float  # family max of per-dim CV improvements
    per_dim: np.ndarray  # improvement per testable target dim
    dim_names: List[str]
    untestable: List[str]  # zero-variance targets: reported, never "passed" (S8)
    n_episodes: int
    n_rows: int
    b_draws: int
    k_folds: int
    seed: int
    # The reward channel's improvement and placebo quantile — reported, never
    # in the family: its history-dependence is a CONFOUNDING signal (see the
    # module docstring), not an observability violation. None when R is
    # untestable (zero variance).
    reward_channel: Optional[dict] = None
    # Held-out R^2 of the lag-k BASE model per target dim (NaN if untestable).
    # First-class diagnostic: a verdict from a base that cannot predict its
    # own environment's dynamics is reporting on its own misspecification,
    # and the reader must be able to see that on the verdict itself.
    base_r2: Optional[np.ndarray] = None
    # Dims where base_r2 < 0: the memoryless model is worse than the mean
    # predictor on held-out episodes, so that dim's Delta-R^2 is NOT a
    # variance fraction (it can exceed 1) and must not be read as one. The
    # DETECTION stays valid — a base that cannot beat the mean IS the signal —
    # but the number's interpretation changes, and the verdict says so rather
    # than letting the number carry it (S8).
    scale_invalid: List[str] = field(default_factory=list)
    # The capacity-shrink diagnostic (reported, never a gate): the observed
    # family statistic recomputed with a 4x-capacity base. Approximation error
    # is capacity-dependent by definition; information is not — measured: the
    # true-MDP effect shrank 56x from RFF 64->256 while the masked effect was
    # stable at ~1x. "Small and capacity-shrinking" corroborates an immaterial
    # violation; "small but capacity-stable" is the case this field exists to
    # catch. Keys: n_rff_hi, stat_hi, shrink (stat/stat_hi; None if stat_hi
    # is not positive).
    capacity: Optional[dict] = None

    def rejected(self, alpha: float) -> bool:
        return self.p_value <= alpha

    def record(self, alpha: float) -> dict:
        """The verdict as a C3 RECORD on the served value — report-only, no
        threshold, nothing branches on it (ruled 2026-09-03: a verdict that
        changes no behaviour needs no binary, and the Delta-R^2 cut it once
        carried was a per-environment constant, which A2 forbids). Flat
        scalars/strings so the artifact writers keep every field:

        * the effect size and its p (``l5_dr2``, ``l5_p``, ``l5_rejected`` —
          the statistical tier's read at the stated ``alpha``, reported as a
          fact about a point null, see S18);
        * the capacity-shrink ratio (``l5_shrink``): the mechanistic,
          dimensionless separator — approximation error shrinks with base
          capacity (measured ~56x on a true MDP), information does not (~1x);
        * base R^2 per dimension and ``l5_scale_invalid`` where negative;
        * the reward-channel diagnostic.

        The reader takes the magnitude with its evidence; no constant is
        required because no decision is made here.
        """
        cap = self.capacity or {}
        rc = self.reward_channel
        out = dict(
            l5_lag=int(self.lag),
            l5_alpha=float(alpha),
            l5_p=float(self.p_value),
            l5_dr2=float(self.statistic),
            l5_rejected=bool(self.rejected(alpha)),
            l5_n_episodes=int(self.n_episodes),
            l5_n_rows=int(self.n_rows),
            l5_b_draws=int(self.b_draws),
            l5_shrink=(
                float(cap["shrink"]) if cap.get("shrink") is not None else float("nan")
            ),
            l5_stat_hi=float(cap.get("stat_hi", float("nan"))),
            l5_untestable=",".join(self.untestable),
            l5_scale_invalid=",".join(self.scale_invalid),
            l5_base_r2=(
                " ".join(f"{n}={r:.4f}" for n, r in zip(self.dim_names, self.base_r2))
                if self.base_r2 is not None
                else ""
            ),
            l5_reward_improvement=(
                float(rc["improvement"]) if rc is not None else float("nan")
            ),
            l5_reward_placebo_quantile=(
                float(rc["placebo_quantile"]) if rc is not None else float("nan")
            ),
            l5_reward_draw_q95=(
                float(rc["draw_q95"]) if rc is not None else float("nan")
            ),
            l5_label=self.label(alpha),
        )
        return out

    def label(self, alpha: float) -> str:
        state = "FALSIFIED" if self.rejected(alpha) else "not-rejected"
        extra = f" untestable={','.join(self.untestable)}" if self.untestable else ""
        br = ""
        if self.base_r2 is not None:
            finite = self.base_r2[np.isfinite(self.base_r2)]
            if finite.size:
                br = f" base_r2_min={float(finite.min()):.4f}"
        if self.scale_invalid:
            br += f" NOT-A-VARIANCE-FRACTION[{','.join(self.scale_invalid)}]"
        return (
            f"l5=markov[lag={self.lag}] {state} p={self.p_value:.4f} "
            f"dR2={self.statistic:.3e} alpha={alpha:g} n_ep={self.n_episodes}{br}{extra}"
        )


@dataclass
class Episode:
    """One episode's raw arrays: obs ``[T+1, D]``, actions ``[T]``, rewards ``[T]``."""

    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray

    def __post_init__(self) -> None:
        self.obs = np.asarray(self.obs, dtype=np.float64)
        self.act = np.asarray(self.act).reshape(-1)
        self.rew = np.asarray(self.rew, dtype=np.float64).reshape(-1)
        if self.obs.ndim != 2 or self.obs.shape[0] != self.act.shape[0] + 1:
            raise ValueError(
                f"episode shapes inconsistent: obs {self.obs.shape}, act {self.act.shape}"
            )
        if self.rew.shape[0] != self.act.shape[0]:
            raise ValueError("rewards and actions must have equal length")


def _one_hot(a: np.ndarray, n_actions: int) -> np.ndarray:
    out = np.zeros((a.shape[0], n_actions), dtype=np.float64)
    out[np.arange(a.shape[0]), a.astype(int)] = 1.0
    return out


@dataclass
class _Design:
    """Row-aligned design for the paired lag-k vs lag-(k+1) comparison.

    Rows are transitions t with k+1 lags available (t = k+1 .. T-1) — the SAME
    row set for the base model, the history block and every placebo block, so
    all comparisons are paired by construction.
    """

    x0: np.ndarray  # [n, p0] intercept + expanded lag-0..k blocks
    hist: np.ndarray  # [n, p_e] expanded lag-(k+1) block (the tested features)
    y: np.ndarray  # [n, d_y] targets: next-obs dims, then reward
    episode_of: np.ndarray  # [n] episode index per row (fold + scramble unit)
    dim_names: List[str] = field(default_factory=list)


def _build_design(
    episodes: Sequence[Episode],
    lag: int,
    *,
    seed: int = 0,
    n_rff: int = _N_RFF,
    history_reward: bool = False,
) -> Optional[_Design]:
    """``history_reward=False`` (the FAMILY): history blocks carry lagged
    (O, A) only — the served augmentation's columns (S19). ``True`` (the
    reward-channel diagnostic): history blocks also carry lagged R. The j=0
    conditioning block is (O_t, A_t) either way, so at lag 0 both designs
    share ``x0`` exactly (the caller reuses the base fit)."""
    n_actions = int(max(int(e.act.max()) for e in episodes if e.act.size)) + 1
    raw0, raw1, ys, eps = [], [], [], []
    for ei, e in enumerate(episodes):
        t_len = e.act.shape[0]
        lo = lag + 1  # first t with lag+1 history
        if t_len - 1 < lo:  # need t+1 <= T for the target
            continue
        ts = np.arange(lo, t_len)  # predict O_{t+1}, R_t for these t
        a_oh = _one_hot(e.act, n_actions)

        def lag_block(j):
            # j = 0 conditions on (O_t, A_t); j >= 1 history blocks carry
            # lagged (O, A) — the served augmentation's columns (S19) — and,
            # for the reward-channel diagnostic ONLY, the lagged REWARD (a
            # hidden state visible only through past rewards is otherwise
            # invisible; found by the reward-relevant-hidden-state unit test
            # reading dr2_fast = 0). The shift placebo nets the
            # episode-constant U that lagged rewards carry, by the same
            # argument as lagged actions.
            cols = [e.obs[ts - j], a_oh[ts - j]]
            if history_reward and j >= 1:
                cols.append(e.rew[ts - j, None])
            return np.concatenate(cols, axis=1)

        raw0.append([lag_block(j) for j in range(0, lag + 1)])
        raw1.append(lag_block(lag + 1))
        ys.append(np.concatenate([e.obs[ts + 1], e.rew[ts, None]], axis=1))
        eps.append(np.full(ts.shape[0], ei))
    if not raw0:
        return None

    # Stack per-lag blocks across episodes, standardise each (RFF bandwidth is
    # defined on unit-variance inputs), then expand: [linear, cos(zW+b)] per
    # block, one (W, b) shared by every lag — identical featurisation.
    n_lag_blocks = lag + 1
    blocks0 = [
        np.concatenate([raw0[i][j] for i in range(len(raw0))])
        for j in range(n_lag_blocks)
    ]
    block1 = np.concatenate(raw1)
    rng = np.random.default_rng(seed)
    # Two block widths exist: the j=0 conditioning block (no reward) and the
    # j>=1 history blocks (+R). One (W, b) pair per WIDTH, shared by every
    # block of that width — so all history blocks and the shifted placebo get
    # the identical featurisation, which is what makes the placebo a placebo.
    dim_c = blocks0[0].shape[1]
    dim_h = block1.shape[1]
    w_c = rng.standard_normal((dim_c, n_rff)) * _RFF_SCALE
    b_c = rng.uniform(0, 2 * np.pi, n_rff)
    # ONE BASIS, SHARED BY CONSTRUCTION. The history weights are always drawn
    # at the R-inclusive width and truncated to the block's width, so the
    # family block (O, A) and the reward-channel block (O, A, R) use the same
    # first dim_c weight rows. That is what makes the two expansions one
    # coherent basis rather than two unrelated draws — and the reason they
    # COINCIDE wherever R is constant (its standardised column is identically
    # zero, so the extra row is multiplied by nothing). The check on that
    # reason: constant-reward data reproduces the pre-S19 (O, A, R) family
    # statistic to 1e-15 at lags 0 and 1, so calibration rows scored before
    # S19 on constant-reward datasets are samples of the SAME procedure. An
    # independent draw here would break both — measured before this was
    # fixed: 26% statistic difference at identical p, draw noise not
    # information. Do not replace with an independent draw.
    w_h = (rng.standard_normal((dim_c + 1, n_rff)) * _RFF_SCALE)[:dim_h]
    b_h = rng.uniform(0, 2 * np.pi, n_rff)

    def standardise(blk: np.ndarray) -> np.ndarray:
        mu, sd = blk.mean(axis=0), blk.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        return (blk - mu) / sd

    z0 = [standardise(b_) for b_ in blocks0]
    z1 = standardise(block1)

    def expand(z: np.ndarray, hist: bool) -> np.ndarray:
        # One (W, b) pair per block ROLE: the j=0 conditioning block uses
        # (w_c, b_c); every history block (in x0 for j >= 1, the tested block,
        # every placebo block) shares (w_h, b_h) — identical featurisation is
        # what makes the placebo a placebo. Role, not width: with (O, A)-only
        # history the two widths coincide.
        w, b_ph = (w_h, b_h) if hist else (w_c, b_c)
        return np.concatenate([z, np.cos(z @ w + b_ph)], axis=1)

    n_rows = block1.shape[0]
    x0 = np.concatenate(
        [np.ones((n_rows, 1))] + [expand(z, j > 0) for j, z in enumerate(z0)],
        axis=1,
    )

    d_obs = episodes[0].obs.shape[1]
    return _Design(
        x0=x0,
        hist=expand(z1, True),
        y=np.concatenate(ys),
        episode_of=np.concatenate(eps),
        dim_names=[f"O[{j}]" for j in range(d_obs)] + ["R"],
    )


def _fold_of_episode(episode_ids: np.ndarray, k_folds: int, rng) -> np.ndarray:
    """Fold per EPISODE, shuffled — the unit of observation is the episode
    (S1); a fold split at row level would leak within-episode structure across
    the train/test boundary and the test would be judging memorisation."""
    uniq = np.unique(episode_ids)
    perm = rng.permutation(uniq.shape[0])
    fold_of_uniq = np.empty(uniq.shape[0], dtype=int)
    fold_of_uniq[perm] = np.arange(uniq.shape[0]) % k_folds
    lookup = {int(u): int(f) for u, f in zip(uniq, fold_of_uniq)}
    return np.array([lookup[int(e)] for e in episode_ids])


def _cv_base(x0: np.ndarray, y: np.ndarray, folds: np.ndarray, k_folds: int):
    """The base (lag-k) model's episode-blocked CV, with per-fold caches.

    Returns ``(sse0 [d_y], caches)``; each cache carries what `_cv_extra`
    needs to score ANY extra block through the identical two-block
    (Frisch-Waugh) decomposition: the base projector, the base predictions,
    and the base training residuals. Computing these once is what makes ~100
    placebo draws affordable — each draw costs one small residualised
    regression, not a full refit.
    """
    sse = np.zeros(y.shape[1])
    caches = []
    for f in range(k_folds):
        te = folds == f
        tr = ~te
        xtr, xte = x0[tr], x0[te]
        a_map = _lstsq_map(xtr)  # [p0, n_tr]; SVD of X, never the gram
        beta_y = a_map @ y[tr]
        yhat_te = xte @ beta_y
        sse += ((y[te] - yhat_te) ** 2).sum(axis=0)
        caches.append(
            dict(
                tr=tr,
                te=te,
                xtr=xtr,
                xte=xte,
                a_map=a_map,
                resid_tr=y[tr] - xtr @ beta_y,
                yhat_te=yhat_te,
            )
        )
    return sse, caches


def _cv_extra(caches, extra: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Held-out SSE of base + ``extra`` block, via the cached decomposition.

    Identical code path for the history block and every placebo block — the
    "same procedure" property of the null is enforced structurally here.
    """
    sse = np.zeros(y.shape[1])
    for c in caches:
        etr, ete = extra[c["tr"]], extra[c["te"]]
        b_map = c["a_map"] @ etr  # base-projection of the extra block
        er_tr = etr - c["xtr"] @ b_map
        er_te = ete - c["xte"] @ b_map
        gamma = _lstsq_map(er_tr) @ c["resid_tr"]
        resid = y[c["te"]] - (c["yhat_te"] + er_te @ gamma)
        sse += (resid**2).sum(axis=0)
    return sse


def _family_stat(
    sse0: np.ndarray, sse1: np.ndarray, sst: np.ndarray, testable: np.ndarray
):
    """Per-dim VARIANCE-normalised improvement (Delta-R^2) and its family max.

    ``(sse0 - sse1) / sst``, never ``/ sse0``: the residual-normalised ratio
    amplified floor-level residuals into headline effects (failure (e) in the
    module docstring). On the variance scale, zero information reads as zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = (sse0 - sse1) / sst
    imp = np.where(testable, imp, -np.inf)
    return imp, float(np.max(imp[testable])) if testable.any() else float("nan")


def markov_test(
    episodes: Sequence[Episode],
    *,
    lag: int = 0,
    b: int = _B_DRAWS,
    k_folds: int = _K_FOLDS,
    seed: int = 0,
    n_rff: int = _N_RFF,
    reward_channel: bool = True,
) -> MarkovVerdict:
    """Test H0: lag-``lag`` features are sufficient (one more lag of history
    does not improve held-out one-step prediction beyond what a matched-
    capacity uninformative block buys). ``lag=0`` is the declared-MDP
    falsifier; ``lag=k`` is the window selector's stage-k test.

    The FAMILY is scored on (O, A)-only history — the served features (S19).
    ``reward_channel=True`` additionally scores the reward target against an
    R-INCLUSIVE history block (the confounding / reward-visible-hidden-state
    diagnostic); at lag 0 that costs one extra block per placebo draw (the
    base fit is shared), at lag >= 1 a second base fit. The selector requests
    it at stage 0 only, where ``serving_material`` reads it.
    """
    rng = np.random.default_rng(seed)
    design = _build_design(episodes, lag, seed=seed, n_rff=n_rff)
    if design is None:
        raise ValueError(
            f"no episode long enough for lag {lag} (need length >= {lag + 2})"
        )
    folds = _fold_of_episode(design.episode_of, k_folds, rng)
    design_r = (
        _build_design(episodes, lag, seed=seed, n_rff=n_rff, history_reward=True)
        if reward_channel
        else None
    )

    # Testable = target varies in the data. A zero-variance target returns the
    # most confident possible pass on no evidence — reported untestable (S8/S9).
    var = design.y.var(axis=0)
    testable = var > 0.0
    untestable = [n for n, t in zip(design.dim_names, testable) if not t]
    # The FAMILY is the next-observation dims only; R is reported separately
    # (module docstring: on confounded arms R's history-dependence is the
    # DECLARED confounding, not an observability violation).
    is_obs_dim = np.array([n != "R" for n in design.dim_names])
    family = testable & is_obs_dim
    r_idx = design.dim_names.index("R")

    sse0, caches = _cv_base(design.x0, design.y, folds, k_folds)
    sst = ((design.y - design.y.mean(axis=0)) ** 2).sum(axis=0)
    sse_hist = _cv_extra(caches, design.hist, design.y)
    per_dim, stat = _family_stat(sse0, sse_hist, sst, family)

    # The reward-channel diagnostic's own base + tested block. At lag 0 the
    # conditioning block is (O_t, A_t) in both designs, so the base fit is
    # shared exactly; at lag >= 1 the R-inclusive history enters the base too.
    if design_r is not None:
        y_r = design_r.y  # same targets, same rows; only the history differs
        if lag == 0:
            assert design_r.x0.shape == design.x0.shape
            sse0_r, caches_r = sse0, caches
        else:
            sse0_r, caches_r = _cv_base(design_r.x0, y_r, folds, k_folds)
        sse_hist_r = _cv_extra(caches_r, design_r.hist, y_r)

    # Placebo draws: the SAME history block, rows circularly shifted by a
    # random nonzero per-episode offset — alignment destroyed, everything else
    # preserved. Row ranges per episode are contiguous by construction.
    bounds = np.flatnonzero(np.diff(design.episode_of, prepend=-1))
    bounds = np.append(bounds, design.episode_of.shape[0])
    spans = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    base_idx = np.arange(design.episode_of.shape[0])
    draws = np.empty(b)
    r_draws = np.empty(b)
    for i in range(b):
        idx = base_idx.copy()
        for st, en in spans:
            length = en - st
            if length > 1:
                s_e = int(rng.integers(1, length))
                idx[st:en] = st + (np.arange(length) + s_e) % length
        sse_shift = _cv_extra(caches, design.hist[idx], design.y)
        _, draws[i] = _family_stat(sse0, sse_shift, sst, family)
        # The reward diagnostic reads against the same shifts, on its own
        # R-inclusive block: a shifted lagged action (and reward) carries the
        # same episode-constant U, so this quantile isolates ALIGNED reward
        # predictability (the confounding signal).
        if design_r is not None:
            sse_shift_r = _cv_extra(caches_r, design_r.hist[idx], y_r)
            with np.errstate(divide="ignore", invalid="ignore"):
                r_draws[i] = (sse0_r[r_idx] - sse_shift_r[r_idx]) / sst[r_idx]

    # Quantile reading with the +1 correction; never a z-score.
    p = float((1 + np.sum(draws >= stat)) / (b + 1))

    reward_channel = None
    if design_r is not None and testable[r_idx]:
        with np.errstate(divide="ignore", invalid="ignore"):
            r_imp = float((sse0_r[r_idx] - sse_hist_r[r_idx]) / sst[r_idx])
        reward_channel = dict(
            improvement=r_imp,
            placebo_quantile=float((1 + np.sum(r_draws >= r_imp)) / (b + 1)),
            # The draws' upper quantile IN Delta-R^2 UNITS — what
            # ``serving_material`` nets out as the episode-constant part
            # (0.95 is the stated reporting convention matching alpha=0.05).
            draw_q95=float(np.quantile(r_draws, 0.95)),
            sd_r=float(np.sqrt(sst[r_idx] / design.y.shape[0])),
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        base_r2 = np.where(testable, 1.0 - sse0 / sst, np.nan)
    scale_invalid = [
        n
        for n, r2, t in zip(design.dim_names, base_r2, testable)
        if t and np.isfinite(r2) and r2 < 0
    ]

    # Capacity-shrink diagnostic: the observed statistic at a 4x base (no
    # placebo loop — this is a reported corroboration, never a gate).
    capacity = None
    if np.isfinite(stat):
        d_hi = _build_design(episodes, lag, seed=seed, n_rff=4 * n_rff)
        sse0_hi, caches_hi = _cv_base(d_hi.x0, d_hi.y, folds, k_folds)
        _, stat_hi = _family_stat(
            sse0_hi, _cv_extra(caches_hi, d_hi.hist, d_hi.y), sst, family
        )
        capacity = dict(
            n_rff_hi=4 * n_rff,
            stat_hi=float(stat_hi),
            shrink=(float(stat / stat_hi) if stat_hi > 0 else None),
        )

    return MarkovVerdict(
        lag=lag,
        p_value=p,
        statistic=stat,
        per_dim=per_dim,
        dim_names=design.dim_names,
        untestable=untestable,
        base_r2=base_r2,
        n_episodes=int(np.unique(design.episode_of).shape[0]),
        n_rows=int(design.y.shape[0]),
        b_draws=b,
        k_folds=k_folds,
        seed=seed,
        reward_channel=reward_channel,
        capacity=capacity,
        scale_invalid=scale_invalid,
    )


def serving_material(verdict: MarkovVerdict, *, w: float) -> dict:
    """Verdict 2 — "does the violation change what GRACE serves?"

    The DERIVED tolerance (docs/l5_equivalence_tolerance.md): the fast
    (non-episode-constant) reward-channel predictive mass bounds the served
    contrast's perturbation by ``2 sd(R) sqrt(dR2_fast)`` (Cauchy-Schwarz,
    RMS -> mean); material iff that bound exceeds L4's half-width ``w`` — the
    uncertainty GRACE already serves the value inside. Every term is measured:
    ``w`` per fit by L4, ``sd(R)`` from the data, the episode-constant netting
    level from the shift-placebo draws. Returns the full record, never a bare
    bool (C3: conditions travel with the number).
    """
    rc = verdict.reward_channel
    if rc is None:
        return dict(
            material=False,
            reason="reward channel untestable (zero variance)",
            w=float(w),
        )
    dr2_fast = max(0.0, rc["improvement"] - max(rc["draw_q95"], 0.0))
    bound = 2.0 * rc["sd_r"] * float(np.sqrt(dr2_fast))
    tau_r = (float(w) / (2.0 * rc["sd_r"])) ** 2 if rc["sd_r"] > 0 else float("inf")
    return dict(
        material=bool(bound > float(w)),
        dr2_fast=float(dr2_fast),
        contrast_bound=float(bound),
        tau_r=float(tau_r),
        w=float(w),
        sd_r=rc["sd_r"],
    )


def episodes_from_minari(dataset, mask_indices: Sequence[int] = ()) -> List[Episode]:
    """Adapt a Minari dataset; ``mask_indices`` drops observation columns at
    ANALYSIS time (the load-time-mask construction — note this realises S->A,
    not D-F's O->A; fine for the statistic's power characterisation, and the
    distinction is stamped on the dataset for the RL grid)."""
    keep = None
    out: List[Episode] = []
    for ep in dataset.iterate_episodes():
        obs = np.asarray(ep.observations, dtype=np.float64)
        if mask_indices:
            if keep is None:
                keep = [j for j in range(obs.shape[1]) if j not in set(mask_indices)]
            obs = obs[:, keep]
        out.append(
            Episode(obs=obs, act=np.asarray(ep.actions), rew=np.asarray(ep.rewards))
        )
    return out

"""L3 — continuous, NBN-native estimation with a discrete finite-mixture latent.

The estimand is a per-``(s, a)`` reward law under a latent class ``U``, fitted by
EM. Two properties of the problem drive every design decision here:

**The latent is EPISODE-STATIC.** One ``U`` is drawn per episode and shared by
every transition in it. So responsibilities are computed *per episode* — the
complete-data log-likelihood of an episode under class ``k`` is the SUM over its
rows — and then broadcast back to rows for the weighted M-step. Computing them
per transition would treat one episode's rows as independent draws of ``U``,
inflating the effective sample size by the episode length and collapsing the
posterior onto whichever class the noise favoured. This is the same episode
granularity rule as S1 in ``docs/grace_v2.md``, applied to estimation rather
than to a null.

**Two channels, not one.** ``U`` enters through ``P(A | S, U)`` and
``P(R | S, A, U)``. Both belong in the E-step likelihood — dropping the action
channel would discard exactly the information that identifies ``U`` when the
reward is weakly informative — and L5 needs them *separately*, which
``log_prob(per_node=True)`` gives directly.

Written against NBN v0.14.0 the library's own way, per ``docs/nbn_requirements``:

* ``fit(..., weights=)`` for the weighted M-step (verified exact against
  replication to 4.8e-07). NOT ``update_local``, which refuses weights (N2) —
  so a refresh REFITS.
* ``log_prob(..., per_node=True)`` for the complete-data likelihood and the
  channel split, instead of re-deriving the decomposition.
* Interventional targets through ``sample(n, do=)``, the only differentiable
  path (N1): ``query``/``query_batch`` run under ``inference_mode`` and
  ``intervene()`` deep-copies, severing the caller's graph.
* Parametric mechanisms only in the EM path (N3): KDE's bandwidth rule is
  unweighted, which would bias strata toward each other and cost L5 detection
  power in the quiet direction.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence

# Required by torch for deterministic cuBLAS; must be in the environment
# before the first CUDA matmul creates the workspace, so it is set at
# import. Without it, ``use_deterministic_algorithms(True)`` raises at the
# first matmul with an error naming this variable -- fail-loud, but at a
# confusing distance from the cause.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from nbn import (
    ContinuousVariable,
    DiscreteVariable,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralBayesianNetwork,
    NeuralCategoricalMechanism,
)
from nbn.mechanisms.parametric.dirac_gaussian import DiracGaussianMechanism

__all__ = ["EpisodeData", "Estimate", "LatentClassFit", "LatentClassEstimator"]

# Mechanisms whose weighted fit is exact. KDE is excluded deliberately (N3).
_WEIGHTED_SAFE = (MDNMechanism, LinearGaussianMechanism, NeuralCategoricalMechanism)


def stacked_rows(stacked: Dict[str, torch.Tensor]) -> int:
    """Row count of a stacked M-step frame — K copies of every transition."""
    return int(next(iter(stacked.values())).shape[0])


def _posterior_mean(out, value_fn=None) -> torch.Tensor:
    """Collapse ``query_batch``'s return into a per-row posterior mean.

    For a DISCRETE target it hands back ``[B, K]`` — a posterior over CLASSES,
    from which ``E[R] = sum_k p_k * level_k`` needs the level values. For a
    CONTINUOUS one the engine returns the likelihood-weighting particle
    representation ``(weights [B, N], samples [B, N, D])`` instead, and the
    posterior mean is the weighted average over particles. Documented as
    returning a tensor, so the tuple is worth naming here rather than
    rediscovering at each call site.
    """
    if not isinstance(out, tuple):
        probs = out.reshape(out.shape[0], -1)
        if value_fn is None:
            # A LATENT BUG UNTIL R BECAME DISCRETE. This branch used to average
            # the [B, K] posterior, which is not an expectation of anything --
            # for K = 2 it returns 0.5 for every row regardless of the
            # distribution, and it did exactly that the first time a categorical
            # R reached it. Averaging a probability vector is never the answer;
            # without the level values there IS no answer, so say so.
            raise ValueError(
                "query_batch returned a [B, K] categorical posterior but no "
                "value map was supplied; E[R] = sum_k p_k * level_k needs the "
                "levels, and averaging the probabilities is meaningless"
            )
        levels = value_fn(torch.arange(probs.shape[1], device=probs.device)).reshape(
            1, -1
        )
        return (probs * levels).sum(dim=1)
    w, samples = out
    if value_fn is not None:
        # Map particles into reward units BEFORE weighting. Averaging class
        # indices and mapping afterwards would return the level at the mean
        # index, which is a different (and meaningless) quantity.
        samples = value_fn(samples).reshape(samples.shape[0], samples.shape[1], -1)
    # Log-weights (any negative entry) versus raw weights: normalise the right
    # way for each rather than assuming, since an unnormalised exp would silently
    # bias the mean toward whichever particles happen to dominate.
    w = (
        torch.softmax(w, dim=1)
        if bool((w < 0).any())
        else w / w.sum(dim=1, keepdim=True).clamp_min(1e-30)
    )
    return (
        (w.unsqueeze(-1) * samples).sum(dim=1).reshape(samples.shape[0], -1).mean(dim=1)
    )


# DIAGNOSTIC reporting levels, not estimator calibration (A2). Nothing in the
# fit depends on either: the raw saturation FRACTION is reported, and these only
# decide when a human-readable flag is printed.
_SATURATION_EPS = 1e-3  # "a responsibility within eps of 0 or 1"
# The level at which the human-readable flag fires: a MAJORITY of episodes
# frozen. Measured values are not as bimodal as first assumed (0.00 at T = 12,
# 0.867 on the T = 500 fixture), so this is a reporting choice and is documented
# as one -- it decides only what gets printed, never what the estimator does.
_SATURATION_FLAG = 0.5
# Consecutive sub-tolerance iterations required to declare convergence. Not a
# calibration constant: it is the WINDOW LENGTH of an existing test, and 2 is the
# smallest window that can distinguish a trend from a single draw.
_CONVERGENCE_WINDOW = 2


@dataclass
class EpisodeData:
    """Transitions plus the episode blocks that own them.

    ``episode_ids`` is not bookkeeping — it is the structure the whole estimator
    rests on. Every array is transition-aligned and float32 on the model device.
    """

    state: torch.Tensor  # (n, d_s)
    action: torch.Tensor  # (n,) long for discrete
    reward: torch.Tensor  # (n,)
    episode_ids: torch.Tensor  # (n,) long
    proxy: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.state.shape[0]
        for name in ("action", "reward", "episode_ids"):
            v = getattr(self, name)
            if v.shape[0] != n:
                raise ValueError(
                    f"{name} has {v.shape[0]} rows against {n} states; every array "
                    "must be transition-aligned"
                )
        for k, v in self.proxy.items():
            if v.shape[0] != n:
                raise ValueError(f"proxy {k!r} has {v.shape[0]} rows against {n}")

    @property
    def n(self) -> int:
        return int(self.state.shape[0])

    def blocks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(unique_episode_ids, inverse_index)`` — the row→episode map."""
        return torch.unique(self.episode_ids, return_inverse=True)

    def episode_sum(self, per_row: torch.Tensor) -> torch.Tensor:
        """Sum a per-row quantity within each episode. ``(n, K) -> (E, K)``."""
        uniq, inv = self.blocks()
        out = torch.zeros(
            uniq.numel(), per_row.shape[1], device=per_row.device, dtype=per_row.dtype
        )
        return out.index_add_(0, inv, per_row)

    def broadcast(self, per_episode: torch.Tensor) -> torch.Tensor:
        """Expand a per-episode quantity back over its rows. ``(E, K) -> (n, K)``."""
        _, inv = self.blocks()
        return per_episode[inv]


@dataclass
class Estimate:
    """A number that carries the conditions it was produced under (C3).

    ``value`` stays a tensor, so the differentiable path is unaffected. The
    labels ride alongside because a fit that decreased its likelihood, or that
    never separated, can still produce a perfectly plausible-looking value —
    recovery was 0.980 on a fixture WITH a decrease present. Non-monotone does
    not mean wrong, which is exactly why it must be labelled rather than either
    trusted or discarded.
    """

    value: torch.Tensor
    monotone: bool
    converged: bool
    separability: float
    backtracks: int = 0
    backtrack_exhausted: bool = False
    # The correctness diagnostic. ``separability`` above is retained as
    # TELEMETRY only: it saturates to 1.0 on long episodes regardless of whether
    # the fit is right, so it cannot answer the question C3 attaches it for.
    separation_per_step: float = 0.0
    initial_saturation: float = 0.0
    saturated_at_init: bool = False
    reached_tau_one: bool = True
    degenerate_mechanism: bool = False
    mechanism_degeneracy: Dict[str, float] = field(default_factory=dict)
    reward_mechanism: str = ""
    algorithm: str = "restart-EM"
    deterministic: bool = False
    tau1_budget_bound: bool = False

    def label(self) -> str:
        # The resolved R mechanism rides on every number: a likelihood read
        # without knowing whether R was categorical or an MDN is not comparable
        # to one read the other way.
        bits = [f"R={self.reward_mechanism or '?'}"]
        if self.algorithm != "gem":
            # PROVISIONAL, and the label is how that travels. Under restart-EM
            # the parameter sequence may never settle even after the objective
            # plateaus, so ``converged`` can fire on delta-LL while the
            # parameters still jump between iterations. Any value-level quantity
            # -- interventional values, L4 bounds -- reads the PARAMETERS, not
            # the objective, and inherits that instability.
            bits.append(f"{self.algorithm.upper()}-PARAMS-PROVISIONAL")
        bits.append(f"sep/step={self.separation_per_step:.3f}")
        if self.degenerate_mechanism:
            worst = max(self.mechanism_degeneracy.items(), key=lambda kv: kv[1])
            bits.append(f"DEGENERATE-SCALE({worst[0]}:{worst[1]:.2f})")
        if not self.reached_tau_one:
            # Stopped while still tempered: the parameters maximise a smoothed
            # surrogate, not the likelihood. Louder than NOT-CONVERGED, because
            # a not-converged fit is at least optimising the right thing.
            bits.append("STOPPED-WHILE-TEMPERED")
        if self.saturated_at_init:
            # Frozen at its initialisation: the E-step was a step function from
            # the first iteration, so this number is a property of the init, not
            # of the likelihood. It must travel with every value downstream.
            bits.append(f"SATURATED-AT-INIT({self.initial_saturation:.2f})")
        if not self.deterministic:
            # The unusual case now flags, not the default one: under default
            # kernels identical fits differed by 12-58 nats run-to-run, so a
            # number produced that way is path-dependent on kernel scheduling
            # and comparable to nothing without a calibrated floor.
            bits.append("NONDETERMINISTIC-KERNELS")
        if not self.monotone:
            bits.append("NON-MONOTONE-EM")
        if not self.converged:
            bits.append("NOT-CONVERGED")
        if self.tau1_budget_bound:
            # The iteration cap acted as a TUNING KNOB for this fit: it ran
            # out of budget mid-ascent. The number is budget-truncated (S11's
            # broken-experiment direction) and any consumer must know.
            bits.append("TAU1-BUDGET-BOUND")
        if self.backtrack_exhausted:
            bits.append("BACKTRACK-EXHAUSTED")
        if self.backtracks:
            bits.append(f"backtracks={self.backtracks}")
        bits.append(f"sep(telemetry)={self.separability:.2f}")
        return " ".join(bits)

    def __float__(self) -> float:
        return float(self.value.detach())


@dataclass
class LatentClassFit:
    """What EM converged to, and the evidence for it."""

    model: NeuralBayesianNetwork
    prior: torch.Tensor  # (K,) P(U)
    responsibilities: torch.Tensor  # (E, K) per EPISODE
    log_likelihood: List[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    label_permutation: tuple = ()
    backtracks: int = 0
    backtrack_exhausted: bool = False
    temperatures: List[float] = field(default_factory=list)
    n_anneal: int = 0
    # Temperatures whose surrogate the M-step could not improve. Reported, not
    # fatal: the anneal is a warm-up and skipping a rung costs at most the
    # warm-up's benefit.
    anneal_exhaustions: int = 0
    # Cumulative learning-rate reduction the backtracking needed. Reported
    # because a fit that had to shrink by 2^-6 is telling you the M-step's
    # default step size is wrong for this data, which is actionable.
    stationary: bool = False
    # Relative worsening of the BEST step available when the fit gave up. Near
    # numerical zero => a numerical fixed point; ~tol => the optimiser simply
    # cannot ascend. ``stationary`` is the same flag either way, but not the
    # same strength of claim.
    rejected_step_rel: float | None = None
    backtracks_per_iter: List[int] = field(default_factory=list)
    final_lr_scale: float = 1.0
    lr_reductions: int = 0
    epoch_escalations: int = 0
    # WHICH ALGORITHM PRODUCED THIS FIT. "gem" when the M-step warm-starts
    # (NBN >= v0.15.0, fit(warm_start=True), the default): each M-step is a
    # partial maximisation from theta_old, so GEM's guarantee holds and the
    # monotone guard compares genuine steps. "restart-EM" when
    # warm_start=False: fit_local rebuilds per call, the M-step is an
    # independent refit, and even ACCEPTED steps are stochastic -- the label
    # carries PARAMS-PROVISIONAL downstream (see Estimate.label()).
    algorithm: str = "restart-EM"
    # Were deterministic kernels enforced for this fit? Default False so
    # only a fit actually run under the flag claims it; ``fit()`` sets it
    # explicitly. Pre- and post-determinism numbers are not naively
    # comparable (deterministic kernels are not bit-identical to the
    # default ones), which is exactly why the mode must travel.
    deterministic: bool = False
    initial_saturation: float = 0.0
    final_saturation: float = 0.0
    # DID THE tau=1 ITERATION CAP BIND? True iff the loop spent its whole
    # budget without reaching an end state (converged / stationary /
    # exhausted). The binding rule (method-parameter audit, 2026-08-21): a
    # limit that never binds across the grid is a safety guard; one that
    # binds is a tuning knob and must be derived or disclosed. This flag is
    # what makes that checkable per run instead of rhetorical. Note the
    # anneal already sits OUTSIDE the cap by construction (total iterations =
    # max_iter + derived rungs), so the cap has no hidden tau0 dependence.
    tau1_budget_bound: bool = False
    separation_per_step: float = 0.0
    mechanism_degeneracy: Dict[str, float] = field(default_factory=dict)
    reward_mechanism: str = ""

    def estimate(self, value: torch.Tensor) -> "Estimate":
        """Wrap a value with this fit's conditions — the ONLY way estimates are
        produced, so a number cannot escape without its labels."""
        return Estimate(
            value=value,
            monotone=self.monotone,
            converged=self.converged,
            separability=self.separability(),
            backtracks=self.backtracks,
            backtrack_exhausted=self.backtrack_exhausted,
            separation_per_step=self.separation_per_step,
            initial_saturation=self.initial_saturation,
            saturated_at_init=self.saturated_at_init,
            reached_tau_one=self.reached_tau_one,
            degenerate_mechanism=self.degenerate_mechanism,
            mechanism_degeneracy=dict(self.mechanism_degeneracy),
            reward_mechanism=self.reward_mechanism,
            algorithm=self.algorithm,
            deterministic=self.deterministic,
            tau1_budget_bound=self.tau1_budget_bound,
        )

    @property
    def degenerate_mechanism(self) -> bool:
        """Has any channel's fitted scale collapsed onto its declared floor?

        Detected against the mechanism's own ``min_scale``, not against a
        magnitude of ``separation_per_step`` — which is unbounded above, so a
        huge value there means degeneracy rather than confidence and cannot
        itself be the test.
        """
        return any(v > 0.0 for v in self.mechanism_degeneracy.values())

    @property
    def finished(self) -> bool:
        """Did the fit reach an end state, by either legitimate route?

        Read ``rejected_step_rel`` alongside a stationary stop: it says
        whether the best available step worsened the objective by numerical
        noise (a fixed point) or by a real margin (an optimiser that cannot
        ascend). Same flag, different strength of claim.

        Two of them, and they are different claims: the TOLERANCE window
        (improvements below ``tol`` twice consecutively) and STATIONARITY (no
        improving step exists at any step size tried, with the last improvement
        already sub-tolerance). A fit that runs out of step sizes while still
        improving by percent is neither -- it is stuck, and is reported as such.
        """
        return bool(self.converged or self.stationary)

    @property
    def reached_tau_one(self) -> bool:
        """Did the anneal finish, or did the fit stop while still tempered?

        A fit that stops mid-anneal is **not an estimate of anything**: its last
        objective is a tempered one, so its parameters maximise a smoothed
        surrogate rather than the likelihood. Observed at small iteration
        budgets, where the backtrack guard exhausts before the schedule
        flattens. It reads exactly like a converged fit unless asked.
        """
        return bool(self.temperatures) and self.temperatures[-1] == 1.0

    @property
    def saturated_at_init(self) -> bool:
        """Did the FIRST untempered E-step already assign hard 0/1 labels?

        A REPORTING flag over ``initial_saturation``, which is the number that
        matters and is always available. Nothing in the fit reads this.

        If it did, EM never had a choice of basin: the M-step fitted whatever
        partition the initialisation supplied and every later E-step confirmed
        it. The resulting number is a function of the init. Detected directly
        rather than inferred, and propagated under C3 like ``monotone`` and
        ``converged``, so it is visible wherever it occurs rather than only
        where someone thought to look."""
        return self.initial_saturation >= _SATURATION_FLAG

    @property
    def monotone(self) -> bool:
        """Did the observed-data log-likelihood never decrease?

        Textbook EM guarantees this. **Ours does not, and the guarantee does not
        apply**: the M-step maximises the weighted likelihood by SGD for a fixed
        epoch budget, so it is a PARTIAL maximisation — generalized EM, whose
        guarantee is only that a step which does not decrease the objective
        cannot decrease the likelihood. A partial M-step can and does decrease
        it, and it was observed decreasing on the recovery fixture.

        This is reported rather than suppressed because a non-monotone run is
        the honest signal that the M-step budget is too small for the problem,
        and the alternative — reading a plausible final value and assuming the
        ascent behind it — is exactly the failure this whole layer exists to
        avoid. It is a diagnostic, not an error: recovery on the fixture is
        0.980 with a decrease present, so non-monotone does NOT imply wrong.
        """
        # EVALUATED ONLY OVER THE tau = 1 PHASE. During annealing the M-step
        # maximises the TEMPERED objective, and the true likelihood may
        # legitimately decrease while it does -- judging the annealing steps by
        # the untempered likelihood would report every annealed fit as
        # non-monotone and, worse, the guard would reject the very steps the
        # anneal exists to take.
        tail = self.log_likelihood[self.n_anneal :]
        return all(b >= a - 1e-6 for a, b in zip(tail, tail[1:]))

    @property
    def final_ll(self) -> float:
        return self.log_likelihood[-1] if self.log_likelihood else float("nan")

    def hard_assignment(self) -> torch.Tensor:
        return self.responsibilities.argmax(dim=1)

    def separability(self) -> float:
        """Mean max-responsibility. **TELEMETRY ONLY — not the correctness
        diagnostic. Use ``separation_per_step``.**

        It was the correctness diagnostic, and it cannot be: the posterior it
        summarises is built from a SUM over the episode's rows, so it is
        confident for long episodes whether or not the fit is right. Measured on
        D-D Acrobot at T = 500: **1.0000 at 0.53 recovery**, in every one of six
        fits. A diagnostic that reads perfect exactly where the estimator fails
        is worse than none, because C3 makes it travel with the number and lends
        it authority.

        Retained because the saturation level is itself informative — read it
        next to ``initial_saturation``, never as evidence a fit is good.
        """
        return float(self.responsibilities.max(dim=1).values.mean())


class LatentClassEstimator:
    """EM over a discrete latent ``U`` with continuous NBN mechanisms.

    ``u_card`` is the declared cardinality — a modelling commitment named in the
    catalogue as ``finite_K_latent_class``, not something inferred here.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        n_actions: int,
        u_card: int = 2,
        proxy_names: Sequence[str] = (),
        device: str = "cpu",
        reward_mechanism: str = "auto",
        seed: int = 0,
    ) -> None:
        if u_card < 2:
            raise ValueError(f"u_card must be at least 2, got {u_card}")
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.u_card = int(u_card)
        self.proxy_names = tuple(proxy_names)
        self.device = device
        self.seed = int(seed)
        self._reward_mechanism = reward_mechanism
        # Resolved from the DATA at fit time when "auto"; None until then. The
        # provisional build below lets callers inspect ``self.model`` before
        # fitting, and ``fit`` rebuilds if the resolution disagrees.
        self._reward_levels: torch.Tensor | None = None
        # Set when R has SUPPORT CARDINALITY 1 (a constant channel — the null
        # arms): the Dirac routing (ruled 2026-08-23). An MDN on a constant
        # drives its scale to the floor and burned 100+ backtracks per fit on
        # d_a_null; a Dirac at the constant is the honest mechanism — finite
        # log-density, nothing to estimate, zero class discrimination.
        self._reward_const: float | None = None
        self.resolved_reward_mechanism = (
            None if reward_mechanism == "auto" else reward_mechanism
        )
        self.model = self._build()

    # ---------------------------------------------------------------- build --
    @property
    def _reward_is_discrete(self) -> bool:
        return self._reward_levels is not None

    def _reward_mech(self):
        """The mechanism for ``R`` and for the proxies.

        **``R`` IS NOT CONTINUOUS ON THESE ARMS, and fitting an MDN to it is a
        modelling error whose symptom is the ``min_scale`` floor.** Every arm in
        this benchmark has a reward that is deterministic given ``(S, A, U)`` --
        CartPole pays 1 (+ the gated bonus), Acrobot pays -1 (+ the bonus) -- so
        ``R`` has finite support and zero conditional noise. An MDN fitted to it
        correctly drives its scale onto the floor, which pins the reward
        log-density at an arbitrary constant.

        That is not a cosmetic problem. **Both calibration layers are
        likelihood-based**: L4's compatible set and L5's likelihood-ratio
        statistics would each be partly measuring ``min_scale`` rather than the
        data. A categorical mechanism gives proper probabilities over the
        observed support, no floor, and meaningful likelihood magnitudes.
        """
        if self._reward_const is not None:
            # Constant support: a do-intervention CPD at the constant. Its
            # fit_local is a documented no-op, so the M-step spends nothing on
            # a channel with nothing to learn, and its log-density is finite
            # rather than a min_scale floor.
            return DiracGaussianMechanism(value=self._reward_const)
        if self._reward_is_discrete:
            return NeuralCategoricalMechanism(
                n_classes=int(self._reward_levels.numel())
            )
        if self._reward_mechanism in ("mdn", "auto"):
            return MDNMechanism(num_components=3, hidden=(64, 64))
        if self._reward_mechanism == "linear_gaussian":
            return LinearGaussianMechanism()
        raise ValueError(
            f"unknown reward mechanism {self._reward_mechanism!r}; the EM path "
            "admits only mechanisms whose weighted fit is exact (N3) — KDE's "
            "bandwidth rule is unweighted and would bias strata toward each "
            "other, costing L5 detection power in the quiet direction"
        )

    def _proxy_mech(self):
        """Proxies stay continuous: they are Gaussian noise around ``U`` by
        construction, so the finite-support argument above does not apply."""
        if self._reward_mechanism == "linear_gaussian":
            return LinearGaussianMechanism()
        return MDNMechanism(num_components=3, hidden=(64, 64))

    def _resolve_reward_type(self, data: EpisodeData) -> None:
        """Decide continuous-vs-discrete for ``R`` FROM THE DATA, and record it.

        The criterion is derived, not a magnitude threshold on the number of
        distinct values (which would be an A2 constant): **a finite-support
        variable's support does not grow when you look at more of it.** Compare
        the distinct values in a half sample against the full sample -- equal
        means finite support, while a continuous variable roughly doubles.

        Errs toward CONTINUOUS: a discrete variable whose rarest level happens
        to fall only in the second half is misread as continuous, which merely
        restores the previous behaviour. The reverse error would be the damaging
        one.
        """
        if self._reward_mechanism != "auto":
            return
        r = data.reward.reshape(-1)
        n = int(r.numel())
        half = int(torch.unique(r[: max(n // 2, 1)]).numel())
        levels = torch.unique(r)
        full = int(levels.numel())
        if full == 1:
            const = float(levels[0])
            changed = (
                self.resolved_reward_mechanism is None
                or not str(self.resolved_reward_mechanism).startswith("dirac")
                or self._reward_const != const
            )
            self._reward_levels = None
            self._reward_const = const
            self.resolved_reward_mechanism = f"dirac[{const:g}]"
            if changed:
                self.model = self._build()
            return
        self._reward_const = None
        discrete = full == half and 1 < full < n
        new_levels = levels.sort().values if discrete else None
        changed = (new_levels is None) != (self._reward_levels is None) or (
            new_levels is not None
            and self._reward_levels is not None
            and not torch.equal(new_levels, self._reward_levels)
        )
        self._reward_levels = new_levels
        self.resolved_reward_mechanism = f"categorical[{full}]" if discrete else "mdn"
        if changed:
            self.model = self._build()

    def _build(self) -> NeuralBayesianNetwork:
        """The two channels, plus any declared covariate-free proxies.

        Proxies are given ``U`` as their ONLY parent, which is the structural
        content of D-D's covariate-freeness: ``P(Z|U)`` does not depend on
        ``(s, a)``, so the measurement matrix is global and pins the labelling
        globally. Wiring ``S`` in here would silently turn D-D into D-B.
        """
        edges = [
            ("S", "A"),
            ("U", "A"),  # action channel
            ("S", "R"),
            ("A", "R"),
            ("U", "R"),  # reward channel
        ]
        variables: Dict[str, object] = {
            "S": ContinuousVariable("S", dim=self.state_dim),
            "A": DiscreteVariable("A", cardinality=self.n_actions),
            "R": (
                DiscreteVariable("R", cardinality=int(self._reward_levels.numel()))
                if self._reward_is_discrete
                else ContinuousVariable("R", dim=1)
            ),
            "U": DiscreteVariable("U", cardinality=self.u_card),
        }
        for p in self.proxy_names:
            edges.append(("U", p))
            variables[p] = ContinuousVariable(p, dim=1)

        model = NeuralBayesianNetwork(edges, variables, device=self.device)
        # Assign the EM-path mechanisms EXPLICITLY rather than accepting
        # auto-selection: N3 admits only mechanisms whose weighted fit is exact,
        # and an auto-chosen KDE would bias strata toward each other silently.
        # Roots too: fit() requires a mechanism on EVERY node. S's marginal is
        # not part of any estimand (it cancels in the posterior -- the same S
        # under every class) but it must be fittable; U's is fitted on the
        # stacked weighted data, which reproduces the responsibility-mean prior
        # exactly, so the two sources of P(U) agree by construction rather than
        # by coincidence.
        model.set_mechanism("S", LinearGaussianMechanism())
        model.set_mechanism("U", NeuralCategoricalMechanism(n_classes=self.u_card))
        model.set_mechanism("A", NeuralCategoricalMechanism(n_classes=self.n_actions))
        model.set_mechanism("R", self._reward_mech())
        for p in self.proxy_names:
            model.set_mechanism(p, self._proxy_mech())
        self._assert_weighted_safe(model)
        return model

    def _assert_weighted_safe(self, model) -> None:
        """Fail FAST on a mechanism that cannot take weights, rather than let the
        M-step silently ignore them (N3). ``supports_weights`` is False for KNN,
        so a mistaken choice raises here instead of biasing the fit."""
        mechs = dict(getattr(model, "mechanisms", {}) or {})
        for name in ("S", "U", "A", "R") + tuple(self.proxy_names):
            mech = mechs.get(name)
            if mech is None:
                continue
            if not getattr(mech, "supports_weights", False):
                raise TypeError(
                    f"node {name!r} uses {type(mech).__name__}, which does not "
                    "support sample weights; the EM M-step is inherently weighted "
                    "by per-episode responsibilities and would silently ignore them"
                )

    # ------------------------------------------------------------------ data --
    def _frame(self, data: EpisodeData, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One NBN data dict with ``U`` clamped to the given per-row class."""
        frame = {
            "S": data.state,
            "A": data.action.reshape(-1),
            "R": self._encode_reward(data.reward),
            "U": u.reshape(-1),
        }
        for name, val in data.proxy.items():
            frame[name] = val.reshape(-1, 1)
        return frame

    def _encode_reward(self, r: torch.Tensor) -> torch.Tensor:
        """Reward in the units the R node expects: a class INDEX when discrete.

        Exact matching, not bucketing: the levels are the observed values, so a
        reward that fails to match one is a data/levels mismatch and must raise
        rather than be snapped to a neighbour.
        """
        r = r.reshape(-1)
        if not self._reward_is_discrete:
            return r.reshape(-1, 1)
        levels = self._reward_levels.to(r.device)
        idx = torch.bucketize(r, levels)
        idx = idx.clamp(0, levels.numel() - 1)
        # bucketize returns the insertion point; snap to the nearer neighbour.
        lo = (idx - 1).clamp_min(0)
        pick = torch.where((r - levels[lo]).abs() <= (levels[idx] - r).abs(), lo, idx)
        if not torch.allclose(levels[pick], r, atol=1e-6):
            raise ValueError(
                "a reward value does not match any resolved level; the discrete "
                "R mechanism was resolved on different data than it is being "
                "scored on"
            )
        return pick.long()

    def _decode_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Model output back into REWARD UNITS.

        Without this the interventional paths would average class INDICES and
        return a plausible number in the wrong units -- E[index] rather than
        E[R] -- which is precisely the kind of silently-wrong value the C3
        discipline exists to keep out.
        """
        if not self._reward_is_discrete:
            return x
        levels = self._reward_levels.to(x.device)
        return levels[x.reshape(-1).long().clamp(0, levels.numel() - 1)]

    # ---------------------------------------------------------------- E-step --
    def _episode_log_liks(self, data: EpisodeData) -> torch.Tensor:
        """``(E, K)`` complete-data log-likelihood of each EPISODE under each class.

        Per row we take the log-density of the OBSERVED children of ``U`` only —
        the action channel, the reward channel and any proxies. ``P(S)`` and the
        ``U`` prior are excluded here: ``S`` is the same under every class so it
        cancels in the posterior, and the prior is added once per episode rather
        than once per row. Adding a per-row prior is the classic bug that makes
        the posterior scale with episode length.
        """
        channels = ["A", "R"] + list(self.proxy_names)
        cols = []
        # The E-step is evaluated with the CURRENT parameters held fixed: in EM
        # the responsibilities are constants that the M-step maximises against,
        # not a function to differentiate through. Leaving the graph attached
        # makes the M-step backward through the E-step's graph a second time
        # (and would be the wrong objective even if it did not).
        for k in range(self.u_card):
            u_k = torch.full((data.n,), k, dtype=torch.long, device=data.state.device)
            with torch.no_grad():
                per_node = self.model.log_prob(self._frame(data, u_k), per_node=True)
            row = sum(per_node[c].reshape(-1).detach() for c in channels)
            cols.append(row)
        per_row = torch.stack(cols, dim=1)  # (n, K)
        return data.episode_sum(per_row)  # (E, K)

    def e_step(
        self, data: EpisodeData, prior: torch.Tensor, temperature: float = 1.0
    ) -> tuple:
        """``(resp, true_ll, tempered_objective, saturation)``.

        **Why a temperature.** The episode log-likelihood is a SUM over the
        episode's rows, so a between-class difference of d nats per step becomes
        ``T * d`` nats per episode. At ``T = 500`` the softmax over classes is a
        step function: responsibilities are 0/1 after the FIRST E-step, the
        M-step fits two hard clusters, and EM can never move between basins
        again. It is frozen at whatever its initialisation happened to pick.
        Measured on D-D Acrobot at ``T = 500``: **6 of 6 fits at chance recovery
        (0.53-0.59), every one reporting separability 1.0000.** At ``T = 18-38``
        the same code recovers 0.997-1.000. This is a property of the estimator
        on long episodes, not of any cell.

        Tempering is the standard remedy (deterministic annealing): divide the
        class log-likelihoods by ``tau`` before the softmax, so early E-steps
        cannot saturate and the basin is chosen by the likelihood rather than by
        the initialisation. The annealed objective
        ``tau * sum_e logsumexp(ll_e / tau)`` reduces to the true log-likelihood
        at ``tau = 1``.

        ``saturation`` is ALWAYS computed from the UNTEMPERED posterior, because
        the question it answers is "would this fit have frozen", which tempering
        is meant to prevent rather than to hide.
        """
        ll = self._episode_log_liks(data) + torch.log(prior).reshape(1, -1)
        # log-sum-exp per episode: the observed-data likelihood, and the
        # normaliser for the posterior.
        marginal = torch.logsumexp(ll, dim=1)  # (E,)
        resp_true = torch.exp(ll - marginal.reshape(-1, 1))
        sat = float(
            (resp_true.max(dim=1).values > 1.0 - _SATURATION_EPS).float().mean()
        )
        if temperature == 1.0:
            return resp_true, float(marginal.sum()), float(marginal.sum()), sat
        tempered = ll / float(temperature)
        resp = torch.softmax(tempered, dim=1)
        obj = float(temperature) * float(torch.logsumexp(tempered, dim=1).sum())
        return resp, float(marginal.sum()), obj, sat

    def _density_ceiling(self, name: str) -> float:
        """The largest per-row log-density this mechanism can legally produce.

        A continuous mechanism declares a ``min_scale`` floor, so its density is
        bounded above by the value a Gaussian of exactly that scale attains at
        its mode: ``-log(min_scale * sqrt(2*pi))`` per dimension. For a mixture
        the bound still holds, since the component weights sum to one.

        **This introduces no constant.** The ceiling is DERIVED from the
        library's own declared parameter, which is what makes it an admissible
        detector under A2 — a magnitude cut-off on ``separation_per_step`` would
        have been a tuned threshold wearing a diagnostic's clothes.
        """
        mech = dict(getattr(self.model, "mechanisms", {}) or {}).get(name)
        ms = float(getattr(mech, "min_scale", 0.0) or 0.0)
        if ms <= 0.0:
            return float("inf")  # no declared floor: nothing to detect against
        var = getattr(self.model, "variables", {}).get(name)
        dim = int(getattr(var, "dim", 1) or 1)
        return -math.log(ms * math.sqrt(2.0 * math.pi)) * dim

    def _min_fitted_scale(self, name: str, pa_tensor, n: int):
        """The SMALLEST scale this mechanism actually emits on the fitted rows.

        Detecting the scale is strictly better than detecting its consequence.
        ``separation_per_step`` blows up from EITHER direction — the winning
        class spiking up (bounded by the density ceiling) or the losing class
        collapsing down (unbounded below, so no ceiling catches it) — but both
        are the same underlying event: a fitted scale sitting on ``min_scale``.
        Reading the scale is two-sided by construction and does not depend on
        which class happens to win at a given row.

        Still no constant: the floor compared against is the mechanism's own
        declared ``min_scale``.
        """
        mech = dict(getattr(self.model, "mechanisms", {}) or {}).get(name)
        if mech is None:
            return None
        # MDN: heteroscedastic, so the scale has to be evaluated ON THE ROWS.
        params = getattr(mech, "_params_from_parents", None)
        if callable(params):
            with torch.no_grad():
                _, _, scale = params(pa_tensor, (n,))
            return float(scale.min())
        # LinearGaussian and friends: homoscedastic, one scale per node.
        flat = getattr(mech, "_scale", None)
        if callable(flat):
            with torch.no_grad():
                return float(flat().min())
        return None

    def _mechanism_degeneracy(self, data: EpisodeData) -> Dict[str, float]:
        """Fraction of rows whose fitted density has hit the ``min_scale`` FLOOR.

        A class whose scale collapses onto the floor is a spike, not a fit: it
        scores its own support astronomically and everything else catastrophically.
        Observed on D-D Acrobot at T = 500, where one fit reported
        ``separation_per_step = 287,155`` nats/step alongside
        ``separability = 1.0000`` and recovery 0.55 — perfect on the old
        diagnostic, visibly broken on the new one, and neither of them said WHY.

        **``separation_per_step`` is unbounded above, so a very large value means
        DEGENERACY rather than confidence.** The two are told apart by this
        detector — the floor — and never by the magnitude.
        """
        from nbn.core.network import pack_parents

        channels = ["R"] + list(self.proxy_names)
        out: Dict[str, float] = {}
        for k in range(self.u_card):
            u_k = torch.full((data.n,), k, dtype=torch.long, device=data.state.device)
            frame = self._frame(data, u_k)
            with torch.no_grad():
                per_node = self.model.log_prob(frame, per_node=True)
            for c in channels:
                mech = dict(getattr(self.model, "mechanisms", {}) or {}).get(c)
                floor = float(getattr(mech, "min_scale", 0.0) or 0.0)
                # PRIMARY, two-sided: is the emitted scale sitting on the floor?
                if floor > 0.0:
                    pa = pack_parents(frame, self.model.dag.parents(c))
                    smin = self._min_fitted_scale(c, pa, data.n)
                    if smin is not None and smin <= floor * (1.0 + 1e-6):
                        out[c] = 1.0
                        continue
                # SECONDARY, one-sided: rows sitting at the density ceiling the
                # floor implies. Two independent detectors for one event are
                # cheap, and either may see a case the other misses.
                ceiling = self._density_ceiling(c)
                if not math.isfinite(ceiling):
                    continue
                lp = per_node[c].reshape(-1)
                frac = float((lp >= ceiling - 1e-6).float().mean())
                out[c] = max(out.get(c, 0.0), frac)
        return out

    def _separation_per_step(self, data: EpisodeData) -> float:
        """Class separation in NATS PER STEP — the length-normalised replacement
        for ``separability`` as the correctness diagnostic.

        ``separability`` (mean max-responsibility) is a posterior confidence, and
        a posterior built from a sum over ``T`` rows is confident for large ``T``
        whether or not it is right: it read **1.0000 at 0.53 recovery**. C3
        attaches a diagnostic so an unseparated fit can be told from a separated
        one, and that one fails exactly in the regime where it is needed.

        This is the gap between the best and second-best class in the per-STEP
        average log-likelihood, so it does not grow with episode length and a
        weak-separation fit reads weak however long the episodes are.
        """
        ll = self._episode_log_liks(data)  # (E, K), episode-summed
        ones = torch.ones(data.n, 1, device=ll.device, dtype=ll.dtype)
        counts = data.episode_sum(ones).clamp_min(1.0)  # (E, 1)
        per_step = ll / counts
        if per_step.shape[1] < 2:
            return 0.0
        top2 = per_step.topk(2, dim=1).values
        return float((top2[:, 0] - top2[:, 1]).mean())

    def _temperature_schedule(
        self, data: EpisodeData, max_iter: int, tau0: float | None, n_anneal: int | None
    ) -> list:
        """Geometric anneal from ``tau0`` down to 1, then 1 for the rest.

        **``tau0`` DEFAULTS TO THE MEAN EPISODE LENGTH, and that is a derivation
        rather than a tuning choice.** At ``tau = T`` the tempered episode
        log-likelihood is exactly the per-STEP average, which is the natural
        scale on which classes differ; annealing from there to 1 walks from
        "one effective observation per episode" to the full episode. It is read
        off the data, never fitted, and it is REPORTED on the fit
        (``temperatures``) so a reader can see the schedule that produced a
        number. Tuning it against recovery would be choosing the method by the
        answer.
        """
        if tau0 is None:
            ones = torch.ones(data.n, 1, device=data.state.device)
            tau0 = float(data.episode_sum(ones).mean())
        tau0 = max(1.0, float(tau0))
        if tau0 <= 1.0:
            return [1.0] * (max_iter + 1)
        if n_anneal is None:
            # RUNGS ARE DERIVED FROM tau0, NOT FROM THE ITERATION BUDGET. Tying
            # them to ``max_iter`` was a real bug and the cost run found it: at
            # ``max_iter = 60`` the anneal claimed 30 rungs, consumed the ENTIRE
            # 65-minute CartPole fit, and the run died on its first tau = 1
            # iteration having spent everything getting there. The number of
            # rungs is a property of how far the temperature has to travel, so
            # halve it each rung -- ``ceil(log2(tau0))`` reaches 1 by
            # construction and introduces no free parameter (5 rungs at
            # tau0 = 18, 9 at tau0 = 500).
            n_anneal = max(1, math.ceil(math.log2(tau0)))
        n_anneal = max(0, int(n_anneal))
        if n_anneal == 0:
            return [1.0] * (max_iter + 1)
        sched = [tau0 ** (1.0 - i / n_anneal) for i in range(n_anneal)]
        return sched + [1.0] * (max_iter + 1)

    # ---------------------------------------------------------------- M-step --
    def m_step(
        self, data: EpisodeData, resp: torch.Tensor, **fit_kwargs
    ) -> torch.Tensor:
        """Refit every mechanism on the responsibility-weighted data.

        The K classes are stacked into ONE weighted dataset — K copies of every
        row, copy ``k`` carrying ``U = k`` with weight ``r_e(k)`` — which is the
        standard EM M-step for a discrete latent and lets a single ``fit`` call
        do all of it. Weights are per-EPISODE, broadcast across that episode's
        rows, because one responsibility governs the whole block.
        """
        # ``warm_start`` rides through fit_kwargs to NBN: True makes this a
        # STEP from the current parameters (GEM); False makes it a fresh refit
        # (restart-EM). The caller (``fit``) decides; the first M-step of every
        # fit is explicitly cold.
        weights_per_row = data.broadcast(resp)  # (n, K)
        frames, weights = [], []
        for k in range(self.u_card):
            u_k = torch.full((data.n,), k, dtype=torch.long, device=data.state.device)
            frames.append(self._frame(data, u_k))
            weights.append(weights_per_row[:, k])
        stacked = {key: torch.cat([f[key] for f in frames], dim=0) for key in frames[0]}
        w = torch.cat(weights, dim=0)
        # FIXED STEP BUDGET. ``epochs`` makes the M-step cost O(n * epochs), so a
        # production-scale dataset pays proportionally more per EM iteration for
        # no extra guarantee -- GEM asks only that the M-step INCREASE the
        # objective, not that it maximise it, so a fixed number of gradient steps
        # is admissible and makes the M-step O(steps) instead. ``epochs`` is
        # derived from it here rather than passed, so the two cannot disagree.
        # NO EWC CONSOLIDATION IN AN M-STEP. NBN defaults ``consolidate=True``,
        # which runs a diagonal-Fisher pass -- up to ``sample_cap = 4096``
        # SEQUENTIAL per-sample backward passes, per node, per call. It exists
        # for continual learning, where a mechanism must retain earlier tasks.
        # An M-step is a fresh weighted fit of the same nodes on the same rows,
        # and GRACE never calls ``update()``, so the snapshot it produces is
        # never read. Overridable, so a caller who wants it can ask.
        fit_kwargs.setdefault("consolidate", False)
        budget = fit_kwargs.pop("m_step_budget", None)
        if budget is not None:
            bs = int(fit_kwargs.get("batch_size") or 1024)
            fit_kwargs["batch_size"] = bs
            steps_per_epoch = max(1, math.ceil(stacked_rows(stacked) / bs))
            fit_kwargs["epochs"] = max(1, round(int(budget) / steps_per_epoch))
        self.model.fit(stacked, weights=w, **fit_kwargs)
        # Prior from the EPISODE responsibilities, never the row ones.
        return resp.mean(dim=0)

    # -------------------------------------------------------------------- EM --
    def _snapshot(self) -> dict:
        """A parameter snapshot the M-step CANNOT mutate.

        ``state_dict()`` returns tensors SHARING STORAGE with the live
        parameters, so an in-place optimiser step mutates the snapshot too and
        a restore silently reinstates the already-stepped values. Verified on
        this vendored copy: LinearGaussian's ``_bias`` read -0.012977, then
        -0.017710 through the naive snapshot after a step, while a deepcopy
        still read -0.012977. Nothing raises.

        Written the natural way, the backtrack below would therefore ACCEPT
        every step it meant to reject — the exact failure the guard exists to
        prevent, reachable through the obvious spelling. ``copy.deepcopy`` is
        the supported idiom.
        """
        return copy.deepcopy(self.model.state_dict())

    def _restore(self, snapshot: dict) -> None:
        self.model.load_state_dict(snapshot)

    def fit(
        self,
        data: EpisodeData,
        *,
        max_iter: int = 30,
        tol: float = 1e-4,
        init: str = "proxy",
        # 6, raised from 3 by the 2026-08-21 binding audit: at depth 3 a
        # production-config CartPole fit exhausted MID-ASCENT on a step-size
        # cliff (best rejected step worsened the objective by 3.3e-2 while
        # accepted improvements were still 0.3-4%), and the deeper search
        # recovered ~98 further nats before reaching a genuine fixed point.
        # Depth 6 spans a 64x lr range; measured never-binding at the
        # production configuration once the fixed-point grant below landed.
        max_backtracks: int = 6,
        temperature: float | None = None,
        n_anneal: int | None = None,
        deterministic: bool = True,
        warm_start: bool = True,
        verbose: bool = False,
        **fit_kwargs,
    ) -> LatentClassFit:
        """Monotone-guarded generalized EM.

        Our M-step is a PARTIAL (SGD) maximisation, so the textbook EM
        monotonicity guarantee does not apply and decreases were observed. GEM
        requires only that the M-step INCREASE the objective, not maximise it —
        so a guard restores the guarantee outright rather than merely reporting
        its absence: if an M-step decreases the observed-data log-likelihood,
        revert to the pre-step parameters and retry with a halved learning rate.

        This also repairs the CONVERGENCE TEST, which was unsound without it:
        if the likelihood may decrease, ``|dLL| < tol`` cannot distinguish
        convergence from oscillating across a maximum or overshooting and
        returning. With decreases rejected, the sequence is non-decreasing by
        construction and a small increment means what it says again.

        Cost is one extra E-step evaluation per iteration in principle, and
        NONE in the happy path: the post-M-step E-step that checks the objective
        IS the next iteration's E-step, so it is computed once and reused.

        **DETERMINISTIC ANNEALING, APPLIED ONLY WHERE IT IS NEEDED.** The first
        E-step is run untempered; it both measures ``initial_saturation`` and
        decides whether to anneal. Where the E-step does not saturate the
        schedule is flat at ``tau = 1`` and this method behaves exactly as it did
        before, on the full iteration budget — which matters, because annealing
        spends iterations and a short-episode fit has no pathology to spend them
        on (it drove the T = 12 fixture into an exhausted backtrack budget).

        ``temperature`` is ``tau0`` and defaults to the mean episode length — a
        derivation, not a tuning choice (see ``_temperature_schedule``);
        ``n_anneal`` defaults to half the iteration budget. The realised schedule
        is stored on the fit and reported, never fitted against recovery. Pass
        ``temperature=1.0`` to force annealing off.
        """
        # DETERMINISTIC KERNELS, ON BY DEFAULT for every reported run
        # (decision 2026-08-19; measured cost ~5%). Not only an instrument for
        # equivalence tests: (a) it removes the whole "signal or noise" class
        # of question -- run-to-run gaps of 12-58 nats were measured on
        # IDENTICAL fits under default kernels; (b) it restores the bootstrap
        # null to measuring resampling variance rather than resampling + fit
        # noise; (c) the noise is not benign -- 5e-4 nats of evaluation noise
        # reaches the backtrack/convergence/stationarity decisions, each a
        # DISCRETE decision on a continuous quantity, and amplifies into
        # macroscopically different execution paths. Reported numbers should
        # not be path-dependent on kernel scheduling.
        #
        # FAIL-LOUD BY DESIGN: torch raises if an op with no deterministic
        # implementation enters the fit path (none does today). If a future
        # change trips it, that is this flag working, not a bug in it.
        # The mode travels on the fit and its estimates under C3.
        torch.use_deterministic_algorithms(deterministic)
        torch.manual_seed(self.seed)
        # Resolve R's TYPE from the data before anything is fitted, and rebuild
        # if it disagrees with the provisional model. Recorded on the fit, so no
        # likelihood is ever read without knowing which mechanism produced it.
        self._resolve_reward_type(data)
        resp = self._init_responsibilities(data, init)
        # The INITIAL M-step is explicitly COLD even when warm_start=True.
        # On a fresh estimator the mechanisms are unfitted and cold-build
        # anyway; on a REUSED estimator an implicit warm start would continue
        # from the PREVIOUS fit's parameters -- across-fit leakage, the exact
        # trap the bootstrap's symmetry rule forbids (a replicate warm-started
        # from the null-generating parameters). Cold here also means the
        # standardisation buffers are derived from THIS fit's weighted data,
        # then freeze for the rest of the EM loop, per the upstream contract.
        prior = self.m_step(data, resp, warm_start=False, **fit_kwargs)
        # The untempered first E-step measures the pathology. It is a PURE
        # DIAGNOSTIC and no control decision reads it.
        #
        # An earlier spelling gated the anneal on it, justified as "the statistic
        # is bimodal, so any cut in (0, 1) decides identically". That was
        # asserted from two observations and is false: the T = 500 fixture
        # measures 0.867, not ~1.0, so the cut-point would have been a tuned
        # constant deciding real cases -- precisely what A2 forbids. Instead the
        # anneal is a PREFIX OF EXTRA iterations rather than a slice of
        # ``max_iter``, which removes the conflict that made a gate look
        # necessary: a short-episode fit keeps its full tau = 1 budget and merely
        # pays a cheap warm-up, so nothing has to decide whether to skip it.
        # ``max_iter`` now means "iterations at tau = 1".
        resp, ll, obj, sat0 = self.e_step(data, prior, temperature=1.0)
        taus = self._temperature_schedule(data, max_iter, temperature, n_anneal)
        n_anneal_used = sum(1 for t in taus if t > 1.0)
        if n_anneal_used:
            resp, ll, obj, _ = self.e_step(data, prior, temperature=taus[0])
        total_iter = max_iter + n_anneal_used

        history: List[float] = [ll]
        base_lr = float(fit_kwargs.pop("lr", 1e-3))
        backtracks, exhausted, converged = 0, False, False
        anneal_exhaustions = 0
        small_steps = 0
        lr_scale = 1.0
        lr_reductions = 0
        epoch_escalations = 0
        stationary = False
        improvement_rel = None
        rejected_step_rel = None
        backtracks_per_iter: List[int] = []
        sat = sat0
        it = 0
        tau_prev = taus[0]
        for it in range(1, total_iter + 1):
            tau = taus[min(it, len(taus) - 1)]
            # WHEN tau MOVES, THE REFERENCE MUST MOVE WITH IT. The guard compares
            # the objective before and after the M-step, and at a temperature
            # change those are DIFFERENT FUNCTIONS -- comparing an objective at
            # tau = 500 against one at tau = 63 is not a comparison at all. Left
            # unhandled it read as a catastrophic decrease at the first anneal
            # step, exhausted the backtrack budget on iteration 1, and stopped
            # the fit before it ever reached tau = 1: the anneal present in the
            # code and absent in effect, which is the exact way this fix could
            # have failed silently. So the previous objective is re-evaluated at
            # the new tau, at the cost of one extra E-step per temperature change
            # and none at all once the schedule flattens.
            if tau != tau_prev:
                _, ll, obj, _ = self.e_step(data, prior, temperature=tau)
                tau_prev = tau
            snapshot = self._snapshot()
            tried = 0
            while True:
                # THE RETRY STRATEGY IS A FUNCTION OF THE ALGORITHM.
                #
                # Under warm_start (GEM, the default since NBN v0.15.0): the
                # M-step CONTINUES from the restored theta_old, so a smaller
                # learning rate is a genuinely gentler step from the current
                # point -- the premise of a backtracking line search holds
                # again, and a rejected step retries HALVED. This is what the
                # guard was originally written for.
                #
                # Under warm_start=False (restart-EM, kept as the labelled
                # fallback): fit_local rebuilds from a fresh random init, a
                # smaller lr is a WORSE FRESH FIT -- measured on the recovery
                # fixture, objective vs incumbent at lr x1 / x0.5 / x0.25 /
                # x0.0625 / x2.4e-4 = -26 / -110 / -1224 / -3148 / -4573,
                # monotone -- so there the retry doubles the EPOCHS instead
                # (more optimisation is what approaches the maximiser). That
                # measurement is the record of why the two regimes must not
                # share a retry rule; do not "simplify" them back together.
                retry_kwargs = dict(fit_kwargs)
                if warm_start:
                    step_lr = base_lr * (0.5**tried)
                else:
                    step_lr = base_lr
                    if tried:
                        retry_kwargs["epochs"] = int(
                            retry_kwargs.get("epochs", 30) * (2**tried)
                        )
                new_prior = self.m_step(
                    data, resp, lr=step_lr, warm_start=warm_start, **retry_kwargs
                )
                new_resp, new_ll, new_obj, sat = self.e_step(
                    data, new_prior, temperature=tau
                )
                # THE GUARD IS APPLIED TO THE OBJECTIVE THE M-STEP IS ACTUALLY
                # MAXIMISING. During annealing that is the TEMPERED objective;
                # judging those steps by the untempered likelihood would reject
                # exactly the moves the anneal exists to make, and the guard
                # would quietly undo the fix.
                if new_obj >= obj - 1e-9 or tried >= max_backtracks:
                    break
                self._restore(snapshot)
                tried += 1
                backtracks += 1
                if verbose:
                    print(
                        f"  EM {it:3d} tau={tau:8.2f} backtrack {tried}: "
                        f"{obj:.4f} -> {new_obj:.4f}"
                    )
            if new_obj < obj - 1e-9:
                self._restore(snapshot)
                if tau > 1.0:
                    # EXHAUSTION DURING THE ANNEAL IS NOT A REASON TO STOP. The
                    # anneal is a warm-up: failing to improve one temperature's
                    # SURROGATE says nothing about the likelihood, and stopping
                    # there leaves the fit maximising a smoothed objective it was
                    # only ever meant to pass through. Observed doing exactly
                    # that on the T = 10 fixture -- it exhausted mid-anneal,
                    # returned a tempered fit, and the do-effect came out at 1.22
                    # against a true 0.75, while every other diagnostic looked
                    # ordinary. Advance to the next temperature instead; the
                    # worst case is that the anneal contributes nothing and the
                    # tau = 1 phase runs exactly as it did before this change.
                    anneal_exhaustions += 1
                    continue
                # AT tau = 1, EXHAUSTION IS AMBIGUOUS -- "the fit is finished"
                # and "the step size is wrong" look identical from here.
                #
                # Under warm_start the extra-attempts loop below would be a
                # bitwise REPLAY: the step was rejected, so prior and resp are
                # unchanged, the parameters were restored, and with
                # deterministic kernels the next iteration would retry the
                # IDENTICAL halving sequence and reject it identically. Go
                # straight to the stationary/stuck classification instead --
                # re-derived for GEM, not inherited from restart-EM, where each
                # retry was a NEW stochastic refit with more epochs and so
                # could genuinely land somewhere else.
                if not warm_start:
                    lr_reductions += 1
                    if lr_reductions <= max_backtracks:
                        if verbose:
                            print(
                                f"  EM {it:3d} tau=1 retry {lr_reductions} with "
                                "more optimisation"
                            )
                        continue
                # STATIONARY vs STUCK -- two different stops, both legitimate,
                # and the distinction must be earned rather than assumed in
                # either direction.
                #
                # Having spent the persistent reductions, no improving step
                # exists at any step size this optimiser will try. If the recent
                # improvements were ALREADY sub-tolerance, that is the objective
                # being stationary at the optimiser's resolution -- a second
                # legitimate route to "finished", distinct from the tolerance
                # window. Measured on CartPole at production scale: last relative
                # delta 2.7e-5 against tol 1e-4, with no improving step
                # available.
                #
                # If instead the fit was still improving by percent when it ran
                # out of step sizes, it is STUCK, and calling that convergence
                # would be relabelling a failure to make a number look finished.
                # The test is on measured quantities only; no new constant.
                # HOW BADLY did the smallest step tried worsen the objective?
                # "No step worked" and "no step worked, and the best available
                # step worsens it by 1e-9" are different claims. At
                # lr_scale ~ 1e-4 the parameter movement is tiny, so a rejected
                # step may be NUMERICAL NOISE rather than a real worsening -- in
                # which case the fit is at a numerical fixed point and
                # ``stationary`` holds for a stronger reason than stated. A
                # rejection at ~1e-4 relative means the optimiser genuinely
                # cannot ascend, which is a weaker claim and must read as one.
                rejected_step_rel = float((obj - new_obj) / max(abs(obj), 1e-12))
                # TWO routes to stationary, both at resolution ``tol`` and
                # neither introducing a constant (A2): (a) the recent ACCEPTED
                # improvements were already sub-tolerance -- the original
                # test; (b) the best REJECTED step's worsening is itself below
                # tolerance -- the objective is flat at the optimiser's
                # resolution in both directions. (b) closes the
                # ABRUPT-CONVERGENCE GAP the binding audit measured: a fit
                # whose final accepted step was large (5e-4 rel) reached a
                # fixed point (rejected step flat to 3e-6) and was classified
                # STUCK because test (a) only reads the accepted tail. Without
                # (b), max_backtracks binds on exactly those fits and becomes
                # a tuning knob; with it, exhaustion at a flat point is what
                # finishing LOOKS like under a line search.
                if (improvement_rel is not None and improvement_rel < tol) or abs(
                    rejected_step_rel
                ) < tol:
                    stationary = True
                exhausted = True
                break
            # PERSIST THE LEARNING-RATE REDUCTION ACROSS ITERATIONS.
            #
            # ``tried`` used to reset every iteration, so the halvings were
            # DISCARDED and each iteration restarted at ``base_lr``. The guard
            # could therefore reject a bad step but never adapt: if the objective
            # needs ``base_lr / 16``, every iteration halves ``max_backtracks``
            # times, fails, and the fit stops. Measured at production scale --
            # Acrobot ended with 9 backtracks, exactly 3 iterations x 3 halvings,
            # each hitting the same wall, and it stopped the moment it reached
            # tau = 1 with the tail still 8x tolerance. CartPole did the same with
            # 3.
            #
            # The transition into tau = 1 is where it bites: the step size that
            # worked on the SMOOTHED objective overshoots on the sharper true
            # one, and a per-iteration reset guarantees it overshoots again.
            # Carrying the reduction forward is ordinary backtracking-line-search
            # behaviour and adds no constant. Monotone decrease, no re-growth:
            # re-inflating is what caused this.
            # NOTHING PERSISTS ACROSS ITERATIONS, in either regime --
            # re-derived under warm-start rather than inherited. The persistent
            # lr reduction was withdrawn under restart-EM (a smaller fresh fit
            # is a worse fresh fit); under GEM the halvings are a per-iteration
            # line search in the classical sense: the curvature that forced a
            # small step at iteration t says nothing binding about t+1, the
            # measured pathology that once motivated persistence (the same wall
            # every iteration) was a restart-EM artefact of comparing
            # independent refits, and a monotone-only persisted reduction would
            # trap the whole tail of the fit at the smallest step one sharp
            # region ever needed. Cost of the reset: at most max_backtracks
            # cheap warm M-steps in an iteration that needs the small step
            # again. ``epoch_escalations`` counts restart-mode retries only;
            # under GEM it stays 0 and ``backtracks`` carries the story.
            if not warm_start:
                epoch_escalations += tried
            backtracks_per_iter.append(tried)
            prior, resp = new_prior, new_resp
            improvement = new_obj - obj
            improvement_rel = improvement / max(abs(new_obj), 1e-12)
            ll, obj = new_ll, new_obj
            history.append(ll)
            if verbose:
                print(
                    f"  EM {it:3d} tau={tau:8.2f} ll={ll:.4f} obj={obj:.4f} "
                    f"sat={sat:.3f} prior={prior.tolist()}"
                )
            # Convergence is only meaningful once the objective has stopped
            # changing underneath it: during the anneal tau itself moves, so a
            # small increment says nothing.
            #
            # TWO CONSECUTIVE ITERATIONS, not one. The per-iteration improvement
            # is NOT monotone even under the guard -- measured on the converging
            # CartPole fit, the relative delta ran
            # 1.5e-2 -> 1.6e-4 -> 5.8e-4 -> 2.6e-4 -> 3.7e-5 against tol = 1e-4.
            # A single-iteration test can therefore fire mid-oscillation: had
            # that second delta come in at 9e-5 the fit would have stopped with
            # 5.8e-4 of real improvement still ahead, and every number
            # downstream would have carried a ``converged: True`` meaning
            # "paused" rather than "finished". It came within a factor of two of
            # doing exactly that. A window turns the flag from a snapshot into a
            # claim, and costs at most one iteration.
            if tau == 1.0 and improvement < tol * abs(obj):
                small_steps += 1
                if small_steps >= _CONVERGENCE_WINDOW:
                    converged = True
                    break
            else:
                small_steps = 0

        fit = LatentClassFit(
            model=self.model,
            prior=prior,
            responsibilities=resp,
            log_likelihood=history,
            n_iter=it,
            converged=converged,
            backtracks=backtracks,
            backtrack_exhausted=exhausted,
            temperatures=list(taus[: it + 1]),
            anneal_exhaustions=anneal_exhaustions,
            stationary=stationary,
            rejected_step_rel=rejected_step_rel,
            backtracks_per_iter=list(backtracks_per_iter),
            final_lr_scale=lr_scale,
            lr_reductions=lr_reductions,
            epoch_escalations=epoch_escalations,
            algorithm="gem" if warm_start else "restart-EM",
            deterministic=deterministic,
            tau1_budget_bound=(it >= total_iter and not converged and not exhausted),
            n_anneal=n_anneal_used,
            initial_saturation=sat0,
            final_saturation=sat,
            separation_per_step=self._separation_per_step(data),
            mechanism_degeneracy=self._mechanism_degeneracy(data),
            reward_mechanism=str(self.resolved_reward_mechanism),
        )
        return self._canonicalise(fit, data)

    def _init_responsibilities(self, data: EpisodeData, init: str) -> torch.Tensor:
        """Break the symmetry. EM on a symmetric init cannot move: every class
        gets identical responsibilities, the M-step fits identical mechanisms,
        and the fixed point is the one-class solution."""
        uniq, _ = data.blocks()
        n_ep = uniq.numel()
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        # EVERY tensor built here must be created on the DATA's device. The
        # random branch below always did (it ends in ``.to(device)``), which is
        # why the omission on the proxy branch went unnoticed -- and the proxy
        # branch is the production DEFAULT, so the default init path had simply
        # never been run on GPU. It raises rather than degrading, but only at
        # the first line that mixes the two.
        dev = data.state.device
        if init == "proxy" and self.proxy_names:
            # A covariate-free proxy is a direct (noisy) read of U, so its
            # episode mean orders the episodes far better than noise does --
            # fewer iterations and no dependence on a lucky seed.
            name = self.proxy_names[0]
            per_ep = data.episode_sum(
                torch.stack(
                    [
                        data.proxy[name].reshape(-1),
                        torch.ones(data.n, device=dev),
                    ],
                    dim=1,
                )
            )
            mean = (per_ep[:, 0] / per_ep[:, 1]).reshape(-1)
            qs = torch.quantile(
                mean, torch.linspace(0, 1, self.u_card + 1, device=dev)[1:-1]
            )
            hard = torch.bucketize(mean, qs)
            resp = torch.full((n_ep, self.u_card), 0.1, device=dev)
            resp[torch.arange(n_ep, device=dev), hard] = 0.9
            return resp / resp.sum(dim=1, keepdim=True)
        resp = torch.rand(n_ep, self.u_card, generator=g).to(dev)
        return resp / resp.sum(dim=1, keepdim=True)

    def _canonicalise(self, fit: LatentClassFit, data: EpisodeData) -> LatentClassFit:
        """Fix the label-swap symmetry so two runs are comparable.

        A mixture's likelihood is invariant to relabelling the classes, so EM
        returns an arbitrary permutation and any across-seed or across-arm
        comparison of per-class quantities is meaningless without a convention.
        Ours: order classes by their fitted mean reward, ascending. It is
        arbitrary but FIXED, which is all a canonicalisation has to be.
        """
        # Ordered by each class's RESPONSIBILITY-WEIGHTED mean reward, read off
        # the data rather than off samples from the fitted model: it is the same
        # ordering, needs no sampler round-trip, and cannot be perturbed by a
        # sampling shape or a mechanism that samples differently from how it
        # scores.
        w = data.broadcast(fit.responsibilities)  # (n, K)
        r = data.reward.reshape(-1, 1)
        means = ((w * r).sum(dim=0) / w.sum(dim=0).clamp_min(1e-12)).tolist()
        order = tuple(int(i) for i in np.argsort(means))
        if order == tuple(range(self.u_card)):
            return fit
        idx = torch.tensor(order, device=fit.responsibilities.device)
        # ``replace`` rather than a fresh construction, and that is not a style
        # preference. The explicit form listed the fields it knew about and
        # SILENTLY RESET THE REST to their defaults -- so a fit whose classes
        # happened to need swapping reported ``backtracks = 0`` and
        # ``backtrack_exhausted = False`` however badly it had struggled, and the
        # C3 labels lied on exactly half the runs at random. Every field added
        # here since would have been dropped the same way. ``replace`` copies by
        # construction, so the bug class is unreachable rather than fixed.
        return replace(
            fit,
            prior=fit.prior[idx],
            responsibilities=fit.responsibilities[:, idx],
            label_permutation=order,
        )

    # -------------------------------------------------------------- channels --
    def channel_log_probs(self, data: EpisodeData, u: torch.Tensor) -> Dict[str, float]:
        """Mean per-row log-density, split by channel — L5's diagnostic.

        Taken from ``log_prob(per_node=True)`` rather than re-derived, so the
        split cannot drift from the model's own decomposition. It raises on a
        missing node, which is what stops a silently marginalised variable
        masquerading as a fitted one.
        """
        with torch.no_grad():
            per_node = self.model.log_prob(self._frame(data, u), per_node=True)
        return {
            "action_channel": float(per_node["A"].mean()),
            "reward_channel": float(per_node["R"].mean()),
            **{p: float(per_node[p].mean()) for p in self.proxy_names},
        }

    # ---------------------------------------------------------- interventions --
    def interventional_reward(
        self,
        state: torch.Tensor,
        action: int,
        fit: "LatentClassFit",
        *,
        n_samples: int = 2048,
    ) -> "Estimate":
        """``E[R | do(A = a), s]``, marginalising ``U`` over the EXOGENOUS prior.

        **Path choice: ``sample(do=)``, looped over intervention values, because
        this value feeds a loss.** The discriminator between the two
        interventional APIs is GRADIENTS, not samples-versus-posterior:
        ``query``/``query_batch`` are non-differentiable by design and return
        inference-mode tensors, which are worse than merely detached ones
        because they RAISE if later used in a differentiable op. Routing this
        computation through ``query_batch`` for its speed would not raise here —
        it would hand back a value with no gradient, presenting downstream as a
        model that will not train. See ``interventional_sweep`` for the
        read-only counterpart, which is two orders of magnitude faster and is
        the right tool for L4's bound evaluation.

        The ~1.5 ms per intervention value is negligible at this call site, so
        the loop costs nothing that matters.

        Two things this must not do, both of which produce plausible wrong
        numbers rather than errors:

        * marginalise over ``P(U | s, a)`` instead of ``P(U)``. The behaviour
          policy tilts the former, and it is exactly that tilt the intervention
          is supposed to remove. This is the same asymmetry that makes D-B's q2
          rest on a strictly stronger assumption than its q1.
        * route through ``intervene()``, which deep-copies and severs the
          caller's graph (N1).

        Shape contract, learned the hard way: ``do`` values are ``[1, D]``, NOT
        0-d scalars and NOT ``(n,)`` vectors — the do-dispatch builds a
        deterministic mechanism that indexes a batch axis, so a 0-d value fails
        inside the sampler rather than at the call. Evidence is expanded to
        ``n`` rows because the sampler reshapes each parent to ``(n, -1)``.
        """
        prior = fit.prior
        state = state.reshape(1, -1).to(self.model.device)
        evidence = {"S": state.expand(n_samples, -1)}
        total = torch.zeros((), device=state.device)
        for k in range(self.u_card):
            out = self.model.sample(
                n_samples,
                evidence=evidence,
                do={
                    "A": torch.tensor([[int(action)]], device=state.device),
                    "U": torch.tensor([[k]], device=state.device),
                },
            )
            total = total + prior[k] * self._decode_reward(out["R"]).reshape(-1).mean()
        # C3: the number leaves carrying the conditions it was produced under.
        return fit.estimate(total)

    def interventional_sweep(
        self,
        states: torch.Tensor,
        actions: Sequence[int],
        fit: "LatentClassFit",
    ) -> "Estimate":
        """READ-ONLY per-row interventional sweep — ``(n_rows,)`` of ``E[R|do(a),s]``.

        **Path choice: ``query_batch(do=)``, because nothing here feeds a loss.**
        It takes a per-row intervention vector and returns the whole sweep in one
        call — measured upstream at 0.3 ms batched against 37.7 ms looped for 256
        interventions, bit-identical to the loop. That is the right trade for
        L4's bound evaluation and any other read-only interventional quantity.

        **The result is NOT differentiable, deliberately.** Do not feed it to an
        optimiser: ``query_batch`` runs under inference mode, so the returned
        tensor raises if used in a differentiable op rather than silently
        producing a zero gradient. If you need a gradient here, you want
        ``interventional_reward`` instead — the choice is per call site.
        """
        prior = fit.prior
        states = states.to(self.model.device)
        n_rows = states.shape[0]
        a = torch.as_tensor(list(actions), device=states.device).reshape(-1, 1)
        if a.shape[0] != n_rows:
            raise ValueError(
                f"{a.shape[0]} actions against {n_rows} states; the sweep is "
                "per-row, so they must correspond"
            )
        total = torch.zeros(n_rows, device=states.device)
        for k in range(self.u_card):
            u_col = torch.full((n_rows,), k, device=states.device, dtype=torch.long)
            out = self.model.query_batch(
                ["R"], evidence={"S": states}, do={"A": a.reshape(-1), "U": u_col}
            )
            total = total + prior[k].item() * _posterior_mean(
                out, value_fn=self._decode_reward if self._reward_is_discrete else None
            )
        return fit.estimate(total)

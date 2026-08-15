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
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

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

__all__ = ["EpisodeData", "Estimate", "LatentClassFit", "LatentClassEstimator"]

# Mechanisms whose weighted fit is exact. KDE is excluded deliberately (N3).
_WEIGHTED_SAFE = (MDNMechanism, LinearGaussianMechanism, NeuralCategoricalMechanism)


def _posterior_mean(out) -> torch.Tensor:
    """Collapse ``query_batch``'s return into a per-row posterior mean.

    For a DISCRETE target it hands back ``[B, K]`` directly. For a CONTINUOUS
    one — which ``R`` is — the engine returns the likelihood-weighting particle
    representation ``(weights [B, N], samples [B, N, D])`` instead, and the
    posterior mean is the weighted average over particles. Documented as
    returning a tensor, so the tuple is worth naming here rather than
    rediscovering at each call site.
    """
    if not isinstance(out, tuple):
        return out.reshape(out.shape[0], -1).mean(dim=1)
    w, samples = out
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

    def label(self) -> str:
        bits = [f"sep={self.separability:.2f}"]
        if not self.monotone:
            bits.append("NON-MONOTONE-EM")
        if not self.converged:
            bits.append("NOT-CONVERGED")
        if self.backtrack_exhausted:
            bits.append("BACKTRACK-EXHAUSTED")
        if self.backtracks:
            bits.append(f"backtracks={self.backtracks}")
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
        )

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
        return all(
            b >= a - 1e-6 for a, b in zip(self.log_likelihood, self.log_likelihood[1:])
        )

    @property
    def final_ll(self) -> float:
        return self.log_likelihood[-1] if self.log_likelihood else float("nan")

    def hard_assignment(self) -> torch.Tensor:
        return self.responsibilities.argmax(dim=1)

    def separability(self) -> float:
        """Mean max-responsibility: 1/K is chance, 1.0 is perfect separation.

        Reported alongside every estimate because a latent-class fit that has
        not separated is not an error — it is a *weak-proxy regime*, and the
        number that distinguishes the two has to be visible rather than inferred
        from a value that looks plausible either way.
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
        reward_mechanism: str = "mdn",
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
        self.model = self._build()

    # ---------------------------------------------------------------- build --
    def _reward_mech(self):
        if self._reward_mechanism == "mdn":
            return MDNMechanism(num_components=3, hidden=(64, 64))
        if self._reward_mechanism == "linear_gaussian":
            return LinearGaussianMechanism()
        raise ValueError(
            f"unknown reward mechanism {self._reward_mechanism!r}; the EM path "
            "admits only mechanisms whose weighted fit is exact (N3) — KDE's "
            "bandwidth rule is unweighted and would bias strata toward each "
            "other, costing L5 detection power in the quiet direction"
        )

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
            "R": ContinuousVariable("R", dim=1),
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
            model.set_mechanism(p, self._reward_mech())
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
            "R": data.reward.reshape(-1, 1),
            "U": u.reshape(-1),
        }
        for name, val in data.proxy.items():
            frame[name] = val.reshape(-1, 1)
        return frame

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

    def e_step(self, data: EpisodeData, prior: torch.Tensor) -> tuple:
        """Episode responsibilities and the observed-data log-likelihood."""
        ll = self._episode_log_liks(data) + torch.log(prior).reshape(1, -1)
        # log-sum-exp per episode: the observed-data likelihood, and the
        # normaliser for the posterior.
        marginal = torch.logsumexp(ll, dim=1)  # (E,)
        resp = torch.exp(ll - marginal.reshape(-1, 1))
        return resp, float(marginal.sum())

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
        weights_per_row = data.broadcast(resp)  # (n, K)
        frames, weights = [], []
        for k in range(self.u_card):
            u_k = torch.full((data.n,), k, dtype=torch.long, device=data.state.device)
            frames.append(self._frame(data, u_k))
            weights.append(weights_per_row[:, k])
        stacked = {key: torch.cat([f[key] for f in frames], dim=0) for key in frames[0]}
        w = torch.cat(weights, dim=0)
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
        max_backtracks: int = 3,
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
        """
        torch.manual_seed(self.seed)
        resp = self._init_responsibilities(data, init)
        prior = self.m_step(data, resp, **fit_kwargs)
        resp, ll = self.e_step(data, prior)

        history: List[float] = [ll]
        base_lr = float(fit_kwargs.pop("lr", 1e-3))
        backtracks, exhausted, converged = 0, False, False
        it = 0
        for it in range(1, max_iter + 1):
            snapshot = self._snapshot()
            tried = 0
            while True:
                new_prior = self.m_step(
                    data, resp, lr=base_lr * (0.5**tried), **fit_kwargs
                )
                new_resp, new_ll = self.e_step(data, new_prior)
                if new_ll >= ll - 1e-9 or tried >= max_backtracks:
                    break
                # The M-step overshot. Revert and take a smaller one.
                self._restore(snapshot)
                tried += 1
                backtracks += 1
                if verbose:
                    print(f"  EM {it:3d}  backtrack {tried}: {ll:.4f} -> {new_ll:.4f}")
            if new_ll < ll - 1e-9:
                # Budget spent and still decreasing: stop on the last good
                # parameters rather than proceed on a decrease.
                self._restore(snapshot)
                exhausted = True
                break
            prior, resp = new_prior, new_resp
            improvement = new_ll - ll
            ll = new_ll
            history.append(ll)
            if verbose:
                print(f"  EM {it:3d}  ll={ll:.4f}  prior={prior.tolist()}")
            if improvement < tol * abs(ll):
                converged = True
                break

        fit = LatentClassFit(
            model=self.model,
            prior=prior,
            responsibilities=resp,
            log_likelihood=history,
            n_iter=it,
            converged=converged,
            backtracks=backtracks,
            backtrack_exhausted=exhausted,
        )
        return self._canonicalise(fit, data)

    def _init_responsibilities(self, data: EpisodeData, init: str) -> torch.Tensor:
        """Break the symmetry. EM on a symmetric init cannot move: every class
        gets identical responsibilities, the M-step fits identical mechanisms,
        and the fixed point is the one-class solution."""
        uniq, _ = data.blocks()
        n_ep = uniq.numel()
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        if init == "proxy" and self.proxy_names:
            # A covariate-free proxy is a direct (noisy) read of U, so its
            # episode mean orders the episodes far better than noise does --
            # fewer iterations and no dependence on a lucky seed.
            name = self.proxy_names[0]
            per_ep = data.episode_sum(
                torch.stack([data.proxy[name].reshape(-1), torch.ones(data.n)], dim=1)
            )
            mean = (per_ep[:, 0] / per_ep[:, 1]).reshape(-1)
            qs = torch.quantile(mean, torch.linspace(0, 1, self.u_card + 1)[1:-1])
            hard = torch.bucketize(mean, qs)
            resp = torch.full((n_ep, self.u_card), 0.1)
            resp[torch.arange(n_ep), hard] = 0.9
            return resp / resp.sum(dim=1, keepdim=True)
        resp = torch.rand(n_ep, self.u_card, generator=g).to(data.state.device)
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
        return LatentClassFit(
            model=fit.model,
            prior=fit.prior[idx],
            responsibilities=fit.responsibilities[:, idx],
            log_likelihood=fit.log_likelihood,
            n_iter=fit.n_iter,
            converged=fit.converged,
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
            total = total + prior[k] * out["R"].reshape(-1).mean()
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
            total = total + prior[k].item() * _posterior_mean(out)
        return fit.estimate(total)

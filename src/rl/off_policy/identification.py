"""Identification strategies for the causal critic (build-order item 2).

A ``CausalCritic`` = base_estimator × ``IdentificationStrategy``. The base
off-policy learner (DQN/CQL/IQL/BCQ) owns the loss skeleton; the strategy owns
ONE hook — ``critic_value(net, x, batch)`` — that routes the learner's
u-conditionable critic evaluations. Deconfounding is then a transform of *what
the critic regresses toward*, not a new learner per cell.

  * ``Observational`` — ``critic_value = net(x)``: a literal pass-through, so a
    learner with the default strategy is byte-identical to the pre-strategy code.
    The floor.
  * ``OracleU`` — ``critic_value = net.q_su(x, batch["confounder_u"])``: conditions
    on the REALIZED latent U (read from the dataset). The fenced oracle REFERENCE
    (``requires_confounder_u = True``), never a reported method. Deploy stays
    ``net(x)`` because the net is ``UMarginalizedQ`` whose ``forward`` returns the
    backdoor-adjusted ``Q_adj`` — so ``act``/``_apparent_q`` need no hook.
  * ``Proximal`` — STUB (build-order item 2 scaffolding). ``needs_episode_grouping
    = True`` (the per-episode latent-class posterior consumes episode-grouped
    sequences), ``requires_confounder_u = False`` (it INFERS U, never reads it).
    ``critic_value`` degrades to ``net(x)`` (no estimator yet); ``statistical_
    diagnostic`` computes the two real gate surrogates from a placeholder split;
    ``target`` degrades to a bound. NO EM, no Q(s,a,u) fit — PR-2 owns the math.
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import torch


@runtime_checkable
class IdentificationStrategy(Protocol):
    """How a critic identifies ``E[R | do(a), s]`` from the five base keys."""

    requires_confounder_u: bool  # reads the realized U (OracleU only — fenced)
    needs_episode_grouping: bool  # consumes episode-grouped sequences (Proximal)

    def critic_value(
        self, net, x: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """The (B, A) critic value the learner regresses through at a learn-time
        site. Observational = ``net(x)``; OracleU = ``net.q_su(x, u)``."""

    def graphical_precondition(self, graph=None) -> bool:
        """Does the assumed graph admit point-ID via this strategy? (decided once)."""

    def statistical_diagnostic(self, batch_or_fit):
        """Finite-data trust score(s) for the identifying functional (per fit)."""


class Observational:
    """No adjustment — the floor. ``critic_value`` is a literal pass-through, so a
    learner defaulting to this strategy is byte-identical to the pre-strategy code."""

    requires_confounder_u = False
    needs_episode_grouping = False

    def critic_value(self, net, x, batch):
        return net(x)

    def graphical_precondition(self, graph=None) -> bool:
        return True

    def statistical_diagnostic(self, batch_or_fit) -> float:
        return 1.0  # no identifying functional to distrust


class OracleU:
    """Backdoor-adjusted oracle reference: conditions the critic on the REALIZED
    per-transition U (``batch["confounder_u"]``). ``requires_confounder_u=True`` —
    the only strategy that reads U; a fenced ceiling, never a reported method."""

    requires_confounder_u = True
    needs_episode_grouping = False
    is_oracle_u = True

    def critic_value(self, net, x, batch):
        # KeyError("confounder_u") if the dataset was not loaded U-aware — the same
        # clear failure the standalone OracleU* learners raised.
        return net.q_su(x, batch["confounder_u"])

    def oracle_anchor_q(self, net, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``Q(s, a_data, u=0)`` over the batch — the u=0 validation anchor the
        runner's value-trace writer records (delegated from the agent)."""
        with torch.no_grad():
            obs = batch["obs"]
            actions = batch["actions"].long()
            q0 = net.q_at(obs, 0.0)
            return q0.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    def graphical_precondition(self, graph=None) -> bool:
        return True

    def statistical_diagnostic(self, batch_or_fit) -> float:
        return 1.0  # identified by construction (it reads the truth)


class Proximal:
    """Latent-class deconfounder via stochastic-EM through the OracleU hook (PR-2b).

    Proximal IS the OracleU critic hook with an INFERRED-and-sampled confounder
    instead of a read one: ``critic_value`` routes to ``net.q_su(x, u)`` exactly
    as OracleU, but ``batch["confounder_u"]`` is written ONLY by the proximal
    agent's E-step sampler (``ProximalEM`` in ``offline/proximal.py``) — a per-step
    Bernoulli draw from the per-episode posterior ``q(U|tau)`` — NEVER the dataset's
    realized U. Hence ``requires_confounder_u = False`` (the five-keys invariant:
    the loader runs ``load_u=False``; proximal never sees the truth). The episode
    posterior needs whole episodes -> ``needs_episode_grouping = True``.

    ``statistical_diagnostic`` computes the two gate surrogates; under the
    known-cell assumption it is LOGGED, not gating (that routing is Stage C)."""

    requires_confounder_u = False  # infers U via the E-step; never reads the truth
    needs_episode_grouping = True

    def critic_value(self, net, x, batch):
        # Identical to OracleU's hook; the difference is the PROVENANCE of
        # batch["confounder_u"] (E-step sample, not the loaded realized U).
        return net.q_su(x, batch["confounder_u"])

    def graphical_precondition(self, graph=None) -> bool:
        return True  # "sole per-episode latent" (asserted for Cell 7)

    def statistical_diagnostic(self, seq_batch: Dict[str, torch.Tensor]) -> dict:
        """Two surrogates on an episode-grouped ``(B, T, *)`` batch:

        1. ``separability`` — median over sequences of ``max_u q(U|seq)`` (gates
           latent RECOVERY). Placeholder posterior: a logistic on the per-sequence
           reward-sum vs the batch median (PR-2 replaces with the real posterior).
        2. ``action_overlap`` — min over (placeholder-stratum, action) of action
           support (gates the deconfounded ARGMAX; thin coverage there is the
           failure the additive-offset cancellation hides).
        """
        rewards = seq_batch["rewards"]  # (B, T)
        actions = seq_batch["actions"].long()  # (B, T)
        reward_sum = rewards.sum(dim=1)  # (B,)
        med = reward_sum.median()
        # Placeholder posterior q(U=1|seq): logistic on centered reward-sum.
        scale = reward_sum.std().clamp_min(1e-6)
        p1 = torch.sigmoid((reward_sum - med) / scale)
        conf = torch.maximum(p1, 1.0 - p1)
        separability = float(conf.median().item())
        # Placeholder strata = reward-sum split at the median; per-stratum action
        # support over the action set; min over (stratum, action).
        n_actions = int(actions.max().item()) + 1
        hi = reward_sum >= med
        overlaps = []
        for stratum in (hi, ~hi):
            if not bool(stratum.any()):
                overlaps.append(0.0)
                continue
            acts = actions[stratum].reshape(-1)
            counts = torch.bincount(acts, minlength=n_actions).float()
            overlaps.append(float((counts / counts.sum()).min().item()))
        action_overlap = min(overlaps)
        return {"separability": separability, "action_overlap": action_overlap}

    def target(self, seq_batch=None) -> str:
        """Degrade to a partial-identification bound — the stub never emits a
        point estimate (no estimator). PR-2 returns the fitted Q_adj when the
        gate passes."""
        return "bound"


class SensitivityBounds:
    """Kallus-Zhou (2020, arXiv:2002.04518) marginal sensitivity model (MSM).

    PESSIMISTIC REWARD REWEIGHTING with a scalar confounding bound ``Γ >= 1``.
    Unlike ``Proximal`` (which needs proxy variables to POINT-identify the causal
    Q), this needs only ``Γ`` and produces a robust ``Q_lower`` valid against any
    confounding of strength up to ``Γ``. The rung between the confounded floor and
    the proxy-precise Proximal: ``Observational -> SensitivityBounds (safe,
    Γ-conservative) -> Proximal (precise) -> OracleU (ceiling)``.

    KEY ARCHITECTURAL FACT (Gate-A finding): the reweighting adjusts the REWARD in
    the Bellman target, which the ``critic_value`` hook CANNOT touch (it returns the
    (B,A) Q-matrix; the learner adds ``rewards`` raw). So this strategy's
    ``critic_value`` is a literal PASS-THROUGH (identical to ``Observational``); the
    pessimism is injected UPSTREAM by ``_SensitivityReweighter`` (offline/sensitivity
    .py), which wraps ``agent.learn`` and mutates ``batch["rewards"]`` before the
    byte-frozen base backup — the ``ProximalEM._install_*`` seam, minus the EM and
    any ``q_su``/network change.

    Five-keys invariant: ``requires_confounder_u = False`` — uses only (S,A,R,S'),
    never reads the realized U. ``needs_episode_grouping = False`` — per-transition,
    so it rides the byte-frozen FLAT offline path.

    KNOWN LIMITATION (documented for paper §5.3): the MSM bounds the ``U -> A``
    action-confounding path but NOT direct ``U -> R`` reward-shift confounding (U
    shifting the reward distribution independently of A). Our confounder is exactly
    ``r += c_r*U`` — a direct ``U -> R`` shift — so SensitivityBounds is SAFE (never
    worse than observational in the pessimistic direction) but does NOT close the gap
    toward OracleU the way Proximal does. That is not a failure: it is an honest
    empirical demonstration of the MSM's theoretical scope — sensitivity bounds
    protect against action-confounding but not reward-confounding; when U enters the
    reward directly, proxy-based methods (Proximal) are necessary."""

    requires_confounder_u = False  # five-keys: only (S,A,R,S'); never reads U
    needs_episode_grouping = False  # per-transition; rides the flat offline path

    def __init__(self, gamma_sensitivity: float = 2.0) -> None:
        if gamma_sensitivity < 1.0:
            raise ValueError(
                f"gamma_sensitivity (Γ) must be >= 1.0 (Γ=1 = no confounding = "
                f"Observational); got {gamma_sensitivity}."
            )
        self.gamma_s = float(gamma_sensitivity)

    def critic_value(self, net, x, batch):
        # Literal pass-through — identical to Observational. The pessimism lives in
        # the reward-reweighting wrapper, NOT here (critic_value can't touch reward).
        return net(x)

    def graphical_precondition(self, graph=None) -> bool:
        return True  # MSM assumes confounding is BOUNDED by Γ, not absent

    def statistical_diagnostic(self, batch_or_fit) -> dict:
        return {"gamma_sensitivity": self.gamma_s}

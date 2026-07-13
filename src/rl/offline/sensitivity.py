"""Sensitivity-bounds deconfounder — Kallus-Zhou (2020) marginal sensitivity model.

Pessimistic REWARD REWEIGHTING with a scalar confounding bound ``Γ >= 1``. The
identification is the ``SensitivityBounds`` strategy (``identification.py``); the
only sensitivity-specific machinery is here:

  * ``_SensitivityReweighter`` — per-batch, before the base backup, it computes the
    residual ``δ = r - Q(s,a)`` under the agent's CURRENT critic, assigns each
    transition a worst-case MSM weight (``1/Γ`` if ``δ>0``: a reward BETTER than
    expected may be confounder-driven luck, so discount it; ``Γ`` if ``δ<0``: a
    reward WORSE than expected is the true causal signal, so amplify it), normalizes
    the weights to mean 1, and writes ``r_adj = r * w`` back into ``batch["rewards"]``.
    The base learner's byte-frozen ``target = r_adj + γ·max Q(s')`` then yields a
    pessimistic ``Q_lower``.
  * ``_install_sensitivity`` — wraps ``agent.learn`` with the reweighter (the
    ``ProximalEM._install_*`` seam), capturing ``agent.q_network`` so the wrapper can
    evaluate the current ``Q(s,a)``. The base ``learn`` and the nets are unchanged.

RESIDUAL-BASED (no behavioral-cloning estimator): the reweighting uses the agent's
own Q to decide which rewards to trust, so there is NO BC pre-training and NO
``fit_from_buffer`` runner seam. (BC-based importance weights — the original KZ
formulation — are a documented follow-up extension.)

Limit behavior:
  * ``Γ = 1.0`` -> ``w ≡ 1`` -> EXACT ``Observational`` (the wrapper early-returns
    ``base_learn(batch)`` with no reward tensor allocated, so it is BYTE-IDENTICAL
    to the plain floor — the Γ=1 gate is bitwise, not merely numeric).
  * ``Γ -> large`` -> mass concentrates on the most-negative-residual (lowest-reward)
    transitions -> ``Q_lower`` collapses toward the low tail of R (conservative).

LEVERAGE PROPERTY (material to reading results). Because the weights are
MEAN-NORMALIZED within the batch, the reweighting has bite ONLY when the residuals
``δ = r - Q(s,a)`` STRADDLE ZERO. If every δ has the same sign — all-positive
(untrained Q below every reward) or all-negative (a converged critic whose Q uniformly
exceeds the one-step reward, e.g. dense +1/step CartPole) — then ``w`` is a single
constant and ``w/mean(w) ≡ 1``, so ``r_adj = r`` exactly: a NO-OP for every Γ.
Empirically the adjusted-reward mean is monotone-decreasing in Γ on mixed-residual
batches (0.50 -> 0.32 -> 0.24 -> 0.20 as Γ: 1->2->4->100) and exactly flat on
one-signed batches. This is a FEATURE aligned with the confounding structure: the
``r += c_r*U`` bonus makes SOME transitions high-surprise and others not, so the
residuals straddle zero exactly where the confounder injected heterogeneity, and the
pessimism localizes there. It also explains why, on the return-preserving Cell-7
CartPole arm (per-episode nuisance U, no return gap by construction), SensitivityBounds
is a near-no-op on return — the honest, expected outcome, not a defect.

Five-keys invariant: uses ONLY (S, A, R, S') — never reads the realized U. Composes
with all four base learners (DQN/CQL/IQL/BCQ) unchanged, since each reads
``batch["rewards"]`` identically. IQL is a PARTIAL MSM application: the reweighted
reward flows into IQL's Q-target (``q_target = r + γ·next_v``) so the Q-side is
pessimistic, but the expectile-V / advantage are computed plain, so the V-side is
NOT — an honestly-reportable known approximation.

KNOWN LIMITATION (paper §5.3): the MSM bounds ``U -> A`` action-confounding but not
direct ``U -> R`` reward-shift confounding. Our confounder is ``r += c_r*U`` (direct
``U -> R``), so SensitivityBounds is safe but does not close the gap toward OracleU —
the honest demonstration that proxy-based Proximal is needed for reward-confounding.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.rl.models.backbone import build_trunk
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.identification import SensitivityBounds
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.bcq import build_bcq
from src.rl.offline.cql import build_cql
from src.rl.offline.dqn import build_offline_dqn
from src.rl.offline.iql import build_iql
from src.rl.offline.oracle_u import _recurrent_trunk_kwargs


class _SensitivityReweighter:
    """Per-batch pessimistic reward reweighting under the MSM with parameter Γ.

    Attached to a base learner by ``_install_sensitivity``; the wrapped ``learn``
    calls ``reweight`` then delegates to the byte-frozen base ``learn``."""

    def __init__(self, q_network, gamma_sensitivity: float) -> None:
        self.q_network = q_network
        self.gamma_s = float(gamma_sensitivity)
        # Recurrent base (Cell 8): batch tensors are (B, T, ·) and q_network returns
        # (q_all, state). Set by _install_sensitivity from the wrapped agent. Default
        # False keeps the flat-MLP path (the primary scope).
        self.is_recurrent = False

    def _q_sa(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The agent's current ``Q(s, a)`` at the data action, detached (the weights
        are TARGETS, not learned — no gradient flows through them)."""
        obs = batch["obs"]
        actions = batch["actions"].long()
        with torch.no_grad():
            q_all = self.q_network(obs)
            if self.is_recurrent:
                # (q_all, state); q_all is (B, T, A), actions (B, T).
                q_all = q_all[0]
            return q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def reweight(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return a shallow-copied batch with ``rewards`` replaced by the pessimistic
        ``r_adj = r * w`` (mean-normalized worst-case MSM weights)."""
        r = batch["rewards"]
        delta = r - self._q_sa(batch)  # residual vs the current critic
        # Worst-case weight over the Γ-ball: downweight good-surprise rewards
        # (possible U-driven luck), upweight bad-surprise rewards (causal signal).
        w = torch.where(
            delta > 0,
            torch.full_like(r, 1.0 / self.gamma_s),
            torch.full_like(r, self.gamma_s),
        )
        w = w / w.mean().clamp_min(1e-8)  # normalize to mean 1
        return {**batch, "rewards": r * w}

    def learn(self, batch: Dict[str, torch.Tensor], base_learn):
        # Γ=1 -> no reweighting -> EXACT Observational: early-return the untouched
        # batch so the floor is byte-identical (no reward tensor allocated).
        if self.gamma_s == 1.0:
            return base_learn(batch)
        return base_learn(self.reweight(batch))


def _install_sensitivity(agent, reweighter: _SensitivityReweighter) -> None:
    """Wrap the base learner's ``learn`` with the reward reweighter (the
    ``_install_proximal_em`` seam). The base ``learn`` and nets are unchanged; the
    wrapper only mutates ``batch["rewards"]`` upstream of the byte-frozen backup."""
    base_learn = agent.learn
    agent.learn = lambda batch: reweighter.learn(batch, base_learn)
    agent._sensitivity_reweighter = reweighter  # keep a ref alive + testable
    reweighter.is_recurrent = bool(getattr(agent, "is_recurrent", False))


def _gamma(kwargs) -> float:
    """Γ from the per-algo ``networks:`` map (threaded via ``network_kwargs`` ->
    builder kwargs). Default 2.0 (a bare ``*_sensitivity`` with no networks block)."""
    return float(kwargs.get("gamma_sensitivity", 2.0))


def _build(plain_builder, **kwargs):
    """Build a plain floor learner with a ``SensitivityBounds`` strategy injected,
    then install the reward reweighter. Reusing the floor builder VERBATIM (same
    nets, same RNG order) is what makes ``*_sensitivity`` @ Γ=1 byte-identical to the
    bare floor algo."""
    gamma_s = _gamma(kwargs)
    kwargs = {**kwargs, "strategy": SensitivityBounds(gamma_s)}
    policy, agent = plain_builder(**kwargs)
    _install_sensitivity(agent, _SensitivityReweighter(agent.q_network, gamma_s))
    return policy, agent


def build_sensitivity_dqn(**kwargs):
    return _build(build_offline_dqn, **kwargs)


def build_sensitivity_cql(**kwargs):
    return _build(build_cql, **kwargs)


def build_sensitivity_iql(**kwargs):
    # IQL: PARTIAL MSM — reweighted reward flows into the Q-target (pessimistic
    # Q-side) but not the expectile-V/advantage (plain V-side). Documented approx.
    return _build(build_iql, **kwargs)


def build_sensitivity_bcq(**kwargs):
    return _build(build_bcq, **kwargs)


def build_sensitivity_dqn_recurrent(**kwargs):
    """Cell 8 (POMDP × confounded) recurrent SensitivityBounds — DQN base only.

    Mirrors ``build_offline_dqn_recurrent`` (the recurrent OBSERVATIONAL floor): a
    PLAIN recurrent trunk (``build_trunk`` over ``critic_network``, NO ``q_su`` — so
    ``DQN._learn_recurrent`` takes its byte-frozen else-branch and reads the reweighted
    ``batch["rewards"]`` directly), but with the ``SensitivityBounds(Γ)`` strategy and
    the reward reweighter installed. ``_install_sensitivity`` auto-detects the recurrent
    agent (``initial_state`` -> ``DQN.is_recurrent``) and flips the reweighter's
    ``is_recurrent`` branch (which unpacks the ``(q_all, state)`` tuple and preserves the
    ``(B, T)`` reward shape) — so the reweighter itself is unchanged from the flat path.

    DQN base only, matching the recurrent proximal/oracle_u builders: CQL/IQL/BCQ have
    no recurrent learn path (their ``learn`` assumes flat ``(B, A)``), so recurrent CQL
    sensitivity is out of scope. Registered WITHOUT ``_offpolicy_recurrent_guard`` (the
    whole point is to ACCEPT ``critic_network=lstm``); the config's ``critic: lstm``
    routes the run to the offline-grouped SequenceReplayBuffer path.

    Byte-identity @ Γ=1: same ``build_trunk`` nets in the same RNG order as the floor,
    the strategy is inert in ``_learn_recurrent`` (``critic_value`` is never called there),
    and the Γ=1 reweighter early-returns the untouched batch -> identical to the floor.
    """
    if kwargs.get("action_type", "discrete") != "discrete":
        raise ValueError(
            "offline_dqn_recurrent_sensitivity is discrete-only (the Cell-8 recurrent "
            "arm); use a continuous offline algo for continuous action spaces."
        )
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    # Recurrent-only, exactly like the floor: mlp would defeat the purpose (that is
    # what offline_dqn_sensitivity is for). Default lstm so a bare recurrent name
    # (no networks block) is still recurrent; networks:{critic: gru} overrides.
    network = kwargs.get("critic_network", "lstm") or "lstm"
    if network == "mlp":
        raise ValueError(
            "offline_dqn_recurrent_sensitivity requires a recurrent critic_network "
            "(lstm/gru/rnn); for an MLP sensitivity critic use offline_dqn_sensitivity."
        )
    tk = _recurrent_trunk_kwargs(kwargs)
    gamma_s = _gamma(kwargs)
    q_net = build_trunk(network, obs_shape, obs_dim, action_dim, **tk)
    target_net = build_trunk(network, obs_shape, obs_dim, action_dim, **tk)
    agent = DQN(
        q_net.to(device),
        target_net.to(device),
        ReplayBuffer(capacity=1_000_000, device=device),
        device=device,
        strategy=SensitivityBounds(gamma_s),
    )
    _install_sensitivity(agent, _SensitivityReweighter(agent.q_network, gamma_s))
    return q_net.to(device), agent

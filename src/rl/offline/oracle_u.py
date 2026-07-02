"""Backdoor-adjusted oracle-U critic — the identification CEILING for the
confounded discrete Cell-7 offline arm (offline_dqn / bcq / cql / iql).

The confounded datasets carry a per-episode latent ``U`` that drives BOTH the
behavior action and a reward bonus (``ConfoundedCollectionWrapper``); ``U`` never
enters the observation, so a naive critic attributes the ``c_r * U`` reward bonus
to whichever action ``U`` biased toward — spurious inflation.

The oracle-U critic learns ``Q(s, a, u)`` on the OBSERVED ``u`` (closing the
``A <- U -> R`` backdoor), then deploys the adjusted estimand

    Q_adj(s, .) = E_u[ Q(s, ., u) ] = (1-p) * Q(s, ., 0) + p * Q(s, ., 1)

with the KNOWN ``P(u) = bernoulli(p)`` (``p = 0.5`` on the discrete arm).
``E_u`` averages the ``c_r * u`` bonus into a constant offset shared by every
action, so ``argmax_a Q_adj`` is deconfounded and the deployed policy needs NO U.

POST-COLLAPSE STRUCTURE: this is no longer four ``OracleU*`` subclasses. The
oracle-U behavior is the base learner (DQN/CQL/IQL/DiscreteBCQ) × the
``OracleU`` identification strategy (``identification.py``): the strategy's
``critic_value`` routes the learner's u-conditionable critic evals to
``net.q_su(x, u)``. This module keeps only (1) ``UMarginalizedQ`` — the
U-conditioned net whose ``forward`` is ``Q_adj`` (so ``act``/value-trace deploy
U-free) — and (2) the ``build_oracle_u_*`` builders (base learner + ``OracleU``
strategy + ``UMarginalizedQ`` nets). The u=0 anchor + the ``is_oracle_u`` marker
live on the ``OracleU`` strategy / ``BaseOffPolicy`` now.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.rl.models.backbone import build_trunk, select_backbone
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.identification import OracleU
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.bcq import DiscreteBCQ
from src.rl.offline.cql import CQL
from src.rl.offline.iql import IQL

# P(u=1) under the confounder's bernoulli(0.5); the E_u weights MUST match
# ConfoundedCollectionWrapper._sample_u (verified bernoulli(0.5)).
_P_U1 = 0.5


def _u_col(u: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-transition ``U`` batch as a ``(B, 1)`` column matching ``ref``'s dtype."""
    return u.reshape(-1, 1).to(dtype=ref.dtype, device=ref.device)


class UMarginalizedQ(nn.Module):
    """Discrete Q-net conditioned on the observed confounder ``U``.

    The inner net takes ``cat([obs, u], -1)`` (``obs_dim + 1`` inputs) and emits
    one Q per action. ``q_su(obs, u)`` is the U-conditioned row ``Q(s, ., u)`` (the
    ``OracleU`` strategy's ``critic_value`` routes learn-time evals here);
    ``forward(obs)`` is the backdoor-adjusted ``Q_adj(s, .)`` (the deployable,
    U-free estimand). Presenting ``forward`` as the public ``(B, A)`` interface
    keeps ``act`` and the runner's value-trace deploy unchanged (they query
    ``Q_adj`` with no ``U``).
    """

    def __init__(self, obs_dim: int, action_dim: int, p_u1: float = _P_U1) -> None:
        super().__init__()
        # U appended as one extra scalar feature -> rank-1 MLP over obs_dim + 1
        # (oracle-U is the discrete vector Cell-7 arm; image obs are out of scope).
        self.inner = select_backbone((obs_dim + 1,), obs_dim + 1, action_dim)
        self.p_u1 = float(p_u1)

    def q_su(self, obs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``Q(s, ., u)`` at the observed per-transition ``u``; shape ``(B, A)``."""
        return self.inner(torch.cat([obs, _u_col(u, obs)], dim=-1))

    def q_at(self, obs: torch.Tensor, u_value: float) -> torch.Tensor:
        """``Q(s, ., u_value)`` at a CONSTANT ``u`` (e.g. the ``u=0`` anchor)."""
        u = torch.full(
            (obs.shape[0], 1), float(u_value), device=obs.device, dtype=obs.dtype
        )
        return self.inner(torch.cat([obs, u], dim=-1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Backdoor-adjusted ``Q_adj(s, .) = E_u[Q(s,.,u)]``; shape ``(B, A)``."""
        return (1.0 - self.p_u1) * self.q_at(obs, 0.0) + self.p_u1 * self.q_at(obs, 1.0)


class RecurrentUMarginalizedQ(nn.Module):
    """Recurrent U-conditioned critic for Cell 8 (POMDP × confounded).

    Same public surface as ``UMarginalizedQ`` (``q_su`` / ``q_at`` / ``forward`` =
    ``Q_adj``) but the inner net is a RECURRENT trunk over ``[obs, u]`` (``build_
    trunk("lstm"/"gru"/"rnn", ...)``) instead of an MLP. Two properties compose:

      * it exposes ``initial_state`` -> ``DQN.is_recurrent`` fires -> the sequence
        ``_learn_recurrent`` path engages (POMDP estimation via BPTT);
      * ``q_su`` threads hidden state through ``(B, T)`` sequences (proximal/oracle
        identification via ``q_su(x, u)``).

    GPU-efficiency requirement: ``u`` is concatenated on the FEATURE axis over the
    WHOLE ``(B, T, obs_dim+1)`` tensor and fed through ONE batched trunk call —
    cuDNN unrolls time internally; there is NO per-timestep Python loop. ``forward``
    (the DEPLOYED ``Q_adj``) stays U-free, so ``act``/value-trace deploy unchanged.

    Actor stays Greedy (nuisance U; Cell 8 matched-cell, not the gated/bound path).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        network: str = "lstm",
        p_u1: float = _P_U1,
        **trunk_kwargs,
    ) -> None:
        super().__init__()
        # Recurrent trunk over obs_dim + 1 (the appended U scalar). build_trunk
        # rejects non-recurrent-capable network types; **trunk_kwargs carries
        # hidden_dim/num_layers from the YAML dict form.
        self.inner = build_trunk(
            network, (obs_dim + 1,), obs_dim + 1, action_dim, **trunk_kwargs
        )
        self.p_u1 = float(p_u1)

    def initial_state(self, batch_size: int, device=None):
        """Delegate to the trunk so ``DQN.is_recurrent`` (hasattr check) fires."""
        return self.inner.initial_state(batch_size, device=device)

    def _cat_u(self, obs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Append ``u`` on the feature axis: ``obs (..., D)`` + ``u`` broadcast to
        ``(..., 1)`` -> ``(..., D+1)``. ``u`` carries one value per obs position
        (``(B, T)`` for sequences, ``(B,)`` for a single step)."""
        u = u.to(dtype=obs.dtype, device=obs.device).reshape(*obs.shape[:-1], 1)
        return torch.cat([obs, u], dim=-1)

    def q_su(self, obs: torch.Tensor, u: torch.Tensor, state=None):
        """``Q(s, ., u)`` over a ``(B, T, D)`` sequence (or ``(B, D)`` step) at the
        per-position ``u``; ONE batched trunk call. Returns ``(q_all, new_state)``."""
        return self.inner(self._cat_u(obs, u), state)

    def q_at(self, obs: torch.Tensor, u_value: float, state=None):
        """``Q(s, ., u_value)`` at a CONSTANT ``u`` (e.g. the u=0 anchor / the
        E_u endpoints). Returns ``(q_all, new_state)``."""
        u = torch.full(obs.shape[:-1], float(u_value), device=obs.device)
        return self.inner(self._cat_u(obs, u), state)

    def forward(self, obs: torch.Tensor, state=None):
        """Backdoor-adjusted ``Q_adj = E_u[Q(s,.,u)]`` (U-free deploy). Returns
        ``(q_all, new_state)`` so ``act``'s recurrent path threads hidden state."""
        q0, new_state = self.q_at(obs, 0.0, state)
        q1, _ = self.q_at(obs, 1.0, state)
        return (1.0 - self.p_u1) * q0 + self.p_u1 * q1, new_state


# --------------------------------------------------------------------------
# Builders: base learner × OracleU strategy, with U-conditioned UMarginalizedQ
# critics. Discrete vector Cell-7 arm only.
# --------------------------------------------------------------------------
def _assert_discrete(name: str, kwargs) -> None:
    if kwargs.get("action_type", "discrete") != "discrete":
        raise NotImplementedError(
            f"oracle_u: discrete Cell-7 arm only (offline_dqn/bcq/cql/iql); "
            f"'{name}' got a non-discrete action space."
        )


def _recurrent_trunk_kwargs(kwargs) -> dict:
    """Optional recurrent-trunk kwargs (hidden_dim, num_layers) from the YAML dict
    form; absent/None entries are dropped so the trunk's own defaults apply."""
    return {
        k: kwargs[k] for k in ("hidden_dim", "num_layers") if kwargs.get(k) is not None
    }


def build_oracle_u_dqn(**kwargs):
    _assert_discrete("offline_dqn", kwargs)
    obs_dim, action_dim, device = (
        kwargs["obs_dim"],
        kwargs["action_dim"],
        kwargs["device"],
    )
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = DQN(q, tgt, buffer, device=device, strategy=OracleU())
    return q, agent


def build_recurrent_oracle_u_dqn(**kwargs):
    """Cell 8 oracle-U ceiling: recurrent × OracleU. ``RecurrentUMarginalizedQ``
    (LSTM/GRU/RNN over [obs,u]) fires ``is_recurrent`` -> ``_learn_recurrent``
    routes the sequence TD eval through ``q_su(x, u)`` on the READ realized U (the
    fenced reference). DQN base only (no recurrent cql/iql/bcq)."""
    _assert_discrete("offline_dqn", kwargs)
    obs_dim, action_dim, device = (
        kwargs["obs_dim"],
        kwargs["action_dim"],
        kwargs["device"],
    )
    network = kwargs.get("critic_network", "lstm")
    tk = _recurrent_trunk_kwargs(kwargs)
    q = RecurrentUMarginalizedQ(obs_dim, action_dim, network=network, **tk).to(device)
    tgt = RecurrentUMarginalizedQ(obs_dim, action_dim, network=network, **tk).to(device)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = DQN(q, tgt, buffer, device=device, strategy=OracleU())
    return q, agent


def build_oracle_u_cql(**kwargs):
    _assert_discrete("cql", kwargs)
    obs_dim, action_dim, device = (
        kwargs["obs_dim"],
        kwargs["action_dim"],
        kwargs["device"],
    )
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = CQL(q, tgt, buffer, device=device, strategy=OracleU())
    return q, agent


def build_oracle_u_iql(**kwargs):
    _assert_discrete("iql", kwargs)
    obs_dim, action_dim, device = (
        kwargs["obs_dim"],
        kwargs["action_dim"],
        kwargs["device"],
    )
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    policy_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)  # U-free
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    value_net = select_backbone(obs_shape, obs_dim, 1).to(device)  # U-free
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = IQL(
        policy_net, q, tgt, value_net, buffer, device=device, strategy=OracleU()
    )
    return policy_net, agent


def build_oracle_u_bcq(**kwargs):
    _assert_discrete("bcq", kwargs)
    obs_dim, action_dim, device = (
        kwargs["obs_dim"],
        kwargs["action_dim"],
        kwargs["device"],
    )
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    q = UMarginalizedQ(obs_dim, action_dim).to(device)
    tgt = UMarginalizedQ(obs_dim, action_dim).to(device)
    behavior_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)  # U-free
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = DiscreteBCQ(q, tgt, behavior_net, buffer, device=device, strategy=OracleU())
    return q, agent

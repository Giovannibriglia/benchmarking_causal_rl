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

from src.rl.models.backbone import select_backbone
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

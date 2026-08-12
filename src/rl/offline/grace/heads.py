"""GRACE Q-network heads: the trainable Q_obs plus the routed serving surface.

The TRAINING path and the SERVING path are separated by an explicit
``route_enabled`` flag toggled by the wrapped ``learn`` (builders.py):

  * inside ``learn`` (flag off) every call — direct ``net(x)``, target-net
    evals, CQL regularizer terms — resolves to the plain trainable ``obs_net``,
    so the base learner's training is exactly the observational floor's
    (base-parity; the reduction/no-harm construction B6);
  * outside ``learn`` (flag on) — ``act``, eval rollouts, the ablation's
    ``predict_q_adj`` scoring — ``forward`` serves the ROUTED estimand per the
    router verdict (B5), with Q_obs fallback on unvisited state bins.

``state_dict`` contains only the wrapped net's parameters (the machinery's
tables are plain attributes), so target-net ``load_state_dict`` syncs keep
working when both nets are wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.rl.offline.grace.machinery import GraceMachinery


class GraceQNetwork(nn.Module):
    """Flat (MLP-base) grace head. ``q_obs`` is the strategy hook's target;
    ``forward`` is the deployed estimand."""

    def __init__(self, obs_net: nn.Module) -> None:
        super().__init__()
        self.obs_net = obs_net
        self.route_enabled = True
        self._machinery: Optional[GraceMachinery] = None  # plain attr, no params

    def attach(self, machinery: GraceMachinery) -> None:
        self._machinery = machinery

    def q_obs(self, x: torch.Tensor) -> torch.Tensor:
        return self.obs_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._machinery
        if self.route_enabled and m is not None and m.ready:
            return m.serve(x, self.q_obs)
        return self.obs_net(x)


@dataclass
class GraceRecurrentState:
    """Composite recurrent state: the trunk's hidden state plus the belief
    filter's running belief and the previous greedy action (used to drive the
    predict step of the filter — exact when eval is greedy; the reward stream
    is not available on the serving path, so U-evidence stays prior there,
    documented in docs/grace.md)."""

    trunk: object
    belief: Optional[torch.Tensor] = None
    last_action: Optional[torch.Tensor] = None


class GraceRecurrentQNetwork(nn.Module):
    """Recurrent (POMDP) grace head wrapping a plain recurrent trunk.

    Contract-compatible with the recurrent observational floor: exposes
    ``initial_state`` (so ``DQN.is_recurrent`` fires) and returns
    ``(q, new_state)``. Deliberately does NOT expose ``q_su`` — the recurrent
    learn path dispatches on that attribute and must take its plain
    (observational-floor) branch.
    """

    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.route_enabled = True
        self._machinery: Optional[GraceMachinery] = None

    def attach(self, machinery: GraceMachinery) -> None:
        self._machinery = machinery

    def initial_state(self, batch_size: int, device=None):
        return GraceRecurrentState(
            trunk=self.trunk.initial_state(batch_size, device=device)
        )

    def _unwrap(self, state):
        if isinstance(state, GraceRecurrentState):
            return state
        return GraceRecurrentState(trunk=state)

    def forward(self, x: torch.Tensor, state=None):
        st = self._unwrap(state)
        q_trunk, trunk_state = self.trunk(x, st.trunk)
        m = self._machinery
        routed = (
            self.route_enabled
            and m is not None
            and m.ready
            and x.dim() == 2  # serving is stepwise; (B, T) windows are learn-path
            and m.verdict is not None
            and m.verdict.serve != "obs"
        )
        if not routed:
            return q_trunk, GraceRecurrentState(
                trunk=trunk_state, belief=st.belief, last_action=st.last_action
            )
        with torch.no_grad():
            sym = m.discretizer.encode(x.reshape(x.shape[0], -1)).to(m.device)
            if st.belief is None:
                belief = m.belief_filter.condition_obs(
                    m.belief_filter.reset(x.shape[0]), sym
                )
            else:
                belief = m.belief_filter.step(
                    st.belief,
                    sym,
                    st.last_action,
                    reward_class=None,
                    next_obs_symbol=sym,
                )
            b = belief.reshape(x.shape[0], m.cbn.n_s, m.cbn.n_u)
            if m.verdict.serve == "do":
                table = m.q_do_table.to(x.device)
                q = torch.einsum("bsu,usa->ba", b, table)
            else:  # pessimistic lower bound
                bs = b.sum(dim=2)
                q = torch.einsum("bs,sa->ba", bs, m.q_lo.to(x.device))
            new_state = GraceRecurrentState(
                trunk=trunk_state, belief=belief, last_action=q.argmax(dim=1)
            )
        return q, new_state

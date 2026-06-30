"""FROZEN pre-collapse OracleU* subclasses (verbatim from commit 6a5bf0f) —
the live A/B reference for the strategy-collapse parity test (test_strategy_parity).
NOT production code: it exists only so the parity test can instantiate the OLD
implementation alongside the NEW (base x OracleU strategy) in one process and
assert bitwise equality. Once parity is trusted this can be retired.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from src.rl.off_policy.dqn import DQN
from src.rl.offline.bcq import DiscreteBCQ
from src.rl.offline.cql import CQL
from src.rl.offline.iql import (  # expectile_loss used via the IQL base
    expectile_loss,
    IQL,
)


class _OracleUMixin:
    """Shared marker + the ``u=0`` validation anchor (AM3).

    At ``u=0`` the wrapper's reward bonus and the behavior action bias both
    vanish, so ``Q(s, a_data, 0)`` should track the CLEAN value — the built-in
    check that conditioning on ``U`` works. ``is_oracle_u`` lets the runner add
    the anchor columns to the value trace without widening the baseline schema.
    """

    is_oracle_u = True

    def oracle_anchor_q(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``Q(s, a_data, u=0)`` over the batch's data actions; shape ``(B,)``."""
        with torch.no_grad():
            obs = batch["obs"]
            actions = batch["actions"].long()
            q0 = self.q_network.q_at(obs, 0.0)
            return q0.gather(1, actions.unsqueeze(-1)).squeeze(-1)


def _require_u(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    u = batch.get("confounder_u")
    if u is None:
        raise KeyError(
            "oracle_u critic requires batch['confounder_u']; the dataset must be "
            "loaded U-aware (load_u=True) — regenerate with per-transition U infos."
        )
    return u


class OracleUDQN(_OracleUMixin, DQN):
    """Oracle-U DQN: TD on ``Q(s,a,u)``; bootstrap carries the SAME ``u`` (AM1)."""

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        u = _require_u(batch)

        q_su = self.q_network.q_su(obs, u)
        q_values = q_su.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_network.q_su(next_obs, u).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for p, tp in zip(self.q_network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        return {"loss": loss.item(), "critic_loss": loss.item(), "q_loss": loss.item()}


class OracleUCQL(_OracleUMixin, CQL):
    """Oracle-U CQL: TD on ``Q(s,a,u)`` + conservative penalty on the same row."""

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        u = _require_u(batch)

        q_su = self.q_network.q_su(obs, u)
        q_sa = q_su.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_network.q_su(next_obs, u).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        td_loss = F.mse_loss(q_sa, target)
        # Conservative penalty on the U-conditioned row (logsumexp_a Q(s,.,u) - q_sa).
        penalty = self.conservative_penalty(q_su, actions)
        loss = td_loss + self.alpha * penalty

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        for p, tp in zip(self.q_network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_penalty": float(penalty.item()),
            "critic_loss": float(td_loss.item()),
            "q_loss": float(td_loss.item()),
        }


class OracleUIQL(_OracleUMixin, IQL):
    """Oracle-U IQL: ``Q(s,a,u)`` trained U-conditioned; ``V``/``pi`` stay U-free.

    Only the Q-net training is U-conditioned (``q_su``); ``V`` and the AWR
    advantage use the MARGINALIZED ``Q_adj`` (the wrapper's ``forward``) so both
    sides of the advantage are on the SAME U-free footing. Using ``q_su(.,u)`` for
    the advantage would leak the confounder: ``Q(s,a,u=1) = Q(s,a,0) + c_r·D``, so
    every high-U transition would get a systematically larger advantage and AWR
    would re-upweight exactly the U-biased actions we are deconfounding. The
    deployed policy never sees ``U`` (AM1)."""

    def _marginalized_q_at_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Target ``Q_adj(s, a_data) = E_u[Q(s, a_data, u)]`` — the U-independent
        signal that feeds BOTH the V target and the AWR advantage. Read via the
        wrapper's ``forward`` (NOT ``q_su``): using ``q_su(.,u)`` here would leak
        the confounder back into the policy (see class docstring)."""
        a_idx = batch["actions"].long().unsqueeze(-1)
        return self.target_network(batch["obs"]).gather(1, a_idx).squeeze(-1)

    def v_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Expectile-regression target for ``V``; U-independent by construction."""
        with torch.no_grad():
            return self._marginalized_q_at_data(batch)

    def awr_advantage(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """AWR advantage ``Q_adj(s,a_data) - V(s)``; U-independent by construction."""
        with torch.no_grad():
            return self._marginalized_q_at_data(batch) - self.value_net(
                batch["obs"]
            ).squeeze(-1)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        u = _require_u(batch)
        a_idx = actions.unsqueeze(-1)

        # Value: expectile regression toward the MARGINALIZED target Q_adj(s,a_data)
        # (U-independent), so V is on the same footing as the advantage below.
        v = self.value_net(obs).squeeze(-1)
        v_loss = expectile_loss(self.v_target(batch) - v, self.expectile).mean()
        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()

        # Q: TD target bootstrapped from the U-free V(next) (no OOD max). The Q-net
        # itself stays U-conditioned (q_su) so the c_r·U reward bonus lands on U.
        with torch.no_grad():
            next_v = self.value_net(next_obs).squeeze(-1)
            q_target = rewards + self.gamma * next_v * (1.0 - dones)
        q_sa = self.q_network.q_su(obs, u).gather(1, a_idx).squeeze(-1)
        q_loss = F.mse_loss(q_sa, q_target)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        for p, tp in zip(self.q_network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        # Policy: AWR with a MARGINALIZED, U-independent advantage (Q_adj - V), so
        # AWR cannot upweight U-biased actions. U-free policy net.
        with torch.no_grad():
            weight = torch.clamp(
                torch.exp(self.beta * self.awr_advantage(batch)), max=self.adv_clip
            )
        log_pi = F.log_softmax(self.policy_net(obs), dim=1).gather(1, a_idx).squeeze(-1)
        pi_loss = -(weight * log_pi).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()
        return {
            "loss": float(q_loss.item()),
            "q_loss": float(q_loss.item()),
            "critic_loss": float(q_loss.item()),
            "value_loss": float(v_loss.item()),
            "actor_loss": float(pi_loss.item()),
        }


class OracleUBCQ(_OracleUMixin, DiscreteBCQ):
    """Oracle-U discrete BCQ: behavior net ``G(a|s)`` stays U-free; the Q target's
    constrained next-argmax is taken over ``Q(s',.,u)`` (AM1)."""

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        u = _require_u(batch)

        bc_loss = F.cross_entropy(self.behavior_net(obs), actions)
        self.behavior_opt.zero_grad(set_to_none=True)
        bc_loss.backward()
        self.behavior_opt.step()

        with torch.no_grad():
            next_q_online = self.q_network.q_su(next_obs, u)
            next_constrained = self._constrained_q(
                next_q_online, self.allowed_mask(next_obs)
            )
            next_actions = torch.argmax(next_constrained, dim=1, keepdim=True)
            next_q = (
                self.target_network.q_su(next_obs, u)
                .gather(1, next_actions)
                .squeeze(-1)
            )
            target = rewards + self.gamma * next_q * (1.0 - dones)
        q_sa = self.q_network.q_su(obs, u).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q_loss = F.mse_loss(q_sa, target)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        for p, tp in zip(self.q_network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        return {
            "loss": float(q_loss.item()),
            "q_loss": float(q_loss.item()),
            "critic_loss": float(q_loss.item()),
            "bc_loss": float(bc_loss.item()),
        }


# --------------------------------------------------------------------------
# Builders (NEW; the existing build_* are untouched). Discrete vector arm only.
# --------------------------------------------------------------------------
def _assert_discrete(name: str, kwargs) -> None:
    if kwargs.get("action_type", "discrete") != "discrete":
        raise NotImplementedError(
            f"oracle_u: discrete Cell-7 arm only (offline_dqn/bcq/cql/iql); "
            f"'{name}' got a non-discrete action space."
        )

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.base import ActionOutput
from src.rl.models.backbone import select_backbone
from src.rl.off_policy.base_off_policy import BaseOffPolicy
from src.rl.off_policy.identification import IdentificationStrategy, Observational
from src.rl.off_policy.replay_buffer import ReplayBuffer


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric (expectile) L2: ``|expectile - 1(diff<0)| * diff^2`` elementwise.

    For ``expectile=0.5`` this is symmetric (ordinary L2 / 2); for
    ``expectile>0.5`` it penalizes positive residuals more, biasing the value
    fit toward an upper expectile of Q(s,a).
    """
    weight = torch.where(
        diff < 0,
        torch.full_like(diff, 1.0 - expectile),
        torch.full_like(diff, expectile),
    )
    return weight * diff.pow(2)


class IQL(BaseOffPolicy):
    """Implicit Q-Learning (discrete actions). Kostrikov et al. 2021
    (arXiv:2110.06169).

    Three networks, actor/critic separated:
      * value critic ``V(s)`` trained by expectile regression toward Q(s,a_data),
      * Q critic ``Q(s,a)`` trained with a V-bootstrapped target (no OOD max),
      * actor ``pi(a|s)`` extracted by advantage-weighted regression (clipped).

        L_V = E[ expectile_loss(Q'(s,a) - V(s), tau) ]
        L_Q = E[ (r + gamma (1-d) V(s') - Q(s,a))^2 ]
        L_pi = E[ min(exp(beta (Q'(s,a) - V(s))), 100) * (-log pi(a|s)) ]

    The policy is greedy argmax over ``pi`` at eval.
    """

    action_type = "discrete"

    def __init__(
        self,
        policy_net: nn.Module,
        q_network: nn.Module,
        target_network: nn.Module,
        value_net: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        beta: float = 3.0,
        adv_clip: float = 100.0,
        strategy: IdentificationStrategy | None = None,
    ) -> None:
        super().__init__(device, gamma=gamma)
        # actor / critics kept as distinct modules.
        self.actor = policy_net
        self.policy_net = policy_net
        self.q_network = q_network
        self.target_network = target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.value_net = value_net
        self.buffer = buffer
        self.q_opt = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.pi_opt = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.tau = tau
        self.expectile = expectile
        self.beta = beta
        self.adv_clip = adv_clip
        # Default Observational => critic_value is net(x), byte-identical to the
        # pre-strategy IQL (golden unchanged). ONLY the q_sa data-action site is
        # routed through the strategy; V-target/advantage stay plain (see below).
        self._strategy = strategy if strategy is not None else Observational()

    def v_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Expectile-regression target for V: the (target) Q at the data action.

        PLAIN ``target_network(obs)`` — deliberately NOT routed through the
        strategy hook. For the OracleU net this forward is the marginalized
        ``Q_adj`` (U-independent); routing it through ``q_su`` would re-introduce
        the confounder leak into V and the policy (the cb751d6 fix)."""
        with torch.no_grad():
            a_idx = batch["actions"].long().unsqueeze(-1)
            return self.target_network(batch["obs"]).gather(1, a_idx).squeeze(-1)

    def awr_advantage(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """AWR advantage ``Q(s,a_data) - V(s)`` using the PLAIN target_network(obs)
        (U-independent for the OracleU net) — same leak-avoidance as v_target."""
        with torch.no_grad():
            a_idx = batch["actions"].long().unsqueeze(-1)
            return self.target_network(batch["obs"]).gather(1, a_idx).squeeze(
                -1
            ) - self.value_net(batch["obs"]).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> ActionOutput:
        with torch.no_grad():
            logits = self.policy_net(obs)
            return ActionOutput(action=torch.argmax(logits, dim=1), state=state)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        a_idx = actions.unsqueeze(-1)

        # Value: expectile regression toward the (target) Q of the data action.
        # v_target is the plain target_network(obs) (NOT hooked) — preserves the
        # original compute order (target-Q forward, then V forward).
        q_target_sa = self.v_target(batch)
        v = self.value_net(obs).squeeze(-1)
        v_loss = expectile_loss(q_target_sa - v, self.expectile).mean()
        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()

        # Q: TD target bootstrapped from V (avoids the OOD max over actions). ONLY
        # the data-action Q (q_sa) is routed through the strategy hook.
        with torch.no_grad():
            next_v = self.value_net(next_obs).squeeze(-1)
            q_target = rewards + self.gamma * next_v * (1.0 - dones)
        q_sa = (
            self._strategy.critic_value(self.q_network, obs, batch)
            .gather(1, a_idx)
            .squeeze(-1)
        )
        q_loss = F.mse_loss(q_sa, q_target)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        for p, tp in zip(self.q_network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        # Policy: advantage-weighted regression with clipped weights. The advantage
        # uses the plain (marginalized) target-Q, NOT the strategy hook.
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


def build_iql(**kwargs):
    if kwargs.get("action_type", "discrete") != "discrete":
        raise ValueError(
            "iql is discrete-only; use iql_continuous (Gaussian AWR policy) for "
            "continuous action spaces. (Without this guard a continuous env would "
            "silently build a single-output discrete Q-net.)"
        )
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    # Backbone by obs rank: vector -> MLP (bitwise-identical to the previous
    # MLP(obs_dim, *)); image -> Nature-CNN. The value head is a 1-output net.
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    policy_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    q_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    target_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    value_net = select_backbone(obs_shape, obs_dim, 1).to(device)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = IQL(
        policy_net,
        q_net,
        target_net,
        value_net,
        buffer,
        device=device,
        strategy=kwargs.get("strategy"),
    )
    return policy_net, agent

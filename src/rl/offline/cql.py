from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.base import ActionOutput
from src.rl.nets.mlp import MLP
from src.rl.off_policy.base_off_policy import BaseOffPolicy
from src.rl.off_policy.replay_buffer import ReplayBuffer


class CQL(BaseOffPolicy):
    """Conservative Q-Learning on a DQN backbone (discrete actions).

    CQL(H), Kumar et al. 2020 (arXiv:2006.04779): standard DQN TD loss plus a
    conservative penalty that pushes down out-of-distribution action values and
    pushes up dataset action values. For discrete actions the penalty is exact:

        L = 0.5 * E[(Q(s,a) - (r + gamma (1-d) max_a' Q'(s',a')))^2]
            + alpha * E_s[ logsumexp_a Q(s,a) - Q(s,a_data) ]

    The policy is greedy argmax over Q. Q-centric (no separate actor).
    """

    action_type = "discrete"

    def __init__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.q_network = q_network
        self.target_network = target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.tau = tau
        self.alpha = alpha

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> ActionOutput:
        with torch.no_grad():
            q = self.q_network(obs)
            return ActionOutput(action=torch.argmax(q, dim=1), state=state)

    @staticmethod
    def conservative_penalty(
        q_values: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """CQL(H) penalty: mean over the batch of
        ``logsumexp_a Q(s,a) - Q(s,a_data)``.

        Always >= 0 (logsumexp >= max >= the data-action value); equals 0 only
        in the degenerate case where the data action dominates everywhere.
        """
        logsumexp_q = torch.logsumexp(q_values, dim=1)
        data_q = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        return (logsumexp_q - data_q).mean()

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        q_values = self.q_network(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        td_loss = F.mse_loss(q_sa, target)
        penalty = self.conservative_penalty(q_values, actions)
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


def build_cql(**kwargs):
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    q_net = MLP(obs_dim, action_dim).to(device)
    target_net = MLP(obs_dim, action_dim).to(device)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = CQL(q_net, target_net, buffer, device=device)
    return q_net, agent

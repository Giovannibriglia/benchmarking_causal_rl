from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.base import ActionOutput
from src.rl.models.backbone import select_backbone
from src.rl.off_policy.base_off_policy import BaseOffPolicy
from src.rl.off_policy.replay_buffer import ReplayBuffer


class DiscreteBCQ(BaseOffPolicy):
    """Discrete Batch-Constrained Q-Learning. Fujimoto et al. 2019
    (arXiv:1910.01708).

    A behavior model ``G(a|s)`` (softmax classifier, NLL-imitated from the
    dataset) restricts the action set the Q-policy may choose from. Only actions
    whose normalized behavior probability clears a threshold are allowed:

        allowed(s) = { a : G(a|s) / max_a' G(a'|s) >= threshold }

    The Q target uses the constrained argmax (double-DQN style); the eval policy
    is the constrained argmax over Q. No VAE — that is only needed for the
    continuous variant (deferred).
    """

    action_type = "discrete"

    def __init__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        behavior_net: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        threshold: float = 0.3,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.q_network = q_network
        self.target_network = target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.behavior_net = behavior_net
        self.buffer = buffer
        self.q_opt = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.behavior_opt = torch.optim.Adam(self.behavior_net.parameters(), lr=lr)
        self.tau = tau
        self.threshold = threshold

    def allowed_mask(
        self, obs: torch.Tensor, threshold: float | None = None
    ) -> torch.Tensor:
        """Boolean ``[batch, n_actions]`` mask of behavior-supported actions.

        ``threshold=0`` keeps every action (== unconstrained DQN); ``threshold>=1``
        keeps only the behavior argmax (BC-greedy), since the normalized prob of
        the most-likely action is exactly 1.
        """
        thr = self.threshold if threshold is None else threshold
        with torch.no_grad():
            probs = F.softmax(self.behavior_net(obs), dim=1)
            max_prob = probs.max(dim=1, keepdim=True).values
            return (probs / (max_prob + 1e-8)) >= thr

    @staticmethod
    def _constrained_q(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(q_values.dtype).min
        return torch.where(mask, q_values, torch.full_like(q_values, neg_inf))

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
            constrained = self._constrained_q(q, self.allowed_mask(obs))
            return ActionOutput(action=torch.argmax(constrained, dim=1), state=state)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Behavior model: cross-entropy imitation of dataset actions.
        bc_loss = F.cross_entropy(self.behavior_net(obs), actions)
        self.behavior_opt.zero_grad(set_to_none=True)
        bc_loss.backward()
        self.behavior_opt.step()

        # Q: target uses the behavior-constrained next-action argmax.
        with torch.no_grad():
            next_q_online = self.q_network(next_obs)
            next_constrained = self._constrained_q(
                next_q_online, self.allowed_mask(next_obs)
            )
            next_actions = torch.argmax(next_constrained, dim=1, keepdim=True)
            next_q = self.target_network(next_obs).gather(1, next_actions).squeeze(-1)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        q_sa = self.q_network(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
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


def build_bcq(**kwargs):
    if kwargs.get("action_type", "discrete") != "discrete":
        raise ValueError(
            "bcq is discrete-only; use bcq_continuous (CVAE-BCQ) for continuous "
            "action spaces. (Without this guard a continuous env would silently "
            "build a single-output discrete Q-net — action_dim = act_space.shape[0].)"
        )
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    # Backbone by obs rank: vector -> MLP (bitwise-identical to the previous
    # MLP(obs_dim, action_dim)); image -> Nature-CNN.
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    q_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    target_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    behavior_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = DiscreteBCQ(q_net, target_net, behavior_net, buffer, device=device)
    return q_net, agent

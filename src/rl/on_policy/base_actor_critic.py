from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..base import ActionOutput, Algorithm


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class BaseActorCritic(Algorithm):
    paradigm = "on_policy"
    action_type = "both"

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__()
        self.policy = policy
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def act(
        self,
        obs: torch.Tensor,
        state: Optional[Any] = None,
        *,
        deterministic: bool = False,
    ) -> ActionOutput:
        if deterministic and hasattr(self.policy, "act_deterministic"):
            return ActionOutput(action=self.policy.act_deterministic(obs), state=state)
        action, logp = self.policy.act(obs)
        return ActionOutput(action=action, log_prob=logp, state=state)

    def compute_gae(
        self, rewards, dones, values, next_values
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    @abc.abstractmethod
    def learn(self, batch: RolloutBatch) -> Dict[str, float]: ...

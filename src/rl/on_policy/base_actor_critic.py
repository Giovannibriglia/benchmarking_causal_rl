from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class BaseActorCritic(abc.ABC):
    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.policy = policy
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

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
    def update(self, batch: RolloutBatch) -> Dict[str, float]: ...

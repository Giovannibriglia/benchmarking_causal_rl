from __future__ import annotations

from typing import Tuple

import torch
import torch.distributions as dist
import torch.nn as nn

from ..base_policy import BasePolicy
from ..nets.mlp import MLP


class ActorCriticMLP(BasePolicy):
    """Shared-encoder actor-critic with separate heads."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str,
        device: torch.device,
        hidden_dims=(64, 64),
    ) -> None:
        super().__init__(device)
        self.action_type = action_type
        self.encoder = MLP(
            obs_dim,
            hidden_dims[-1],
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else (),
        )
        if action_type == "discrete":
            self.actor = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std = None
        else:
            self.actor = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dims[-1], 1)
        self.to(device)

    def distribution(self, obs: torch.Tensor) -> dist.Distribution:
        if obs.dim() == 1:
            obs = obs.unsqueeze(-1)
        feat = self.encoder(obs)
        if self.action_type == "discrete":
            logits = self.actor(feat)
            return dist.Categorical(logits=logits)
        mean = self.actor(feat)
        std = torch.exp(self.log_std).expand_as(mean)
        return dist.Normal(mean, std)

    def log_prob(
        self, distribution: dist.Distribution, actions: torch.Tensor
    ) -> torch.Tensor:
        logp = distribution.log_prob(actions)
        if self.action_type != "discrete":
            logp = logp.sum(-1)
        return logp

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        action = distribution.sample()
        logp = self.log_prob(distribution, action)
        return action.to(self.device), logp.to(self.device)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(obs)
        if self.action_type == "discrete":
            return torch.argmax(distribution.logits, dim=-1)
        return distribution.mean

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(-1)
        feat = self.encoder(obs)
        return self.critic(feat).squeeze(-1)

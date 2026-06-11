from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.rl.models.backbone import select_backbone
from src.rl.models.encoding import encode_action


class TransitionModel(nn.Module):
    """Learned next-state model s'(s, a) -> next-state vector.

    A separate auxiliary module (NOT folded into the RL loss): trained alongside
    the RL update on its batch, logged. Predicts the flat next observation;
    backbone is chosen by observation rank via the shared selector (MLP for
    rank-1 today; CNN is PR6, RNN/GRU a seam).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str,
        obs_shape: Sequence[int],
        hidden_dims: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.action_type = action_type
        self.action_dim = action_dim
        self.backbone = select_backbone(
            obs_shape,
            input_dim=obs_dim + action_dim,
            output_dim=obs_dim,
            hidden_dims=hidden_dims,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        a = encode_action(actions, self.action_type, self.action_dim)
        x = torch.cat([obs.float(), a], dim=-1)
        return self.backbone(x)

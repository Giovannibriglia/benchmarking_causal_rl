from __future__ import annotations

import torch
import torch.nn.functional as F


def encode_action(
    actions: torch.Tensor, action_type: str, action_dim: int
) -> torch.Tensor:
    """Encode actions as model input features.

    Discrete actions become one-hot vectors of width ``action_dim``; continuous
    actions are returned as a float tensor (already a per-dim vector). Output is
    always 2-D ``[batch, action_dim]`` so it concatenates with flat observations.
    """
    if action_type == "discrete":
        return F.one_hot(actions.long().reshape(-1), num_classes=action_dim).float()
    a = actions.float()
    if a.dim() == 1:
        a = a.unsqueeze(-1)
    return a

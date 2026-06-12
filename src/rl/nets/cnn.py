from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """Nature-CNN image encoder (Mnih et al. 2015), full ``obs -> output_dim`` net.

    Conv stack -> flatten -> Linear(conv_out, 512) -> ReLU -> Linear(512,
    output_dim). The 512 feature width is carried INTERNALLY so the head still
    sits on ``output_dim`` (crux-1 option a): callers and the actor/critic heads
    are unchanged. For DQN (``output_dim = action_dim``) this is exactly
    Nature-DQN; for the on-policy encoder (``output_dim = 64``) the 512 feeds a
    thin embedding, fine for a smoke fixture.

    The conv-flatten dimension is inferred by a dummy forward, so the net is
    robust to the input ``(C, H, W)``. ``variant="impala"`` is a documented seam.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        output_dim: int,
        variant: str = "nature",
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        if variant != "nature":
            raise NotImplementedError(
                f"CNN variant '{variant}' is a documented seam (e.g. IMPALA "
                "residual blocks); only 'nature' is implemented."
            )
        channels = int(obs_shape[0])
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Infer the flattened conv output via a dummy forward (no RNG draw).
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(conv_out, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.conv(x))

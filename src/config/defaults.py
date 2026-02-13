from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch

from .device import detect_device


@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    n_train_envs: int = 16
    n_eval_envs: int = 16
    rollout_len: int = 1024
    seed: int = 42


@dataclass
class TrainingConfig:
    n_episodes: int = 250
    n_checkpoints: int = 25
    eval_interval: Optional[int] = None  # derived from n_episodes / n_checkpoints
    deterministic: bool = False
    device: str = field(default_factory=lambda: str(detect_device()))
    algorithm: str = "ppo"
    checkpoint_dir: Optional[str] = None
    aggregation: str = "iqm"

    def checkpoint_episodes(self) -> list[int]:
        """Compute uniformly spaced checkpoint episodes including first and last."""
        count = max(2, min(self.n_checkpoints, self.n_episodes))
        if count == 2:
            return [0, self.n_episodes - 1]
        # linear spacing over episode indices
        indices = torch.linspace(0, self.n_episodes - 1, steps=count)
        unique = sorted({int(round(x.item())) for x in indices})
        # ensure first and last present
        unique[0] = 0
        unique[-1] = self.n_episodes - 1
        return unique


@dataclass
class RunConfig:
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir: Optional[str] = None

    def resolve_run_dir(self) -> str:
        if self.run_dir is not None:
            return self.run_dir
        return f"runs/benchmark_{self.timestamp}"

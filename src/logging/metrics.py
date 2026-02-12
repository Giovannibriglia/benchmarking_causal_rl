from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


def to_cpu_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return float(x)


@dataclass
class EpisodeMetrics:
    rewards: List[float]

    def mean_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)

    def to_dict(self) -> Dict[str, float]:
        return {"reward_mean": self.mean_reward()}

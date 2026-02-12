from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import torch


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.storage: Deque[Dict[str, torch.Tensor]] = deque(maxlen=capacity)

    def add(self, transition: Dict[str, torch.Tensor]) -> None:
        self.storage.append({k: v.detach().cpu() for k, v in transition.items()})

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        import random

        batch = random.sample(self.storage, batch_size)
        out: Dict[str, torch.Tensor] = {}
        for k in batch[0].keys():
            out[k] = torch.stack([b[k] for b in batch]).to(self.device)
        return out

    def __len__(self) -> int:
        return len(self.storage)

from __future__ import annotations

import torch

Tensor = torch.Tensor


def to_tensor(x, device=None, dtype=None) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    t = torch.as_tensor(x)
    return t.to(device=device or t.device, dtype=dtype or t.dtype)

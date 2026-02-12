from __future__ import annotations

import torch


def detect_device() -> torch.device:
    """Select CUDA if available else CPU.

    Detection is performed once at startup and passed everywhere to avoid scattered .to(device).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

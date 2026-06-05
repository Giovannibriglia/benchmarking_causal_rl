from __future__ import annotations

import torch
from src.causal_metrics.estimators import mmd_gauss


def test_mmd_identical_samples_is_near_zero() -> None:
    x = torch.randn(128, 4)
    mmd = mmd_gauss(x, x.clone())
    assert mmd.item() <= 1e-6

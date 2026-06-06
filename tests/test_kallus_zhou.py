"""Kallus-Zhou sensitivity-interval tests (Cell-8 variant machinery)."""

from __future__ import annotations

import pytest
import torch
from src.eval.kallus_zhou import _extremal_value, kz_interval
from tests.test_confounding_gate import _synthetic_confounded_source
from tests.test_ope import TabularPolicy


def test_extremal_value_brackets_nominal():
    torch.manual_seed(0)
    w = torch.rand(200) + 0.5
    g = torch.randn(200)
    nominal = float((w * g).sum() / w.sum())
    lo = _extremal_value(w, g, gamma=2.0, maximize=False)
    hi = _extremal_value(w, g, gamma=2.0, maximize=True)
    assert lo <= nominal <= hi
    assert hi > lo


def test_gamma_one_collapses_to_point():
    torch.manual_seed(0)
    w = torch.rand(100) + 0.5
    g = torch.randn(100)
    lo = _extremal_value(w, g, gamma=1.0, maximize=False)
    hi = _extremal_value(w, g, gamma=1.0, maximize=True)
    assert abs(hi - lo) < 1e-9


def test_interval_widens_with_gamma():
    src = _synthetic_confounded_source(beta=1.5, delta=0.5, n_episodes=300)
    g = torch.Generator().manual_seed(5)
    target = TabularPolicy(torch.randn(2, 2, generator=g))

    class ObsPolicy(TabularPolicy):
        def _z(self, obs):  # continuous obs -> bucket on sign of dim 0
            return (obs[..., 0] > 0).long()

    target = ObsPolicy(target.logits)
    widths = []
    for gamma in (1.5, 2.0, 3.0):
        iv = kz_interval(src, target, gamma=gamma)
        assert iv.lower <= iv.upper
        widths.append(iv.upper - iv.lower)
    assert widths[0] < widths[1] < widths[2]


def test_gamma_below_one_rejected():
    src = _synthetic_confounded_source(beta=1.5, delta=0.5, n_episodes=50)
    g = torch.Generator().manual_seed(5)

    class ObsPolicy(TabularPolicy):
        def _z(self, obs):
            return (obs[..., 0] > 0).long()

    with pytest.raises(ValueError, match="gamma must be >= 1"):
        kz_interval(src, ObsPolicy(torch.randn(2, 2, generator=g)), gamma=0.5)

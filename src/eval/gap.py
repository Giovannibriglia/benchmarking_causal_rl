"""Identifiability-gap metrics: divergence between observational and
interventional reward distributions (secondary, explanatory metric).

Adapted from the recovered ``causal_metrics/gap.py`` (Phase-3 gate verdict
(b)) with two changes: (1) decoupled from the old ``CausalEnv`` ABC — the
caller supplies ``p_obs`` (from the dataset / model) and ``p_do`` (from
``src/eval/oracle.py`` or an analytic fixture); (2) the silent broad
``try/except`` fallback is removed — callers choose the tabular or the
sample-based (MMD) path explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from .estimators import mmd_gauss, plugin_tv

DivergenceName = Literal["tv", "kl", "chi2", "sup", "js_normalized"]

_EPS = 1e-8


@dataclass
class GapResult:
    delta: float
    divergence: str
    n_samples: int
    is_oracle: bool


def divergence_from_probs(
    p_obs: torch.Tensor, p_do: torch.Tensor, divergence: DivergenceName
) -> torch.Tensor:
    """φ-divergence between row-wise tabular distributions [B, K]."""
    if divergence == "tv":
        return plugin_tv(p_obs, p_do)
    if divergence == "kl":
        return (
            (p_obs * (p_obs.add(_EPS).log() - p_do.add(_EPS).log())).sum(dim=-1).mean()
        )
    if divergence == "chi2":
        return (((p_obs - p_do) ** 2) / p_do.add(_EPS)).sum(dim=-1).mean()
    if divergence == "sup":
        return (p_obs - p_do).abs().max(dim=-1).values.mean()
    if divergence == "js_normalized":
        m = 0.5 * (p_obs + p_do)
        js = 0.5 * (
            (p_obs * (p_obs.add(_EPS).log() - m.add(_EPS).log())).sum(dim=-1)
            + (p_do * (p_do.add(_EPS).log() - m.add(_EPS).log())).sum(dim=-1)
        )
        return (js / torch.log(torch.tensor(2.0))).mean()
    raise ValueError(f"Unsupported divergence '{divergence}'")


def compute_gap(
    p_obs: torch.Tensor,
    p_do: torch.Tensor,
    divergence: DivergenceName = "tv",
    *,
    is_oracle: bool = True,
) -> GapResult:
    """Tabular gap Δ_φ between aligned [B, K] distributions."""
    if p_obs.shape != p_do.shape:
        raise ValueError(
            f"p_obs {tuple(p_obs.shape)} and p_do {tuple(p_do.shape)} must align; "
            "the caller resolves support mismatches explicitly."
        )
    with torch.no_grad():
        delta = divergence_from_probs(p_obs, p_do, divergence)
    return GapResult(
        delta=float(delta.item()),
        divergence=divergence,
        n_samples=int(p_obs.shape[0]),
        is_oracle=is_oracle,
    )


def gap_from_samples(
    obs_rewards: torch.Tensor,
    do_rewards: torch.Tensor,
    sigma: Optional[float] = None,
    *,
    is_oracle: bool = True,
) -> GapResult:
    """Sample-based gap via unbiased Gaussian MMD² — the continuous-reward
    path (e.g. HalfCheetah oracle reward samples vs logged rewards)."""
    with torch.no_grad():
        delta = mmd_gauss(
            obs_rewards.reshape(-1, 1), do_rewards.reshape(-1, 1), sigma=sigma
        )
    return GapResult(
        delta=float(delta.item()),
        divergence="mmd",
        n_samples=int(obs_rewards.numel()),
        is_oracle=is_oracle,
    )

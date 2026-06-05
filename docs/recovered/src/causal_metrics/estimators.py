"""Estimators for causal identifiability-gap divergences (paper Section 5)."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def plugin_tv(p_obs: torch.Tensor, p_do: torch.Tensor) -> torch.Tensor:
    """Plug-in total-variation divergence for tabular distributions."""
    return 0.5 * (p_obs - p_do).abs().sum(dim=-1).mean()


def mmd_gauss(
    samples_obs: torch.Tensor, samples_do: torch.Tensor, sigma: Optional[float] = None
) -> torch.Tensor:
    """Gaussian-kernel MMD^2 with an unbiased finite-sample estimator."""
    x = samples_obs.reshape(samples_obs.shape[0], -1)
    y = samples_do.reshape(samples_do.shape[0], -1)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)

    if sigma is None:
        with torch.no_grad():
            joined = torch.cat([x, y], dim=0)
            d2 = torch.cdist(joined, joined).pow(2)
            med = torch.median(d2[d2 > 0])
            sigma = float(torch.sqrt(med + 1e-8).item()) if torch.isfinite(med) else 1.0
            sigma = max(sigma, 1e-4)
    gamma = 1.0 / (2.0 * sigma * sigma)

    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))

    n = x.shape[0]
    m = y.shape[0]
    sum_xx = (kxx.sum() - torch.diagonal(kxx).sum()) / (n * (n - 1))
    sum_yy = (kyy.sum() - torch.diagonal(kyy).sum()) / (m * (m - 1))
    sum_xy = kxy.mean()
    return (sum_xx + sum_yy - 2.0 * sum_xy).clamp_min(0.0)


def dice_chi2(
    behavior_buffer: Dict[str, torch.Tensor],
    target_policy: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Simple χ² proxy using clipped importance ratios over replay samples."""
    obs = behavior_buffer.get("obs")
    actions = behavior_buffer.get("actions")
    logp_b = behavior_buffer.get("behavior_logprob")
    if obs is None or actions is None or logp_b is None:
        return torch.tensor(0.0)

    logp_t_all = target_policy(obs)
    if logp_t_all.ndim != 2:
        raise ValueError("target_policy(obs) must return log-prob matrix [B, A].")
    logp_t = logp_t_all.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
    w = torch.exp(logp_t - logp_b).clamp(0.0, 50.0)
    return ((w - 1.0) ** 2).mean()

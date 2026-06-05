"""Divergence estimators for identifiability-gap metrics.

Ported from the recovered ``causal_metrics/estimators.py`` (Phase-3 gate
verdict (a)) with an observability audit: every function consumes ONLY
observed quantities — tabular distributions supplied by the caller, sample
batches, or LOGGED behavior propensities (available exactly when the cell
declares pi_b known via ``ExperienceSource.behavior_logprob``). No function
reads latent confounders or true propensities; keep it that way (§8 grep).
"""

from __future__ import annotations

from typing import Optional

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
    obs: torch.Tensor,
    actions: torch.Tensor,
    behavior_logprob: torch.Tensor,
    target_log_probs: torch.Tensor,
    clip: float = 50.0,
) -> torch.Tensor:
    """χ² overlap proxy from clipped importance ratios.

    ``behavior_logprob`` must be the LOGGED propensities of the dataset
    (``ExperienceSource.behavior_logprob``); ``target_log_probs`` is the
    [B, A] log-prob matrix of the target policy at ``obs``. Useful as a
    coverage/overlap diagnostic for tier sweeps and the confounding gate.
    """
    _ = obs
    if target_log_probs.ndim != 2:
        raise ValueError("target_log_probs must be a [B, A] log-prob matrix.")
    logp_t = target_log_probs.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
    w = torch.exp(logp_t - behavior_logprob).clamp(0.0, clip)
    return ((w - 1.0) ** 2).mean()

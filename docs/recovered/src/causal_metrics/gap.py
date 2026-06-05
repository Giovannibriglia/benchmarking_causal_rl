"""Computation of φ-divergence identifiability-gap metrics (paper Section 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
from src.envs.causal_base import CausalEnv

from .estimators import mmd_gauss, plugin_tv

DivergenceName = Literal["tv", "kl", "chi2", "sup"]


@dataclass
class GapResult:
    delta: float
    divergence: DivergenceName
    n_samples: int
    is_oracle: bool


def _obs_reward_distribution(
    reward: torch.Tensor, support: Optional[torch.Tensor]
) -> torch.Tensor:
    r = reward.reshape(-1, 1)
    if support is None:
        values = torch.unique(reward.detach())
        support = values.sort().values.to(reward.device).float()
    else:
        support = support.to(reward.device).float()
    dist = torch.argmin((r - support.reshape(1, -1)).abs(), dim=-1)
    one_hot = torch.nn.functional.one_hot(dist, num_classes=support.numel()).float()
    return one_hot


def _divergence(
    p_obs: torch.Tensor, p_do: torch.Tensor, divergence: DivergenceName
) -> torch.Tensor:
    eps = 1e-8
    if divergence == "tv":
        return plugin_tv(p_obs, p_do)
    if divergence == "kl":
        return (p_obs * (p_obs.add(eps).log() - p_do.add(eps).log())).sum(dim=-1).mean()
    if divergence == "chi2":
        return (((p_obs - p_do) ** 2) / p_do.add(eps)).sum(dim=-1).mean()
    if divergence == "sup":
        return (p_obs - p_do).abs().max(dim=-1).values.mean()
    raise ValueError(f"Unsupported divergence '{divergence}'")


def compute_gap(
    env: CausalEnv,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_obs: torch.Tensor,
    divergence: DivergenceName = "tv",
    estimator: Optional[Callable] = None,
) -> GapResult:
    """Compute Δ_φ on a transition batch."""
    _ = obs, next_obs
    n = int(reward.numel())
    with torch.no_grad():
        try:
            p_do = env.do_reward(action)
            if p_do.ndim == 1:
                p_do = p_do.unsqueeze(-1)
            p_obs = env.observed_reward_distribution(action=action, reward=reward)
            if p_obs is None:
                p_obs = _obs_reward_distribution(reward, env.reward_support)
            if p_obs.shape[-1] != p_do.shape[-1]:
                k = min(p_obs.shape[-1], p_do.shape[-1])
                p_obs = p_obs[:, :k]
                p_do = p_do[:, :k]
                p_obs = p_obs / p_obs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                p_do = p_do / p_do.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            delta = (
                estimator(p_obs, p_do)
                if estimator
                else _divergence(p_obs, p_do, divergence)
            )
            return GapResult(
                delta=float(delta.item()),
                divergence=divergence,
                n_samples=n,
                is_oracle=bool(env.supports_oracle()),
            )
        except Exception:
            do_samples = env.do_reward(action)
            if do_samples.ndim == 2 and do_samples.shape[-1] > 1:
                probs = do_samples / do_samples.sum(dim=-1, keepdim=True).clamp_min(
                    1e-8
                )
                idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
                support = (
                    env.reward_support.to(reward.device)
                    if env.reward_support is not None
                    else torch.arange(probs.shape[-1], device=reward.device).float()
                )
                do_samples = support[idx]
            if estimator is not None:
                delta = estimator(reward.reshape(-1, 1), do_samples.reshape(-1, 1))
            else:
                delta = mmd_gauss(reward.reshape(-1, 1), do_samples.reshape(-1, 1))
            return GapResult(
                delta=float(delta.item()),
                divergence=divergence,
                n_samples=n,
                is_oracle=False,
            )

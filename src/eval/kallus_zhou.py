"""Kallus–Zhou-style sensitivity intervals for confounded OPE (Cell 8).

Marginal sensitivity model (MSM) adapted to our DGP: the confounder U is
fixed PER EPISODE, so the unobserved multiplicative correction to each
episode's importance weight is a single factor ``lambda_ep`` with odds-ratio
budget ``lambda_ep ∈ [1/Gamma, Gamma]``. The identified set for the target
policy's value is then

    V(lambda) = sum_ep w_ep·lambda_ep·G_ep / sum_ep w_ep·lambda_ep,

minimized / maximized over the box — a linear-fractional program whose
optimum assigns ``Gamma`` above a return threshold and ``1/Gamma`` below
(or vice versa), so an exact solution is found by scanning the n+1
thresholds of the return-sorted episodes.

The interval is REPORTED, never collapsed to a point (gate condition 2):
under Cell-8 non-identifiability the honest answer is a set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from src.data.experience_source import OfflineDatasetSource
from src.eval.ope import clone_behavior_policy, TargetPolicy


@dataclass
class KZInterval:
    lower: float
    upper: float
    gamma: float
    n_episodes: int


def _episode_weights_returns(
    source: OfflineDatasetSource, target: TargetPolicy, clone_seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nominal per-episode weights w_ep = prod_t pi_t/pi_bar (pi_bar = logged
    propensities when known, else the cloned behavior policy)."""
    lengths = [int(ep["rewards"].shape[0]) for ep in source.episodes]
    logp_t = target.log_prob(source.obs.float(), source.actions)
    if source.behavior_logprob is not None:
        logp_b = source.behavior_logprob.float()
    else:
        cloned = clone_behavior_policy(source, seed=clone_seed)
        logp_b = cloned(source.obs.float(), source.actions)
    log_ratio = (logp_t - logp_b).cpu()
    weights, returns = [], []
    for ep_ratio, ep in zip(torch.split(log_ratio, lengths), source.episodes):
        weights.append(torch.exp(ep_ratio.sum().clamp(min=-30.0, max=30.0)))
        returns.append(ep["rewards"].sum().cpu())
    return torch.stack(weights), torch.stack(returns).float()


def _extremal_value(
    w: torch.Tensor, g: torch.Tensor, gamma: float, maximize: bool
) -> float:
    """Exact box-constrained optimum of the self-normalized weighted mean by
    threshold scan over return-sorted episodes."""
    order = torch.argsort(g, descending=maximize)
    w_s, g_s = w[order].double(), g[order].double()
    hi, lo = float(gamma), 1.0 / float(gamma)
    n = len(w_s)
    # prefix sums: episodes before the threshold get the boosting multiplier
    # (toward the favorable extreme), the rest get the opposite one.
    best = None
    w_hi, w_lo = w_s * hi, w_s * lo
    num_hi = torch.cumsum(w_hi * g_s, dim=0)
    den_hi = torch.cumsum(w_hi, dim=0)
    num_lo_total = (w_lo * g_s).sum()
    den_lo_total = w_lo.sum()
    num_lo_prefix = torch.cumsum(w_lo * g_s, dim=0)
    den_lo_prefix = torch.cumsum(w_lo, dim=0)
    for k in range(n + 1):
        num = (
            (num_hi[k - 1] if k > 0 else 0.0)
            + num_lo_total
            - (num_lo_prefix[k - 1] if k > 0 else 0.0)
        )
        den = (
            (den_hi[k - 1] if k > 0 else 0.0)
            + den_lo_total
            - (den_lo_prefix[k - 1] if k > 0 else 0.0)
        )
        val = float(num / den) if float(den) > 0 else float("nan")
        if best is None or (maximize and val > best) or (not maximize and val < best):
            best = val
    return float(best)


def kz_interval(
    source: OfflineDatasetSource,
    target: TargetPolicy,
    gamma: float = 2.0,
    clone_seed: int = 0,
) -> KZInterval:
    """Sensitivity interval [V_lb, V_ub] for the target policy's value under
    the per-episode odds-ratio budget ``gamma``."""
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1 (odds-ratio budget)")
    w, g = _episode_weights_returns(source, target, clone_seed=clone_seed)
    lb = _extremal_value(w, g, gamma, maximize=False)
    ub = _extremal_value(w, g, gamma, maximize=True)
    return KZInterval(
        lower=lb, upper=ub, gamma=float(gamma), n_episodes=len(source.episodes)
    )

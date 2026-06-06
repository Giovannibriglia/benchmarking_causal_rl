"""Regret protocol (§6.6): per-episode J, normalized regret, IQM + bootstrap.

Headline metric: ``regret = J(ref) − J(learned)`` against the per-task Cell-1
reference, normalized by ``J(ref) − J(random)``. J is always measured with
TRUE per-episode returns in the genuine environment (full observability,
unconfounded); policies trained on masked views receive their masked
observation through the same wrapper, which does not alter dynamics/rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

# CSV schema for causal_cells_metrics.csv (§6.6)
CAUSAL_CELLS_COLUMNS = [
    "cell",
    "task",
    "anchor",
    "tier",
    "algo",
    "role",  # basic | variant | reference | random
    "seed",
    "J",
    "regret",
    "normalized_regret",
    "gap_kl",
    "gap_js_normalized",
    "ope_naive",
    "ope_dm",
    "ope_ipw",
    "ope_dr",
    "gate_passed",
]


def evaluate_policy(
    env_id: str,
    act_fn: Callable[[np.ndarray], np.ndarray],
    n_episodes: int = 100,
    seed_base: int = 10_000,
    env_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """TRUE per-episode returns of ``act_fn`` (obs -> action) over full
    episodes. ``env_id`` may be a ``causal/`` id so masked policies receive
    their masked view; rewards/dynamics are those of the base env."""
    env = gym.make(env_id, **(env_kwargs or {}))
    returns = []
    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=seed_base + ep)
        done, total = False, 0.0
        while not done:
            action = act_fn(np.asarray(obs, dtype=np.float32))
            obs, r, term, trunc, _ = env.step(action)
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return np.asarray(returns, dtype=np.float64)


def random_policy_returns(
    env_id: str, n_episodes: int = 100, seed_base: int = 77_000
) -> np.ndarray:
    """Once-per-task random-policy floor."""
    env = gym.make(env_id)
    env.action_space.seed(seed_base)
    returns = []
    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=seed_base + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return np.asarray(returns, dtype=np.float64)


# ---------------------------------------------------------------------------
# Aggregation (rliable methodology, hand-rolled)
# ---------------------------------------------------------------------------


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean (mean of the middle 50%)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    mid = v[lo:hi] if hi > lo else v
    return float(mid.mean())


def stratified_bootstrap_iqm_ci(
    matrix: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """95% CI of the IQM over a (task × seed) score matrix, resampling seeds
    within each task stratum (rliable-style)."""
    rng = np.random.default_rng(seed)
    n_tasks, n_seeds = matrix.shape
    stats = []
    for _ in range(int(n_boot)):
        resampled = np.stack(
            [matrix[t, rng.integers(0, n_seeds, n_seeds)] for t in range(n_tasks)]
        )
        stats.append(iqm(resampled.reshape(-1)))
    return float(np.quantile(stats, alpha / 2)), float(
        np.quantile(stats, 1 - alpha / 2)
    )


def probability_of_improvement(
    variant_scores: np.ndarray, basic_scores: np.ndarray, seed: int = 0
) -> float:
    """P(variant > basic) over all score pairs (higher = better)."""
    _ = seed
    v = np.asarray(variant_scores, dtype=np.float64).reshape(-1, 1)
    b = np.asarray(basic_scores, dtype=np.float64).reshape(1, -1)
    return float((v > b).mean() + 0.5 * (v == b).mean())


# ---------------------------------------------------------------------------
# Regret
# ---------------------------------------------------------------------------


@dataclass
class RegretResult:
    j: float
    j_ref: float
    j_random: float
    regret: float
    normalized_regret: float


def compute_regret(j: float, j_ref: float, j_random: float) -> RegretResult:
    regret = j_ref - j
    denom = j_ref - j_random
    normalized = regret / denom if abs(denom) > 1e-9 else float("nan")
    return RegretResult(
        j=float(j),
        j_ref=float(j_ref),
        j_random=float(j_random),
        regret=float(regret),
        normalized_regret=float(normalized),
    )

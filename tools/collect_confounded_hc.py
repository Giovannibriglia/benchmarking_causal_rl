#!/usr/bin/env python3
"""Continuous confounded collection on HalfCheetah (6C item 1).

Logging policy = cached Cell-1 SAC actor mean + U-coupling
(a ~ N(mu(s)+gamma*U*v, sigma^2)), exact biased Gaussian log-probs and U into
Minari infos, FULL observations stored, overwrite guard active. The
ConfoundedEnv adds the per-episode U and the reward shift r' = r + delta*U
(delta = 1.5 ~= one per-step reward std under the reference; reported with
reasoning in the 6C notes).

Usage:
    python -m tools.collect_confounded_hc [--episodes 200] [--gamma 1.0] \
        [--delta 1.5] [--neutered] [--force]
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import torch
from src.benchmarking.causal_cells import _env_dims
from src.causal.confounding import (
    assert_confounded,
    ConfoundedEnv,
    GaussianConfoundedExplorer,
)
from src.data.minari_io import collect_dataset, to_offline_source
from src.envs.registry import register_default_env_wrappers

REF_CKPT = (
    "references/causal-halfcheetah-cell1_sac_seed0_ep125_rl1024_ne8_relu/"
    "reference.pt"
)


def _sac_mean_fn(device):
    from src.benchmarking.checkpoints import load_checkpoint
    from src.rl.off_policy.sac import SquashedGaussianActor

    obs_dim, action_dim, _ = _env_dims("HalfCheetah-v5")
    actor = SquashedGaussianActor(obs_dim, action_dim).to(device)
    actor.load_state_dict(load_checkpoint(REF_CKPT)["policy_state"])
    actor.eval()

    def mean_fn(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = actor(obs)
            return torch.tanh(mu)  # squashed mean in [-1, 1]

    return mean_fn, action_dim


def _dataset_id(gamma: float, delta: float, version: int = 0) -> str:
    if gamma == 0.0 and delta == 0.0:
        return f"causal/halfcheetah/cell7-neutered-v{version}"
    g = f"{gamma:g}".replace(".", "p")
    d = f"{delta:g}".replace(".", "p")
    return f"causal/halfcheetah/cell7-g{g}-d{d}-v{version}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--neutered", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    register_default_env_wrappers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_fn, action_dim = _sac_mean_fn(device)

    configs = [(args.gamma, args.delta)]
    if args.neutered:
        configs.append((0.0, 0.0))

    for gamma, delta in configs:
        run_gate = not args.no_gate and not (gamma == 0.0 and delta == 0.0)
        policy = GaussianConfoundedExplorer(
            mean_fn, gamma=gamma, sigma=args.sigma, action_dim=action_dim
        )
        dataset_id = _dataset_id(gamma, delta)

        def env_factory(delta=delta):
            return ConfoundedEnv(gym.make("HalfCheetah-v5"), delta=delta, seed=12345)

        collect_dataset(
            env_id="HalfCheetah-v5",
            behavior_policy=policy,
            dataset_id=dataset_id,
            n_episodes=args.episodes,
            seed=999_000,
            device=device,
            env_factory=env_factory,
            collection_config={
                "gamma": float(gamma),
                "delta": float(delta),
                "sigma": float(args.sigma),
                "u_dist": "bernoulli_pm1",
                "base_policy": "sac_relu_1M",
            },
            force=args.force,
        )
        if run_gate:
            src = to_offline_source(dataset_id, device, behavior_policy="known")
            rep = assert_confounded(src)  # raises on failure
            print(
                f"[gate PASS] {dataset_id}: |naive-ipw|={rep.naive_ipw_gap:.2f} "
                f"A-U z={rep.action_u_zscore:.1f} R-U z={rep.reward_u_zscore:.1f}"
            )
        else:
            print(f"[no gate] {dataset_id} collected")


if __name__ == "__main__":
    main()

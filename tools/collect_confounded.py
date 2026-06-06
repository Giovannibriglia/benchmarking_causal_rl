#!/usr/bin/env python3
"""Confounded dataset collection + gate (§6.2/§6.3), discrete anchor.

Collects ``causal/cartpole/cell7-b<beta>-d<delta>-v0`` datasets with a
U-biased epsilon-soft logging policy on a ConfoundedEnv (per-episode
U ∈ {−1,+1}; U→action logit bias beta; U→reward shift delta), exact BIASED
propensities + U in Minari infos, FULL observations stored. Every dataset is
passed through ``assert_confounded`` immediately — collection fails hard if
the gate rejects.

A deliberately neutered control (beta=delta=0) is collected WITHOUT gating
(it must FAIL the gate; tests assert that).

Usage:
    python tools/collect_confounded.py [--episodes 300] [--seed 42] \
        [--configs 1.0:0.5 2.0:1.0] [--neutered]
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import torch
from src.causal.confounding import assert_confounded, ConfoundedEnv, ConfoundedExplorer
from src.data.minari_io import collect_dataset, to_offline_source

from tools.collect_offline_tiers import train_logging_policies


def _dataset_id(beta: float, delta: float, version: int = 0) -> str:
    if beta == 0.0 and delta == 0.0:
        return f"causal/cartpole/cell7-neutered-v{version}"
    # Minari forbids '.' in dataset names; encode 0.5 -> 0p5
    b = f"{beta:g}".replace(".", "p")
    d = f"{delta:g}".replace(".", "p")
    return f"causal/cartpole/cell7-b{b}-d{d}-v{version}"


def collect_confounded_cartpole(
    beta: float,
    delta: float,
    n_episodes: int,
    seed: int,
    device: torch.device,
    q_net,
    version: int = 0,
    run_gate: bool = True,
):
    """Collect one (beta, delta) config; optionally enforce the gate."""

    def base_logits(obs: torch.Tensor) -> torch.Tensor:
        # epsilon-soft base policy: temperature-scaled Q logits keep every
        # action's propensity bounded away from 0 (IPW overlap).
        with torch.no_grad():
            return q_net(obs) / 10.0

    policy = ConfoundedExplorer(base_logits, beta=beta)
    dataset_id = _dataset_id(beta, delta, version)

    def env_factory() -> gym.Env:
        return ConfoundedEnv(gym.make("CartPole-v1"), delta=delta, seed=seed * 7 + 1)

    collect_dataset(
        env_id="CartPole-v1",
        behavior_policy=policy,
        dataset_id=dataset_id,
        n_episodes=n_episodes,
        seed=seed * 1000 + 400_000,
        device=device,
        env_factory=env_factory,
    )
    if run_gate:
        source = to_offline_source(dataset_id, device, behavior_policy="known")
        report = assert_confounded(source)  # raises on failure
        print(
            f"[gate PASS] {dataset_id}: |naive-ipw|={report.naive_ipw_gap:.2f} "
            f"A-U z={report.action_u_zscore:.1f} R-U z={report.reward_u_zscore:.1f}"
        )
    else:
        print(f"[no gate] {dataset_id} collected (control)")
    return dataset_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["1.0:0.5", "2.0:1.0"],
        help="beta:delta pairs to collect (gated)",
    )
    parser.add_argument(
        "--neutered",
        action="store_true",
        help="also collect the beta=delta=0 negative control (ungated)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    snapshots = train_logging_policies(args.seed, device)
    q_net = snapshots["medium"]

    for cfg in args.configs:
        beta, delta = (float(x) for x in cfg.split(":"))
        collect_confounded_cartpole(
            beta, delta, args.episodes, args.seed, device, q_net
        )
    if args.neutered:
        collect_confounded_cartpole(
            0.0, 0.0, args.episodes, args.seed, device, q_net, run_gate=False
        )


if __name__ == "__main__":
    main()

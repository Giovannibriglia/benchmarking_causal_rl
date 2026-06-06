#!/usr/bin/env python3
"""Self-collected CartPole tiers: causal/cartpole/{simple,medium,expert}-v0.

Trains one DQN logging policy and snapshots it under-trained (simple),
partially trained (medium) and well-trained (expert); each tier is collected
as a Minari dataset with EXACT epsilon-greedy propensities in
``infos["behavior_logprob"]`` (§6.3). Expert uses a low epsilon, producing
the narrow-support tier for the coverage sweep.

Usage:
    python tools/collect_offline_tiers.py [--episodes 300] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy

import gymnasium as gym
import numpy as np
import torch
from src.config.seeding import set_seed
from src.data.behavior_policies import EpsilonGreedyExplorer
from src.data.minari_io import collect_dataset
from src.rl.nets.mlp import MLP
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer

# tier -> (training steps for the logging policy, collection epsilon,
#          deterministic collection-seed offset)
TIERS = {
    "simple": (2_000, 0.30, 1),
    "medium": (12_000, 0.20, 2),
    "expert": (40_000, 0.05, 3),
}


def train_logging_policies(seed: int, device: torch.device) -> dict[str, MLP]:
    """Train DQN on CartPole-v1, snapshotting the Q-net at tier stages."""
    set_seed(seed, deterministic=True)
    env = gym.make("CartPole-v1")
    q_net = MLP(4, 2).to(device)
    target = MLP(4, 2).to(device)
    agent = DQN(q_net, target, ReplayBuffer(50_000, device), device=device, epsilon=0.1)

    snapshots: dict[str, MLP] = {}
    stages = sorted((steps, tier) for tier, (steps, _, _) in TIERS.items())
    obs, _ = env.reset(seed=seed)
    ep_ret, returns = 0.0, []
    for step in range(1, stages[-1][0] + 1):
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device
        )
        action = int(agent.act(obs_t).action.item())
        next_obs, r, term, trunc, _ = env.step(action)
        ep_ret += r
        agent.buffer.add(
            {
                "obs": obs_t.squeeze(0),
                "actions": torch.tensor(action),
                "rewards": torch.tensor(float(r)),
                "next_obs": torch.as_tensor(
                    np.asarray(next_obs, dtype=np.float32), device=device
                ),
                "dones": torch.tensor(float(term or trunc)),
            }
        )
        obs = next_obs
        if term or trunc:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        if len(agent.buffer) > 256:
            agent.update(agent.buffer.sample(64))
        for steps_needed, tier in stages:
            if step == steps_needed:
                snapshots[tier] = copy.deepcopy(q_net).cpu().eval()
                recent = np.mean(returns[-20:]) if returns else 0.0
                print(
                    f"[tier:{tier}] snapshot at step {step}; "
                    f"recent train return ~{recent:.0f}"
                )
    env.close()
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300, help="episodes per tier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")  # tiny nets; CPU keeps collection portable
    snapshots = train_logging_policies(args.seed, device)

    for tier, (_, eps, seed_offset) in TIERS.items():
        policy = EpsilonGreedyExplorer(snapshots[tier], epsilon=eps)
        dataset_id = f"causal/cartpole/{tier}-v{args.version}"
        collect_dataset(
            env_id="CartPole-v1",
            behavior_policy=policy,
            dataset_id=dataset_id,
            n_episodes=args.episodes,
            seed=args.seed * 1000 + seed_offset * 100_000,
            device=device,
        )
        print(f"collected {dataset_id}: {args.episodes} episodes (epsilon={eps})")


if __name__ == "__main__":
    main()

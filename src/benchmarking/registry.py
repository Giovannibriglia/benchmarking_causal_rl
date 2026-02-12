from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.rl.nets.mlp import MLP
from src.rl.off_policy.ddpg import DDPG
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.on_policy.a2c import A2C
from src.rl.on_policy.policy import ActorCriticMLP
from src.rl.on_policy.ppo import PPO
from src.rl.on_policy.trpo import TRPO
from src.rl.on_policy.vanilla import VanillaPolicyGradient
from .runner import AlgorithmSpec


class Registry:
    def __init__(self):
        self._algos: Dict[str, AlgorithmSpec] = {}

    def register(self, name: str, spec: AlgorithmSpec) -> None:
        self._algos[name.lower()] = spec

    def get(self, name: str) -> AlgorithmSpec:
        key = name.lower()
        if key not in self._algos:
            raise KeyError(f"Algorithm {name} not registered")
        return self._algos[key]


registry = Registry()


def register_default_algorithms() -> None:
    # On-policy builders
    def build_vanilla(**kwargs):
        policy = ActorCriticMLP(
            kwargs["obs_dim"],
            kwargs["action_dim"],
            kwargs["action_type"],
            kwargs["device"],
        )
        agent = VanillaPolicyGradient(policy, device=kwargs["device"])
        return policy, agent

    def build_a2c(**kwargs):
        policy = ActorCriticMLP(
            kwargs["obs_dim"],
            kwargs["action_dim"],
            kwargs["action_type"],
            kwargs["device"],
        )
        agent = A2C(policy, device=kwargs["device"])
        return policy, agent

    def build_ppo(**kwargs):
        policy = ActorCriticMLP(
            kwargs["obs_dim"],
            kwargs["action_dim"],
            kwargs["action_type"],
            kwargs["device"],
        )
        agent = PPO(policy, device=kwargs["device"])
        return policy, agent

    def build_trpo(**kwargs):
        policy = ActorCriticMLP(
            kwargs["obs_dim"],
            kwargs["action_dim"],
            kwargs["action_type"],
            kwargs["device"],
        )
        agent = TRPO(policy, device=kwargs["device"])
        return policy, agent

    # Off-policy builders
    def build_dqn(**kwargs):
        obs_dim = kwargs["obs_dim"]
        action_dim = kwargs["action_dim"]
        device = kwargs["device"]
        q_net = MLP(obs_dim, action_dim)
        target_net = MLP(obs_dim, action_dim)
        buffer = ReplayBuffer(capacity=100_000, device=device)
        agent = DQN(q_net.to(device), target_net.to(device), buffer, device=device)
        return q_net.to(device), agent

    def build_ddpg(**kwargs):
        obs_dim = kwargs["obs_dim"]
        action_dim = kwargs["action_dim"]
        device = kwargs["device"]
        action_space = kwargs["action_space"]
        actor = MLP(obs_dim, action_dim, output_activation=nn.Tanh)
        critic = MLP(obs_dim + action_dim, 1)
        target_actor = MLP(obs_dim, action_dim, output_activation=nn.Tanh)
        target_critic = MLP(obs_dim + action_dim, 1)
        buffer = ReplayBuffer(capacity=100_000, device=device)
        agent = DDPG(
            actor.to(device),
            critic.to(device),
            target_actor.to(device),
            target_critic.to(device),
            buffer,
            device=device,
        )
        # set action bounds if available
        try:
            low = torch.as_tensor(action_space.low, device=device)
            high = torch.as_tensor(action_space.high, device=device)
            agent.action_low = low
            agent.action_high = high
        except Exception:
            pass
        return actor.to(device), agent

    registry.register("vanilla", AlgorithmSpec(builder=build_vanilla, kind="on_policy"))
    registry.register("a2c", AlgorithmSpec(builder=build_a2c, kind="on_policy"))
    registry.register("ppo", AlgorithmSpec(builder=build_ppo, kind="on_policy"))
    registry.register("trpo", AlgorithmSpec(builder=build_trpo, kind="on_policy"))
    registry.register("dqn", AlgorithmSpec(builder=build_dqn, kind="off_policy"))
    registry.register("ddpg", AlgorithmSpec(builder=build_ddpg, kind="off_policy"))

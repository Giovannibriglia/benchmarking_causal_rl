from __future__ import annotations

from typing import Callable, Dict, List

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


# Central env set registry. Values can be static lists or callables returning a list.
ENV_SETS: Dict[str, List[str] | Callable[[], List[str]]] = {
    # Dynamically enumerates all installed Gymnasium env ids.
    "gymnasium": [
        "Blackjack-v1",
        # "CarRacing-v3",
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Acrobot-v1",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "FrozenLake-v1",
        "FrozenLake8x8-v1",
        "CliffWalking-v1",
        "Taxi-v3",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
    ],
    "mujoco": [
        "Ant-v5",
        "Reacher-v5",
        "Pusher-v5",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Swimmer-v5",
        "Walker2d-v5",
        "Humanoid-v5",
        "HumanoidStandup-v5",
    ],
    "gymnasium-robotics": [
        # Fetch (multi-goal, Dict obs)
        "FetchReach-v4",
        "FetchPush-v4",
        "FetchSlide-v4",
        "FetchPickAndPlace-v4",
        # Shadow Dexterous Hand (multi-goal, Dict obs)
        "HandReach-v3",
        "HandManipulateBlock-v1",
        "HandManipulateEgg-v1",
        "HandManipulatePen-v1",
        # Touch-sensor variants (optional; only if you want the bigger obs)
        # "HandManipulateEgg_BooleanTouchSensors-v1",
        # "HandManipulateEgg_ContinuousTouchSensors-v1",
    ],
}


def expand_env_set(name: str) -> List[str]:
    key = name.lower()
    if key not in ENV_SETS:
        raise KeyError(
            f"Unknown env_set '{name}'. Add it to ENV_SETS in src/benchmarking/registry.py."
        )
    entries = ENV_SETS[key]
    envs = entries() if callable(entries) else list(entries)
    if len(envs) == 0:
        raise ValueError(
            f"Environment set '{name}' is defined but empty. Populate ENV_SETS or install the provider."
        )
    return envs


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
    # vanilla_ac: alias of vanilla (same builder/spec); keep the vanilla key.
    registry.register(
        "vanilla_ac", AlgorithmSpec(builder=build_vanilla, kind="on_policy")
    )
    registry.register("a2c", AlgorithmSpec(builder=build_a2c, kind="on_policy"))
    registry.register("ppo", AlgorithmSpec(builder=build_ppo, kind="on_policy"))
    registry.register("trpo", AlgorithmSpec(builder=build_trpo, kind="on_policy"))

    def build_sac(**kwargs):
        from src.rl.off_policy.sac import SAC, SquashedGaussianActor

        obs_dim = kwargs["obs_dim"]
        action_dim = kwargs["action_dim"]
        device = kwargs["device"]
        action_space = kwargs["action_space"]
        actor = SquashedGaussianActor(obs_dim, action_dim).to(device)
        mk_q = lambda: MLP(  # noqa: E731
            obs_dim + action_dim, 1, hidden_dims=(256, 256), activation=nn.ReLU
        )
        q1, q2, q1t, q2t = (mk_q().to(device) for _ in range(4))
        # SAC needs a large buffer for million-step reference training
        buffer = ReplayBuffer(capacity=1_000_000, device=device)
        try:
            scale = float(abs(action_space.high[0]))
        except Exception:
            scale = 1.0
        agent = SAC(
            actor,
            q1,
            q2,
            q1t,
            q2t,
            buffer,
            device=device,
            action_dim=action_dim,
            action_scale=scale,
        )
        return actor, agent

    registry.register("dqn", AlgorithmSpec(builder=build_dqn, kind="off_policy"))
    registry.register("sac", AlgorithmSpec(builder=build_sac, kind="off_policy"))
    registry.register("ddpg", AlgorithmSpec(builder=build_ddpg, kind="off_policy"))

    # Offline (fixed-dataset) algorithms: data_regime="offline" routes run() to
    # _train_offline. Online dqn is left untouched.
    from src.rl.offline.bcq import build_bcq
    from src.rl.offline.cql import build_cql
    from src.rl.offline.dqn import build_offline_dqn
    from src.rl.offline.iql import build_iql

    for _name, _builder in (
        ("offline_dqn", build_offline_dqn),
        ("bcq", build_bcq),
        ("cql", build_cql),
        ("iql", build_iql),
    ):
        registry.register(
            _name,
            AlgorithmSpec(builder=_builder, kind="off_policy", data_regime="offline"),
        )

from typing import Dict, List

from src.base import BaseEnv
from src.envs.gymnasium import GymnasiumEnv

ENV_CLASSES: Dict[str, type[BaseEnv]] = {"gymnasium": GymnasiumEnv}
ENV_NAMES: Dict[str, List[str]] = {
    "gymnasium": [
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Acrobot-v1",
        # "phys2d/CartPole-v1",
        # "phys2d/Pendulum-v0",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
        # "CarRacing-v3",
        # "Blackjack-v1",
        "FrozenLake-v1",
        "FrozenLake8x8-v1",
        "CliffWalking-v0",
        "Taxi-v3",
        # "tabular/Blackjack-v0",
        # "tabular/CliffWalking-v0",
        "Reacher-v5",
        "Pusher-v5",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Swimmer-v5",
        "Walker2d-v5",
        "Ant-v5",
        # "Humanoid-v5",
        # "HumanoidStandup-v5",
    ]
}

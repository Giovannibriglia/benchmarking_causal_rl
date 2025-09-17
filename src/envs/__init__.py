from typing import Dict, List

from src.base import BaseEnv
from src.envs.gymnasium import GymnasiumEnv

ENV_CLASSES: Dict[str, type[BaseEnv]] = {"gymnasium": GymnasiumEnv}
ENV_NAMES: Dict[str, List[str]] = {
    "gymnasium": [
        "FrozenLake-v1",
        "FrozenLake8x8-v1",
        "CliffWalking-v0",
        "Taxi-v3",
        "Ant-v5",
        "Reacher-v5",
        "Pusher-v5",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Swimmer-v5",
        "Walker2d-v5",
        "Blackjack-v1",
        "Humanoid-v5",
        "HumanoidStandup-v5",
        # "CarRacing-v3",
    ]
}
"""
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Acrobot-v1",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",

"""

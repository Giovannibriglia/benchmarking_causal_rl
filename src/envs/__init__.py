from typing import Dict, List

from src.base import BaseEnv
from src.envs.gymnasium import GymnasiumEnv

ENV_CLASSES: Dict[str, type[BaseEnv]] = {
    "gymnasium": GymnasiumEnv,
    "gymnasium-robotics": GymnasiumEnv,
}
ENV_NAMES: Dict[str, List[str]] = {
    "gymnasium": [
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
"""


"""

from typing import Dict, List

from src.base import BaseEnv
from src.envs.gymnasium import GymnasiumEnv

ENV_CLASSES: Dict[str, type[BaseEnv]] = {"gymnasium": GymnasiumEnv}
ENV_NAMES: Dict[str, List[str]] = {"gymnasium": ["FrozenLake-v1", "CartPole-v1"]}

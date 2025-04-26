from datetime import datetime

import gymnasium
import numpy as np
import torch
import yaml
from gymnasium.spaces import Box

from gymnasium_suite.base import BasePolicy

from gymnasium_suite.dqn import make_dqn_policy
from gymnasium_suite.ppo import make_ppo_policy
from gymnasium_suite.sac import make_sac_policy

# map algo‑name → extra scalar keys to log
EXTRA_KEYS = {
    "dqn": ["td_loss", "avg_q", "epsilon"],
    "causal_dqn": ["td_loss", "avg_q", "epsilon"],
    "sac": ["td_loss", "avg_q", "epsilon"],
    "causal_sac": ["td_loss", "avg_q", "epsilon"],
    "a2c": ["entropy", "actor_loss", "value_loss", "kl"],
    "causal_a2c": ["entropy", "actor_loss", "value_loss", "kl"],
    "ppo": ["entropy", "actor_loss", "value_loss", "kl"],
    "causal_ppo": ["entropy", "actor_loss", "value_loss", "kl"],
}


def safe_tensor(x, dtype, device):
    """Clone‑detach if tensor, else construct a new tensor."""
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def find_max_N(n_envs, n_features, leave_free_gb=1.0):
    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory  # in bytes
    available_memory = total_memory - int(leave_free_gb * 1024**3)  # leave N GB free

    memory_per_element = 4  # float32 is 4 bytes
    n_max_elements = available_memory / 4
    max_N = (available_memory / (memory_per_element * n_envs)) ** (1 / n_features)
    max_N = int(max_N)  # floor it to an integer

    if max_N >= 16:
        max_N = 16
    elif max_N >= 8:
        max_N = max_N - (max_N % 8)
    elif max_N >= 4:
        max_N = max_N - (max_N % 4)
    else:
        max_N = 2
    print(
        f"Max_N: {max_N} - max #elements: {n_max_elements} - #features: {n_features} - available memory: {round(available_memory/(1024**3),2)}/{round(total_memory/(1024**3),2)}"
    )
    return max_N


def generate_simulation_name(prefix: str = "benchmarking") -> str:
    """
    Generates a simulation name based on the current date and time.

    Args:
        prefix (str): Prefix for the simulation name (default is "Simulation").

    Returns:
        str: A string containing the prefix and a timestamp.
    """
    # Get the current date and time
    now = datetime.now()
    # Format the date and time into a readable string
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Combine the prefix and timestamp
    simulation_name = f"{prefix}_{timestamp}"
    return simulation_name


def _augment_observation_space(
    orig_dim: int, action_space_shape: int, **kwargs
) -> gymnasium.spaces.Space:
    if action_space_shape > 0:
        N_max = kwargs["causality_init"]["N_max"]
        aug_dim = N_max**action_space_shape
    else:
        aug_dim = 0
    aug_space = Box(
        low=-np.inf, high=np.inf, shape=(orig_dim + aug_dim,), dtype=np.float32
    )

    return aug_space


def get_space_n_features(space):
    if isinstance(space, gymnasium.spaces.Discrete):
        return 1
    elif isinstance(space, gymnasium.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gymnasium.spaces.Tuple):
        return len(space)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def make_policy(
    algo_name: str,
    action_space,
    observation_space,
    n_envs,
    n_episodes,
    **kwargs,
) -> BasePolicy:
    is_causal = True if "causal" in algo_name else False

    if is_causal:

        with open("causality_hyperparams.yaml", "r") as file:
            causality_init = yaml.safe_load(file)

        action_space_shape = get_space_n_features(action_space)
        observation_space_shape = get_space_n_features(observation_space)

        n_features = action_space_shape + observation_space_shape

        causality_init["N_max"] = find_max_N(n_envs, n_features, 4)
        kwargs["causality_init"] = causality_init

        observation_space = _augment_observation_space(
            observation_space_shape, action_space_shape, **kwargs
        )
    elif isinstance(observation_space, gymnasium.spaces.Tuple):
        observation_space_shape = get_space_n_features(observation_space)
        observation_space = _augment_observation_space(
            observation_space_shape, 0, **kwargs
        )

    if "ppo" in algo_name:
        PPO_cls = make_ppo_policy(is_causal)
        return PPO_cls(algo_name, action_space, observation_space, n_envs, **kwargs)
    elif "a2c" in algo_name:
        A2C_cls = make_ppo_policy(is_causal)
        return A2C_cls(algo_name, action_space, observation_space, n_envs, **kwargs)
    elif "dqn" in algo_name:
        DQN_cls = make_dqn_policy(is_causal)
        return DQN_cls(
            algo_name, action_space, observation_space, n_envs, n_episodes, **kwargs
        )
    elif "sac" in algo_name:
        SAC_cls = make_sac_policy(is_causal)
        return SAC_cls(
            algo_name, action_space, observation_space, n_envs, n_episodes, **kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

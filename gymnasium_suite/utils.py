from datetime import datetime

import gymnasium
import numpy as np
import torch
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
    observation_space: gymnasium.spaces.Space, **kwargs
) -> gymnasium.spaces.Space:
    causality_init = kwargs.pop("causality_init", {})

    N_max = causality_init.get("N_max", 16)

    orig_dim = int(np.prod(observation_space.shape))

    # augmented
    aug_dim = N_max**2
    aug_space = Box(
        low=-np.inf, high=np.inf, shape=(orig_dim + aug_dim,), dtype=np.float32
    )

    return aug_space


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
        observation_space = _augment_observation_space(observation_space)

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

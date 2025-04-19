from datetime import datetime

from gymnasium_suite.a2c_minimal import A2CPolicy
from gymnasium_suite.base import BasePolicy, RandomPolicy
from gymnasium_suite.dqn_minimal import DQNPolicy
from gymnasium_suite.ppo_minimal import PPOPolicy


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


def make_policy(
    algo_name: str, action_space, observation_space, n_envs, n_episodes, **kwargs
) -> BasePolicy:
    if algo_name == "random":
        return RandomPolicy(algo_name, action_space, observation_space, n_envs)
    elif algo_name == "ppo":
        return PPOPolicy(algo_name, action_space, observation_space, n_envs, **kwargs)
    elif algo_name == "a2c":
        return A2CPolicy(algo_name, action_space, observation_space, n_envs, **kwargs)
    elif algo_name == "dqn":
        return DQNPolicy(
            algo_name, action_space, observation_space, n_envs, n_episodes, **kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

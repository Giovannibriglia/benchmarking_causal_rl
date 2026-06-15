"""feat/masked-observation-wrapper (PR1) — observation masking machinery.

Covers the MaskedObservationWrapper in isolation (smoke + index validation) and
its composition under ConfoundedCollectionWrapper (the Cell 8 stack: base ->
Confounded -> Masked, docs/experimental_design.md §8). CSV writers and dataset
metadata are out of scope here (PR2/PR3).
"""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch
from gymnasium.spaces import Box, Discrete
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper
from src.envs.wrappers.gymnasium_env import GymnasiumEnv
from src.envs.wrappers.masked import MaskedObservationWrapper

CPU = torch.device("cpu")


def test_masked_wrapper_smoke():
    env = gym.make("CartPole-v1")
    wrapped = MaskedObservationWrapper(env, indices=(1, 3))

    obs, _ = wrapped.reset()
    assert obs.shape == (2,)

    next_obs, reward, terminated, truncated, _ = wrapped.step(0)
    assert next_obs.shape == (2,)

    # Observation space projected to the surviving dims; action space and reward
    # are untouched.
    assert isinstance(wrapped.observation_space, Box)
    assert wrapped.observation_space.shape == (2,)
    assert isinstance(wrapped.action_space, Discrete)
    assert wrapped.action_space.n == env.action_space.n
    assert float(reward) == 1.0


def test_masked_wrapper_validates_indices():
    env = gym.make("CartPole-v1")  # obs dim 4

    with pytest.raises(ValueError):
        MaskedObservationWrapper(env, indices=(7,))  # out of range

    with pytest.raises(ValueError):
        MaskedObservationWrapper(env, indices=(1, 1))  # duplicate


def test_masked_wrapper_composes_with_confounded():
    base = GymnasiumEnv("CartPole-v1", n_envs=1, device=CPU, seed=0)
    confounded = ConfoundedCollectionWrapper(base, c_a=1.0, c_r=1.0)
    wrapped = MaskedObservationWrapper(confounded, indices=(1, 3))

    obs, _ = wrapped.reset(seed=0)
    # Vectorized stack -> obs is (n_envs, obs_dim); masking drops 2 of 4 dims.
    assert obs.shape[-1] == 2

    action = torch.zeros(base.n_envs, dtype=torch.long)
    next_obs, _, _, _, _ = wrapped.step(action)
    assert next_obs.shape[-1] == 2

    # The confounded latent U is still reachable through the mask (attribute
    # delegation), and the eval/agent stack can read it.
    assert wrapped.current_u is not None
    base.close()

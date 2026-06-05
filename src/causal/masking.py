"""Partial-observability transform for the causal cell experiments (§6.1).

``ObservationMasking`` hides a declared subset of observation dimensions from
the LEARNER while the environment dynamics stay untouched. The full
observation is always available to oracles and propensity models through
``info["full_obs"]`` — masking happens at learner-input time only, so
datasets collected through this wrapper still store complete observations.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

MaskSpec = Union[Sequence[int], str]

# Velocity columns for classic-control envs without a MuJoCo
# observation_structure. CartPole: [x, x_dot, theta, theta_dot].
_KNOWN_VELOCITY_INDICES = {
    "CartPole": [1, 3],
    "Acrobot": [4, 5],
    "MountainCar": [1],
    "Pendulum": [2],
}


def resolve_mask_indices(env: gym.Env, mask_indices: MaskSpec) -> List[int]:
    """Resolve a mask spec to concrete observation indices for ``env``.

    ``"velocities"`` resolves via the MuJoCo v5 ``observation_structure``
    (qvel block follows the qpos block) or, for classic-control anchors, a
    known per-env table.
    """
    if not isinstance(mask_indices, str):
        return sorted(int(i) for i in mask_indices)
    if mask_indices != "velocities":
        raise ValueError(
            f"Unknown symbolic mask spec '{mask_indices}'. "
            "Supported: 'velocities' or an explicit index list."
        )

    structure = getattr(env.unwrapped, "observation_structure", None)
    if isinstance(structure, dict) and "qvel" in structure:
        qpos_len = int(structure.get("qpos", 0))
        qvel_len = int(structure["qvel"])
        return list(range(qpos_len, qpos_len + qvel_len))

    spec_id = env.spec.id if env.spec is not None else ""
    for key, indices in _KNOWN_VELOCITY_INDICES.items():
        if key.lower() in spec_id.lower():
            return list(indices)
    raise ValueError(
        f"Cannot resolve 'velocities' for env '{spec_id}': no MuJoCo "
        "observation_structure and no known velocity table entry. "
        "Pass explicit mask indices."
    )


class ObservationMasking(gym.Wrapper):
    """Hide observation dimensions from the learner; expose the full
    observation via ``info["full_obs"]``.

    Emits a flat reduced ``Box`` so it composes under the existing vectorized
    wrapper (§6.1). Only flat Box observation spaces are supported — both
    anchors (CartPole-v1, HalfCheetah-v5) satisfy this.
    """

    def __init__(self, env: gym.Env, mask_indices: MaskSpec = "velocities") -> None:
        super().__init__(env)
        if (
            not isinstance(env.observation_space, Box)
            or len(env.observation_space.shape) != 1
        ):
            raise TypeError(
                "ObservationMasking requires a flat Box observation space, "
                f"got {env.observation_space}."
            )
        obs_dim = env.observation_space.shape[0]
        self.mask_indices = resolve_mask_indices(env, mask_indices)
        if not self.mask_indices:
            raise ValueError("mask_indices resolved to an empty set.")
        if min(self.mask_indices) < 0 or max(self.mask_indices) >= obs_dim:
            raise ValueError(
                f"mask indices {self.mask_indices} out of range for obs dim {obs_dim}."
            )
        if len(self.mask_indices) >= obs_dim:
            raise ValueError("Masking every observation dimension is not allowed.")
        self.keep_indices = [i for i in range(obs_dim) if i not in self.mask_indices]
        self._keep = np.asarray(self.keep_indices, dtype=np.int64)
        self.observation_space = Box(
            low=env.observation_space.low[self._keep],
            high=env.observation_space.high[self._keep],
            dtype=env.observation_space.dtype,
        )

    def _mask(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs)[self._keep]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["full_obs"] = np.asarray(obs).copy()
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["full_obs"] = np.asarray(obs).copy()
        return self._mask(obs), reward, terminated, truncated, info

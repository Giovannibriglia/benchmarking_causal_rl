from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - gymnasium is a hard dependency
    raise ImportError("""Gymnasium is not installed.
Please install it with:
    pip install gymnasium[all]""")

from gymnasium.spaces import Box

# 7-cell egocentric agent view * 12 px/cell = exactly 84x84, so the partial
# render lands on the Nature-CNN's expected input with NO ResizeObservation
# (and therefore no opencv dependency). Holds for the standard small MiniGrid
# envs (agent_view_size = 7).
_TILE_SIZE = 12


def make_minigrid_env(env_id: str, render_mode: str | None = None) -> "gym.Env":
    """Build a ``MiniGrid-*`` env delivering ``(3, 84, 84)`` uint8 RGB obs.

    Parallel to ``make_atari_env``: a single source for the MiniGrid image
    pipeline. Chain:

      ``RGBImgPartialObsWrapper(tile_size=12)`` -> 84x84x3 egocentric RGB render
      ``ImgObsWrapper``                          -> drop the Dict, keep the image
      HWC -> CHW via ``TransformObservation``    -> (3, 84, 84), channels-first

    The channels-first transpose matches the convention ``normalize_image_obs``
    already expects (Atari delivers ``(4,84,84)``; MiniGrid delivers
    ``(3,84,84)`` -- the CNN derives channels from ``obs_shape[0]``, so 3 vs 4
    just works). No frame-stacking: MiniGrid is near-Markovian and the
    egocentric view encodes orientation. The render is genuine uint8 ``[0,255]``,
    so the shared ``/255`` normalization is correct -- unlike the native
    *symbolic* obs (object/color/state indices), where ``/255`` is meaningless.
    """
    from minigrid.wrappers import (  # noqa: F401  (import also registers MiniGrid-* ids)
        ImgObsWrapper,
        RGBImgPartialObsWrapper,
    )

    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env, tile_size=_TILE_SIZE)
    env = ImgObsWrapper(env)
    chw_space = Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.transpose(obs, (2, 0, 1)), chw_space
    )
    return env

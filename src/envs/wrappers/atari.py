from __future__ import annotations

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - gymnasium is a hard dependency
    raise ImportError("""Gymnasium is not installed.
Please install it with:
    pip install gymnasium[all]""")


def make_atari_env(env_id: str, render_mode: str | None = None) -> "gym.Env":
    """Build an ``ALE/*`` env with the canonical Atari preprocessing chain.

    SINGLE SOURCE of the Atari preprocessing config, consumed by both the online
    wrapper (``GymnasiumEnv``) and the offline fixture generator
    (``tools/make_atari_offline.py``) so an offline frame is bit-for-bit the
    representation the online path produces. Re-specifying these params anywhere
    else would let online and offline frames silently drift.

    ``frameskip=1`` at ``gym.make`` so ``AtariPreprocessing`` owns the single
    source of frameskip (ALE/*-v5 defaults to 4; stacking both would give an
    effective 16-frame skip). Output is frame-stacked ``(4, 84, 84)`` uint8.
    """
    import ale_py  # noqa: F401  (registers the ALE/* env ids)

    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def normalize_image_obs(obs, device: torch.device) -> torch.Tensor:
    """Convert frame-stacked uint8 image obs to float CHW normalized to ``[0, 1]``.

    SINGLE SOURCE of the image-obs normalization, consumed by both the online
    wrapper (``GymnasiumEnv._image_obs``) and the offline Minari loader
    (``fill_replay_buffer_from_minari``), so offline frames feeding the CNN are
    byte-identical to the online representation. The loader has no ``/255`` of
    its own; routing both through this helper is what makes that equality
    provable rather than hopeful.

    Frame-stacked obs are already channels-first (``..., C, H, W``) uint8.
    """
    arr = np.asarray(obs)
    return torch.from_numpy(arr).to(device).float().div_(255.0)

# gymnasium_env.py
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # or "osmesa" if EGL isn’t available

from pathlib import Path
from typing import Optional

import gymnasium as gym
import gymnasium_robotics as gr

gym.register_envs(gr)  # makes Fetch*/Hand*/*Maze*, etc. visible to gym.make(...)

import imageio
import numpy as np
from gymnasium.vector import SyncVectorEnv

from src.base import BaseEnv


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    # assume float in [0,1] or general float – convert safely
    if np.issubdtype(frame.dtype, np.floating):
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255)
    return frame.astype(np.uint8)


def _pad_even_hw(frame: np.ndarray) -> np.ndarray:
    """FFmpeg (yuv420p) needs even width/height."""
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    pad = (
        ((0, pad_h), (0, pad_w), (0, 0))
        if frame.ndim == 3
        else ((0, pad_h), (0, pad_w))
    )
    return np.pad(frame, pad, mode="edge")


class SingleVideoRecorder(gym.Wrapper):
    """
    Record ALL frames across episodes into ONE mp4 using imageio-ffmpeg.
    Requires env with render_mode='rgb_array'.
    """

    def __init__(self, env: gym.Env, video_path: str | Path, fps: int = 30):
        super().__init__(env)
        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[imageio.Writer] = None
        self.fps = int(fps)
        # sanity: env should advertise rgb_array
        supported = getattr(env, "metadata", {}).get("render_modes", [])
        if "rgb_array" not in supported:
            raise RuntimeError(
                f"{type(env).__name__} does not support 'rgb_array'. render_modes={supported}"
            )

    # ------------- internals -------------
    def _open_if_needed(self, frame: np.ndarray):
        if self._writer is not None:
            return
        frame = _pad_even_hw(_to_uint8(frame))
        # Use FFmpeg backend; yuv420p for broad compatibility
        self._writer = imageio.get_writer(
            str(self.video_path),
            fps=self.fps,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,  # keep exact resolution
            pixelformat="yuv420p",
        )
        self._writer.append_data(frame)

    def _write_frame(self):
        frame = self.env.render()
        if frame is None:
            return  # some envs return None immediately after reset; just skip
        frame = _pad_even_hw(_to_uint8(frame))
        if self._writer is None:
            self._open_if_needed(frame)
        else:
            self._writer.append_data(frame)

    # ------------- Gym API -------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # not all envs render a valid frame right after reset — best-effort
        self._write_frame()
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self._write_frame()
        return out

    def close(self):
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            self._writer = None
            super().close()


class GymnasiumEnv(BaseEnv):
    def __init__(
        self,
        *args,
        record_indices=None,
        video_name_prefix: str | None = None,
        video_fps: int = 30,
        **kwargs,
    ):
        self.record_indices = set(record_indices or {0})
        self.video_name_prefix = video_name_prefix or "run"
        self.video_fps = int(video_fps)
        super().__init__(*args, **kwargs)

    def _make_single(self, idx: int, record: bool):
        def thunk():
            kwargs = dict(self.make_kwargs)
            if record and (idx in self.record_indices):
                kwargs.pop("render_mode", None)
                kwargs["render_mode"] = "rgb_array"

            env = gym.make(self.env_id, **kwargs)

            if record and (idx in self.record_indices):
                video_file = self.video_dir / f"{self.video_name_prefix}.mp4"
                env = SingleVideoRecorder(
                    env, video_path=video_file, fps=self.video_fps
                )
            """
            if record and (idx in self.record_indices):
                video_file = self.video_dir / f"{self.video_name_prefix}.mp4"
                env = SingleVideoRecorder(
                    env, video_path=video_file, fps=self.video_fps
                )
            """

            env.reset(seed=self.seed + idx)
            return env

        return thunk

    def _setup_env(self):
        fns = []
        for i in range(self.n_envs):
            record_this = bool(self.record_video and (i in self.record_indices))
            fns.append(self._make_single(i, record_this))
        self.vec_env = SyncVectorEnv(fns)
        self.numpy_actions = True

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import gymnasium as gym
except ImportError:
    raise ImportError("""Gymnasium is not installed.
Please install it with:
    pip install gymnasium[all]""")


def _maybe_import_robotics(env_id: str) -> None:
    """Import gymnasium-robotics when a robotics env is requested.

    Gymnasium only registers robotics environments when the package is imported.
    This helper avoids NameNotFound errors for env ids like Fetch*, Hand*, Adroit*,
    and Franka* by loading the plugin on demand and surfacing a clear message if it
    isn't installed.
    """

    prefixes = ("fetch", "hand", "adroit", "franka", "kitchen")
    if not any(env_id.lower().startswith(p) for p in prefixes):
        return

    try:
        import gymnasium_robotics  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "gymnasium-robotics is required for robotics environments like "
            f"'{env_id}'. Install via `pip install gymnasium-robotics`."
        ) from exc


import numpy as np
import torch
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten, flatten_space

from ..base import BaseEnv


@dataclass
class SingleVideoRecorder:
    path: str
    fps: int = 30

    def __post_init__(self) -> None:
        import imageio

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._writer = imageio.get_writer(
            self.path, fps=self.fps, macro_block_size=None
        )

    def add_frame(self, frame: np.ndarray) -> None:
        # enforce even dimensions for FFmpeg
        h, w = frame.shape[:2]
        if h % 2 == 1:
            frame = frame[:-1, :]
        if w % 2 == 1:
            frame = frame[:, :-1]
        self._writer.append_data(frame)

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


class GymnasiumEnv(BaseEnv):
    def __init__(
        self,
        env_id: str,
        n_envs: int,
        device: torch.device,
        seed: int,
        render: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None,
        numpy_actions: bool = False,
    ) -> None:
        self.env_id = env_id
        self.n_envs = n_envs
        self.device = device
        self.render = render or record_video
        self.record_video = record_video
        self.video_path = video_path
        self.numpy_actions = numpy_actions
        self.base_seed = seed

        def make_env(rank: int) -> Callable[[], gym.Env]:
            def _thunk():
                _maybe_import_robotics(self.env_id)
                env = gym.make(
                    self.env_id, render_mode="rgb_array" if self.render else None
                )
                env.reset(seed=self.base_seed + rank)
                return env

            return _thunk

        self.env = gym.vector.SyncVectorEnv([make_env(i) for i in range(self.n_envs)])
        self.obs_space: Space = flatten_space(self.env.single_observation_space)
        self.act_space: Space = self.env.single_action_space

        self.video_recorder: Optional[SingleVideoRecorder] = None
        if self.record_video and self.video_path:
            self.video_recorder = SingleVideoRecorder(self.video_path)

    def _flatten_obs(self, obs) -> torch.Tensor:
        # Handle structured observations by flattening per environment
        single_space = self.env.single_observation_space
        flat_sample = flatten(single_space, single_space.sample())
        target_shape = np.asarray(flat_sample).shape

        if isinstance(obs, dict):
            flat_list = []
            for i in range(self.n_envs):
                single = {k: v[i] for k, v in obs.items()}
                flat_list.append(flatten(single_space, single))
            flat = np.stack(flat_list)
        elif isinstance(obs, (list, tuple)):
            flat_list = [flatten(single_space, o) for o in obs]
            flat = np.stack(flat_list)
        else:
            arr = np.asarray(obs)
            # If already flattened to target shape per env, accept directly
            if arr.shape[1:] == target_shape:
                flat = arr.astype(np.float32)
            else:
                flat_list = [flatten(single_space, o) for o in arr]
                flat = np.stack(flat_list)
        flat = flat.astype(np.float32)
        return torch.from_numpy(flat).to(self.device)

    def _format_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        act_space = self.act_space
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)

        if hasattr(act_space, "n"):  # discrete
            action_np = action_np.astype(np.int64).reshape(-1)
            if action_np.shape[0] == 1 and self.n_envs > 1:
                action_np = np.repeat(action_np, self.n_envs)
            return action_np

        # continuous Box
        action_np = action_np.astype(np.float32)
        if action_np.ndim == 1:
            action_np = np.expand_dims(action_np, 0)
        if action_np.shape[0] == 1 and self.n_envs > 1:
            action_np = np.repeat(action_np, self.n_envs, axis=0)
        return action_np

    def reset(self, seed: int | None = None):
        if seed is not None:
            # update seeds for all envs sequentially
            for i, env in enumerate(self.env.envs):
                env.reset(seed=seed + i)
        obs, info = self.env.reset()
        obs_tensor = self._flatten_obs(obs)
        return obs_tensor, info

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        act_np = self._format_action(action)

        obs, reward, terminated, truncated, info = self.env.step(act_np)

        done = np.logical_or(terminated, truncated)
        if done.any():
            if hasattr(self.env, "reset_done"):
                reset_obs, reset_info = self.env.reset_done(done)
            else:
                reset_obs, reset_info = self.env.reset()
            # replace obs for done envs
            if isinstance(obs, dict):
                for k, v in obs.items():
                    src = reset_obs[k] if reset_obs is not None else v
                    v[done] = src[done]
            else:
                obs[done] = reset_obs[done] if reset_obs is not None else obs[done]
            # merge infos
            for i, d in enumerate(done):
                if d:
                    info["final_info"] = info.get("final_info", {})
                    info["final_info"][i] = (
                        reset_info[i]
                        if isinstance(reset_info, dict) and i in reset_info
                        else {}
                    )

        obs_tensor = self._flatten_obs(obs)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        term_tensor = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        trunc_tensor = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        if self.record_video and self.video_recorder is not None:
            frame = self.env.envs[0].render()
            if frame is not None:
                self.video_recorder.add_frame(frame)

        return obs_tensor, reward_tensor, term_tensor, trunc_tensor, info

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
        try:
            self.env.close()
        except Exception as exc:
            # Mujoco EGL contexts can raise EGL_NOT_INITIALIZED during interpreter teardown; ignore to avoid noisy exit.
            if "EGL_NOT_INITIALIZED" not in str(exc):
                raise

    def start_video(self, path: str) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
        self.video_recorder = SingleVideoRecorder(path)
        self.record_video = True

    def stop_video(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
        self.video_recorder = None
        self.record_video = False

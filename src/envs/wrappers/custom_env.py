from __future__ import annotations

import importlib
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten, flatten_space

from ..base import BaseEnv
from .video import SingleVideoRecorder


def load_entry_point(entry_point: str) -> Callable[..., object]:
    if ":" not in entry_point:
        raise ValueError(
            f"Invalid entry point '{entry_point}'. Expected format 'module:callable'."
        )
    module_name, attr = entry_point.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if factory is None or not callable(factory):
        raise TypeError(
            f"Entry point '{entry_point}' did not resolve to a callable object."
        )
    return factory


def _call_factory(
    factory: Callable[..., object], env_id: str, env_kwargs: dict
) -> object:
    if env_kwargs:
        try:
            return factory(env_id=env_id, **env_kwargs)
        except TypeError:
            return factory(**env_kwargs)
    try:
        return factory(env_id=env_id)
    except TypeError:
        return factory()


def _extract_spaces(env: object) -> Tuple[Space, Space]:
    obs_space = getattr(env, "observation_space", None) or getattr(
        env, "obs_space", None
    )
    act_space = getattr(env, "action_space", None) or getattr(env, "act_space", None)
    if obs_space is None or act_space is None:
        raise ValueError(
            "Custom env must define 'observation_space' and 'action_space' (Gymnasium spaces)."
        )
    return obs_space, act_space


class CustomEnv(BaseEnv):
    """Vectorized wrapper for user-defined environments.

    The custom environment must provide:
    - observation_space: gymnasium Space
    - action_space: gymnasium Space
    - reset(seed=...) -> obs or (obs, info)
    - step(action) -> (obs, reward, terminated, truncated, info) or (obs, reward, done, info)
    - optional render() and close()
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int,
        device: torch.device,
        seed: int,
        render: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None,
        env_entry_point: str | Callable[..., object] | None = None,
        env_kwargs: Optional[dict] = None,
    ) -> None:
        if env_entry_point is None:
            raise ValueError(
                "CustomEnv requires env_entry_point to build environments."
            )
        factory = (
            load_entry_point(env_entry_point)
            if isinstance(env_entry_point, str)
            else env_entry_point
        )
        self.env_id = env_id
        self.n_envs = n_envs
        self.device = device
        self.render = render or record_video
        self.record_video = record_video
        self.video_path = video_path
        self.base_seed = seed
        self.env_kwargs = env_kwargs or {}
        self._factory = factory

        self.envs = [self._make_env() for _ in range(self.n_envs)]
        for i, env in enumerate(self.envs):
            self._reset_env(
                env, self.base_seed + i if self.base_seed is not None else None
            )
        obs_space, act_space = _extract_spaces(self.envs[0])
        self._single_obs_space: Space = obs_space
        self.obs_space: Space = flatten_space(obs_space)
        self.act_space: Space = act_space

        self.video_recorder: Optional[SingleVideoRecorder] = None
        if self.record_video and self.video_path:
            self.video_recorder = SingleVideoRecorder(self.video_path)

    def _make_env(self) -> object:
        return _call_factory(self._factory, self.env_id, self.env_kwargs)

    def _reset_env(self, env: object, seed: int | None) -> Tuple[object, dict]:
        if seed is None:
            result = getattr(env, "reset")()
        else:
            try:
                result = getattr(env, "reset")(seed=seed)
            except TypeError:
                if hasattr(env, "seed"):
                    try:
                        getattr(env, "seed")(seed)
                    except Exception:
                        pass
                result = getattr(env, "reset")()
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1] if result[1] is not None else {}
            if len(result) == 1:
                return result[0], {}
        return result, {}

    def _flatten_obs(self, obs) -> torch.Tensor:
        single_space = self._single_obs_space
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

        if hasattr(act_space, "n"):
            action_np = action_np.astype(np.int64).reshape(-1)
            if action_np.shape[0] == 1 and self.n_envs > 1:
                action_np = np.repeat(action_np, self.n_envs)
            return action_np

        action_np = action_np.astype(np.float32)
        if action_np.ndim == 1:
            action_np = np.expand_dims(action_np, 0)
        if action_np.shape[0] == 1 and self.n_envs > 1:
            action_np = np.repeat(action_np, self.n_envs, axis=0)
        return action_np

    def reset(self, seed: int | None = None):
        base_seed = seed if seed is not None else None
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            obs, info = self._reset_env(
                env, base_seed + i if base_seed is not None else None
            )
            obs_list.append(obs)
            info_list.append(info)
        obs_tensor = self._flatten_obs(obs_list)
        info = {"infos": info_list} if any(info_list) else {}
        return obs_tensor, info

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        act_np = self._format_action(action)

        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        final_info = {}

        for i, env in enumerate(self.envs):
            env_action = act_np[i] if self.n_envs > 1 else act_np[0]
            result = getattr(env, "step")(env_action)
            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, terminated, truncated, info = result
            elif isinstance(result, tuple) and len(result) == 4:
                obs, reward, done, info = result
                terminated, truncated = bool(done), False
            else:
                raise ValueError("Custom env step must return 4 or 5 values.")

            done = bool(terminated) or bool(truncated)
            if done:
                reset_obs, reset_info = self._reset_env(env, None)
                if reset_obs is not None:
                    obs = reset_obs
                final_info[i] = reset_info

            obs_list.append(obs)
            reward_list.append(float(reward))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))

        obs_tensor = self._flatten_obs(obs_list)
        reward_tensor = torch.as_tensor(
            reward_list, dtype=torch.float32, device=self.device
        )
        term_tensor = torch.as_tensor(
            terminated_list, dtype=torch.bool, device=self.device
        )
        trunc_tensor = torch.as_tensor(
            truncated_list, dtype=torch.bool, device=self.device
        )

        info = {}
        if final_info:
            info["final_info"] = final_info

        if self.record_video and self.video_recorder is not None:
            frame = getattr(self.envs[0], "render", lambda: None)()
            if frame is not None:
                self.video_recorder.add_frame(frame)

        return obs_tensor, reward_tensor, term_tensor, trunc_tensor, info

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

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

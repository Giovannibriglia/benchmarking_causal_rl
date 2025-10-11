# gymnasium_env.py
from __future__ import annotations

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo

from src.base import BaseEnv


class GymnasiumEnv(BaseEnv):
    def __init__(
        self, *args, record_indices=None, video_name_prefix: str | None = None, **kwargs
    ):
        # record only env 0 unless overridden
        self.record_indices = set(record_indices or {0})
        # IMPORTANT: we will pass an explicit prefix from Benchmark per checkpoint
        self.video_name_prefix = video_name_prefix or "run"
        super().__init__(*args, **kwargs)

    def _make_single(self, idx: int, record: bool):
        def thunk():
            kwargs = dict(self.make_kwargs)
            if record and (idx in self.record_indices):
                kwargs.pop("render_mode", None)
                kwargs["render_mode"] = "rgb_array"

            env = gym.make(self.env_id, **kwargs)

            if record and (idx in self.record_indices):
                supported = getattr(env, "metadata", {}).get("render_modes", [])
                if "rgb_array" not in supported:
                    raise RuntimeError(
                        f"{self.env_id} lacks 'rgb_array' in metadata.render_modes={supported}."
                    )
                env = RecordVideo(
                    env,
                    video_folder=str(self.video_dir),
                    episode_trigger=lambda ep: True,
                    name_prefix=self.video_name_prefix,  # ← exactly what Benchmark sets (e.g., ckpt003)
                )

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

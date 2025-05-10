import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo

from src.base import BaseEnv


class GymnasiumEnv(BaseEnv):
    def _make_single(self, idx: int, record: bool):
        def thunk():
            kwargs = dict(self.make_kwargs)
            if record:
                kwargs.setdefault("render_mode", "rgb_array")
            env = gym.make(self.env_id, **kwargs)
            if record:
                env = RecordVideo(
                    env,
                    video_folder=self.video_dir,
                    name_prefix=f"idx{idx}",
                )
            env.reset(seed=self.seed + idx)
            return env

        return thunk

    def _setup_env(self):
        self.vec_env = SyncVectorEnv(
            [self._make_single(i, self.record_video) for i in range(self.n_envs)]
        )

        self.numpy_actions = True

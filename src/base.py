from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings(
    "ignore",
    message=".*WARN: Overwriting existing videos at*",
    category=UserWarning,
)


class MetricBuffer(dict):
    def add(self, **kv):
        for k, v in kv.items():
            self[k] = self.get(k, 0.0) + float(v)

    def pop(self, divisor: int | None = None) -> Dict[str, float]:
        div = divisor or 1
        out = {k: v / max(div, 1) for k, v in self.items()}
        self.clear()
        return out


class BaseEnv(ABC):
    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        seed: int = 42,
        device: str | torch.device = DEFAULT_DEVICE,
        record_video: bool = False,
        video_dir: str | Path | None = None,
        **make_kwargs,
    ) -> None:
        self.env_id = env_id
        self.n_envs = n_envs
        self.seed = seed
        self.device = torch.device(device)
        self.record_video = record_video
        self.video_dir = Path(video_dir or "videos").expanduser()
        self.make_kwargs = make_kwargs

        self.vec_env = None
        self.numpy_actions = False
        self._setup_env()

    # ------------------------------------------------------------------
    def _to_tensor(self, x, *, dtype=None):
        """Return *x* as torch.Tensor on self.device (avoid copies if possible)."""
        if isinstance(x, torch.Tensor):
            if x.device is self.device and (dtype is None or x.dtype == dtype):
                return x  # nothing to do
            return x.to(device=self.device, dtype=dtype or x.dtype)
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------
    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        if self.numpy_actions:
            obs, rew, term, trunc, info = self.vec_env.step(action.cpu().numpy())
        else:
            obs, rew, term, trunc, info = self.vec_env.step(action)
        return (
            self._to_tensor(obs),
            self._to_tensor(rew, dtype=torch.float32),
            self._to_tensor(term, dtype=torch.bool),
            self._to_tensor(trunc, dtype=torch.bool),
            info,
        )

    def reset(self) -> torch.Tensor:
        obs, _ = self.vec_env.reset(seed=self.seed)
        return self._to_tensor(obs)

    # ------------------------------------------------------------------
    @property
    def action_space(self):
        return self.vec_env.single_action_space

    @property
    def observation_space(self):
        return self.vec_env.single_observation_space

    # ------------------------------------------------------------------
    @abstractmethod
    def _setup_env(self):
        raise NotImplementedError

    def close(self):
        self.vec_env.close()


class BasePolicy(ABC):
    def __init__(
        self,
        env: BaseEnv,
        eval_env: BaseEnv,
        rollout_len: int = 1024,
        device: str | torch.device = DEFAULT_DEVICE,
        **kwargs_agent,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.rollout_len = rollout_len
        self.device = torch.device(device)
        self.train_metrics, self.eval_metrics = MetricBuffer(), MetricBuffer()

    # ------------------------------------------------------------------
    def _rollout_evaluation(self, env: BaseEnv) -> Tuple[List[int], List[float]]:
        obs = env.reset().to(self.device)
        total_r = torch.zeros(env.n_envs, device=self.device)
        total_l = torch.zeros(env.n_envs, device=self.device)

        for _ in range(self.rollout_len):
            with torch.no_grad():
                act = self._get_action(obs)
            nxt_obs, rew, term, trunc, _ = env.step(act)
            nxt_obs, rew = nxt_obs.to(self.device), rew.to(self.device)

            total_r += rew
            total_l += 1
            obs = nxt_obs

            # stop when **all** envs are done
            if (term | trunc).all():
                break

        # move to CPU & convert to Python lists
        lengths = total_l.cpu().tolist()  # [len_env0, len_env1, ...]
        returns = total_r.cpu().tolist()  # [ret_env0, ret_env1, ...]
        return lengths, returns

    def evaluate(self):
        lengths, returns = self._rollout_evaluation(self.eval_env)

        # log each env separately, so we know “where” each reward/length came from
        for idx, (l, r) in enumerate(zip(lengths, returns)):
            self.eval_metrics.add(
                **{f"evaluation_length_{idx}": l, f"evaluation_return_{idx}": r}
            )

    # ------------------------------------------------------------------
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def _get_action(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _ensure_pt_path(path: Union[str, Path]) -> Path:
        p = Path(path)
        return p.with_suffix(".pt") if p.suffix != ".pt" else p

    def save_policy(self, path: Union[str, Path]):
        torch.save({}, self._ensure_pt_path(path))

    def load_policy(self, path: Union[str, Path]):
        _ = torch.load(self._ensure_pt_path(path))

    def _setup_actor_prior(self):
        raise NotImplementedError

    def _setup_critic_prior(self):
        raise NotImplementedError

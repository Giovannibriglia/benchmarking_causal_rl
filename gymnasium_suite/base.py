import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical, Normal


class MetricBuffer(dict):
    def add(self, **k):
        for key, val in k.items():
            self[key] = self.get(key, 0.0) + float(val)

    def dump(self, divisor: int) -> dict:
        out = {k: v / max(divisor, 1) for k, v in self.items()}
        self.clear()
        return out


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(indices.long().view(-1), num_classes=num_classes).float()


def safe_clone(x, dtype=None, device: torch.device = "cuda"):
    """clone‑detach tensors, wrap numpy → tensor"""
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


class BasePolicy(ABC):
    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces,
        observation_space: gym.spaces,
        n_envs: int,
        **kwargs,
    ):
        self.algo_name = algo_name
        self.action_space = action_space
        self.observation_space = observation_space

        self.n_envs = n_envs

        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.metrics = MetricBuffer()

    @abstractmethod
    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def setup_actions(
        self,
        actions_tensor: torch.Tensor,  # (n_envs, ...) – raw policy output
        done_mask: torch.Tensor,  # (n_envs,)     – True where episode finished
        out_type: str = "numpy",  # "numpy" | "torch"
    ):
        """
        Convert `actions_tensor` to the exact datatype/shape expected by
        `SyncVectorEnv.step`.  Always returns a 1‑D container of length n_envs.
        """
        if out_type not in {"numpy", "torch"}:
            raise ValueError("out_type must be 'numpy' or 'torch'")

        # ---------------------------------------------------------------------
        # DISCRETE ACTION SPACE  (CartPole, MountainCar, etc.)
        # ---------------------------------------------------------------------
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions_tensor = (actions_tensor > 0).to(torch.long)

            # Case A ─ logits / Q‑values  [n_envs, n]  → choose arg‑max
            if actions_tensor.ndim > 1:
                actions_tensor = torch.argmax(actions_tensor, dim=1)

            # Case B ─ single float per env  [n_envs]  → threshold at 0
            elif actions_tensor.dtype.is_floating_point:
                actions_tensor = (actions_tensor > 0).to(torch.long)

            actions_tensor.to(device=self.device)

            # zero‑out finished envs
            actions_tensor = torch.where(
                done_mask,
                torch.zeros_like(actions_tensor, device=self.device),
                actions_tensor,
            )

            if out_type == "torch":
                return actions_tensor  # shape (n_envs,)  torch.long
            else:
                return actions_tensor.cpu().numpy().astype(np.int32)  # shape (n_envs,)

        # ---------------------------------------------------------------------
        # CONTINUOUS (BOX) ACTION SPACE
        # ---------------------------------------------------------------------
        elif isinstance(self.action_space, gym.spaces.Box):
            # If the Box has ≥1 dimension but you only need the first scalar,
            # keep actions_tensor[:, 0]; otherwise keep the full vector.
            if self.action_space.shape == ():
                # scalar Box → squeeze to (n_envs,)
                actions_tensor = actions_tensor.squeeze(-1)

            # zero‑out finished envs
            actions_tensor = torch.where(
                done_mask.view(-1, *([1] * (actions_tensor.ndim - 1))),
                torch.zeros_like(actions_tensor),
                actions_tensor,
            )

            if out_type == "torch":
                return actions_tensor  # dtype matches policy output
            else:
                return actions_tensor.cpu().numpy()  # np.ndarray, shape (n_envs, …)

        # ---------------------------------------------------------------------
        else:
            raise NotImplementedError(
                f"Unsupported action space type: {type(self.action_space)}"
            )

    @abstractmethod
    def pop_metrics(self) -> dict:  # <── call at episode end
        raise NotImplementedError

    @staticmethod
    def _ensure_pt_path(path: str | os.PathLike) -> Path:
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def save_policy(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load_policy(self, path: str):
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces,
        observation_space: gym.spaces,
        n_envs: int,
    ):
        super().__init__(algo_name, action_space, observation_space, n_envs)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        pass

    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:

        if isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (self.n_envs,)

            actions_tensor = torch.randint(
                0, self.action_space.n, size=action_shape, device=self.device
            )
        elif isinstance(self.action_space, gym.spaces.Box):
            action_shape = (self.n_envs, self.action_space.shape[0])

            # Original low/high from the action space
            low = torch.tensor(self.action_space.low, device=self.device)
            high = torch.tensor(self.action_space.high, device=self.device)

            # Replace infs with large finite values
            finite_low = torch.where(
                torch.isinf(low), torch.full_like(low, -1e6, device=self.device), low
            )
            finite_high = torch.where(
                torch.isinf(high), torch.full_like(high, 1e6, device=self.device), high
            )

            actions_tensor = finite_low + (finite_high - finite_low) * torch.rand(
                size=action_shape, device=self.device
            )
        else:
            raise ValueError(
                f"Unsupported action space type: {type(self.action_space)}"
            )

        return actions_tensor


class BaseACPolicy(BasePolicy, ABC):
    def __init__(
        self,
        algo_name: str,
        act_space: gym.spaces.Space,
        obs_space: gym.spaces.Space,
        n_envs: int,
        *,
        rollout_len: int = 128,
        gamma: float = 0.98,
        lr: float = 3e-4,
        **kwargs,
    ):
        super().__init__(algo_name, act_space, obs_space, n_envs, **kwargs)

        # ---- obs encoder -----------------------------------------------------
        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(obs_space.shape))
            self._enc: Callable = lambda x: safe_clone(
                x, torch.float32, self.device
            ).view(-1, self.obs_dim)
        elif isinstance(obs_space, gym.spaces.Discrete):
            self.obs_dim = obs_space.n
            self._enc = lambda x: one_hot(
                safe_clone(x, torch.long, self.device), self.obs_dim
            )
        else:
            raise NotImplementedError("Unsupported observation space")

        # ---- net, optimiser, buffer -----------------------------------------
        self.net = ACNet(self.obs_dim, act_space).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.rollout_len = rollout_len

        self.metrics = MetricBuffer()
        self._reset_buf()

    # ─── buffer helpers ─────────────────────────────────────────────────────
    def _reset_buf(self):
        self.buf: Dict[str, List] = {k: [] for k in ("s", "a", "r", "d", "logp", "v")}

    def _store(self, s, a, r, d, logp, v):
        self.buf["s"].append(s)
        self.buf["a"].append(a)
        self.buf["r"].append(safe_clone(r, torch.float32, self.device))
        self.buf["d"].append(safe_clone(d, torch.float32, self.device))
        self.buf["logp"].append(logp)
        self.buf["v"].append(v)

    # ─── common action selection  (called by env loop) ──────────────────────
    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        enc = self._enc(observations)  # (n_envs, obs_dim)
        dist = self.net.dist(enc)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if log_prob.ndim > 1:  # continuous: sum over action dims
            log_prob, entropy = log_prob.sum(-1), entropy.sum(-1)

        value = self.net.value(enc)

        # track entropy per step
        self.metrics.add(entropy=entropy.mean().item())

        # store with dummy reward/done (will be overwritten in update)
        self._store(enc, actions.detach(), 0.0, 0.0, log_prob.detach(), value.detach())
        return actions  # let caller format via setup_actions

    # ─── abstract: subclasses implement algorithm‑specific update ───────────
    @abstractmethod
    def update(self, obs, acts, rews, next_obs, dones): ...

    # pop averaged metrics once per episode
    def pop_metrics(self) -> Dict[str, float]:
        return self.metrics.dump(divisor=self.rollout_len)

    def _extra_to_save(self) -> dict:
        """Sub‑classes may override to add algorithm‑specific scalars."""
        return {}

    def _load_extra(self, d: dict):
        """Sub‑classes may override to read what they saved."""
        pass

    def save_policy(self, path: str):
        path = self._ensure_pt_path(path)
        payload = {
            "net": self.net.state_dict(),
            "optim": self.opt.state_dict(),
            "buffer": self.buf,  # you may skip if memory heavy
            **self._extra_to_save(),
        }
        torch.save(payload, path)
        print(f"[{self.algo_name.upper()}] policy saved → {path}")

    def load_policy(self, path: str):
        path = self._ensure_pt_path(path)
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["net"])
        self.opt.load_state_dict(payload["optim"])
        self.buf = payload.get(
            "buffer", {k: [] for k in ("s", "a", "r", "d", "logp", "v")}
        )
        self._load_extra(payload)
        print(f"[{self.algo_name.upper()}] policy loaded ← {path}")


class ACNet(nn.Module):
    """Common actor–critic network for Box *or* Discrete actions."""

    def __init__(self, obs_dim: int, act_space: gym.spaces.Space):
        super().__init__()
        self.discrete = isinstance(act_space, gym.spaces.Discrete)
        act_dim = act_space.n if self.discrete else int(np.prod(act_space.shape))

        self.torso = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        if self.discrete:
            self.logits = nn.Linear(128, act_dim)
        else:
            self.mu = nn.Linear(128, act_dim)
            self.std = nn.Linear(128, act_dim)
            self.act_bd = float(act_space.high[0])
        self.v_head = nn.Linear(128, 1)

    def dist(self, x: torch.Tensor):
        h = self.torso(x)
        return (
            Categorical(logits=self.logits(h))
            if self.discrete
            else Normal(
                self.act_bd * torch.tanh(self.mu(h)), F.softplus(self.std(h)) + 1e-5
            )
        )

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_head(self.torso(x)).squeeze(-1)

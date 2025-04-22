import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal


class MetricBuffer(dict):
    def add(self, **kv):
        for k, v in kv.items():
            self[k] = self.get(k, 0.0) + float(v)

    def dump(self, div: int = 1):  # average & reset
        out = {k: v / max(div, 1) for k, v in self.items()}
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

        # holder for the current extra state (set each env step)
        self._extra_state: torch.Tensor | None = None

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
    def get_actions(
        self, observations: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
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

    def set_extra_state(self, x: torch.Tensor | None):
        """
        Store a tensor of shape (n_envs, *extra_dims).
        Pass None to disable augmentation for the next step.
        The tensor is **not copied**; keep it alive until after `update`.
        """
        self._extra_state = x.to(self.device) if x is not None else None


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
        extra_input_dim: int = 0,  # N*M if you have extra tensor (flattened)
        **kwargs,
    ):
        super().__init__(algo_name, act_space, obs_space, n_envs, **kwargs)

        # ---------- obs encoder ------------------------------------------
        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(obs_space.shape))
            self._base_enc: Callable = lambda x: safe_clone(
                x, torch.float32, self.device
            ).view(-1, self.obs_dim)
        elif isinstance(obs_space, gym.spaces.Discrete):
            self.obs_dim = obs_space.n
            self._base_enc = lambda x: one_hot(
                safe_clone(x, torch.long, self.device), self.obs_dim
            )
        else:
            raise NotImplementedError

        in_dim = self.obs_dim + extra_input_dim
        self.net = ACNet(in_dim, act_space).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.rollout_len = gamma, rollout_len

        # buffer & metrics
        self._reset_buf()
        self.metrics = MetricBuffer()

        # placeholder for extra tensor (set each step)
        self._extra_state: torch.Tensor | None = None

        # wrap encoder with augmentation
        def _enc(x):
            out = self._base_enc(x)
            if self._extra_state is not None:
                aug = self._extra_state.reshape(self._extra_state.size(0), -1).float()
                out = torch.cat((out, aug.to(self.device)), dim=1)
            return out

        self._enc = _enc

    # ---------------------------------------------------------------- buffer
    def _reset_buf(self):
        self.buf: Dict[str, List] = {k: [] for k in ("s", "a", "r", "d", "logp", "v")}

    # ---------------------------------------------------------------- setter
    def set_extra_state(self, x: torch.Tensor | None):
        self._extra_state = x.to(self.device) if x is not None else None

    # ---------------------------------------------------------------- common act
    def get_actions(self, obs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        enc = self._enc(obs)
        dist = self.net.dist(enc)
        acts = dist.sample()
        logp = dist.log_prob(acts)
        ent = dist.entropy()
        if logp.ndim > 1:
            logp, ent = logp.sum(-1), ent.sum(-1)

        val = self.net.value(enc)
        self.metrics.add(entropy=ent.mean().item())
        self._store(enc, acts.detach(), 0.0, 0.0, logp.detach(), val.detach())
        return acts

    def _store(self, s, a, r, d, lp, v):
        self.buf["s"].append(s)
        self.buf["a"].append(a)
        self.buf["r"].append(safe_clone(r, torch.float32, self.device))
        self.buf["d"].append(safe_clone(d, torch.float32, self.device))
        self.buf["logp"].append(lp)
        self.buf["v"].append(v)

    # ─── abstract: subclasses implement algorithm‑specific update ───────────
    @abstractmethod
    def update(self, obs, acts, rews, next_obs, dones): ...

    # pop averaged metrics once per episode
    def pop_metrics(self) -> Dict[str, float]:
        return self.metrics.dump(div=self.rollout_len)

    def _extra_to_save(self) -> dict:
        """Sub‑classes may override to add algorithm‑specific scalars."""
        return {}

    def _load_extra(self, d: dict):
        """Sub‑classes may override to read what they saved."""
        pass

    def save_policy(self, path: str):
        p = self._ensure_pt_path(path)
        torch.save(
            {
                "net": self.net.state_dict(),
                "opt": self.opt.state_dict(),
                "extra": self._extra_to_save(),
            },
            p,
        )
        print(f"[{self.algo_name}] saved → {p}")

    def load_policy(self, path: str):
        p = self._ensure_pt_path(path)
        d = torch.load(p, map_location=self.device)
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        self._load_extra(d.get("extra", {}))
        print(f"[{self.algo_name}] loaded ← {p}")


class ACNet(nn.Module):
    """Actor–critic head for Box *or* Discrete action spaces."""

    def __init__(self, in_dim: int, act_space: gym.spaces.Space):
        super().__init__()
        self.discrete = isinstance(act_space, gym.spaces.Discrete)
        act_dim = act_space.n if self.discrete else int(np.prod(act_space.shape))

        self.torso = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())
        if self.discrete:
            self.logits = nn.Linear(128, act_dim)
        else:
            self.mu = nn.Linear(128, act_dim)
            self.std = nn.Linear(128, act_dim)
            self.bd = float(act_space.high[0])

        self.v_head = nn.Linear(128, 1)

    def dist(self, x):
        h = self.torso(x)
        if self.discrete:
            return Categorical(logits=self.logits(h))
        else:
            mu = self.bd * torch.tanh(self.mu(h))
            std = F.softplus(self.std(h)) + 1e-5
            return Normal(mu, std)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_head(self.torso(x)).squeeze(-1)

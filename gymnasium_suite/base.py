import os
import random
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        self._update(observations, actions, rewards, next_observations, dones)

    @abstractmethod
    def _update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        raise NotImplementedError

    def get_actions(
        self, observations: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        return self._get_actions(observations, mask=mask)

    @abstractmethod
    def _get_actions(
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
                return (
                    actions_tensor.detach().cpu().numpy().astype(np.int32)
                )  # shape (n_envs,)

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
                return (
                    actions_tensor.detach().cpu().numpy()
                )  # np.ndarray, shape (n_envs, …)

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


class BaseCausalPolicy(BasePolicy, ABC):
    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces,
        observation_space: gym.spaces,
        n_envs: int,
        **kwargs,
    ):
        super().__init__(algo_name, action_space, observation_space, n_envs, **kwargs)

        # TODO: define bn

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        # TODO: augment observation (and next_observation?)
        augmented_obs = observations.to(self.device)

        self._update(augmented_obs, actions, rewards, next_observations, dones)

    def get_actions(
        self, observations: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # TODO: causal mask
        causal_mask = mask

        return self._get_actions(observations, causal_mask)


def build_base_acnet(is_causal: bool = False):
    base_policy = BaseCausalPolicy if is_causal else BasePolicy

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

    class BaseACPolicy(base_policy, ABC):
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
                    aug = self._extra_state.reshape(
                        self._extra_state.size(0), -1
                    ).float()
                    out = torch.cat((out, aug.to(self.device)), dim=1)
                return out

            self._enc = _enc

        # ---------------------------------------------------------------- buffer
        def _reset_buf(self):
            self.buf: Dict[str, List] = {
                k: [] for k in ("s", "a", "r", "d", "logp", "v")
            }

        # ---------------------------------------------------------------- setter
        def set_extra_state(self, x: torch.Tensor | None):
            self._extra_state = x.to(self.device) if x is not None else None

        # ---------------------------------------------------------------- common act
        def _get_actions(
            self, obs: torch.Tensor, mask: torch.Tensor = None
        ) -> torch.Tensor:
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
        def _update(self, obs, acts, rews, next_obs, dones): ...

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

    return BaseACPolicy


def build_base_q_policy(is_causal: bool = False):
    base_policy = BaseCausalPolicy if is_causal else BasePolicy

    # Replay buffer shared by all off-policy methods
    class ReplayBuffer:
        def __init__(self, capacity: int = 20000):
            self.buf: deque = deque(maxlen=capacity)

        def __len__(self):
            return len(self.buf)

        def put(self, obs, act, rew, next_obs, non_terminal: float):
            self.buf.append((obs, act, rew, next_obs, non_terminal))

        def sample(
            self, batch: int, device
        ) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            batch_items = random.sample(self.buf, batch)
            o, a, r, o2, m = zip(*batch_items)
            obs = torch.as_tensor(np.array(o), dtype=torch.float32, device=device)
            actions = torch.as_tensor(
                a,
                dtype=torch.float32 if isinstance(a[0], float) else torch.long,
                device=device,
            )
            rewards = torch.as_tensor(r, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(np.array(o2), dtype=torch.float32, device=device)
            mask = torch.as_tensor(m, dtype=torch.float32, device=device)
            return obs, actions, rewards, next_obs, mask

    # Generic Q-network: input_dim -> hidden -> ... -> output_dim
    class QNetwork(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128)):
            super().__init__()
            layers = []
            dims = [input_dim] + list(hidden_dims)
            for i in range(len(hidden_dims)):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            layers.append(nn.Linear(dims[-1], output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # Continuous-action policy network for SAC
    class ActorNetwork(nn.Module):
        def __init__(self, input_dim: int, action_dim: int, hidden_dims=(256, 256)):
            super().__init__()
            layers = []
            dims = [input_dim] + list(hidden_dims)
            for i in range(len(hidden_dims)):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            self.net = nn.Sequential(*layers)
            self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        def forward(self, x):
            x = self.net(x)
            mu = self.mu_head(x)
            log_std = self.log_std_head(x).clamp(-20, 2)
            return mu, log_std

    # Unified BaseQPolicy handling both discrete and continuous actions
    class BaseQPolicy(base_policy, ABC):
        def __init__(
            self,
            algo_name: str,
            act_space: gym.spaces.Space,
            obs_space: gym.spaces.Space,
            n_envs: int,
            n_episodes: int,
            *,
            buffer_size=50_000,
            batch=32,
            gamma=0.99,
            lr=5e-4,
            pi_lr=5e-4,
            tgt_sync=100,
            entropy_coef=0.01,
            device=None,
            extra_input_dim: int = 0,
        ):
            super().__init__(algo_name, act_space, obs_space, n_envs)
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.algo_name = algo_name
            self.n_envs = n_envs
            self.cur_ep = 0
            self.update_ct = 0

            # Encoder setup
            if isinstance(obs_space, gym.spaces.Box):
                self.obs_dim = int(np.prod(obs_space.shape))
                self._base_enc = lambda x: safe_clone(
                    x, torch.float32, self.device
                ).view(-1, self.obs_dim)
            else:
                self.obs_dim = obs_space.n
                self._base_enc = lambda x: one_hot(
                    safe_clone(x, torch.long, self.device), self.obs_dim
                )

            def _enc(x):
                out = self._base_enc(x)
                if hasattr(self, "_extra_state") and self._extra_state is not None:
                    aug = self._extra_state.reshape(
                        self._extra_state.size(0), -1
                    ).float()
                    out = torch.cat((out, aug.to(self.device)), dim=1)
                return out

            self._enc = _enc
            self.mem = ReplayBuffer(buffer_size)
            self.batch, self.gamma, self.tgt_sync, self.entropy_coef = (
                batch,
                gamma,
                tgt_sync,
                entropy_coef,
            )
            self.extra_input_dim = extra_input_dim

            # Action-space specific networks
            self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
            input_dim = self.obs_dim + extra_input_dim

            if self.is_discrete:
                action_dim = act_space.n
                # Q-network outputs one Q-value per action
                self.q_net = QNetwork(input_dim, action_dim).to(self.device)
                self.target_q = QNetwork(input_dim, action_dim).to(self.device)
                self.target_q.load_state_dict(self.q_net.state_dict())
                self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=lr)
            else:
                action_dim = act_space.shape[0]
                # Actor network for continuous actions
                self.actor = ActorNetwork(input_dim, action_dim).to(self.device)
                # Twin critics: input (obs + act) -> scalar
                critic_in = input_dim + action_dim
                self.q_net1 = QNetwork(critic_in, 1).to(self.device)
                self.q_net2 = QNetwork(critic_in, 1).to(self.device)
                self.target_q1 = QNetwork(critic_in, 1).to(self.device)
                self.target_q2 = QNetwork(critic_in, 1).to(self.device)
                self.target_q1.load_state_dict(self.q_net1.state_dict())
                self.target_q2.load_state_dict(self.q_net2.state_dict())
                self.opt_q = torch.optim.Adam(
                    list(self.q_net1.parameters()) + list(self.q_net2.parameters()),
                    lr=lr,
                )
                self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)

        def set_extra_state(self, x):
            self._extra_state = x.to(self.device) if x is not None else None

        def update_episode(self, ep):
            self.cur_ep = ep

        def pop_metrics(self):
            return {}

        @abstractmethod
        def _get_actions(self, observations: torch.Tensor, mask: torch.Tensor = None):
            raise NotImplementedError

        @abstractmethod
        def _update(self, obs, acts, rews, next_obs, dones):
            raise NotImplementedError

        def save_policy(self, path: str):
            p = self._ensure_pt_path(path)
            data = {
                "cur_ep": self.cur_ep,
                "update_ct": self.update_ct,
            }
            if self.is_discrete:
                data.update(
                    {
                        "q_net": self.q_net.state_dict(),
                        "target_q": self.target_q.state_dict(),
                        "opt_q": self.opt_q.state_dict(),
                    }
                )
            else:
                data.update(
                    {
                        "actor": self.actor.state_dict(),
                        "q_net1": self.q_net1.state_dict(),
                        "q_net2": self.q_net2.state_dict(),
                        "target_q1": self.target_q1.state_dict(),
                        "target_q2": self.target_q2.state_dict(),
                        "opt_q": self.opt_q.state_dict(),
                        "opt_pi": self.opt_pi.state_dict(),
                    }
                )
            torch.save(data, p)

        def load_policy(self, path: str):
            p = self._ensure_pt_path(path)
            data = torch.load(p, map_location=self.device)
            self.cur_ep = data["cur_ep"]
            self.update_ct = data["update_ct"]
            if self.is_discrete:
                self.q_net.load_state_dict(data["q_net"])
                self.target_q.load_state_dict(data["target_q"])
                self.opt_q.load_state_dict(data["opt_q"])
            else:
                self.actor.load_state_dict(data["actor"])
                self.q_net1.load_state_dict(data["q_net1"])
                self.q_net2.load_state_dict(data["q_net2"])
                self.target_q1.load_state_dict(data["target_q1"])
                self.target_q2.load_state_dict(data["target_q2"])
                self.opt_q.load_state_dict(data["opt_q"])
                self.opt_pi.load_state_dict(data["opt_pi"])

    return BaseQPolicy

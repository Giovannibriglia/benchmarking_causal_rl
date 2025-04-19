import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gymnasium_suite.base import BasePolicy
from torch.distributions import Categorical, Normal


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer observation (…,1) → one‑hot (…, num_classes)."""
    return F.one_hot(indices.long().view(-1), num_classes=num_classes).float()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space):
        super().__init__()
        self.is_cont = isinstance(action_space, gym.spaces.Box)
        action_dim = (
            int(np.prod(action_space.shape)) if self.is_cont else action_space.n
        )
        self.fc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        # outputs
        if self.is_cont:
            self.mu = nn.Linear(128, action_dim)
            self.std = nn.Linear(128, action_dim)
            self.act_bound = float(action_space.high[0])
        else:
            self.logits = nn.Linear(128, action_dim)
        self.v = nn.Linear(128, 1)

    # ---------- helpers ----------
    def dist(self, x):
        h = self.fc(x)
        if self.is_cont:
            mu = self.act_bound * torch.tanh(self.mu(h))
            std = F.softplus(self.std(h)) + 1e-5
            return Normal(mu, std)
        else:
            return Categorical(logits=self.logits(h))

    def value(self, x):
        return self.v(self.fc(x)).squeeze(-1)


class PPOPolicy(BasePolicy):
    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        n_envs: int,
        rollout_len=128,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        lr=3e-4,
        k_epochs=10,
        **kwargs,
    ):
        super().__init__(algo_name, action_space, observation_space, n_envs, **kwargs)

        # ---------- observation encoder ----------
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(observation_space.shape))
            self._enc = lambda x: self._safe_tensor(x, dtype=torch.float32).view(
                -1, self.obs_dim
            )
        elif isinstance(observation_space, gym.spaces.Discrete):
            self.obs_dim = observation_space.n
            self._enc = lambda x: one_hot(
                self._safe_tensor(x, dtype=torch.long), self.obs_dim
            )
        else:
            raise NotImplementedError

        self.model = ActorCritic(self.obs_dim, action_space).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.is_cont = isinstance(action_space, gym.spaces.Box)

        # hyper‑params
        self.rollout_len, self.gamma, self.lam = rollout_len, gamma, gae_lambda
        self.eps_clip, self.k_epochs = eps_clip, k_epochs
        self.reset_buf()

    def _safe_tensor(self, x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.clone().detach().to(dtype=dtype, device=self.device)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    def _safe_copy(self, x, dtype=torch.float32):
        return (
            x.clone().detach().to(dtype=dtype)
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=dtype)
        )

    # ---------- Buffer & GAE ----------
    def reset_buf(self):
        self.buf = {k: [] for k in ("s", "a", "r", "d", "logp", "v")}

    def _store_step(self, s, a, r, d, logp, v):
        self.buf["s"].append(s)
        self.buf["a"].append(a)
        self.buf["r"].append(self._safe_copy(r))
        self.buf["d"].append(self._safe_copy(d))
        self.buf["logp"].append(logp)
        self.buf["v"].append(v)

    def _process_rollout(self):
        # stack [T, n_envs, ...] ➜ [T*n_envs, ...]
        for k in ("s", "a", "logp", "v"):
            self.buf[k] = torch.cat(self.buf[k], dim=0).to(self.device)
        r = torch.stack(self.buf["r"]).to(self.device)  # [T, n_envs]
        d = torch.stack(self.buf["d"]).to(self.device)
        v = self.buf["v"].view(self.rollout_len, self.n_envs)
        with torch.no_grad():
            adv, g = torch.zeros_like(r), 0
            next_v = torch.zeros(self.n_envs, device=self.device)
            for t in reversed(range(self.rollout_len)):
                delta = r[t] + self.gamma * next_v * (1 - d[t]) - v[t]
                g = delta + self.gamma * self.lam * (1 - d[t]) * g
                adv[t] = g
                next_v = v[t]
            ret = adv + v
        return (
            self.buf["s"],
            self.buf["a"],
            self.buf["logp"],
            adv.view(-1),
            ret.view(-1),
        )

    # ---------- Policy API ----------
    def get_actions(self, obs: torch.Tensor) -> torch.Tensor:
        enc = self._enc(obs)
        dist = self.model.dist(enc)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        logp = logp.sum(dim=-1) if self.is_cont else logp  # [n_envs]
        v = self.model.value(enc)

        # store
        self._store_step(
            enc,
            actions.detach(),
            obs.new_tensor(0),  # r placeholder
            0,
            logp.detach(),
            v.detach(),
        )
        return actions

    def update(self, observations, actions, rewards, next_observations, dones):
        # safer copy with gradient tracking off
        rewards = rewards.clone().detach().float()
        dones = dones.clone().detach().float()

        self.buf["r"][-1] = rewards
        self.buf["d"][-1] = dones

        if len(self.buf["r"]) >= self.rollout_len:
            s, a, old_logp, adv, ret = self._process_rollout()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(self.k_epochs):
                dist = self.model.dist(s)
                logp = (
                    dist.log_prob(a).sum(dim=-1) if self.is_cont else dist.log_prob(a)
                )
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                v = self.model.value(s)
                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * F.smooth_l1_loss(v, ret)
                    - 0.01 * dist.entropy().mean()
                )

                self.opt.zero_grad()
                loss.mean().backward()
                self.opt.step()

            self.reset_buf()

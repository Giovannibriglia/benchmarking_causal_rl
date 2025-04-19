import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from gymnasium_suite.base import BasePolicy
from torch import nn


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer observation (…,1) → one‑hot (…, num_classes)."""
    return F.one_hot(indices.long().view(-1), num_classes=num_classes).float()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def put(self, *items):
        self.buf.append(items)

    def sample(self, batch):
        samples = random.sample(self.buf, batch)
        stacked = list(zip(*samples))  # list of tuples per field
        tensors = [torch.tensor(x) for x in stacked]  # convert to tensors
        return tensors  # keeps order


class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNPolicy(BasePolicy):
    """
    ‑ Discrete **action** spaces only (Box actions require other algorithms).
    ‑ Handles Box or Discrete observation spaces.
    """

    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        n_envs: int,
        n_episodes: int = 1_000,
        buffer_limit: int = 50_000,
        batch_size: int = 32,
        gamma: float = 0.99,
        lr: float = 5e-4,
        **kwargs,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("DQN supports *discrete* actions only.")
        super().__init__(algo_name, action_space, observation_space, n_envs, **kwargs)

        # ---------- observation dimension ----------
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(observation_space.shape))
            self._obs_enc = lambda x: torch.tensor(
                x, dtype=torch.float32, device=self.device
            ).view(-1, self.obs_dim)
        elif isinstance(observation_space, gym.spaces.Discrete):
            self.obs_dim = observation_space.n
            self._obs_enc = lambda x: one_hot(
                torch.tensor(x, device=self.device), self.obs_dim
            )
        else:
            raise NotImplementedError

        self.act_dim = action_space.n
        self.q_net = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q_tgt = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q_tgt.load_state_dict(self.q_net.state_dict())

        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_limit)
        self.batch = batch_size
        self.gamma = gamma
        # ---------- ε‑greedy schedule ----------
        self.eps_hi = 0.10
        self.eps_lo = 0.01
        self.eps_decay = (self.eps_hi - self.eps_lo) / n_episodes
        self.cur_ep = 0
        self.update_cnt = 0
        self.tgt_freq = 100

    # ------------------------------------------------------------------ public API
    def update_episode(self, ep: int):
        self.cur_ep = ep

    def _epsilon(self) -> float:
        return max(self.eps_lo, self.eps_hi - self.eps_decay * self.cur_ep)

    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        obs_flat = self._obs_enc(observations)
        if random.random() < self._epsilon():
            return torch.randint(0, self.act_dim, (self.n_envs,), device=self.device)
        with torch.no_grad():
            q = self.q_net(obs_flat)
            return q.argmax(dim=1)

    def update(self, obs, acts, rews, next_obs, dones):
        # ‑‑ store N envs step‑wise
        for i in range(self.n_envs):
            self.memory.put(
                obs[i],
                acts[i],
                rews[i],
                next_obs[i],
                0.0 if dones[i] else 1.0,
            )
        if len(self.memory) < self.batch:
            return

        # ‑‑ sample & train
        s, a, r, s2, m = self.memory.sample(self.batch)
        s, s2 = self._obs_enc(s), self._obs_enc(s2)
        a, r, m = (
            a.long().to(self.device),
            r.float().to(self.device),
            m.float().to(self.device),
        )

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        tgt = r + self.gamma * self.q_tgt(s2).max(1)[0] * m
        loss = F.smooth_l1_loss(q, tgt.detach())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.update_cnt += 1
        if self.update_cnt % self.tgt_freq == 0:
            self.q_tgt.load_state_dict(self.q_net.state_dict())

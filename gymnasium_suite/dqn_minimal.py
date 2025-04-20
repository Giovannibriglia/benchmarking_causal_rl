import random
from collections import deque
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from gymnasium_suite.base import BasePolicy, one_hot
from torch import nn


class ReplayBuffer:
    """Simple FIFO replay with NumPy arrays → torch tensors on sample."""

    def __init__(self, capacity: int = 20000):
        self.buf: deque = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def put(self, obs, act: int, rew: float, next_obs, non_terminal: float):
        self.buf.append((obs, act, rew, next_obs, non_terminal))

    def sample(self, batch: int, device) -> Tuple[torch.Tensor, ...]:
        """Return 5 tensors on given device with correct dtypes."""
        batch_items = random.sample(self.buf, batch)
        o, a, r, o2, m = zip(*batch_items)  # tuple of lists

        obs = torch.as_tensor(np.array(o), dtype=torch.float32, device=device)
        actions = torch.as_tensor(a, dtype=torch.long, device=device)
        rewards = torch.as_tensor(r, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(np.array(o2), dtype=torch.float32, device=device)
        mask = torch.as_tensor(m, dtype=torch.float32, device=device)
        return obs, actions, rewards, next_obs, mask


class QNet(nn.Module):
    def __init__(self, in_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNPolicy(BasePolicy):
    """
    Vanilla DQN (Double/dueling/prioritized could be added later).
    • Requires Discrete action space
    • Handles Box or Discrete observations (flatten / one‑hot)
    """

    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Space,
        n_envs: int,
        n_episodes: int,
        *,
        buffer_size: int = 20000,
        batch_size: int = 32,
        gamma: float = 0.99,
        lr: float = 5e-4,
        target_sync: int = 100,
    ):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("DQNPolicy supports *discrete* action spaces only.")

        super().__init__(algo_name, action_space, observation_space, n_envs)
        self.batch = batch_size
        self.gamma = gamma
        self.tgt_sync = target_sync

        # --- observation encoder -------------------------------------------
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(observation_space.shape))
            self._enc = lambda x: x.view(-1, self.obs_dim).float()  # already float32
        elif isinstance(observation_space, gym.spaces.Discrete):
            self.obs_dim = observation_space.n
            self._enc = lambda x: one_hot(x.long().view(-1), self.obs_dim)
        else:
            raise NotImplementedError("Unsupported observation space.")

        # --- networks & optimiser ------------------------------------------
        self.q_net = QNet(self.obs_dim, action_space.n).to(self.device)
        self.q_tgt = QNet(self.obs_dim, action_space.n).to(self.device)
        self.q_tgt.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)

        # --- replay & ε‑schedule -------------------------------------------
        self.mem = ReplayBuffer(buffer_size)
        self.eps_hi, self.eps_lo = 0.10, 0.01
        self.decay = (self.eps_hi - self.eps_lo) / n_episodes
        self.cur_ep = 0
        self.update_ct = 0

    # ------------------------------------------------------------------ utils
    def _epsilon(self):  # linear decay
        return max(self.eps_lo, self.eps_hi - self.decay * self.cur_ep)

    def update_episode(self, ep):  # called from training loop
        self.cur_ep = ep
        self.metrics.add(epsilon=self._epsilon())

    # ---------------------------------------------------------------- actions
    def get_actions(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """obs_tensor still raw; we encode internally."""
        with torch.no_grad():
            q_vals = self.q_net(self._enc(obs_tensor.to(self.device)))
        if random.random() < self._epsilon():
            acts = torch.randint(
                0, self.action_space.n, (self.n_envs,), device=self.device
            )
        else:
            acts = q_vals.argmax(dim=1)
        return acts

    # ---------------------------------------------------------------- update
    def update(self, obs_t, act_t, rew_t, next_obs_t, done_t):
        # store transitions (CPU numpy for replay efficiency)
        for i in range(self.n_envs):
            self.mem.put(
                obs_t[i].cpu().numpy(),  # store as np
                int(act_t[i].item()),
                float(rew_t[i].item()),
                next_obs_t[i].cpu().numpy(),
                0.0 if done_t[i] else 1.0,
            )

        if len(self.mem) < self.batch:  # not enough yet
            return

        s, a, r, s2, m = self.mem.sample(self.batch, self.device)
        q_sa = self.q_net(self._enc(s)).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_q_sp = self.q_tgt(self._enc(s2)).max(1)[0]
            target = r + self.gamma * max_q_sp * m

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_ct += 1
        if self.update_ct % self.tgt_sync == 0:
            self.q_tgt.load_state_dict(self.q_net.state_dict())

        # metrics
        self.metrics.add(td_loss=loss.item(), avg_q=q_sa.mean().item())

    def pop_metrics(self):
        return self.metrics.dump(divisor=1)

    def save_policy(self, path: str):
        path = self._ensure_pt_path(path)
        payload = {
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_tgt.state_dict(),
            "optim": self.opt.state_dict(),
            # resume‑support small scalars
            "cur_episode": self.cur_ep,
            "update_count": self.update_ct,
            "epsilon_hi": self.eps_hi,
            "epsilon_lo": self.eps_lo,
            "eps_decay": self.decay,
        }
        torch.save(payload, path)
        print(f"[DQN] policy saved → {path}")

    def load_policy(self, path: str):
        path = self._ensure_pt_path(path)
        payload = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(payload["q_net"])
        self.q_tgt.load_state_dict(payload["q_target"])
        self.opt.load_state_dict(payload["optim"])
        self.cur_ep = payload.get("cur_episode", 0)
        self.update_ct = payload.get("update_count", 0)
        self.eps_hi = payload.get("epsilon_hi", self.eps_hi)
        self.eps_lo = payload.get("epsilon_lo", self.eps_lo)
        self.decay = payload.get("eps_decay", self.decay)
        print(f"[DQN] policy loaded ← {path}")

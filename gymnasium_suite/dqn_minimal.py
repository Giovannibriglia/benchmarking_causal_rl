import random
from collections import deque
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from gymnasium_suite.base import BasePolicy, one_hot, safe_clone
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
    def __init__(
        self,
        algo_name: str,
        act_space: gym.spaces.Discrete,
        obs_space: gym.spaces.Space,
        n_envs: int,
        n_episodes: int,
        *,
        extra_input_dim: int = 0,
        buffer_size=50_000,
        batch=32,
        gamma=0.99,
        lr=5e-4,
        tgt_sync=100,
        device=None,
    ):
        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("DQN → discrete actions only.")
        super().__init__(
            algo_name,
            act_space,
            obs_space,
            n_envs,
            device=device
            or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # encoder
        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(obs_space.shape))
            self._base_enc = lambda x: safe_clone(x, torch.float32, self.device).view(
                -1, self.obs_dim
            )
        else:
            self.obs_dim = obs_space.n
            self._base_enc = lambda x: one_hot(
                safe_clone(x, torch.long, self.device), self.obs_dim
            )

        in_dim = self.obs_dim + extra_input_dim
        self.q_net = QNet(in_dim, act_space.n).to(self.device)
        self.tgt = QNet(in_dim, act_space.n).to(self.device)
        self.tgt.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.mem = ReplayBuffer(buffer_size)
        self.batch, self.gamma, self.tgt_sync = batch, gamma, tgt_sync
        self.eps_hi, self.eps_lo = 0.10, 0.01
        self.decay = (self.eps_hi - self.eps_lo) / n_episodes
        self.cur_ep = self.update_ct = 0

        # augmentation holder
        self._extra_state: torch.Tensor | None = None

        def _enc(x):
            out = self._base_enc(x)
            if self._extra_state is not None:
                aug = self._extra_state.reshape(self._extra_state.size(0), -1).float()
                out = torch.cat((out, aug.to(self.device)), dim=1)
            return out

        self._enc = _enc

    def set_extra_state(self, x):  # same signature
        self._extra_state = x.to(self.device) if x is not None else None

    # ------------------------------------------------------------------ utils
    def _epsilon(self):  # linear decay
        return max(self.eps_lo, self.eps_hi - self.decay * self.cur_ep)

    def update_episode(self, ep):  # called from training loop
        self.cur_ep = ep
        self.metrics.add(epsilon=self._epsilon())

    # ---------------------------------------------------------------- actions
    def get_actions(self, obs):
        q_vals = self.q_net(self._enc(obs.to(self.device))).detach()
        if random.random() < self._epsilon():
            acts = torch.randint(
                0, self.action_space.n, (self.n_envs,), device=self.device
            )
        else:
            acts = q_vals.argmax(1)
        return acts

    def update(self, obs, acts, rews, next_obs, dones):
        # store
        for i in range(self.n_envs):
            self.mem.put(
                obs[i].cpu().numpy(),
                int(acts[i]),
                float(rews[i]),
                next_obs[i].cpu().numpy(),
                0.0 if dones[i] else 1.0,
            )
        if len(self.mem) < self.batch:
            return

        s, a, r, s2, m = self.mem.sample(self.batch, self.device)
        q_sa = self.q_net(self._enc(s)).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_q = self.tgt(self._enc(s2)).max(1)[0]
            tgt = r + self.gamma * max_q * m

        loss = F.smooth_l1_loss(q_sa, tgt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.update_ct += 1
        if self.update_ct % self.tgt_sync == 0:
            self.tgt.load_state_dict(self.q_net.state_dict())
        self.metrics.add(td_loss=loss.item(), avg_q=q_sa.mean().item())

    def pop_metrics(self):
        return self.metrics.dump()

    def save_policy(self, path):
        p = self._ensure_pt_path(path)
        torch.save(
            {
                "q": self.q_net.state_dict(),
                "tgt": self.tgt.state_dict(),
                "opt": self.opt.state_dict(),
                "cur_ep": self.cur_ep,
                "upd": self.update_ct,
                "eps_hi": self.eps_hi,
                "eps_lo": self.eps_lo,
                "decay": self.decay,
            },
            p,
        )
        print(f"[DQN] saved → {p}")

    def load_policy(self, path):
        p = self._ensure_pt_path(path)
        d = torch.load(p, map_location=self.device)
        self.q_net.load_state_dict(d["q"])
        self.tgt.load_state_dict(d["tgt"])
        self.opt.load_state_dict(d["opt"])
        self.cur_ep, self.update_ct = d["cur_ep"], d["upd"]
        self.eps_hi, self.eps_lo, self.decay = d["eps_hi"], d["eps_lo"], d["decay"]
        print(f"[DQN] loaded ← {p}")

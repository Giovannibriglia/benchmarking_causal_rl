import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from gymnasium_suite.base import BasePolicy, one_hot, safe_clone
from gymnasium_suite.causal_knowledge import CausalKnowledge

from gymnasium_suite.dqn import QNet, ReplayBuffer


class CausalDQNPolicy(BasePolicy):
    def __init__(
        self,
        algo_name: str,
        act_space: gym.spaces.Discrete,
        obs_space: gym.spaces.Space,
        n_envs: int,
        n_episodes: int,
        *,
        buffer_size=50_000,
        batch=32,
        gamma=0.99,
        lr=5e-4,
        tgt_sync=100,
        N_max_causal: int = 16,
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

        self.N_max_causal = N_max_causal

        # encoder
        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(obs_space.shape))
            self._base_enc = lambda x: safe_clone(x, torch.float32, self.device).view(
                -1, self.obs_dim
            )
            # extra_input_dim = N_max_causal * 2
        else:
            self.obs_dim = obs_space.n
            self._base_enc = lambda x: one_hot(
                safe_clone(x, torch.long, self.device), self.obs_dim
            )
        extra_input_dim = N_max_causal**2

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

        self.causal_knowledge = CausalKnowledge()

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
    def get_actions(self, obs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        1. Build augmented, flattened state:   [obs | extra]
        2. ε‑greedy on Q‑values
        3. Return discrete action indices (n_envs,)
        """
        # ---- concatenate observation + causal extra ---------------------------
        base = self._base_enc(obs.to(self.device))  # (n_envs, obs_dim)
        extra = self.causal_knowledge.get_reward_action_values(obs)  # (n_envs, N, M)
        extra = extra.to(self.device).reshape(obs.size(0), -1).float()  # (n_envs, N*M)
        aug = torch.cat((base, extra), dim=1)  # (n_envs, in_dim)

        # ---- ε‑greedy ---------------------------------------------------------
        q_vals = self.q_net(aug).detach()
        if random.random() < self._epsilon():
            return torch.randint(
                0, self.action_space.n, (self.n_envs,), device=self.device
            )
        else:
            return q_vals.argmax(dim=1)

    def update(self, obs, acts, rews, next_obs, dones):
        """
        Store *augmented* obs/next_obs in replay, then train from replay.
        """
        self.causal_knowledge.store_data(obs, acts, rews)

        # -------- 1. augment & store into replay --------------------------------
        base_cur = self._base_enc(obs.to(self.device))
        extra_cur = (
            self.causal_knowledge.get_reward_action_values(obs)
            .to(self.device)
            .reshape(obs.size(0), -1)
            .float()
        )
        aug_cur = torch.cat((base_cur, extra_cur), dim=1)  # (n_envs, in_dim)

        base_next = self._base_enc(next_obs.to(self.device))
        extra_next = (
            self.causal_knowledge.get_reward_action_values(next_obs)
            .to(self.device)
            .reshape(next_obs.size(0), -1)
            .float()
        )
        aug_next = torch.cat((base_next, extra_next), dim=1)

        for i in range(self.n_envs):
            if dones[i]:  # skip terminal transitions entirely
                continue
            self.mem.put(
                aug_cur[i].cpu().numpy(),
                int(acts[i]),
                float(rews[i]),
                aug_next[i].cpu().numpy(),
                1.0,  # this is the mask: 1 for non-terminal
            )

        # -------- 2. learn from replay -----------------------------------------
        if len(self.mem) < self.batch:  # warm‑up
            return

        s, a, r, s2, m = self.mem.sample(self.batch, self.device)  # already float32
        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tgt_max = self.tgt(s2).max(1)[0]
            target = r + self.gamma * tgt_max * m

        loss = F.smooth_l1_loss(q_sa, target)
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

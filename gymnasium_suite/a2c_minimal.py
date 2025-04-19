import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gymnasium_suite.base import BasePolicy
from torch.distributions import Categorical, Normal


# ───── generic one‑hot helper (same as in DQN/PPO) ────────────────────────────────
def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(indices.long().view(-1), num_classes=num_classes).float()


# ───── shared actor‑critic network (same head shapes as in PPO) ───────────────────
class ACNet(nn.Module):
    def __init__(self, obs_dim: int, act_space: gym.spaces.Space):
        super().__init__()
        self.discrete = isinstance(act_space, gym.spaces.Discrete)
        act_dim = act_space.n if self.discrete else int(np.prod(act_space.shape))

        self.base = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        if self.discrete:
            self.logits = nn.Linear(128, act_dim)
        else:
            self.mu = nn.Linear(128, act_dim)
            self.std = nn.Linear(128, act_dim)
            self.act_bd = float(act_space.high[0])

        self.v_head = nn.Linear(128, 1)

    # distribution over actions
    def dist(self, x: torch.Tensor):
        h = self.base(x)
        if self.discrete:
            return Categorical(logits=self.logits(h))
        else:
            mu = self.act_bd * torch.tanh(self.mu(h))
            std = F.softplus(self.std(h)) + 1e-5
            return Normal(mu, std)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_head(self.base(x)).squeeze(-1)


# ───── A2C policy class compatible with your BasePolicy interface ────────────────
class A2CPolicy(BasePolicy):
    """
    * Synchronous Advantage Actor‑Critic.
    * Handles **Box/Discrete observations** and **Box/Discrete actions**.
    """

    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        n_envs: int,
        rollout_len: int = 5,  # shorter than PPO (on‐policy, frequent updates)
        gamma: float = 0.99,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        **kwargs,
    ):
        super().__init__(algo_name, action_space, observation_space, n_envs, **kwargs)

        # ---- observation encoder (flat float vector or one‑hot) -----------------
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(observation_space.shape))
            self._enc = lambda x: self._safe_tensor(
                x, dtype=torch.float32, device=self.device
            ).view(-1, self.obs_dim)
        elif isinstance(observation_space, gym.spaces.Discrete):
            self.obs_dim = observation_space.n
            self._enc = lambda x: one_hot(
                self._safe_tensor(x, dtype=torch.long, device=self.device), self.obs_dim
            )
        else:
            raise NotImplementedError(
                "A2CPolicy supports Box or Discrete obs spaces only"
            )

        # ---- actor‑critic network ------------------------------------------------
        self.net = ACNet(self.obs_dim, action_space).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # ---- hyper‑params & buffers ---------------------------------------------
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.ent_coef = entropy_coef
        self.val_coef = value_coef
        self.reset_buf()

    def _safe_tensor(self, x, dtype=torch.float32, device="cpu"):
        if isinstance(x, torch.Tensor):
            return x.clone().detach().to(dtype=dtype, device=device)
        else:
            return torch.tensor(x, dtype=dtype, device=device)

    # ───────────── internal rollout buffer ───────────────────────────────────────
    def reset_buf(self):
        self.buf = {k: [] for k in ("s", "a", "r", "d", "logp")}

    def _store(self, s, a, r, d, logp):
        self.buf["s"].append(s)
        self.buf["a"].append(a)
        self.buf["r"].append(torch.tensor(r, dtype=torch.float32))
        self.buf["d"].append(torch.tensor(d, dtype=torch.float32))
        self.buf["logp"].append(logp)

    # ───────────── BasePolicy: get_actions ───────────────────────────────────────
    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        enc_obs = self._enc(observations)
        dist = self.net.dist(enc_obs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = (
            log_prob.sum(-1) if log_prob.ndim > 1 else log_prob
        )  # continuous sum over dims
        self._store(enc_obs, actions.detach(), 0.0, 0.0, log_prob.detach())
        return actions

    # ───────────── BasePolicy: update (called every env step) ────────────────────
    def update(self, observations, actions, rewards, next_observations, dones):
        # safer copy with gradient tracking off
        rewards = rewards.clone().detach().float()
        dones = dones.clone().detach().float()

        self.buf["r"][-1] = rewards
        self.buf["d"][-1] = dones

        if len(self.buf["r"]) >= self.rollout_len:
            # -------- stack rollout [T, n_envs, ...] → [T*n_envs, ...] ----------
            s = torch.cat(self.buf["s"]).to(self.device)
            a = torch.cat(self.buf["a"]).to(self.device)
            # logp_old = torch.cat(self.buf["logp"]).to(self.device)
            r = torch.stack(self.buf["r"]).to(self.device)  # [T, n_envs]
            d = torch.stack(self.buf["d"]).to(self.device)  # [T, n_envs]

            # -------- bootstrap value for last state ----------------------------
            with torch.no_grad():
                next_v = self.net.value(self._enc(next_observations)).view(self.n_envs)

            # -------- compute discounted returns --------------------------------
            T = self.rollout_len
            returns = torch.zeros_like(r)
            running = next_v
            for t in reversed(range(T)):
                running = r[t] + self.gamma * running * (1 - d[t])
                returns[t] = running
            returns = returns.view(-1)  # [T*n_envs]

            # -------- critic loss -----------------------------------------------
            values = self.net.value(s)
            adv = returns - values.detach()

            # -------- actor (policy) loss ---------------------------------------
            dist = self.net.dist(s)
            logp = dist.log_prob(a)
            logp = logp.sum(-1) if logp.ndim > 1 else logp
            entropy = dist.entropy()
            entropy = entropy.sum(-1) if entropy.ndim > 1 else entropy

            actor_loss = -(logp * adv).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = (
                actor_loss
                + self.val_coef * critic_loss
                - self.ent_coef * entropy.mean()
            )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.reset_buf()

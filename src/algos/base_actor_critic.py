from __future__ import annotations

from abc import abstractmethod

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from src.base import BaseEnv, BasePolicy, DEFAULT_DEVICE


class BaseActorCritic(BasePolicy, nn.Module):
    def __init__(
        self,
        env: BaseEnv,
        eval_env: BaseEnv,
        rollout_len: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        hidden_dim: int = 64,
        device: str | torch.device = DEFAULT_DEVICE,
    ) -> None:
        nn.Module.__init__(self)
        BasePolicy.__init__(self, env, eval_env, rollout_len, device)

        self.gamma, self.gae_lambda = gamma, gae_lambda

        obs_space, act_space = env.observation_space, env.action_space
        if isinstance(obs_space, gym.spaces.Discrete):
            self.encoder = nn.Embedding(obs_space.n, hidden_dim)
            latent = hidden_dim
        else:
            flat = int(np.prod(obs_space.shape))
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(flat, hidden_dim), nn.Tanh()
            )
            latent = hidden_dim

        if isinstance(act_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.actor = nn.Sequential(
                nn.Linear(latent, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, act_space.n),
            )
            self.dist_fn = lambda logits: Categorical(logits=logits)
        else:
            self.is_discrete = False
            adim = int(np.prod(act_space.shape))
            self.actor_mu = nn.Sequential(
                nn.Linear(latent, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, adim)
            )
            self.log_std = nn.Parameter(torch.zeros(adim))
            self.dist_fn = self._normal_dist

        self.critic = nn.Sequential(
            nn.Linear(latent, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.to(self.device)
        self.optim = Adam(self.parameters(), lr=lr)

    def train(self):
        mem, L, R = self._collect_rollout()
        self.train_metrics.add(training_length=L, training_return=R)
        self._post_update(mem)
        self._algo_update(mem)

    @abstractmethod
    def _algo_update(self, mem):
        raise NotImplementedError

    def _post_update(self, mem):
        """
        Called once per rollout *after* the algorithm’s parameter update
        and *after* returns / advantages are available.

        Override in a subclass to add extra logging, target‑network sync,
        or whatever you need.  Default: do nothing.
        """
        pass

    # ------------------------------------------------------------------
    def _normal_dist(self, mu):
        return Normal(mu, torch.exp(self.log_std))

    def parameters(self):
        if self.is_discrete:  # categorical actor
            actor_params = self.actor.parameters()
            extras = []
        else:  # Gaussian actor
            actor_params = self.actor_mu.parameters()
            extras = [self.log_std]  # trainable sigma
        return (
            list(self.encoder.parameters())
            + list(actor_params)
            + extras
            + list(self.critic.parameters())
        )

    # ------------------------------------------------------------------
    class Memory(dict):
        pass

    def _collect_rollout(self):
        env, device = self.env, self.device
        obs = env.reset().to(device)

        mem = dict(
            obs=[], actions=[], logp=[], values=[], rewards=[], dones=[], entropy=[]
        )
        for _ in range(self.rollout_len):
            with torch.no_grad():
                latent = self.encoder(obs)
                if self.is_discrete:  # ------------ DISCRETE
                    logits = self.actor(latent)  # [N, n_actions]
                    dist = self.dist_fn(logits)
                    act = dist.sample()  # [N]
                    logp = dist.log_prob(act)  # [N]  (keep per‑env dim!)
                    ent = dist.entropy()  # [N]
                else:  # ----------- CONTINUOUS
                    mu = self.actor_mu(latent)  # [N, act_dim]
                    dist = self.dist_fn(mu)
                    act = dist.sample()  # [N, act_dim]
                    logp = dist.log_prob(act).sum(-1)  # [N]
                    ent = dist.entropy().sum(-1)  # [N]
                val = self.critic(latent).squeeze(-1)  # [N]

            nxt_obs, rew, term, trunc, _ = env.step(act)
            nxt_obs, rew = nxt_obs.to(device), rew.to(device)

            mem["obs"].append(obs)
            mem["actions"].append(act)
            mem["logp"].append(logp)
            mem["values"].append(val)
            mem["rewards"].append(rew)
            mem["dones"].append(term | trunc)
            mem["entropy"].append(ent)
            obs = nxt_obs

        # stack to tensors with shape [T, N, ...]
        for k in mem:
            mem[k] = torch.stack(mem[k], dim=0)

        # ---------- GAE ----------
        T, N = self.rollout_len, env.n_envs
        with torch.no_grad():
            last_val = self.critic(self.encoder(obs)).squeeze(-1)

        returns = torch.zeros(T, N, device=device)
        # adv = torch.zeros_like(returns)
        gae = torch.zeros(N, device=device)

        for t in reversed(range(T)):
            mask = 1.0 - mem["dones"][t].float()
            delta = mem["rewards"][t] + self.gamma * last_val * mask - mem["values"][t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            returns[t] = gae + mem["values"][t]
            last_val = mem["values"][t]

        mem["returns"] = returns
        mem["advantages"] = returns - mem["values"]

        episode_return = mem["rewards"].sum(0).mean().item()

        """print("latents: ", mem["obs"])
        print("actions:", mem["actions"])
        print("returns: ", mem["returns"])"""

        """if self.actor_prior:
            self.actor_prior.update(latents=mem["latents"])
        if self.critic_prior:
            self.critic_prior.update(latents=mem["latents"],
                                     returns=mem["returns"])"""

        return mem, T, episode_return

    def _log_ac_metrics(self, mse, adv_var, entropy):
        self.train_metrics.add(value_mse=mse, adv_var=adv_var, entropy=entropy)

    def extra_actor_loss(
        self,
        states: torch.Tensor | None = None,  # [B, feat]
        logits: torch.Tensor | None = None,  # [B, n_actions]
    ) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

    def extra_critic_loss(
        self,
        states: torch.Tensor = None,
        logits: torch.Tensor = None,
    ):
        return torch.tensor(0.0, device=self.device)

    @staticmethod
    def flat(x):
        return x.reshape(-1, *x.shape[2:])

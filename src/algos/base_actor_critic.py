from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

import gymnasium
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import flatdim, flatten
from torch import nn
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from src.base import BaseEnv, BasePolicy, DEFAULT_DEVICE


class BaseActorCritic(BasePolicy, nn.Module):
    def __init__(
        self,
        env: BaseEnv,
        eval_env: BaseEnv,
        rollout_len: int = 1024,
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
        self._obs_space = obs_space  # keep a handle for flatten()

        # ----- Encoder selection (robust to Tuple/Dict/Multi*) -----
        if isinstance(obs_space, gymnasium.spaces.Discrete):
            self.encoder = nn.Embedding(obs_space.n, hidden_dim)
            latent = hidden_dim
            self._use_space_flatten = False
        elif (
            isinstance(obs_space, gymnasium.spaces.Box) and obs_space.shape is not None
        ):
            flat = int(np.prod(obs_space.shape))
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(flat, hidden_dim), nn.Tanh()
            )
            latent = hidden_dim
            self._use_space_flatten = False
        else:
            # Tuple / Dict / MultiDiscrete / MultiBinary or any space with no .shape
            self._obs_flatdim = flatdim(obs_space)
            self.encoder = nn.Sequential(
                nn.Linear(self._obs_flatdim, hidden_dim), nn.Tanh()
            )
            latent = hidden_dim
            self._use_space_flatten = True

        # ----- Actor heads (unchanged) -----
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

    # ---------- helpers ----------
    def _normal_dist(self, mu):  # unchanged
        return Normal(mu, torch.exp(self.log_std))

    def parameters(self):  # unchanged
        if self.is_discrete:
            actor_params = self.actor.parameters()
            extras = []
        else:
            actor_params = self.actor_mu.parameters()
            extras = [self.log_std]
        return (
            list(self.encoder.parameters())
            + list(actor_params)
            + extras
            + list(self.critic.parameters())
        )

    def _flatten_obs_batch(self, obs) -> torch.Tensor:
        """
        Robustly flatten a batch of observations from arbitrary Gym spaces into float32 [B, flatdim].
        Handles Tuple/Dict spaces whether batch arrives as:
          • tuple/list of component arrays/tensors (len = n_components), each shape [B, ...]
          • a tensor accidentally constructed with shape [n_components, B, ...]
          • normal np.ndarray/torch.Tensor for Box-like spaces
        """
        space = self._obs_space

        # -------- Tuple spaces --------
        if isinstance(space, gymnasium.spaces.Tuple):
            n_comp = len(space.spaces)

            # (A) tuple/list of components, each [B, ...] (can be tensor or ndarray)
            if isinstance(obs, (tuple, list)) and len(obs) == n_comp:
                comps = [
                    (
                        c.detach().cpu().numpy()
                        if isinstance(c, torch.Tensor)
                        else np.asarray(c)
                    )
                    for c in obs
                ]
                B = comps[0].shape[0]
                samples = [tuple(c[i] for c in comps) for i in range(B)]
                flat = np.stack([flatten(space, s) for s in samples], axis=0)
                return torch.as_tensor(flat, device=self.device, dtype=torch.float32)

            # (B) tensor shaped [n_comp, B, ...]
            if (
                isinstance(obs, torch.Tensor)
                and obs.ndim >= 2
                and obs.shape[0] == n_comp
            ):
                comps = [obs[i].detach().cpu().numpy() for i in range(n_comp)]
                B = comps[0].shape[0]
                samples = [tuple(c[i] for c in comps) for i in range(B)]
                flat = np.stack([flatten(space, s) for s in samples], axis=0)
                return torch.as_tensor(flat, device=self.device, dtype=torch.float32)

        # -------- Dict spaces --------
        if isinstance(space, gymnasium.spaces.Dict):
            if isinstance(obs, dict):
                keys = list(space.spaces.keys())
                arrs = {
                    k: (
                        v.detach().cpu().numpy()
                        if isinstance(v, torch.Tensor)
                        else np.asarray(v)
                    )
                    for k, v in obs.items()
                }
                B = arrs[keys[0]].shape[0]
                samples = [{k: arrs[k][i] for k in keys} for i in range(B)]
                flat = np.stack([flatten(space, s) for s in samples], axis=0)
                return torch.as_tensor(flat, device=self.device, dtype=torch.float32)

            if (
                isinstance(obs, torch.Tensor)
                and obs.ndim >= 2
                and obs.shape[0] == len(space.spaces)
            ):
                keys = list(space.spaces.keys())
                comps = [obs[i].detach().cpu().numpy() for i in range(len(keys))]
                B = comps[0].shape[0]
                samples = [
                    {k: comps[j][i] for j, k in enumerate(keys)} for i in range(B)
                ]
                flat = np.stack([flatten(space, s) for s in samples], axis=0)
                return torch.as_tensor(flat, device=self.device, dtype=torch.float32)

        # -------- Default (Box / Multi*) --------
        if isinstance(obs, torch.Tensor):
            arr = obs.detach().cpu().numpy()
        else:
            arr = np.asarray(obs)

        if arr.dtype == object:
            samples = list(arr)
            flat = np.stack([flatten(space, s) for s in samples], axis=0)
        else:
            if arr.ndim == 1:
                arr = arr[:, None]
            B = arr.shape[0]
            flat = arr.reshape(B, -1)

        return torch.as_tensor(flat, device=self.device, dtype=torch.float32)

    def _encode(self, obs: torch.Tensor):
        if isinstance(self.encoder, nn.Embedding):
            return self.encoder(obs.long())
        if self._use_space_flatten:
            x = self._flatten_obs_batch(obs)
            return self.encoder(x)  # float32 -> Linear
        return self.encoder(obs.float())

    # ---------- rollout (use _encode) ----------
    def _collect_rollout(self):
        env, device = self.env, self.device
        obs = env.reset()  # <-- was: env.reset().to(device)

        mem = dict(
            obs=[], actions=[], logp=[], values=[], rewards=[], dones=[], entropy=[]
        )
        for _ in range(self.rollout_len):
            with torch.no_grad():
                latent = self._encode(obs)  # handles tuple/dict/Box uniformly
                if self.is_discrete:
                    logits = self.actor(latent)
                    dist = self.dist_fn(logits)
                    act = dist.sample()
                    logp = dist.log_prob(act)
                    ent = dist.entropy()
                else:
                    mu = self.actor_mu(latent)
                    dist = self.dist_fn(mu)
                    act = dist.sample()
                    logp = dist.log_prob(act).sum(-1)
                    ent = dist.entropy().sum(-1)
                val = self.critic(latent).squeeze(-1)

            nxt_obs, rew, term, trunc, _ = env.step(act)
            # DON'T try to .to() the structured obs:
            nxt_obs = nxt_obs
            rew = rew.to(device)

            mem["obs"].append(obs)  # keep raw structured obs per step
            mem["actions"].append(act)
            mem["logp"].append(logp)
            mem["values"].append(val)
            mem["rewards"].append(rew)
            mem["dones"].append(term | trunc)
            mem["entropy"].append(ent)
            obs = nxt_obs

        # stack numeric tensors as before
        for k in ["actions", "logp", "values", "rewards", "dones", "entropy"]:
            mem[k] = torch.stack(mem[k], dim=0)

        # but flatten tuple/dict observations per step first, then stack to [T, N, flatdim]
        mem["obs"] = torch.stack(
            [self._flatten_obs_batch(o) for o in mem["obs"]], dim=0
        )

        # ---------- GAE ----------
        T, N = self.rollout_len, env.n_envs
        with torch.no_grad():
            last_latent = self._encode(obs)  # works for structured obs
            last_val = self.critic(last_latent).squeeze(-1)

        returns = torch.zeros(T, N, device=device)
        gae = torch.zeros(N, device=device)

        for t in reversed(range(T)):
            mask = 1.0 - mem["dones"][t].float()
            delta = mem["rewards"][t] + self.gamma * last_val * mask - mem["values"][t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            returns[t] = gae + mem["values"][t]
            last_val = mem["values"][t]

        mem["returns"] = returns
        mem["advantages"] = returns - mem["values"]
        self._log_adv_summary(mem)

        episode_return = mem["rewards"].sum(0).mean().item()
        return mem, T, episode_return

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

    def _log_ac_metrics(self, mse, adv_var, entropy):
        self.train_metrics.add(value_mse=mse, adv_var=adv_var, entropy=entropy)

    def _log_adv_summary(self, mem):
        adv = self.flat(mem["advantages"])
        self.train_metrics.add(
            adv_mean=float(adv.mean().item()),
            adv_std=float(adv.std(unbiased=False).item()),
            adv_min=float(adv.min().item()),
            adv_max=float(adv.max().item()),
        )

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

    # A minimal schema of common metrics we want everywhere
    _COMMON_METRICS = (
        "total_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "adv_var",
        "value_mse",
        "extra_actor_loss",
        "extra_critic_loss",
        "grad_norm",
        "lr",
    )

    def _current_lr(self) -> float:
        if hasattr(self, "optim") and self.optim.param_groups:
            return float(self.optim.param_groups[0].get("lr", float("nan")))
        return float("nan")

    def _log_update_metrics(
        self,
        *,
        total_loss: float | None = None,
        actor_loss: float | None = None,
        critic_loss: float | None = None,
        entropy: float | None = None,
        adv_var: float | None = None,
        value_mse: float | None = None,
        extra_actor_loss: float | None = None,
        extra_critic_loss: float | None = None,
        grad_norm: float | None = None,
        # any algo-specific extras will pass through **extras
        **extras: Any,
    ) -> None:
        """Unify metric keys across all algos, while allowing algo-specific extras."""

        def _to_num(x):
            if x is None:
                return float("nan")
            try:
                return float(x)
            except Exception:
                return float("nan")

        payload: Dict[str, float] = {
            "total_loss": _to_num(total_loss),
            "actor_loss": _to_num(actor_loss),
            "critic_loss": _to_num(critic_loss),
            "entropy": _to_num(entropy),
            "adv_var": _to_num(adv_var),
            "value_mse": _to_num(value_mse),
            "extra_actor_loss": _to_num(extra_actor_loss),
            "extra_critic_loss": _to_num(extra_critic_loss),
            "grad_norm": _to_num(grad_norm),
            "lr": _to_num(self._current_lr()),
        }

        # attach per-algo extras (kept flat)
        for k, v in extras.items():
            payload[k] = _to_num(v)

        # your buffer already supports kwargs via .add(...)
        self.train_metrics.add(**payload)

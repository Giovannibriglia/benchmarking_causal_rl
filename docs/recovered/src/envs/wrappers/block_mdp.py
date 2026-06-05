"""Synthetic Block-MDP family used for causal-RL ablations (paper Section 6).

This environment is pure PyTorch and vectorized. It exposes latent factors and
exact interventional reward/transition distributions, making it suitable for
plug-in identifiability-gap estimators on small tabular settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from gymnasium.spaces import Box, Discrete

from src.envs.causal_base import CausalEnv

from ._causal_cell import get_cell_config


@dataclass
class _BlockState:
    z: torch.Tensor  # [B] latent state index
    u: torch.Tensor  # [B] binary confounder in {-1, +1}
    t: torch.Tensor  # [B] step index


class BlockMDPEnv(CausalEnv):
    """Vectorized Block-MDP with configurable observability and confounding."""

    reward_support = torch.tensor([-1.0, 1.0], dtype=torch.float32)

    def __init__(
        self,
        env_id: str,
        n_envs: int,
        device: torch.device,
        seed: int,
        *,
        cell: int = 8,
        d: int = 4,
        D: Optional[int] = None,
        k: int = 8,
        sigma2: float = 0.05,
        alpha: float = 0.0,
        rho: float = 1.0,
        n_actions: int = 4,
        horizon: int = 20,
        **_: object,
    ) -> None:
        self.env_id = env_id
        self.n_envs = n_envs
        self.device = device
        self.cell = int(cell)
        self.cell_cfg = get_cell_config(self.cell)

        self.d = int(d)
        self.obs_dim = int(D if D is not None else self.d + int(k))
        self.k = int(k)
        self.sigma2 = float(max(0.0, sigma2))
        self.alpha = float(alpha)
        self.rho = float(min(max(rho, 0.0), 1.0))
        self.n_actions = int(n_actions)
        self.horizon = int(horizon)

        if self.d <= 0:
            raise ValueError("d must be > 0")
        if self.n_actions <= 1:
            raise ValueError("n_actions must be >= 2")
        if self.obs_dim <= 0:
            raise ValueError("D must be > 0")

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))
        self.base_seed = int(seed)

        self.n_states = 2 ** self.d
        states = torch.arange(self.n_states, device=self.device)
        bits = ((states.unsqueeze(-1) >> torch.arange(self.d, device=self.device)) & 1)
        self.state_bits = bits.float() * 2.0 - 1.0

        leak_mask = torch.rand(self.d, generator=self.rng, device=self.device) < self.rho
        self.leak_mask = leak_mask.float()

        transition_logits = torch.randn(
            self.n_states, self.n_actions, self.n_states, generator=self.rng, device=self.device
        )
        self.transition_probs = torch.softmax(transition_logits, dim=-1)
        self.transition_conf = torch.randn(
            self.n_states, self.n_actions, self.n_states, generator=self.rng, device=self.device
        ) * 0.25

        reward_logits = torch.randn(
            self.n_states, self.n_actions, generator=self.rng, device=self.device
        )
        self.reward_logits = reward_logits
        self.reward_conf = torch.randn(
            self.n_states, self.n_actions, generator=self.rng, device=self.device
        )

        self.obs_space = Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_dim,),
            dtype=float,
        )
        self.act_space = Discrete(self.n_actions)
        self._state = _BlockState(
            z=torch.zeros(self.n_envs, dtype=torch.long, device=self.device),
            u=torch.ones(self.n_envs, dtype=torch.float32, device=self.device),
            t=torch.zeros(self.n_envs, dtype=torch.long, device=self.device),
        )

    def _sample_initial_state(self, batch_idx: torch.Tensor) -> None:
        n = int(batch_idx.numel())
        self._state.z[batch_idx] = torch.randint(
            0, self.n_states, (n,), generator=self.rng, device=self.device
        )
        u_bin = torch.randint(0, 2, (n,), generator=self.rng, device=self.device)
        self._state.u[batch_idx] = u_bin.float() * 2.0 - 1.0
        self._state.t[batch_idx] = 0

    def _build_obs(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if idx is None:
            idx = torch.arange(self.n_envs, device=self.device)
        z = self._state.z[idx]
        u = self._state.u[idx]
        z_bits = self.state_bits[z]
        obs = torch.zeros((idx.numel(), self.obs_dim), device=self.device)

        if self.cell_cfg.z_exposed:
            n = min(self.d, self.obs_dim)
            obs[:, :n] = z_bits[:, :n] * self.leak_mask[:n]

        if self.obs_dim > self.d:
            noise = torch.randn(
                (idx.numel(), self.obs_dim - self.d),
                generator=self.rng,
                device=self.device,
            )
            obs[:, self.d :] = noise

        if self.cell_cfg.u_exposed and self.obs_dim >= 1:
            obs[:, -1] = u

        if self.sigma2 > 0.0:
            obs = obs + torch.randn(
                obs.shape, generator=self.rng, device=self.device
            ) * (self.sigma2 ** 0.5)
        return obs

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self.rng.manual_seed(int(seed))
        idx = torch.arange(self.n_envs, device=self.device)
        self._sample_initial_state(idx)
        return self._build_obs(), {}

    def _factual_reward_prob(self, actions: torch.Tensor) -> torch.Tensor:
        z = self._state.z
        base = self.reward_logits[z, actions]
        conf = self.reward_conf[z, actions]
        return torch.sigmoid(base + self.alpha * self._state.u * conf)

    def _do_reward_prob(self, actions: torch.Tensor) -> torch.Tensor:
        z = self._state.z
        base = self.reward_logits[z, actions]
        conf = self.reward_conf[z, actions]
        plus = torch.sigmoid(base + self.alpha * conf)
        minus = torch.sigmoid(base - self.alpha * conf)
        return 0.5 * (plus + minus)

    def do_reward(self, action: torch.Tensor) -> torch.Tensor:
        actions = action.long().view(-1)
        p_pos = self._do_reward_prob(actions).clamp(1e-6, 1 - 1e-6)
        return torch.stack([1.0 - p_pos, p_pos], dim=-1)

    def observed_reward_distribution(
        self, action: torch.Tensor, reward: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        _ = reward
        actions = action.long().view(-1)
        p_pos = self._factual_reward_prob(actions).clamp(1e-6, 1 - 1e-6)
        return torch.stack([1.0 - p_pos, p_pos], dim=-1)

    def _factual_transition(self, actions: torch.Tensor) -> torch.Tensor:
        z = self._state.z
        base = self.transition_probs[z, actions]
        conf = self.transition_conf[z, actions]
        logits = torch.log(base.clamp_min(1e-8)) + self.alpha * self._state.u.unsqueeze(-1) * conf
        return torch.softmax(logits, dim=-1)

    def do_transition(self, action: torch.Tensor) -> torch.Tensor:
        actions = action.long().view(-1)
        z = self._state.z
        base = self.transition_probs[z, actions]
        conf = self.transition_conf[z, actions]
        logits_base = torch.log(base.clamp_min(1e-8))
        p_plus = torch.softmax(logits_base + self.alpha * conf, dim=-1)
        p_minus = torch.softmax(logits_base - self.alpha * conf, dim=-1)
        return 0.5 * (p_plus + p_minus)

    def latent_state(self) -> torch.Tensor:
        zf = self._state.z.float().unsqueeze(-1)
        uf = self._state.u.float().unsqueeze(-1)
        return torch.cat([zf, uf], dim=-1)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = action.long().view(-1)
        if actions.numel() == 1 and self.n_envs > 1:
            actions = actions.repeat(self.n_envs)
        if actions.numel() != self.n_envs:
            raise ValueError("Action batch size must equal n_envs")

        p_rew = self._factual_reward_prob(actions)
        rew_draw = torch.rand(self.n_envs, generator=self.rng, device=self.device)
        reward = torch.where(
            rew_draw < p_rew,
            torch.ones(self.n_envs, device=self.device),
            -torch.ones(self.n_envs, device=self.device),
        )

        trans = self._factual_transition(actions)
        next_z = torch.multinomial(trans, num_samples=1).squeeze(-1)
        self._state.z = next_z
        self._state.t = self._state.t + 1

        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        truncated = self._state.t >= self.horizon
        info = {
            "latent": self.latent_state().detach().clone(),
            "cell": self.cell,
        }

        if truncated.any():
            done_idx = torch.nonzero(truncated, as_tuple=False).squeeze(-1)
            self._sample_initial_state(done_idx)

        return self._build_obs(), reward, terminated, truncated, info

    def start_video(self, path: str) -> None:
        _ = path

    def stop_video(self) -> None:
        return

    def close(self) -> None:
        return

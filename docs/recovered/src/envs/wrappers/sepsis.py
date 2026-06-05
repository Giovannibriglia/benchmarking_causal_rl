"""Tensorized sepsis-style causal simulator for the eight-cell benchmark.

This wrapper follows the sepsis setup used in the companion causal-RL paper:
discrete latent patient states, binary diabetes confounder, 8 discrete
treatment actions, and cell-dependent observability of hidden factors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from gymnasium.spaces import Box, Discrete

from src.envs.causal_base import CausalEnv

from ._causal_cell import get_cell_config


class SepsisCausalEnv(CausalEnv):
    """Vectorized sepsis-like tabular CausalEnv with exact do-oracles.

    `gamma_msm` controls confounding strength through:
        alpha = log(max(gamma_msm, 1.0))
    where alpha is the coefficient multiplying the hidden confounder in the
    behavior and reward mechanisms (alpha=0 means no confounding).
    """

    reward_support = torch.tensor([-1.0, 1.0], dtype=torch.float32)

    def __init__(
        self,
        env_id: str,
        n_envs: int,
        device: torch.device,
        seed: int,
        *,
        cell: int = 1,
        horizon: int = 20,
        gamma_msm: float = 1.0,
        **_: object,
    ) -> None:
        self.env_id = env_id
        self.n_envs = int(n_envs)
        self.device = device
        self.cell = int(cell)
        self.cell_cfg = get_cell_config(self.cell)
        self.horizon = int(horizon)
        self.alpha = float(torch.log(torch.tensor(max(float(gamma_msm), 1.0))).item())

        self.n_states = 720
        self.n_actions = 8
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

        self._z = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self._u = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
        self._t = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

        base_logits = torch.randn(
            2, self.n_states, self.n_actions, self.n_states, generator=self.rng, device=self.device
        )
        self.transition = torch.softmax(base_logits, dim=-1)
        self.transition_conf = torch.randn(
            self.n_states, self.n_actions, self.n_states, generator=self.rng, device=self.device
        ) * 0.1
        self.reward_logits = torch.randn(
            2, self.n_states, self.n_actions, generator=self.rng, device=self.device
        )
        self.behavior_logits = torch.randn(
            self.n_states, self.n_actions, generator=self.rng, device=self.device
        )
        self.behavior_gate = torch.randn(
            self.n_states, self.n_actions, generator=self.rng, device=self.device
        )

        obs_dim = 720 if self.cell_cfg.z_exposed else 64
        if self.cell_cfg.u_exposed:
            obs_dim += 1
        self.obs_dim = obs_dim
        self.obs_space = Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=float)
        self.act_space = Discrete(self.n_actions)

    def _sample_initial(self, idx: torch.Tensor) -> None:
        n = int(idx.numel())
        self._z[idx] = torch.randint(
            0, self.n_states, (n,), generator=self.rng, device=self.device
        )
        u = torch.randint(0, 2, (n,), generator=self.rng, device=self.device)
        self._u[idx] = u.float() * 2.0 - 1.0
        self._t[idx] = 0

    def _obs(self) -> torch.Tensor:
        if self.cell_cfg.z_exposed:
            obs = torch.zeros((self.n_envs, 720), device=self.device)
            obs[torch.arange(self.n_envs, device=self.device), self._z] = 1.0
        else:
            obs = torch.randn((self.n_envs, 64), generator=self.rng, device=self.device)

        if self.cell_cfg.u_exposed:
            obs = torch.cat([obs, self._u.unsqueeze(-1)], dim=-1)
        return obs

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self.rng.manual_seed(int(seed))
        idx = torch.arange(self.n_envs, device=self.device)
        self._sample_initial(idx)
        return self._obs(), {}

    def _factual_reward_prob(self, actions: torch.Tensor) -> torch.Tensor:
        z = self._z
        u_idx = ((self._u + 1.0) * 0.5).long()
        base = self.reward_logits[u_idx, z, actions]
        conf = self.behavior_gate[z, actions]
        return torch.sigmoid(base + self.alpha * self._u * conf)

    def do_reward(self, action: torch.Tensor) -> torch.Tensor:
        actions = action.long().view(-1)
        z = self._z
        p_u0 = torch.sigmoid(
            self.reward_logits[0, z, actions] - self.alpha * self.behavior_gate[z, actions]
        )
        p_u1 = torch.sigmoid(
            self.reward_logits[1, z, actions] + self.alpha * self.behavior_gate[z, actions]
        )
        p = 0.5 * (p_u0 + p_u1)
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.stack([1.0 - p, p], dim=-1)

    def observed_reward_distribution(
        self, action: torch.Tensor, reward: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        _ = reward
        actions = action.long().view(-1)
        p = self._factual_reward_prob(actions).clamp(1e-6, 1 - 1e-6)
        return torch.stack([1.0 - p, p], dim=-1)

    def _factual_transition(self, actions: torch.Tensor) -> torch.Tensor:
        z = self._z
        u_idx = ((self._u + 1.0) * 0.5).long()
        base = self.transition[u_idx, z, actions]
        conf = self.transition_conf[z, actions]
        logits = torch.log(base.clamp_min(1e-8)) + self.alpha * self._u.unsqueeze(-1) * conf
        return torch.softmax(logits, dim=-1)

    def do_transition(self, action: torch.Tensor) -> torch.Tensor:
        actions = action.long().view(-1)
        z = self._z
        base_u0 = self.transition[0, z, actions]
        base_u1 = self.transition[1, z, actions]
        conf = self.transition_conf[z, actions]
        p0 = torch.softmax(torch.log(base_u0.clamp_min(1e-8)) - self.alpha * conf, dim=-1)
        p1 = torch.softmax(torch.log(base_u1.clamp_min(1e-8)) + self.alpha * conf, dim=-1)
        return 0.5 * (p0 + p1)

    def latent_state(self) -> torch.Tensor:
        return torch.stack([self._z.float(), self._u.float()], dim=-1)

    def behavior_log_probs(self) -> torch.Tensor:
        logits = self.behavior_logits[self._z]
        if self.cell in (7, 8):
            logits = logits + self.alpha * self._u.unsqueeze(-1) * self.behavior_gate[self._z]
        return torch.log_softmax(logits, dim=-1)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = action.long().view(-1)
        if actions.numel() == 1 and self.n_envs > 1:
            actions = actions.repeat(self.n_envs)
        p_rew = self._factual_reward_prob(actions)
        draw = torch.rand(self.n_envs, generator=self.rng, device=self.device)
        reward = torch.where(
            draw < p_rew,
            torch.ones(self.n_envs, device=self.device),
            -torch.ones(self.n_envs, device=self.device),
        )
        trans = self._factual_transition(actions)
        self._z = torch.multinomial(trans, num_samples=1).squeeze(-1)
        self._t = self._t + 1
        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        truncated = self._t >= self.horizon

        info = {"cell": self.cell}
        if self.cell_cfg.pi_b_known:
            logp_all = self.behavior_log_probs()
            info["behavior_logprob"] = logp_all.gather(
                1, actions.unsqueeze(-1)
            ).squeeze(-1)
            info["behavior_logits"] = logp_all
        info["latent"] = self.latent_state().detach().clone()

        if truncated.any():
            idx = torch.nonzero(truncated, as_tuple=False).squeeze(-1)
            self._sample_initial(idx)

        return self._obs(), reward, terminated, truncated, info

    def start_video(self, path: str) -> None:
        _ = path

    def stop_video(self) -> None:
        return

    def close(self) -> None:
        return

"""Soft Actor-Critic (continuous-only) — the Cell-1 continuous reference
trainer (Phase-6B, author decision A).

Standard SAC: squashed-Gaussian actor, twin Q critics with soft target
updates, automatically tuned entropy temperature (target entropy =
−action_dim). Additive: built on the existing Algorithm/BaseOffPolicy
abstractions and the BenchmarkRunner off-policy loop; no edits to existing
algorithms.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ActionOutput
from ..nets.mlp import MLP
from .base_off_policy import BaseOffPolicy
from .replay_buffer import ReplayBuffer

_LOG_STD_MIN, _LOG_STD_MAX = -20.0, 2.0


class SquashedGaussianActor(nn.Module):
    """Standard SAC actor: ReLU trunk (tanh trunks saturate on MuJoCo and
    cost a large fraction of final return - measured: 3.2k vs target 12k)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.trunk = MLP(
            obs_dim,
            hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=nn.ReLU,
        )
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor):
        h = torch.relu(self.trunk(obs))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """Reparameterized squashed sample with exact log-prob."""
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre = dist.rsample()
        action = torch.tanh(pre)
        logp = dist.log_prob(pre) - torch.log1p(-action.pow(2) + 1e-6)
        return action, logp.sum(-1), torch.tanh(mean)


class SAC(BaseOffPolicy):
    """SAC for continuous action spaces (actions scaled to the env bounds)."""

    action_type = "continuous"

    def __init__(
        self,
        actor: SquashedGaussianActor,
        q1: nn.Module,
        q2: nn.Module,
        q1_target: nn.Module,
        q2_target: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        action_scale: float = 1.0,
        updates_per_learn: int = 4,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.actor = actor
        self.q1, self.q2 = q1, q2
        self.q1_target, self.q2_target = q1_target, q2_target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.buffer = buffer
        self.tau = float(tau)
        self.action_scale = float(action_scale)
        # The benchmark off-policy loop calls update() once per VECTOR step
        # (UTD ~= 1/n_envs). SAC needs a higher update-to-data ratio on
        # MuJoCo, so learn() performs extra gradient steps on self-sampled
        # batches - additive, the shared loop stays untouched.
        self.updates_per_learn = max(1, int(updates_per_learn))
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        noise: bool = True,
    ) -> ActionOutput:
        if deterministic or not noise:
            with torch.no_grad():
                mean, _ = self.actor(obs)
                return ActionOutput(
                    action=torch.tanh(mean) * self.action_scale, state=state
                )
        with torch.no_grad():
            action, logp, _ = self.actor.sample(obs)
        return ActionOutput(
            action=action * self.action_scale, log_prob=logp, state=state
        )

    def _q_input(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, actions], dim=-1)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics = self._learn_one(batch)
        batch_size = int(batch["rewards"].shape[0])
        for _ in range(self.updates_per_learn - 1):
            if len(self.buffer) <= batch_size:
                break
            metrics = self._learn_one(self.buffer.sample(batch_size))
        return metrics

    def _learn_one(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"] / self.action_scale
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # critics
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_obs)
            tq = torch.min(
                self.q1_target(self._q_input(next_obs, next_a)),
                self.q2_target(self._q_input(next_obs, next_a)),
            ).squeeze(-1)
            target = rewards + self.gamma * (1.0 - dones) * (
                tq - self.alpha.detach() * next_logp
            )
        q1 = self.q1(self._q_input(obs, actions)).squeeze(-1)
        q2 = self.q2(self._q_input(obs, actions)).squeeze(-1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # actor
        new_a, logp, _ = self.actor.sample(obs)
        q_new = torch.min(
            self.q1(self._q_input(obs, new_a)), self.q2(self._q_input(obs, new_a))
        ).squeeze(-1)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # temperature
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # soft target updates
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "loss": float((critic_loss + actor_loss).item()),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_loss": float(critic_loss.item()),
            "entropy": float(-logp.mean().item()),
            "alpha": float(self.alpha.item()),
        }

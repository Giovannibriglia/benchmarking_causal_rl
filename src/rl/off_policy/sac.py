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
from ..models.backbone import select_backbone
from .base_off_policy import BaseOffPolicy
from .replay_buffer import ReplayBuffer

_LOG_STD_MIN, _LOG_STD_MAX = -20.0, 2.0


class SquashedGaussianActor(nn.Module):
    """Standard SAC actor: ReLU trunk (tanh trunks saturate on MuJoCo and
    cost a large fraction of final return - measured: 3.2k vs target 12k)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
        obs_shape=None,
        network: str = "mlp",
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        # network="mlp" keeps the exact pre-recurrent construction (bitwise / SAC
        # golden); a recurrent type routes the trunk through build_trunk and makes
        # forward/sample state-aware (returning new_state as a trailing element).
        self.is_recurrent = network != "mlp"
        feat_dim = hidden_dims[-1]
        if not self.is_recurrent:
            # Trunk routed through the shared selector; obs_shape defaults to
            # (obs_dim,) -> rank-1 -> the identical MLP (bitwise).
            self.trunk = select_backbone(
                obs_shape if obs_shape is not None else (obs_dim,),
                obs_dim,
                feat_dim,
                hidden_dims=hidden_dims[:-1],
                activation=nn.ReLU,
            )
        else:
            from src.rl.models.backbone import build_trunk

            self.trunk = build_trunk(
                network,
                obs_shape if obs_shape is not None else (obs_dim,),
                obs_dim,
                feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        self.mean_head = nn.Linear(feat_dim, action_dim)
        self.log_std_head = nn.Linear(feat_dim, action_dim)

    def initial_state(self, n: int, device=None):
        return self.trunk.initial_state(n, device=device) if self.is_recurrent else None

    def forward(self, obs: torch.Tensor, state=None):
        if self.is_recurrent:
            feat, new_state = self.trunk(obs, state)
            h = torch.relu(feat)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
            return mean, log_std, new_state
        h = torch.relu(self.trunk(obs))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, state=None):
        """Reparameterized squashed sample with exact log-prob. Returns
        ``(action, logp, tanh(mean))`` for MLP (unchanged), plus ``new_state`` as
        a 4th element when recurrent."""
        if self.is_recurrent:
            mean, log_std, new_state = self(obs, state)
        else:
            mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre = dist.rsample()
        action = torch.tanh(pre)
        logp = dist.log_prob(pre) - torch.log1p(-action.pow(2) + 1e-6)
        if self.is_recurrent:
            return action, logp.sum(-1), torch.tanh(mean), new_state
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
        updates_per_learn: int = 8,  # = n_train_envs -> canonical UTD 1.0
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
        # Recurrent iff the actor or the critics carry hidden state. The MLP path
        # is byte-identical -> SAC golden stays green.
        self.is_recurrent = getattr(actor, "is_recurrent", False) or hasattr(
            q1, "initial_state"
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # -- recurrent hidden-state helpers (collection threads only the ACTOR's
    # state; critics zero-init per sequence at learn time) --
    def initial_state(self, n: int, device=None):
        if getattr(self.actor, "is_recurrent", False):
            return self.actor.initial_state(n, device=device)
        return None

    def reset_state_where(self, state, mask: torch.Tensor):
        if state is None:
            return None
        if isinstance(state, tuple):
            for s in state:
                s[:, mask, :] = 0.0
            return state
        state[:, mask, :] = 0.0
        return state

    def _q_out(self, net, obs, action):
        """Q value, handling MLP (tensor) and recurrent (out, state) critics."""
        o = net(self._q_input(obs, action))
        o = o[0] if isinstance(o, tuple) else o
        return o.squeeze(-1)

    def _actor_sample(self, obs, state=None):
        """Actor sample returning (action, logp, mean_action) regardless of
        whether the actor is recurrent (zero-init when recurrent)."""
        if getattr(self.actor, "is_recurrent", False):
            a, lp, m, _ = self.actor.sample(obs, state)
            return a, lp, m
        return self.actor.sample(obs)

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        noise: bool = True,
    ) -> ActionOutput:
        if getattr(self.actor, "is_recurrent", False):
            with torch.no_grad():
                if deterministic or not noise:
                    mean, _, new_state = self.actor(obs, state)
                    return ActionOutput(
                        action=torch.tanh(mean) * self.action_scale, state=new_state
                    )
                action, logp, _, new_state = self.actor.sample(obs, state)
            return ActionOutput(
                action=action * self.action_scale, log_prob=logp, state=new_state
            )
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
        if self.is_recurrent:
            return self._learn_recurrent(batch)
        metrics = self._learn_one(batch)
        batch_size = int(batch["rewards"].shape[0])
        for _ in range(self.updates_per_learn - 1):
            if len(self.buffer) <= batch_size:
                break
            metrics = self._learn_one(self.buffer.sample(batch_size))
        return metrics

    def _learn_recurrent(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Sequence SAC update over (B, T, ...) within-episode batches. UTD is
        unchanged: updates_per_learn sequence-samples per learn step (mirrors the
        MLP path's updates_per_learn random-samples). (B, T) is inferred from the
        passed batch so internal resamples match shape."""
        metrics = self._learn_one_recurrent(batch)
        b, t = batch["rewards"].shape[0], batch["rewards"].shape[1]
        for _ in range(self.updates_per_learn - 1):
            if not self.buffer.can_sample(t):
                break
            metrics = self._learn_one_recurrent(self.buffer.sample_sequences(b, t))
        return metrics

    def _learn_one_recurrent(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]  # (B, T, obs_dim)
        actions = batch["actions"] / self.action_scale  # (B, T, act_dim)
        rewards = batch["rewards"]  # (B, T)
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # critics — actor + target critics zero-init over the sequence (the buffer
        # guarantees no episode-boundary span); target critics forward next_obs
        # independently of the online critics.
        with torch.no_grad():
            next_a, next_logp, _ = self._actor_sample(next_obs)
            tq = torch.min(
                self._q_out(self.q1_target, next_obs, next_a),
                self._q_out(self.q2_target, next_obs, next_a),
            )
            target = rewards + self.gamma * (1.0 - dones) * (
                tq - self.alpha.detach() * next_logp
            )
        q1 = self._q_out(self.q1, obs, actions)
        q2 = self._q_out(self.q2, obs, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # actor
        new_a, logp, _ = self._actor_sample(obs)
        q_new = torch.min(
            self._q_out(self.q1, obs, new_a), self._q_out(self.q2, obs, new_a)
        )
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

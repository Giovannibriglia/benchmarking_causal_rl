from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ActionOutput
from ..nets.mlp import MLP
from ..off_policy.base_off_policy import BaseOffPolicy


class RunningNorm(nn.Module):
    """Per-feature running mean / std (Welford over batches), as buffers so it
    checkpoints with the model. ``update`` is called on real data only."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * (n / total)
        self.m2 += batch_m2 + delta**2 * (self.count * n / total)
        self.count.copy_(total)

    @property
    def std(self) -> torch.Tensor:
        var = self.m2 / self.count.clamp(min=1)
        return var.sqrt().clamp(min=self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean


class DynamicsEnsemble(nn.Module):
    """``K`` MLPs, each mapping ``(obs, one_hot(a)) -> (delta_obs, r, done_logit)``
    in NORMALISED coordinates (obs and delta each have their own RunningNorm).

    The ensemble's disagreement (std of the predicted delta across members)
    is exposed as an epistemic-uncertainty signal for the optional
    pessimism penalty in the offline regime.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_members: int = 3,
        hidden_dims: Tuple[int, ...] = (200, 200),
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.members = nn.ModuleList(
            [
                MLP(
                    self.obs_dim + self.n_actions,
                    self.obs_dim + 2,
                    hidden_dims=hidden_dims,
                    activation=nn.ReLU,
                )
                for _ in range(int(n_members))
            ]
        )
        self.obs_norm = RunningNorm(self.obs_dim)
        self.delta_norm = RunningNorm(self.obs_dim)

    def _inputs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        a = F.one_hot(actions.long(), self.n_actions).float()
        return torch.cat([self.obs_norm.normalize(obs.float()), a], dim=-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """Raw per-member heads: ``delta_z (K,B,D)``, ``r (K,B)``, ``done_logit (K,B)``."""
        x = self._inputs(obs, actions)
        outs = torch.stack([m(x) for m in self.members])  # (K, B, D+2)
        return (
            outs[..., : self.obs_dim],
            outs[..., self.obs_dim],
            outs[..., self.obs_dim + 1],
        )

    def loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, act = batch["obs"], batch["actions"]
        delta = batch["next_obs"].float() - obs.float()
        self.obs_norm.update(obs)
        self.delta_norm.update(delta)
        dz, r, dl = self.forward(obs, act)
        target_dz = self.delta_norm.normalize(delta).unsqueeze(0).expand_as(dz)
        l_dyn = F.mse_loss(dz, target_dz)
        l_rew = F.mse_loss(r, batch["rewards"].float().unsqueeze(0).expand_as(r))
        l_done = F.binary_cross_entropy_with_logits(
            dl, batch["dones"].float().unsqueeze(0).expand_as(dl)
        )
        total = l_dyn + l_rew + l_done
        return total, {
            "model_loss": float(total.item()),
            "model_dyn_loss": float(l_dyn.item()),
            "model_reward_loss": float(l_rew.item()),
            "model_done_loss": float(l_done.item()),
        }

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, actions: torch.Tensor):
        """Ensemble-mean prediction in raw coordinates:
        ``next_obs (B,D)``, ``reward (B,)``, ``done_prob (B,)``,
        ``disagreement (B,)`` = mean over dims of the members' delta std."""
        dz, r, dl = self.forward(obs, actions)
        delta = self.delta_norm.denormalize(dz)  # (K, B, D)
        next_obs = obs.float() + delta.mean(dim=0)
        disagreement = delta.std(dim=0, unbiased=False).mean(dim=-1)
        return next_obs, r.mean(dim=0), torch.sigmoid(dl).mean(dim=0), disagreement


class MLPValueIteration(BaseOffPolicy):
    """Fitted value iteration through a learned MLP dynamics model.

    Per ``learn(batch)`` (one sampled batch of REAL transitions):

    1. **Model step** -- one gradient step of the ensemble on the batch.
    2. **Value-iteration step** (after ``model_warmup`` model steps) -- for
       every state ``s`` in the batch and EVERY action ``a``, the model gives
       ``(s', r, d)`` and the target ``y(s,a) = r - c*u(s,a) + gamma (1-d) max_a' Q_t(s',a')``;
       the Q-network regresses onto all ``|A|`` targets at once. That is a
       full Bellman backup on the batch's states -- value iteration on the
       buffer's state sample, the continuous analogue of ``TabularVI``.
       ``u`` is the ensemble disagreement and ``c = penalty_coef`` (0 online;
       the offline builder's default is small and positive, MOPO-style, so
       the planner cannot exploit model error where the data is thin).

    The real transition's reward and next state are used ONLY to train the
    model; Q never sees them directly. Acting is epsilon-greedy on Q.
    """

    action_type = "discrete"

    def __init__(
        self,
        model: DynamicsEnsemble,
        q_network: nn.Module,
        target_network: nn.Module,
        buffer,
        device: torch.device,
        lr_model: float = 1e-3,
        lr_q: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon: float = 0.1,
        model_warmup: int = 500,
        penalty_coef: float = 0.0,
        n_actions: int | None = None,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.model = model
        self.q_network = q_network
        self.target_network = target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.buffer = buffer
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_model)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr_q)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.model_warmup = int(model_warmup)
        self.penalty_coef = float(penalty_coef)
        self.n_actions = int(n_actions if n_actions is not None else model.n_actions)
        self.is_recurrent = False
        self._model_steps = 0

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> ActionOutput:
        if deterministic:
            epsilon = 0.0
        eps = self.epsilon if epsilon is None else epsilon
        if torch.rand(1).item() < eps:
            return ActionOutput(
                action=torch.randint(
                    0, self.n_actions, (obs.shape[0],), device=obs.device
                ),
                state=state,
            )
        with torch.no_grad():
            return ActionOutput(
                action=torch.argmax(self.q_network(obs), dim=1), state=state
            )

    def _model_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        loss, metrics = self.model.loss(batch)
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        self._model_steps += 1
        return metrics

    def _vi_step(self, obs: torch.Tensor) -> Dict[str, float]:
        B, A = obs.shape[0], self.n_actions
        s = obs.float().repeat_interleave(A, dim=0)  # (B*A, D): s_0 x A, s_1 x A, ...
        a = torch.arange(A, device=obs.device).repeat(B)  # 0..A-1, 0..A-1, ...
        with torch.no_grad():
            next_obs, r, d, u = self.model.predict(s, a)
            next_q = self.target_network(next_obs).max(dim=1).values
            y = (r - self.penalty_coef * u) + self.gamma * (1.0 - d) * next_q
            y = y.reshape(B, A)
        q = self.q_network(obs)  # (B, A)
        loss = F.mse_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for p, tp in zip(self.q_network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        lv = float(loss.item())
        return {
            "q_loss": lv,
            "critic_loss": lv,
            "model_disagreement": float(u.mean().item()),
            "model_done_prob": float(d.mean().item()),
        }

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics = self._model_step(batch)
        if self._model_steps > self.model_warmup:
            metrics.update(self._vi_step(batch["obs"]))
            metrics["loss"] = metrics["q_loss"]
        else:
            metrics["loss"] = metrics["model_loss"]
        return metrics

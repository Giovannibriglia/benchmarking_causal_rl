from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ActionOutput
from .base_off_policy import BaseOffPolicy
from .identification import IdentificationStrategy, Observational
from .replay_buffer import ReplayBuffer


class DQN(BaseOffPolicy):
    """DQN for discrete action spaces."""

    action_type = "discrete"

    def __init__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon: float = 0.1,
        strategy: IdentificationStrategy | None = None,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.q_network = q_network
        self.target_network = target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.tau = tau
        self.epsilon = epsilon
        # Identification strategy: routes the u-conditionable critic evals.
        # Default Observational => critic_value is net(x), byte-identical to the
        # pre-strategy code (online + offline DQN goldens unchanged).
        self._strategy = strategy if strategy is not None else Observational()
        # Recurrent iff the Q-trunk carries hidden state (build_trunk("lstm"/...)
        # exposes initial_state; build_trunk("mlp") returns a bare MLP). The MLP
        # path below stays byte-identical -> off-policy goldens stay green.
        self.is_recurrent = hasattr(q_network, "initial_state")

    # -- recurrent hidden-state helpers (used by the recurrent collection path) --
    def initial_state(self, n: int, device=None):
        if not self.is_recurrent:
            return None
        return self.q_network.initial_state(n, device=device)

    def reset_state_where(self, state, mask: torch.Tensor):
        """Zero the per-env slots in ``state`` where ``mask`` is True (episode
        boundary). Handles LSTM ``(h, c)`` tuples and bare ``h``; None -> None."""
        if state is None:
            return None
        if isinstance(state, tuple):
            for s in state:
                s[:, mask, :] = 0.0
            return state
        state[:, mask, :] = 0.0
        return state

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
        if self.is_recurrent:
            # Advance the hidden state every step (greedy OR random) so it
            # reflects the observations seen, not the actions taken.
            with torch.no_grad():
                q, new_state = self.q_network(obs, state)
            if torch.rand(1).item() < eps:
                action = torch.randint(
                    0, q.shape[-1], (obs.shape[0],), device=obs.device
                )
            else:
                action = torch.argmax(q, dim=-1)
            return ActionOutput(action=action, state=new_state)
        # MLP path (unchanged / bitwise). NOTE: torch.rand is evaluated
        # unconditionally (even for eps == 0.0) to preserve RNG consumption order.
        if torch.rand(1).item() < eps:
            batch = obs.shape[0]
            return ActionOutput(
                action=torch.randint(
                    0, self.q_network(obs).shape[1], (batch,), device=obs.device
                ),
                state=state,
            )
        with torch.no_grad():
            q = self.q_network(obs)
            return ActionOutput(action=torch.argmax(q, dim=1), state=state)

    def _learn_recurrent(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Sequence TD update over ``(B, T, ...)`` within-episode batches from the
        SequenceReplayBuffer. Both nets zero-init at sequence start (the buffer
        guarantees no episode-boundary span, so no mid-sequence reset is needed);
        the target net forwards next_obs INDEPENDENTLY (its own hidden state).
        loss.backward() propagates through the recurrent cells (the BPTT)."""
        obs = batch["obs"]  # (B, T, D)
        actions = batch["actions"].long()  # (B, T)
        rewards = batch["rewards"]  # (B, T)
        next_obs = batch["next_obs"]  # (B, T, D)
        dones = batch["dones"]  # (B, T)

        q_all, _ = self.q_network(obs)  # (B, T, A), zero-init
        q_values = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, T)
        with torch.no_grad():
            tq_all, _ = self.target_network(next_obs)  # (B, T, A), independent
            next_q = tq_all.max(dim=-1).values  # (B, T)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        return {"loss": loss.item(), "critic_loss": loss.item(), "q_loss": loss.item()}

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.is_recurrent:
            return self._learn_recurrent(batch)
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        q_values = (
            self._strategy.critic_value(self.q_network, obs, batch)
            .gather(1, actions.unsqueeze(-1))
            .squeeze(-1)
        )
        with torch.no_grad():
            next_q = (
                self._strategy.critic_value(self.target_network, next_obs, batch)
                .max(dim=1)
                .values
            )
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        return {"loss": loss.item(), "critic_loss": loss.item(), "q_loss": loss.item()}

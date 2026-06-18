from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn

from ..base_policy import BasePolicy
from ..models.backbone import build_trunk


class _TrunkAdapter(nn.Module):
    """Uniform ``(out, new_state)`` contract over a pooled trunk.

    Recurrent trunks (those exposing ``initial_state``) delegate verbatim;
    bare trunks (MLP / CNN, built via ``select_backbone``) return ``(out, None)``
    and ignore the passed state. This keeps ``build_trunk``'s public MLP contract
    untouched (a bare ``MLP`` is still returned by ``build_trunk("mlp", ...)``,
    preserving PR #48's byte-identical invariant) while giving ``ActorCritic`` a
    single ``(out, state)`` call site for every trunk type.
    """

    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.recurrent = hasattr(trunk, "initial_state")

    def initial_state(self, batch_size: int, device=None):
        if self.recurrent:
            return self.trunk.initial_state(batch_size, device=device)
        return None

    def forward(self, obs, state=None):
        if self.recurrent:
            return self.trunk(obs, state)
        return self.trunk(obs), None


class ActorCritic(BasePolicy):
    """Separate-trunk actor-critic. The actor and critic EACH own a trunk
    (MLP / LSTM / GRU / RNN), selected per-component via ``build_trunk``; heads
    project the trunk embedding to the action distribution / scalar value.

    Replaces the shared-encoder ``ActorCriticMLP``. Even when both components are
    MLP they are separate trunks — so MLP outputs are NOT byte-identical to the
    old shared encoder (different parameter-init order / count); on-policy goldens
    are regenerated for this architectural change.

    Two interfaces:
      * Bare (stateless) — ``distribution(obs)`` / ``value(obs)`` / ``act(obs)``
        / ``act_deterministic(obs)`` — matches ``ActorCriticMLP``'s old surface,
        so the existing (non-recurrent) rollout + flat-minibatch PPO/A2C/TRPO/
        Vanilla updates keep working unchanged with MLP trunks.
      * Stateful (recurrent) — ``initial_state`` / ``act_step`` /
        ``evaluate_sequence`` — used only by the recurrent PPO path
        (``rollout_recurrent`` + truncated-BPTT update). Hidden state is a dict
        ``{"actor": ..., "critic": ...}``; each entry is None for an MLP trunk.
    """

    def __init__(
        self,
        obs_shape,
        obs_dim: int,
        action_dim: int,
        action_type: str,
        device: torch.device,
        actor_network: str = "mlp",
        critic_network: str = "mlp",
        *,
        embed_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (64,),
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__(device)
        self.action_type = action_type
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.embed_dim = embed_dim
        obs_shape = tuple(obs_shape) if obs_shape is not None else (obs_dim,)

        # MLP trunks consume hidden_dims; recurrent trunks consume
        # hidden_dim/num_layers. build_trunk forwards only the kwargs the chosen
        # trunk accepts, so pass per-type sets.
        def _mk(network: str) -> _TrunkAdapter:
            if network == "mlp":
                trunk = build_trunk(
                    network, obs_shape, obs_dim, embed_dim, hidden_dims=hidden_dims
                )
            else:
                trunk = build_trunk(
                    network,
                    obs_shape,
                    obs_dim,
                    embed_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )
            return _TrunkAdapter(trunk)

        self.actor_trunk = _mk(actor_network)
        self.critic_trunk = _mk(critic_network)

        # Heads.
        self.actor_head = nn.Linear(embed_dim, action_dim)
        if action_type == "discrete":
            self.log_std = None
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(embed_dim, 1)
        self.to(device)

    # ------------------------------------------------------------------
    # Recurrence introspection
    # ------------------------------------------------------------------
    @property
    def is_recurrent(self) -> bool:
        """True iff either component uses a non-MLP (recurrent) trunk. The runner
        dispatches to the recurrent rollout/BPTT path on this."""
        return self.actor_network != "mlp" or self.critic_network != "mlp"

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------
    def _actor_dist(self, feat: torch.Tensor) -> dist.Distribution:
        if self.action_type == "discrete":
            return dist.Categorical(logits=self.actor_head(feat))
        mean = self.actor_head(feat)
        std = torch.exp(self.log_std).expand_as(mean)
        return dist.Normal(mean, std)

    def log_prob(
        self, distribution: dist.Distribution, actions: torch.Tensor
    ) -> torch.Tensor:
        logp = distribution.log_prob(actions)
        if self.action_type != "discrete":
            logp = logp.sum(-1)
        return logp

    # ------------------------------------------------------------------
    # Bare (stateless) interface — MLP path, byte-compatible surface
    # ------------------------------------------------------------------
    def distribution(self, obs: torch.Tensor) -> dist.Distribution:
        if obs.dim() == 1:
            obs = obs.unsqueeze(-1)
        feat, _ = self.actor_trunk(obs, None)
        return self._actor_dist(feat)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(-1)
        feat, _ = self.critic_trunk(obs, None)
        return self.critic_head(feat).squeeze(-1)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        action = distribution.sample()
        logp = self.log_prob(distribution, action)
        return action.to(self.device), logp.to(self.device)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(obs)
        if self.action_type == "discrete":
            return torch.argmax(distribution.logits, dim=-1)
        return distribution.mean

    # ------------------------------------------------------------------
    # Stateful (recurrent) interface — recurrent PPO path only
    # ------------------------------------------------------------------
    def initial_state(self, batch_size: int, device=None) -> dict:
        device = device or self.device
        return {
            "actor": self.actor_trunk.initial_state(batch_size, device=device),
            "critic": self.critic_trunk.initial_state(batch_size, device=device),
        }

    def act_step(
        self, obs: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """One step of recurrent collection: advance both trunks, sample an
        action, and return ``(action, log_prob, value, new_state)``."""
        if state is None:
            state = self.initial_state(obs.shape[0], device=obs.device)
        a_feat, a_state = self.actor_trunk(obs, state["actor"])
        c_feat, c_state = self.critic_trunk(obs, state["critic"])
        distribution = self._actor_dist(a_feat)
        action = distribution.sample()
        logp = self.log_prob(distribution, action)
        value = self.critic_head(c_feat).squeeze(-1)
        return action, logp, value, {"actor": a_state, "critic": c_state}

    def value_step(self, obs: torch.Tensor, critic_state=None):
        """Critic-only step (no action sample): ``(value, new_critic_state)``.
        Used for the GAE next-value bootstrap during recurrent collection, where
        V(s_{t+1}) must use the critic hidden state after observing s_t."""
        feat, new_state = self.critic_trunk(obs, critic_state)
        return self.critic_head(feat).squeeze(-1), new_state

    def reset_state_where(self, state: dict, mask: torch.Tensor) -> dict:
        """Return a copy of the ``{actor, critic}`` state with per-env slots in
        ``mask`` (shape ``[N]``) zeroed — used between rollout steps to reset
        recurrent state at episode boundaries."""
        return {
            "actor": self._reset_state(state["actor"], mask),
            "critic": self._reset_state(state["critic"], mask),
        }

    def _reset_state(self, state, mask: torch.Tensor):
        """Zero the per-env slots in ``state`` where ``mask`` (shape ``[N]``) is
        True (episode boundary). Handles LSTM ``(h, c)`` tuples and bare ``h``;
        None (MLP) is left as None."""
        if state is None:
            return None
        if isinstance(state, tuple):
            for s in state:
                s[:, mask, :] = 0.0
            return state
        state[:, mask, :] = 0.0
        return state

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        episode_starts: torch.Tensor,
        init_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Truncated-BPTT forward over ``(T, N, ...)`` sequences with per-env
        hidden-state resets at episode boundaries (``episode_starts[t]`` True).

        Returns ``(log_probs, values, entropy)`` each shaped ``(T, N)`` (entropy
        summed over action dims for continuous). Gradients flow through the
        recurrent connections within each episode — this IS the BPTT.
        """
        T, N = obs_seq.shape[0], obs_seq.shape[1]
        if init_state is None:
            init_state = self.initial_state(N, device=obs_seq.device)
        a_state = init_state["actor"]
        c_state = init_state["critic"]
        logps, values, entropies = [], [], []
        for t in range(T):
            mask = episode_starts[t].bool()
            if mask.any():
                a_state = self._reset_state(a_state, mask)
                c_state = self._reset_state(c_state, mask)
            a_feat, a_state = self.actor_trunk(obs_seq[t], a_state)
            c_feat, c_state = self.critic_trunk(obs_seq[t], c_state)
            distribution = self._actor_dist(a_feat)
            logps.append(self.log_prob(distribution, actions_seq[t]))
            values.append(self.critic_head(c_feat).squeeze(-1))
            ent = distribution.entropy()
            if ent.ndim > 1:
                ent = ent.sum(-1)
            entropies.append(ent)
        return (
            torch.stack(logps),
            torch.stack(values),
            torch.stack(entropies),
        )

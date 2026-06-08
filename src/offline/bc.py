"""Behavior cloning — the basic offline baseline (§5, all offline cells)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.data.experience_source import OfflineDatasetSource
from src.rl.base import ActionOutput, Algorithm
from src.rl.on_policy.policy import ActorCriticMLP


class BehaviorCloning(Algorithm):
    """Maximum-likelihood imitation of the dataset's action distribution.

    Uses :class:`ActorCriticMLP` as the policy container (actor head only is
    trained), so evaluation plumbing (``act`` / ``act_deterministic``) is
    shared with the online algorithms.
    """

    paradigm = "offline"

    def __init__(
        self,
        policy: ActorCriticMLP,
        device: torch.device,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.action_type = policy.action_type
        # Standard observation normalization (uniform with the d3rlpy
        # variants' StandardObservationScaler): fit mean/std on the source in
        # fit_source; identity until then. Correctness config, not tuning.
        self._obs_mean: Optional[torch.Tensor] = None
        self._obs_std: Optional[torch.Tensor] = None

    def _scale(self, obs: torch.Tensor) -> torch.Tensor:
        if self._obs_mean is None:
            return obs
        return (obs - self._obs_mean) / self._obs_std

    def act(
        self,
        obs: torch.Tensor,
        state: Optional[Any] = None,
        *,
        deterministic: bool = False,
    ) -> ActionOutput:
        obs = self._scale(obs.float())
        if deterministic:
            return ActionOutput(action=self.policy.act_deterministic(obs), state=state)
        action, logp = self.policy.act(obs)
        return ActionOutput(action=action, log_prob=logp, state=state)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = self._scale(batch["obs"].float())
        actions = batch["actions"]
        distribution = self.policy.distribution(obs)
        logp = self.policy.log_prob(distribution, actions)
        loss = -logp.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "bc_nll": loss.item()}

    def fit_source(
        self,
        source: OfflineDatasetSource,
        n_steps: int,
        batch_size: int = 256,
        on_step=None,
        on_step_every: int = 0,
    ) -> Dict[str, float]:
        # fit standardization stats on the full source (once)
        obs_all = source.obs.float()
        self._obs_mean = obs_all.mean(dim=0, keepdim=True)
        self._obs_std = obs_all.std(dim=0, keepdim=True).clamp_min(1e-6)
        metrics: Dict[str, float] = {}
        for it in range(int(n_steps)):
            metrics = self.update(source.sample(batch_size), source)
            if on_step and on_step_every and (it + 1) % on_step_every == 0:
                on_step(it + 1)
        return metrics

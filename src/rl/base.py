from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn


@dataclass
class ActionOutput:
    """Result of a single action selection.

    `state` carries recurrent / per-agent state through time (POMDP- and
    MARL-ready); single-agent feed-forward algorithms pass it through
    untouched.
    """

    action: torch.Tensor
    log_prob: Optional[torch.Tensor] = None
    state: Optional[Any] = None


class Algorithm(abc.ABC, nn.Module):
    """Unified interface for all RL algorithms in this repo.

    Class attributes
    ----------------
    paradigm: which data regime the algorithm consumes. ``on_policy``
        algorithms require an online experience source (validated at
        config time, see ``src.data.experience_source.validate_pairing``).
    action_type: action spaces supported by the algorithm.

    Contract
    --------
    ``update`` = ``process_batch`` (advantages / n-step targets) followed by
    ``learn`` (the gradient update(s)). The ``source`` argument provides
    dataset context (e.g. logged propensities) for offline algorithms; online
    algorithms ignore it.

    MARL readiness: tensors may carry a leading agent dimension
    ``(*batch, n_agents, feat)``; single-agent paths squeeze it.
    ``set_agent_group`` is a no-op when ``n_agents == 1``.

    Actor/critic separation: off-policy algorithms keep distinct
    ``self.actor`` / ``self.critic`` modules; on-policy algorithms use the
    separate-trunk ``ActorCritic`` policy (independent actor/critic trunks,
    each MLP or recurrent, with their own heads).
    """

    paradigm: Literal["on_policy", "off_policy", "offline"]
    action_type: Literal["discrete", "continuous", "both"] = "both"

    @abc.abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        state: Optional[Any] = None,
        *,
        deterministic: bool = False,
    ) -> ActionOutput:
        """Select actions for a (vectorized) observation batch."""

    def process_batch(self, batch: Any, source: Any = None) -> Any:
        """Prepare a raw batch for learning.

        Default is identity: on-policy rollouts arrive with GAE already
        computed by the collection path, off-policy replay batches are
        consumed as-is. Offline algorithms (Phase 3+) override this to build
        n-step targets / propensity weights from the source.
        """
        return batch

    @abc.abstractmethod
    def learn(self, batch: Any) -> Dict[str, float]:
        """Run the algorithm's gradient update(s) on a processed batch."""

    def update(self, batch: Any, source: Any = None) -> Dict[str, float]:
        """``process_batch`` -> ``learn``; returns scalar training metrics."""
        return self.learn(self.process_batch(batch, source))

    def set_agent_group(self, mapping: Optional[Dict[str, Any]] = None) -> None:
        """MARL hook: assign agents to parameter groups. No-op when
        ``n_agents == 1`` (always, until ``src/marl`` is populated)."""
        return None

from __future__ import annotations

import abc

import torch

from src.rl.base import ActionOutput, Algorithm


class BehaviorPolicy(abc.ABC):
    """Produces the actions taken during off-policy *collection*.

    This is the seam that decouples the collection loop from *how* exploratory
    actions are drawn. The default reproduces today's behavior exactly (the
    agent's own exploratory ``act``); Stage B injects biased / sub-optimal /
    logged policies here to generate offline datasets, without touching the
    collection loop or the agents.
    """

    @abc.abstractmethod
    def act(self, obs: torch.Tensor) -> ActionOutput:
        """Select actions for the given (batched) observation."""
        raise NotImplementedError


class AgentBehaviorPolicy(BehaviorPolicy):
    """Default behavior policy: delegate verbatim to the agent's ``act``.

    Calling ``agent.act(obs)`` with its default arguments and drawing nothing
    extra keeps the global RNG consumption byte-identical to the pre-seam
    collection loop, so the off-policy golden values stay bitwise unchanged.
    """

    def __init__(self, agent: Algorithm) -> None:
        self.agent = agent

    def act(self, obs: torch.Tensor) -> ActionOutput:
        return self.agent.act(obs)


class BiasedBehaviorPolicy(BehaviorPolicy):
    """Stub for Stage B: a deliberately biased / sub-optimal logging policy.

    Interface only in Stage A — no behavior, never constructed by the runner.
    """

    def act(self, obs: torch.Tensor) -> ActionOutput:
        raise NotImplementedError(
            "biased behavior policies are implemented in Stage B."
        )

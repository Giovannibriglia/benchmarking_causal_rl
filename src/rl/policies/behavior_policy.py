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


def _action_bounds(action_space, device: torch.device):
    """``(low, high)`` float tensors for a continuous (Box) action space."""
    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
    return low, high


class AntiRewardBehaviorPolicy(BehaviorPolicy):
    """Critic-pessimal collection: take the action the agent's OWN critic values
    least (argmin-Q), per-algorithm. No second learner.

      * discrete (DQN): argmin_a Q(s, a), inside the SAME epsilon-greedy
        structure as ``DQN.act`` (the ``torch.rand`` draw is kept unconditional
        so a future agent= run alongside is RNG-consistent).
      * continuous (DDPG/SAC): sample ``n_candidates`` actions from the agent's
        own ``act`` and return the one the critic scores LOWEST. SAC's twin-Q is
        evaluated on the de-scaled action; DDPG's single critic on the action
        as stored.
    """

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        *,
        epsilon: float = 0.1,
        n_candidates: int = 10,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.epsilon = float(epsilon)
        self.n_candidates = max(1, int(n_candidates))

    def _critic_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Critic value ``Q(s, a)`` per env (shape ``[B]``)."""
        if hasattr(self.agent, "q1") and hasattr(self.agent, "q2"):  # SAC twin-Q
            scale = float(getattr(self.agent, "action_scale", 1.0))
            x = torch.cat([obs, action / scale], dim=-1)
            return torch.min(self.agent.q1(x), self.agent.q2(x)).squeeze(-1)
        x = torch.cat([obs, action], dim=-1)  # DDPG single critic
        return self.agent.critic(x).squeeze(-1)

    def _argmin_over_candidates(
        self, obs: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Pick the lowest-Q candidate per env. ``candidates`` is ``[K, B, A]``."""
        k, b = candidates.shape[0], candidates.shape[1]
        q = torch.stack(
            [self._critic_q(obs, candidates[i]) for i in range(k)], dim=0
        )  # [K, B]
        idx = torch.argmin(q, dim=0)  # [B]
        return candidates[idx, torch.arange(b, device=candidates.device)]

    def act(self, obs: torch.Tensor) -> ActionOutput:
        if self.action_type == "discrete":
            # Mirror DQN.act's epsilon-greedy, but argmin instead of argmax.
            if torch.rand(1).item() < self.epsilon:
                batch = obs.shape[0]
                return ActionOutput(
                    action=torch.randint(
                        0,
                        self.agent.q_network(obs).shape[1],
                        (batch,),
                        device=obs.device,
                    )
                )
            with torch.no_grad():
                q = self.agent.q_network(obs)
                return ActionOutput(action=torch.argmin(q, dim=1))
        # Continuous: sample candidates from the agent, return the worst by Q.
        with torch.no_grad():
            candidates = torch.stack(
                [self.agent.act(obs).action for _ in range(self.n_candidates)], dim=0
            )
            return ActionOutput(action=self._argmin_over_candidates(obs, candidates))


class SkewBehaviorPolicy(BehaviorPolicy):
    """Action-skew: with probability ``p`` emit a fixed PREFERRED action (a
    directional bias), else the agent's action. Discrete -> a fixed index
    (``preferred``); continuous -> the low bound of the action space (one
    extreme). ``p=1`` -> always preferred; ``p=0`` -> always the agent.
    """

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        *,
        p: float = 0.5,
        preferred: int = 0,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.p = float(p)
        self.preferred = int(preferred)

    def act(self, obs: torch.Tensor) -> ActionOutput:
        b = obs.shape[0]
        agent_a = self.agent.act(obs).action
        take_pref = torch.rand(b, device=obs.device) < self.p
        if self.action_type == "discrete":
            pref = torch.full(
                (b,), self.preferred, dtype=agent_a.dtype, device=obs.device
            )
            return ActionOutput(action=torch.where(take_pref, pref, agent_a))
        low, _ = _action_bounds(self.action_space, obs.device)
        pref = low.expand_as(agent_a)
        return ActionOutput(action=torch.where(take_pref.unsqueeze(-1), pref, agent_a))


class SuboptimalBehaviorPolicy(BehaviorPolicy):
    """Suboptimality mixture: with probability ``beta`` use the agent, else a
    UNIFORM-random base action (degrade toward random). ``beta=1`` -> all agent;
    ``beta=0`` -> all base.
    """

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        *,
        beta: float = 0.5,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.beta = float(beta)

    def _uniform_base(self, b: int, device: torch.device) -> torch.Tensor:
        if self.action_type == "discrete":
            return torch.randint(0, int(self.action_space.n), (b,), device=device)
        low, high = _action_bounds(self.action_space, device)
        return low + (high - low) * torch.rand(b, *low.shape, device=device)

    def act(self, obs: torch.Tensor) -> ActionOutput:
        b = obs.shape[0]
        agent_a = self.agent.act(obs).action
        base_a = self._uniform_base(b, obs.device)
        use_agent = torch.rand(b, device=obs.device) < self.beta
        if self.action_type == "discrete":
            return ActionOutput(action=torch.where(use_agent, agent_a, base_a))
        return ActionOutput(
            action=torch.where(use_agent.unsqueeze(-1), agent_a, base_a)
        )


# Collection-policy registry. The "agent" entry is handled by the runner with
# the original AgentBehaviorPolicy(self.agent) construction (byte-identical to
# the pre-A1 path), so the off-policy golden is untouched; this factory only
# builds the opt-in policies.
_STRENGTH_PARAM = {
    "anti_reward": "epsilon",  # epsilon-greedy exploration over argmin-Q
    "bias_skew": "p",  # probability of the preferred action
    "bias_suboptimal": "beta",  # probability of using the agent (vs uniform base)
}


def build_collection_policy(
    name: str,
    agent: Algorithm,
    action_type: str,
    action_space,
    strength: float | None = None,
) -> BehaviorPolicy:
    """Build an opt-in collection policy. ``strength`` maps to the policy's
    primary parameter (see ``_STRENGTH_PARAM``); ``None`` keeps the default."""
    if name == "agent":
        return AgentBehaviorPolicy(agent)
    kw = {} if strength is None else {_STRENGTH_PARAM[name]: float(strength)}
    if name == "anti_reward":
        return AntiRewardBehaviorPolicy(agent, action_type, action_space, **kw)
    if name == "bias_skew":
        return SkewBehaviorPolicy(agent, action_type, action_space, **kw)
    if name == "bias_suboptimal":
        return SuboptimalBehaviorPolicy(agent, action_type, action_space, **kw)
    raise ValueError(
        f"Unknown behavior policy '{name}'. Choose from: agent, "
        "anti_reward, bias_skew, bias_suboptimal."
    )

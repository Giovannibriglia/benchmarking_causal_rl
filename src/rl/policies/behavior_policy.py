from __future__ import annotations

import abc
import math

import torch
import torch.nn.functional as F

from src.rl.base import ActionOutput, Algorithm
from src.rl.models.transition_model import TransitionModel


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

    @classmethod
    def affects_on_policy(cls) -> bool:
        """Whether this behavior policy's mechanism touches on-policy algorithms.

        Behavior policies in this codebase have two possible mechanisms:
        - Action-bias only: the policy is consulted in build_collection_policy
          to produce biased actions for off-policy collection. PPO and other
          on-policy algorithms ignore this entirely. Examples: curiosity,
          anti_reward, bias_skew, bias_suboptimal.
        - Reward perturbation (also): the policy adds a wrapper to train_env
          that modifies the reward stream for all algorithms training on that
          env, including on-policy ones. Example: bias_confounded (via
          ConfoundedCollectionWrapper).

        When False (default), listing on-policy algorithms in a YAML with this
        behavior_policy is structurally redundant — those algorithms see no
        effect of the behavior, so their results are equivalent to the same
        algos run with behavior_policy="agent". The runner rejects such YAMLs
        at load time with a clear error message; remove the on-policy algos
        or change the behavior_policy to "agent".

        When True, on-policy algorithms genuinely respond to the behavior's
        mechanism and may be included as a structural control.
        """
        return False


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
    """Critic-pessimal collection: with probability ``strength`` take the action
    the agent's OWN critic values least (argmin-Q); otherwise take the agent's
    action. ``strength=0`` -> pure agent (the unconfounded baseline);
    ``strength=1`` -> fully adversarial. This dial convention matches
    ``CuriosityBehaviorPolicy`` and ``ConfoundedBehaviorPolicy`` (0 = off,
    1 = fully active), so the three behaviors share a single strength grid.

    The adversarial (argmin-Q) branch, per-algorithm, with no second learner:

      * discrete (DQN): argmin_a Q(s, a), inside the SAME epsilon-greedy
        structure as ``DQN.act`` (``epsilon`` is the within-branch random-action
        rate; the ``torch.rand`` draw is kept unconditional so a future agent=
        run alongside is RNG-consistent).
      * continuous (DDPG/SAC): sample ``n_candidates`` actions from the agent's
        own ``act`` and return the one the critic scores LOWEST. SAC's twin-Q is
        evaluated on the de-scaled action; DDPG's single critic on the action
        as stored.

    Note ``strength`` (the agent-vs-adversarial mixture dial, mapped from the
    config's ``behavior_strength``) is distinct from ``epsilon`` (the random
    exploration rate WITHIN the adversarial branch, a fixed secondary knob).
    """

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        *,
        strength: float = 1.0,
        epsilon: float = 0.1,
        n_candidates: int = 10,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.strength = float(strength)
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

    def _adversarial(self, obs: torch.Tensor) -> torch.Tensor:
        """The critic-pessimal (argmin-Q) action per env."""
        if self.action_type == "discrete":
            # Mirror DQN.act's epsilon-greedy, but argmin instead of argmax.
            if torch.rand(1).item() < self.epsilon:
                batch = obs.shape[0]
                return torch.randint(
                    0,
                    self.agent.q_network(obs).shape[1],
                    (batch,),
                    device=obs.device,
                )
            with torch.no_grad():
                q = self.agent.q_network(obs)
                return torch.argmin(q, dim=1)
        # Continuous: sample candidates from the agent, return the worst by Q.
        with torch.no_grad():
            candidates = torch.stack(
                [self.agent.act(obs).action for _ in range(self.n_candidates)], dim=0
            )
            return self._argmin_over_candidates(obs, candidates)

    def act(self, obs: torch.Tensor) -> ActionOutput:
        agent_a = self.agent.act(obs).action
        adversarial = self._adversarial(obs)
        take = torch.rand(obs.shape[0], device=obs.device) < self.strength
        if self.action_type == "discrete":
            return ActionOutput(action=torch.where(take, adversarial, agent_a))
        return ActionOutput(
            action=torch.where(take.unsqueeze(-1), adversarial, agent_a)
        )


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


class CuriosityBehaviorPolicy(BehaviorPolicy):
    """Curiosity-driven collection: steer toward NOVEL transitions via ensemble
    disagreement (the Disagreement / Plan2Explore route).

    A single point-predictor cannot rank actions before observing s' (the
    novelty IS the prediction error, which needs the true next state), so this
    uses a small ENSEMBLE of forward models: the variance across members'
    predictions for ``(s, a_candidate)`` is an epistemic-novelty score
    computable BEFORE acting. N self-supervised dynamics models — no
    intrinsic-reward value/policy head (no second policy learner).

    Online-trains its own ensemble on throttled ``agent.buffer`` samples inside
    ``act`` — reusing the buffer the collection loop already fills, so no
    collection-loop / seam change. The buffer-sampling-for-training draws from
    an ISOLATED generator (``self._gen``), so the agent's own update-sampling
    stream (``random.sample`` in ReplayBuffer) is NOT coupled to how often the
    ensemble trains — curiosity's effect is attributable to the data it
    collects, not an RNG side-channel. The selection draws (continuous sample-K,
    the mixture coin) stay in the main stream: those genuinely are the behavior.

    ``strength`` is the curiosity intensity: with probability ``strength`` emit
    the max-disagreement candidate, else the agent's action (1 = pure curiosity,
    0 = agent). Vector obs only; image-obs (feature-space ICM) is deferred.
    """

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        *,
        strength: float = 0.5,
        n_models: int = 5,
        n_candidates: int = 10,
        lr: float = 1e-3,
        train_every: int = 4,
        train_batch: int = 128,
        min_buffer: int = 256,
        seed: int = 0,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.strength = float(strength)
        self.n_models = int(n_models)
        self.n_candidates = int(n_candidates)
        self.lr = float(lr)
        self.train_every = max(1, int(train_every))
        self.train_batch = int(train_batch)
        self.min_buffer = int(min_buffer)
        self.models: list[TransitionModel] | None = None
        self.opts: list = []
        self.action_dim = 0
        self._step = 0
        # Isolated stream for the training-buffer sampling ONLY (see class doc).
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(int(seed))

    def _build(self, obs: torch.Tensor) -> None:
        """Lazily construct the ensemble from the first ``act`` obs — so the
        factory signature stays unchanged (no obs_dim plumbed through the runner
        call => no snapshot line-shift)."""
        device = obs.device
        obs_shape = tuple(obs.shape[1:]) or (1,)
        obs_dim = int(math.prod(obs_shape))
        self.action_dim = (
            int(self.action_space.n)
            if self.action_type == "discrete"
            else int(self.action_space.shape[0])
        )
        self.models = [
            TransitionModel(obs_dim, self.action_dim, self.action_type, obs_shape).to(
                device
            )
            for _ in range(self.n_models)
        ]
        self.opts = [torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models]

    def _disagreement(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-env epistemic novelty = variance across the ensemble's predicted
        next-states for ``(obs, actions)`` (shape ``[B]``)."""
        preds = torch.stack([m(obs, actions) for m in self.models], dim=0)
        return preds.var(dim=0, unbiased=False).mean(dim=-1)

    def _maybe_train(self) -> None:
        buf = getattr(self.agent, "buffer", None)
        if buf is None or len(buf) < self.min_buffer or self._step % self.train_every:
            return
        storage = list(buf.storage)
        idx = torch.randint(
            0, len(storage), (self.train_batch,), generator=self._gen
        ).tolist()
        device = next(self.models[0].parameters()).device
        obs = torch.stack([storage[i]["obs"] for i in idx]).to(device).float()
        actions = torch.stack([storage[i]["actions"] for i in idx]).to(device)
        next_obs = torch.stack([storage[i]["next_obs"] for i in idx]).to(device).float()
        for m, opt in zip(self.models, self.opts):
            loss = F.mse_loss(m(obs, actions), next_obs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def _most_novel(self, obs: torch.Tensor) -> torch.Tensor:
        """The max-disagreement candidate action per env."""
        b = obs.shape[0]
        if self.action_type == "discrete":
            dis = torch.stack(
                [
                    self._disagreement(
                        obs, torch.full((b,), a, device=obs.device, dtype=torch.long)
                    )
                    for a in range(self.action_dim)
                ],
                dim=0,
            )  # [n_actions, B]
            return dis.argmax(dim=0)
        # Continuous: sample-K candidates from the agent (main-stream behavior).
        cands = torch.stack(
            [self.agent.act(obs).action for _ in range(self.n_candidates)], dim=0
        )  # [K, B, A]
        dis = torch.stack(
            [self._disagreement(obs, cands[k]) for k in range(cands.shape[0])], dim=0
        )  # [K, B]
        idx = dis.argmax(dim=0)
        return cands[idx, torch.arange(b, device=obs.device)]

    def act(self, obs: torch.Tensor) -> ActionOutput:
        if self.models is None:
            self._build(obs)
        self._maybe_train()
        self._step += 1
        agent_a = self.agent.act(obs).action
        with torch.no_grad():
            novel = self._most_novel(obs)
        take = torch.rand(obs.shape[0], device=obs.device) < self.strength
        if self.action_type == "discrete":
            return ActionOutput(action=torch.where(take, novel, agent_a))
        return ActionOutput(action=torch.where(take.unsqueeze(-1), novel, agent_a))


class ConfoundedBehaviorPolicy(BehaviorPolicy):
    """Confounded collection: bias the action by the env wrapper's per-episode
    latent ``U`` (read from ``env.current_u``), which ALSO perturbs the reward —
    so action and reward share an unobserved common cause. Pairs with
    ``ConfoundedCollectionWrapper`` on the train env.

      * continuous: ``agent_action + c_a * U``.
      * discrete: with prob ``min(c_a, 1)`` take the U-indexed preferred action
        (``round(U)`` clamped to the action set), else the agent's action.

    ``c_a = base_c_a * strength`` (the single confounding-strength dial scales
    both this and the wrapper's ``c_r``). U sampling lives in the wrapper (main
    RNG stream); the per-env mixture coin here is the behavior.
    """

    @classmethod
    def affects_on_policy(cls) -> bool:
        # bias_confounded also wraps train_env with ConfoundedCollectionWrapper,
        # which perturbs the reward stream for all algos (the wrapper runs
        # regardless of which collection branch is taken), so on-policy
        # algorithms genuinely respond to it.
        return True

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        env,
        *,
        strength: float = 1.0,
        base_c_a: float = 1.0,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.env = env  # the ConfoundedCollectionWrapper (exposes current_u)
        self.c_a = float(base_c_a) * float(strength)

    def act(self, obs: torch.Tensor) -> ActionOutput:
        agent_a = self.agent.act(obs).action
        u = self.env.current_u.to(obs.device)
        if self.action_type == "discrete":
            n = int(self.action_space.n)
            pref = u.round().long().clamp(0, n - 1)
            take = torch.rand(obs.shape[0], device=obs.device) < min(self.c_a, 1.0)
            return ActionOutput(action=torch.where(take, pref, agent_a))
        return ActionOutput(action=agent_a + self.c_a * u.unsqueeze(-1))


_CONF_EPS = 1e-8
# Default FIXED exploration for pi_basic on discrete/DQN bases. NOT 1.0 (that would
# make the 2-action base UNIFORM RANDOM = the `random` coverage tier, not `basic`):
# 0.5 keeps a real preference (argmax gets 0.75, the other 0.25) while holding p away
# from 0 so corr(1[a=a_bad], U) = sigma*sqrt(p(1-p)) stays visible.
PI_BASIC_EPSILON_DEFAULT = 0.5
# Default continuous confounding scale. NOT 1.0 (delta = min(a0-lo, hi-a0) would push
# every mid-range action exactly onto a bound at sigma=1, destroying the policy): 0.2
# is a small, reported perturbation.
CONF_BASE_SCALE_DEFAULT = 0.2


def _pi_basic_probs(
    agent: Algorithm, obs: torch.Tensor, action_space, epsilon: float
) -> torch.Tensor:
    """``pi_basic(.|s)`` as a normalized ``[B, A]`` categorical at a FIXED ``epsilon``
    (NEVER the learner's decaying epsilon). Resolution order: an explicit
    ``action_probs`` seam (tests + clean integration) or an on-policy ``distribution``
    head are used AS-IS (they already define pi_basic); a DQN ``q_network`` is softened
    to eps-greedy at the fixed ``epsilon``. An opaque agent RAISES — a one-hot base
    would pin ``p in {0,1}`` and silently kill the confounding.

    This is the SINGLE source of pi_basic shared by ``PiBasicBehaviorPolicy`` (the
    basic arm) and ``MarginallyMatchedConfoundedBehaviorPolicy`` (the confounded arm),
    so the (beta=0, sigma=0) origin is one identical policy across subcells."""
    fn = getattr(agent, "action_probs", None)
    if callable(fn):
        probs = fn(obs)
    elif hasattr(agent, "distribution"):
        probs = agent.distribution(obs).probs
    elif hasattr(agent, "q_network"):
        with torch.no_grad():
            q = agent.q_network(obs)
        if isinstance(q, tuple):
            q = q[0]
        n = q.shape[-1]
        eps = min(max(float(epsilon), 0.0), 1.0)
        probs = torch.full_like(q, eps / n)  # eps-greedy at the FIXED epsilon
        rows = torch.arange(q.shape[0], device=q.device)
        probs[rows, q.argmax(dim=-1)] += 1.0 - eps
    else:
        raise ValueError(
            "pi_basic needs the base policy's action distribution (an `action_probs` "
            "method, an on-policy `distribution` head, or a DQN `q_network`); a one-hot "
            "fallback would pin p in {0,1} and silently kill the confounding signal."
        )
    probs = probs.float().clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(_CONF_EPS)


class PiBasicBehaviorPolicy(BehaviorPolicy):
    """The shared base policy ``pi_basic`` — a FIXED policy (fixed epsilon, NOT the
    learner's decaying epsilon) that is the common ORIGIN of all three subcells:

      * ``basic``      collects with ``pi_basic`` directly (this policy);
      * ``biased``     applies ``bias_skew`` ON TOP of ``pi_basic``;
      * ``confounded`` applies the U-swap ON TOP of ``pi_basic``.

    Because the basic arm and the confounded arm read the SAME ``pi_basic`` (same
    fixed epsilon), the ``(beta=0, sigma=0)`` point of both is one identical policy —
    the shared origin the orthogonality claim rests on. Online, the fixed epsilon also
    keeps the basic behavior policy STATIONARY (the learner's epsilon anneals; this one
    does not)."""

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        *,
        epsilon: float = PI_BASIC_EPSILON_DEFAULT,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.epsilon = min(max(float(epsilon), 0.0), 1.0)

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        return _pi_basic_probs(self.agent, obs, self.action_space, self.epsilon)

    def act(self, obs: torch.Tensor) -> ActionOutput:
        if self.action_type == "discrete":
            probs = self.action_probs(obs)
            return ActionOutput(action=torch.multinomial(probs, 1).reshape(-1))
        # continuous: pi_basic is the fixed base policy's own (continuous) action.
        return ActionOutput(action=self.agent.act(obs).action)


class MarginallyMatchedConfoundedBehaviorPolicy(BehaviorPolicy):
    """Action-dependent confounded collection = the U-swap applied ON TOP of the shared
    ``pi_basic`` (see ``PiBasicBehaviorPolicy``), EXACTLY marginally-matched:
    ``E_U[pi_b(a|s,U)] == pi_basic(a|s)`` for all ``s``, ``p``, ``sigma``
    (``U ~ Bernoulli(0.5)``). Pairs with ``ConfoundedCollectionWrapper`` in
    ``action_gated`` mode (``bias_confounded_action``). Replaces
    ``ConfoundedBehaviorPolicy`` for the action-dependent arm only; the additive cells
    7/8 keep the byte-frozen ``ConfoundedBehaviorPolicy`` untouched.

    Construction (discrete, per obs, with ``p = pi_basic(a_bad|s)``,
    ``g = pi_basic(a_good|s)``, ``pbar = p/(p+g)``) — the MIXTURE form of the swap:

      * draw a confounder coin ``c ~ Bernoulli(sigma)``;
      * ``c == 0`` (prob ``1-sigma``): keep ``a0 ~ pi_basic`` — the learner's own
        draw (a genuine ``do(a)`` online);
      * ``c == 1`` (prob ``sigma``): if ``a0 in {a_good, a_bad}`` redraw WITHIN the
        pair with ``P(a_bad | U=1) = pbar*(2-pbar)``, ``P(a_bad | U=0) = pbar^2``
        (other actions pass through). Both lie in ``[0,1]`` by construction — NO
        clipping. Averaged over ``U`` the pair split is ``pbar`` again, so the
        marginal is ``pi_basic`` EXACTLY, and at ``sigma=0`` the policy IS ``pi_basic``.

    ``sigma`` (``strength``) scales the U->A edge ONLY; the U->R reward shift lives in
    the wrapper's fixed ``c_r``. ``pi_basic_epsilon`` is the SHARED fixed exploration
    (default ``0.5``): NOT the learner's decaying epsilon (which would desync the
    origin and shrink the signal online) and NOT uniform (which would be the random
    tier). The U->A gap is ``2*sigma*p*g/(p+g)`` (``= 2*sigma*p*(1-p)`` binary): it
    VANISHES where ``pi_basic`` is deterministic — an action always/never taken cannot
    be confounded without moving the marginal.

    Continuous: a BOUNDED mean-preserving reflection reading ``[lo, hi]`` from the
    action space. ``delta(a0) = base_scale * min(a0-lo, hi-a0)``; ``a = a0`` with prob
    ``1-sigma`` (kept), else ``a = a0 + (2U-1)*delta(a0)``. Both branches lie in
    ``[lo, hi]`` BY CONSTRUCTION (``base_scale in [0,1]``), so the env's clip is a
    no-op and the marginal MEAN is preserved. ``base_scale`` defaults to ``0.2`` (1.0
    would push mid-range actions onto a bound at ``sigma=1``). NOTE: the reflection
    preserves the MEAN, not the full distribution — under a nonlinear reward a
    mean-preserving spread costs return, so the c_r=0 return-equivalence gate
    (orthogonality) holds for the discrete swap but NOT for this continuous
    construction; a distribution-preserving continuous confounder is future work.

    ``intervened`` (per element): ``is_online AND (c == 0)`` — True only where the
    learner's own action executed under an online regime. Offline it is False
    everywhere; hence ``mean(intervened) == 0`` offline, ``~= 1 - sigma`` online.
    """

    @classmethod
    def affects_on_policy(cls) -> bool:
        # Also wraps train_env with ConfoundedCollectionWrapper (reward shift),
        # so on-policy algorithms genuinely respond — same rationale as
        # ConfoundedBehaviorPolicy.
        return True

    def __init__(
        self,
        agent: Algorithm,
        action_type: str,
        action_space,
        env,
        *,
        strength: float = 1.0,
        a_bad: int = 1,
        a_good: int = 0,
        base_scale: float = CONF_BASE_SCALE_DEFAULT,
        pi_basic_epsilon: float = PI_BASIC_EPSILON_DEFAULT,
        is_online: bool = False,
    ) -> None:
        self.agent = agent
        self.action_type = action_type
        self.action_space = action_space
        self.env = env  # the ConfoundedCollectionWrapper (exposes current_u)
        self.strength = float(strength)
        self.a_bad = int(a_bad)
        self.a_good = int(a_good)
        # base_scale in [0,1] so the bounded reflection stays within [lo,hi].
        self.base_scale = min(max(float(base_scale), 0.0), 1.0)
        # SHARED fixed exploration defining pi_basic (same object as the basic arm's).
        self.pi_basic_epsilon = min(max(float(pi_basic_epsilon), 0.0), 1.0)
        self.is_online = bool(is_online)

    def _base_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """``pi_basic(.|s)`` via the SHARED helper at the fixed ``pi_basic_epsilon`` —
        identical to what ``PiBasicBehaviorPolicy`` samples for the basic arm."""
        return _pi_basic_probs(
            self.agent, obs, self.action_space, self.pi_basic_epsilon
        )

    def _action_bounds(self, ref: torch.Tensor):
        """``(lo, hi)`` from the continuous action space, broadcastable to ``ref``."""
        space = self.action_space
        lo = torch.as_tensor(space.low, dtype=ref.dtype, device=ref.device)
        hi = torch.as_tensor(space.high, dtype=ref.dtype, device=ref.device)
        return lo, hi

    def _intervened(self, coin_fired: torch.Tensor) -> torch.Tensor:
        if self.is_online:
            return ~coin_fired
        return torch.zeros_like(coin_fired)

    def act(self, obs: torch.Tensor) -> ActionOutput:
        b = obs.shape[0]
        u = self.env.current_u.to(obs.device).reshape(-1)
        sigma = min(max(self.strength, 0.0), 1.0)
        coin = torch.rand(b, device=obs.device) < sigma  # confounder acts w.p. sigma

        if self.action_type == "discrete":
            probs = self._base_action_probs(obs)  # == pi_basic (shared origin)
            a0 = torch.multinomial(probs, 1).reshape(-1)  # a0 ~ pi_basic
            p = probs[:, self.a_bad]
            g = probs[:, self.a_good]
            pbar = p / (p + g).clamp_min(_CONF_EPS)  # within-pair P(a_bad)
            # confounded within-pair P(a_bad): pbar(2-pbar) if U==1 else pbar^2.
            w = torch.where(u > 0.5, pbar * (2.0 - pbar), pbar * pbar)
            in_pair = (a0 == self.a_bad) | (a0 == self.a_good)
            redraw = coin & in_pair
            pair_action = torch.where(
                torch.rand(b, device=obs.device) < w,
                torch.full_like(a0, self.a_bad),
                torch.full_like(a0, self.a_good),
            )
            action = torch.where(redraw, pair_action, a0)
            return ActionOutput(action=action, intervened=self._intervened(coin))

        # continuous: BOUNDED mean-preserving reflection. delta shrinks to 0 at the
        # bounds so a0 +/- delta stays in [lo, hi] -> the env clip is a no-op and the
        # post-step marginal mean is a0 for every sigma (the +/- cancel under U).
        a0 = self.agent.act(obs).action
        lo, hi = self._action_bounds(a0)
        delta = self.base_scale * torch.minimum(a0 - lo, hi - a0).clamp_min(0.0)
        sign = (2.0 * u - 1.0).unsqueeze(-1)
        perturbed = a0 + sign * delta
        action = torch.where(coin.unsqueeze(-1), perturbed, a0)
        return ActionOutput(action=action, intervened=self._intervened(coin))


# Collection-policy registry. The "agent" entry is handled by the runner with
# the original AgentBehaviorPolicy(self.agent) construction (byte-identical to
# the pre-A1 path), so the off-policy golden is untouched; this factory only
# builds the opt-in policies.
_STRENGTH_PARAM = {
    "anti_reward": "strength",  # P(take argmin-Q action) vs the agent's action
    "bias_skew": "p",  # probability of the preferred action
    "bias_suboptimal": "beta",  # probability of using the agent (vs uniform base)
    "curiosity": "strength",  # probability of emitting the max-disagreement action
    "bias_confounded": "strength",  # confounding-strength dial (scales c_a, c_r)
    "bias_confounded_action": "strength",  # action-dependent confounder (same dial)
}

# name -> BehaviorPolicy subclass, for class-level introspection (e.g.
# affects_on_policy()) without constructing a policy. Mirrors the names accepted
# by build_collection_policy.
_BEHAVIOR_POLICY_CLASSES: dict[str, type[BehaviorPolicy]] = {
    "agent": AgentBehaviorPolicy,
    # The shared fixed-epsilon base policy pi_basic — the basic subcell's collector
    # and the common origin the biased / confounded arms transform on top of.
    "pi_basic": PiBasicBehaviorPolicy,
    "anti_reward": AntiRewardBehaviorPolicy,
    "bias_skew": SkewBehaviorPolicy,
    "bias_suboptimal": SuboptimalBehaviorPolicy,
    "curiosity": CuriosityBehaviorPolicy,
    "bias_confounded": ConfoundedBehaviorPolicy,
    # Action-dependent confounder: the marginally-matched swap policy (sigma scales
    # the U->A edge only, E_U[pi_b] == pi_basic exactly). Distinct class from the
    # additive bias_confounded above, which stays byte-frozen (cells 7/8).
    "bias_confounded_action": MarginallyMatchedConfoundedBehaviorPolicy,
}


def behavior_policy_class(name: str) -> type[BehaviorPolicy] | None:
    """The BehaviorPolicy subclass for a behavior_policy name, or None if unknown
    (unknown names are left for build_collection_policy / other validation)."""
    return _BEHAVIOR_POLICY_CLASSES.get(name)


def build_collection_policy(
    name: str,
    agent: Algorithm,
    action_type: str,
    action_space,
    strength: float | None = None,
    env=None,
    is_online: bool = False,
    pi_basic_epsilon: float | None = None,
) -> BehaviorPolicy:
    """Build an opt-in collection policy. ``strength`` maps to the policy's
    primary parameter (see ``_STRENGTH_PARAM``); ``None`` keeps the default.
    ``env`` is the (confounded) train-env handle, used only by the confounded
    policies to read ``current_u`` (default ``None``, so the existing policies
    build unchanged). ``is_online`` distinguishes the online collection loop
    (learner-controlled = ``do(a)``) from the offline generation loop (fixed
    exogenous logger); it is consumed ONLY by the marginally-matched confounded
    policy's ``intervened`` flag. The runner (online) passes ``True``; the offline
    generator passes ``False`` (default), so the two regimes — which share the
    ``build_collection_policy`` seam — are cleanly distinguished at construction.
    ``pi_basic_epsilon`` is the SHARED fixed exploration defining ``pi_basic``, read
    identically by the ``pi_basic`` (basic) and ``bias_confounded_action`` (confounded)
    arms so their ``(beta=0, sigma=0)`` origin is one identical policy."""
    if name == "agent":
        return AgentBehaviorPolicy(agent)
    if name == "pi_basic":
        eps = {} if pi_basic_epsilon is None else {"epsilon": float(pi_basic_epsilon)}
        return PiBasicBehaviorPolicy(agent, action_type, action_space, **eps)
    kw = {} if strength is None else {_STRENGTH_PARAM[name]: float(strength)}
    if name == "anti_reward":
        return AntiRewardBehaviorPolicy(agent, action_type, action_space, **kw)
    if name == "bias_skew":
        return SkewBehaviorPolicy(agent, action_type, action_space, **kw)
    if name == "bias_suboptimal":
        return SuboptimalBehaviorPolicy(agent, action_type, action_space, **kw)
    if name == "curiosity":
        return CuriosityBehaviorPolicy(agent, action_type, action_space, **kw)
    if name == "bias_confounded":
        # Additive confounder (cells 7/8) — byte-frozen; U-indexed action mixture.
        return ConfoundedBehaviorPolicy(agent, action_type, action_space, env, **kw)
    if name == "bias_confounded_action":
        # Action-dependent confounder — the U-swap on top of the SHARED pi_basic
        # (same fixed pi_basic_epsilon as the basic arm, None -> the policy default).
        eps_kw = (
            {}
            if pi_basic_epsilon is None
            else {"pi_basic_epsilon": float(pi_basic_epsilon)}
        )
        return MarginallyMatchedConfoundedBehaviorPolicy(
            agent, action_type, action_space, env, is_online=is_online, **kw, **eps_kw
        )
    raise ValueError(
        f"Unknown behavior policy '{name}'. Choose from: agent, pi_basic, anti_reward, "
        "bias_skew, bias_suboptimal, curiosity, bias_confounded, "
        "bias_confounded_action."
    )

"""PR 1 — marginally-matched action-dependent confounder (behavior-policy core).

Locks the three fixed definitions the old confounded policy violated:
  T1  marginal matching: E_U[pi_b(.|s,U)] == pi_basic(.|s) EXACTLY (incl. p=0.9,
      the case an additive-shift-with-clamp implementation fails);
  T2  U->A edge is live and monotone in sigma (0 at sigma=0);
  T3  c_r (U->R) is decoupled from sigma -> reward shift on a_bad|U=1 invariant;
  T4  intervened == (online AND agent's own action) -> mean ~= 1-sigma online, 0 offline.
Plus T5: the continuous mean-preserving construction.

All reference ``MarginallyMatchedConfoundedBehaviorPolicy`` / the decoupled
``build_rollout_env(c_r=...)`` seam, both new in this PR, so the module fails on master.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium.spaces import Discrete
from src.envs.offline.generate import build_rollout_env
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper
from src.rl.base import ActionOutput
from src.rl.policies.behavior_policy import (
    MarginallyMatchedConfoundedBehaviorPolicy,
    PiBasicBehaviorPolicy,
)

CPU = torch.device("cpu")
A_GOOD, A_BAD = 0, 1
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N = 40000  # MC batch; error ~ sqrt(.25/N) ~ 0.0025 << the 0.02 tolerances


class _ProbAgent:
    """Discrete agent exposing a fixed pi_basic(a_bad|s) = p over two actions."""

    def __init__(self, p: float) -> None:
        self.p = float(p)

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        return torch.tensor([[1.0 - self.p, self.p]]).repeat(b, 1)


class _MeanAgent:
    """Continuous agent emitting a fixed action (the pi_basic mean)."""

    def __init__(self, mean: float, dim: int = 2) -> None:
        self.mean = float(mean)
        self.dim = dim

    def act(self, obs: torch.Tensor, *a, **k) -> ActionOutput:
        return ActionOutput(action=torch.full((obs.shape[0], self.dim), self.mean))


class _UEnv:
    """Minimal confounder-env stand-in: exposes ``current_u`` (the test resamples
    it to Bernoulli(0.5) each round so the marginalization is over U)."""

    def __init__(self, n: int) -> None:
        self.n_envs = n
        self.device = CPU
        self.current_u = torch.zeros(n)

    def draw_u(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.current_u = torch.bernoulli(torch.full((self.n_envs,), 0.5), generator=g)


def _policy(p: float, sigma: float, env, *, is_online: bool):
    # A mock exposing action_probs IS pi_basic directly (pi_basic_epsilon is ignored
    # for the action_probs / distribution seams), so the marginal / corr assertions can
    # target a known p.
    return MarginallyMatchedConfoundedBehaviorPolicy(
        _ProbAgent(p),
        "discrete",
        Discrete(2),
        env,
        strength=sigma,
        a_bad=A_BAD,
        a_good=A_GOOD,
        is_online=is_online,
    )


# ---------------------------------------------------------------------------
# T1 — marginal matching (EXACT, no clipping), including p=0.9.
# ---------------------------------------------------------------------------
def test_t1_marginal_matches_pi_basic_across_sigma_and_p():
    obs = torch.zeros(N, 2)
    for p in (0.1, 0.5, 0.9):
        for sigma in SIGMAS:
            torch.manual_seed(0)
            env = _UEnv(N)
            env.draw_u(seed=1)  # U ~ Bernoulli(0.5), iid across the N envs
            pol = _policy(p, sigma, env, is_online=False)
            a = pol.act(obs).action
            p_bad = (a == A_BAD).float().mean().item()
            tv = abs(p_bad - p)  # binary TV == |P(a_bad) - p|
            assert tv < 0.02, f"p={p} sigma={sigma}: TV={tv:.4f} (marginal drifted)"


# ---------------------------------------------------------------------------
# T2 — U->A edge: ~0 at sigma=0, monotone increasing in sigma.
# ---------------------------------------------------------------------------
def _corr_abad_u(p: float, sigma: float) -> float:
    torch.manual_seed(0)
    env = _UEnv(N)
    env.draw_u(seed=2)
    pol = _policy(p, sigma, env, is_online=False)
    a = pol.act(torch.zeros(N, 2)).action
    ab = (a == A_BAD).float().numpy()
    u = env.current_u.numpy()
    if ab.std() < 1e-8 or u.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(ab, u)[0, 1])


def test_t2_u_to_a_edge_live_and_monotone():
    corrs = [_corr_abad_u(0.5, s) for s in SIGMAS]
    assert abs(corrs[0]) < 0.02, f"sigma=0 should have no U->A edge, got {corrs[0]:.4f}"
    for lo, hi in zip(corrs, corrs[1:]):
        assert hi > lo - 1e-3, f"non-monotone U->A edge across sigma: {corrs}"
    assert corrs[-1] > 0.2, f"sigma=1 U->A edge too weak: {corrs[-1]:.4f}"


# ---------------------------------------------------------------------------
# T3 — c_r (U->R) decoupled from sigma: reward shift on a_bad|U=1 invariant.
# ---------------------------------------------------------------------------
class _ZeroRewardEnv:
    """Clean reward 0 -> the whole reward IS the confounder's shift."""

    def __init__(self, n: int = 4) -> None:
        self.n_envs = n
        self.device = CPU

    def reset(self, seed=None):
        return torch.zeros(self.n_envs, 2), {}

    def step(self, action):
        z = torch.zeros(self.n_envs, dtype=torch.bool)
        return torch.zeros(self.n_envs, 2), torch.zeros(self.n_envs), z, z, {}


def test_t3_c_r_invariant_across_sigma():
    # Direct wrapper: fixed c_r -> shift on a_bad|U=1 is c_r for every sigma.
    shifts = []
    for sigma in SIGMAS:
        w = ConfoundedCollectionWrapper(
            _ZeroRewardEnv(),
            c_a=sigma,
            c_r=1.0,
            confounder_kind="action_gated",
            a_bad=A_BAD,
        )
        w.current_u = torch.ones(4)
        _, reward, *_ = w.step(torch.full((4,), A_BAD))
        shifts.append(float(reward[0].item()))
    assert all(abs(s - 1.0) < 1e-9 for s in shifts), f"c_r drifted with sigma: {shifts}"

    # Construction seam: build_rollout_env keeps c_r fixed while sigma sweeps
    # (action_gated), and STILL couples c_r=sigma for the byte-frozen additive path.
    from src.envs.registry import register_default_env_wrappers

    register_default_env_wrappers()
    for sigma in SIGMAS:
        env_ag = build_rollout_env(
            "CartPole-v1", 1, CPU, 0, "bias_confounded_action", sigma, c_r=1.0
        )
        assert abs(float(env_ag.c_r) - 1.0) < 1e-9, (sigma, env_ag.c_r)
        env_add = build_rollout_env("CartPole-v1", 1, CPU, 0, "bias_confounded", sigma)
        assert abs(float(env_add.c_r) - sigma) < 1e-9, "additive c_r must stay = sigma"


# ---------------------------------------------------------------------------
# T4 — interventional fraction: online mean ~= 1-sigma, offline == 0.
# ---------------------------------------------------------------------------
def test_t4_intervened_fraction():
    obs = torch.zeros(N, 2)
    for sigma in SIGMAS:
        torch.manual_seed(0)
        env = _UEnv(N)
        env.draw_u(seed=3)
        online = _policy(0.5, sigma, env, is_online=True).act(obs)
        assert online.intervened is not None and online.intervened.dtype == torch.bool
        mean_iv = online.intervened.float().mean().item()
        assert abs(mean_iv - (1.0 - sigma)) < 0.02, (sigma, mean_iv)

        offline = _policy(0.5, sigma, env, is_online=False).act(obs)
        assert offline.intervened is not None
        assert offline.intervened.float().mean().item() == 0.0, sigma


# ---------------------------------------------------------------------------
# T5 — continuous BOUNDED reflection: mean preserved AFTER env clipping, a0 near
# the bound, on a REAL bounded env (Pendulum). This asserts the post-step
# quantity (clip(action) = what the env applies), the thing that broke under the
# old unbounded `a0 + base_scale*(2U-1)`.
# ---------------------------------------------------------------------------
def _continuous_pol(agent, action_space, env, sigma, *, is_online=True):
    return MarginallyMatchedConfoundedBehaviorPolicy(
        agent,
        "continuous",
        action_space,
        env,
        strength=sigma,
        base_scale=1.0,
        is_online=is_online,
    )


def test_t5_continuous_bounded_reflection_preserves_post_clip_mean():
    import gymnasium as gym

    penv = gym.make("Pendulum-v1")  # real bounded env, action_space = Box([-2],[2])
    space = penv.action_space
    lo = torch.as_tensor(space.low)
    hi = torch.as_tensor(space.high)
    a0_val = 1.9  # NEAR the +2.0 bound: the confident case an env clip would wreck
    obs = torch.zeros(N, 1)
    for sigma in SIGMAS:
        torch.manual_seed(0)
        env = _UEnv(N)
        env.draw_u(seed=4)
        out = _continuous_pol(_MeanAgent(a0_val, dim=1), space, env, sigma).act(obs)
        act = out.action
        # bounded BY CONSTRUCTION -> the env's clip is a no-op -> post-step == emitted.
        assert (act >= lo - 1e-5).all() and (act <= hi + 1e-5).all(), sigma
        realized = torch.clamp(act, lo, hi)  # exactly what Pendulum applies in step()
        assert abs(realized.mean().item() - a0_val) < 0.02, (sigma, realized.mean())
        assert abs(out.intervened.float().mean().item() - (1.0 - sigma)) < 0.02

    # A near-bound action is accepted and applied UNCLIPPED by a real env.step.
    penv.reset(seed=0)
    env1 = _UEnv(1)
    env1.current_u = torch.ones(1)
    a = (
        _continuous_pol(_MeanAgent(a0_val, dim=1), space, env1, 1.0)
        .act(torch.zeros(1, 1))
        .action
    )
    assert (a >= lo).all() and (a <= hi).all()
    penv.step(a.reshape(-1).numpy())
    penv.close()


# ---------------------------------------------------------------------------
# R2 — the confounding-signal magnitude and its entropy pinning.
# ---------------------------------------------------------------------------
def test_r2_corr_matches_sigma_sqrt_p_1_minus_p():
    """corr(1[a==a_bad], U) == sigma*sqrt(p(1-p)) within MC error (the visible signal)."""
    obs = torch.zeros(N, 2)
    for p in (0.1, 0.3, 0.5):
        for sigma in (0.25, 0.5, 1.0):
            torch.manual_seed(0)
            env = _UEnv(N)
            env.draw_u(seed=5)
            a = _policy(p, sigma, env, is_online=False).act(obs).action
            ab = (a == A_BAD).float().numpy()
            u = env.current_u.numpy()
            corr = float(np.corrcoef(ab, u)[0, 1])
            expected = sigma * np.sqrt(p * (1.0 - p))
            assert abs(corr - expected) < 0.015, (p, sigma, corr, expected)


class _FixedDQN:
    """DQN-like base: a_bad is the greedy action; ANNEALED epsilon is tiny. Only
    q_network + the FIXED pi_basic_epsilon should be read, never `epsilon`."""

    epsilon = 0.001  # if inherited, p ~= 0.0005 -> confounding invisible

    def q_network(self, obs):
        return torch.tensor([[0.0, 1.0]]).repeat(obs.shape[0], 1)  # argmax = a_bad


def test_r2_dqn_uses_fixed_pi_basic_epsilon_not_decaying():
    obs = torch.zeros(N, 2)
    torch.manual_seed(0)
    env = _UEnv(N)
    env.draw_u(seed=7)
    # default pi_basic_epsilon = 0.5 => eps-greedy: p(a_bad) = 1-0.5 + 0.5/2 = 0.75.
    pol = MarginallyMatchedConfoundedBehaviorPolicy(
        _FixedDQN(),
        "discrete",
        Discrete(2),
        env,
        strength=1.0,
        a_bad=A_BAD,
        a_good=A_GOOD,
    )
    p_eff = (pol.act(obs).action == A_BAD).float().mean().item()
    # decaying epsilon (0.001) would give p ~= 0.9995; fixed 0.5 gives 0.75.
    assert abs(p_eff - 0.75) < 0.02, p_eff


def test_r2_one_hot_fallback_raises():
    class _Opaque:
        def act(self, obs, *a, **k):
            return ActionOutput(action=torch.zeros(obs.shape[0], dtype=torch.long))

    env = _UEnv(4)
    env.current_u = torch.ones(4)
    pol = MarginallyMatchedConfoundedBehaviorPolicy(
        _Opaque(), "discrete", Discrete(2), env, strength=1.0
    )
    with pytest.raises(ValueError):
        pol.act(torch.zeros(4, 2))


# ---------------------------------------------------------------------------
# B1 — SHARED ORIGIN: the basic arm (pi_basic) and the confounded arm at sigma=0
# read the SAME fixed-epsilon pi_basic, so their action distributions coincide.
# (On the pre-fix code the basic arm used the learner's decaying epsilon while the
# confounded arm pinned the pair to 0.5 -> different policies -> the origin was
# contaminated; PiBasicBehaviorPolicy did not exist, so this errors on master.)
# ---------------------------------------------------------------------------
def test_b1_shared_origin_basic_equals_confounded_sigma0():
    obs = torch.zeros(N, 2)
    agent = _FixedDQN()  # argmax = a_bad
    torch.manual_seed(0)
    a_basic = (
        PiBasicBehaviorPolicy(agent, "discrete", Discrete(2), epsilon=0.5)
        .act(obs)
        .action
    )
    env = _UEnv(N)
    env.draw_u(seed=8)
    a_conf = (
        MarginallyMatchedConfoundedBehaviorPolicy(
            agent,
            "discrete",
            Discrete(2),
            env,
            strength=0.0,  # sigma = 0
            pi_basic_epsilon=0.5,
            a_bad=A_BAD,
            a_good=A_GOOD,
        )
        .act(obs)
        .action
    )
    p_basic = (a_basic == A_BAD).float().mean().item()
    p_conf = (a_conf == A_BAD).float().mean().item()
    assert abs(p_basic - p_conf) < 0.02, (p_basic, p_conf)  # identical within MC error
    assert abs(p_basic - 0.75) < 0.02, p_basic  # eps-greedy(0.5) on argmax=a_bad


# ---------------------------------------------------------------------------
# B2 — the default pi_basic_epsilon (0.5) retains real PREFERENCE (not the uniform
# random tier eps=1.0, not the annealed ~0).
# ---------------------------------------------------------------------------
def test_b2_default_epsilon_retains_preference():
    obs = torch.zeros(N, 2)
    env = _UEnv(N)

    def p_at(eps):
        torch.manual_seed(0)
        env.draw_u(seed=10)
        pol = MarginallyMatchedConfoundedBehaviorPolicy(
            _FixedDQN(),
            "discrete",
            Discrete(2),
            env,
            strength=1.0,
            pi_basic_epsilon=eps,
            a_bad=A_BAD,
            a_good=A_GOOD,
        )
        return (pol.act(obs).action == A_BAD).float().mean().item()

    assert abs(p_at(0.5) - 0.75) < 0.02  # default: real preference (0.75)
    assert abs(p_at(1.0) - 0.5) < 0.02  # eps=1.0 IS the uniform random tier


# ---------------------------------------------------------------------------
# B3 — continuous default base_scale (0.2) does NOT saturate at the bounds at
# sigma=1 (base_scale=1.0 does — every mid-range action lands on a bound).
# ---------------------------------------------------------------------------
def test_b3_continuous_default_scale_not_degenerate_at_bounds():
    import gymnasium as gym

    space = gym.make("Pendulum-v1").action_space  # Box([-2], [2])
    hi = float(space.high[0])
    obs = torch.zeros(N, 1)
    env = _UEnv(N)
    env.draw_u(seed=11)

    # default base_scale=0.2, a0=0 mid-range, sigma=1 -> a in {+/-0.4}, far from +/-2.
    a = (
        MarginallyMatchedConfoundedBehaviorPolicy(
            _MeanAgent(0.0, dim=1), "continuous", space, env, strength=1.0
        )
        .act(obs)
        .action.reshape(-1)
    )
    assert a.abs().max().item() < 0.5, a.abs().max().item()
    assert ((a.abs() >= hi - 1e-3).float().mean().item()) < 0.01  # ~none at the bound

    # base_scale=1.0 SATURATES: delta = min(0-lo, hi-0) = 2 -> every action on +/-2.
    a2 = (
        MarginallyMatchedConfoundedBehaviorPolicy(
            _MeanAgent(0.0, dim=1),
            "continuous",
            space,
            env,
            strength=1.0,
            base_scale=1.0,
        )
        .act(obs)
        .action.reshape(-1)
    )
    assert ((a2.abs() - hi).abs() < 1e-4).float().mean().item() > 0.99


# ---------------------------------------------------------------------------
# B4 — c_r=0 RETURN EQUIVALENCE (the empirical orthogonality gate). With c_r=0, U
# is causally inert on reward, so a confounded ACTION policy that preserves the
# marginal action distribution at every state cannot change return.
# ---------------------------------------------------------------------------
class _AngleQ:
    """State-DEPENDENT CartPole base policy: prefer the action that reduces the pole
    angle (obs[2]). A realistic non-degenerate pi_basic for the return gate."""

    def q_network(self, obs):
        ang = obs[:, 2]
        return torch.stack([-ang, ang], dim=-1)  # action 1 preferred when angle > 0


def _cartpole_returns(kind, sigma, per, n_eps=500, seed=0):
    import gymnasium as gym

    penv = gym.make("CartPole-v1")
    agent = _AngleQ()
    env_u = _UEnv(1)
    if kind == "basic":
        pol = PiBasicBehaviorPolicy(agent, "discrete", Discrete(2), epsilon=0.5)
    else:
        pol = MarginallyMatchedConfoundedBehaviorPolicy(
            agent,
            "discrete",
            Discrete(2),
            env_u,
            strength=sigma,
            pi_basic_epsilon=0.5,
            a_bad=A_BAD,
            a_good=A_GOOD,
        )
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    rets = []
    for ep in range(n_eps):
        obs, _ = penv.reset(seed=seed + ep)
        if kind != "basic" and per == "episode":
            env_u.current_u = torch.bernoulli(torch.tensor([0.5]), generator=g)
        ret = 0.0
        for _ in range(500):
            if kind != "basic" and per == "step":
                env_u.current_u = torch.bernoulli(torch.tensor([0.5]), generator=g)
            o = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            a = int(pol.act(o).action.item())
            obs, r, term, trunc, _ = penv.step(a)
            ret += r
            if term or trunc:
                break
        rets.append(ret)
    penv.close()
    return np.array(rets)


def _diff_sem(basic, conf):
    return abs(conf.mean() - basic.mean()), np.sqrt(basic.var() + conf.var()) / np.sqrt(
        len(basic)
    )


def test_b4_discrete_perstep_marginalization_holds():
    """Under PER-STEP marginalization the swap preserves the per-step action dist ->
    the trajectory (hence return) distribution is EXACTLY pi_basic's. This validates
    the swap MECHANISM (diff ~ MC noise for every sigma)."""
    basic = _cartpole_returns("basic", 0.0, "step")
    for sigma in (0.25, 0.5, 1.0):
        conf = _cartpole_returns("conf", sigma, "step")
        diff, sem = _diff_sem(basic, conf)
        assert diff < 3.0 * sem, (sigma, basic.mean(), conf.mean(), diff, sem)


@pytest.mark.xfail(
    strict=True,
    reason="STOP / DESIGN DECISION: the REAL confounder uses PER-EPISODE U (a stable "
    "per-episode common cause; the additive cells 7/8 are byte-frozen on it). Under "
    "per-episode U the confounded policy is a per-episode MIXTURE of two biased "
    "policies, and return is nonlinear in the policy (Jensen), so E_U[J(pi_U)] != "
    "J(pi_basic): the c_r=0 action confounder BIASES return even in the DISCRETE case "
    "(CartPole: diff ~ -9 sem at sigma=0.5, ~ -25 sem at sigma=1.0). The brief's "
    "'discrete passes by construction' assumes per-STEP marginalization. Resolving "
    "this (per-step U for the action-gated arm, or redefining orthogonality at the "
    "marginal-action-dist level) is a design change pending approval.",
)
def test_b4_discrete_per_episode_confounder_biases_return():
    basic = _cartpole_returns("basic", 0.0, "episode")
    conf = _cartpole_returns("conf", 1.0, "episode")
    diff, sem = _diff_sem(basic, conf)
    assert diff < 3.0 * sem, (basic.mean(), conf.mean(), diff, sem)


class _PendCtrl:
    """Deterministic state-dependent Pendulum controller (a non-trivial pi_basic)."""

    def act(self, obs, *a, **k):
        sin, td = obs[:, 1], obs[:, 2]
        return ActionOutput(
            action=(2.0 * sin - 0.5 * td).clamp(-2.0, 2.0).unsqueeze(-1)
        )


def _pendulum_returns(kind, sigma, base_scale, n_eps=200, seed=0):
    import gymnasium as gym

    penv = gym.make("Pendulum-v1")
    ctrl = _PendCtrl()
    env_u = _UEnv(1)
    pol = (
        None
        if kind == "basic"
        else MarginallyMatchedConfoundedBehaviorPolicy(
            ctrl,
            "continuous",
            penv.action_space,
            env_u,
            strength=sigma,
            base_scale=base_scale,
        )
    )
    g = torch.Generator().manual_seed(seed + 1)
    rets = []
    for ep in range(n_eps):
        obs, _ = penv.reset(seed=seed + ep)
        if pol is not None:
            env_u.current_u = torch.bernoulli(torch.tensor([0.5]), generator=g)
        ret = 0.0
        for _ in range(200):
            o = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            a = (ctrl if pol is None else pol).act(o).action
            obs, r, term, trunc, _ = penv.step(a.reshape(-1).numpy())
            ret += r
            if term or trunc:
                break
        rets.append(ret)
    penv.close()
    return np.array(rets)


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN LIMITATION (STOP): the continuous reflection preserves the MEAN, "
    "not the DISTRIBUTION. Under Pendulum's nonlinear reward a mean-preserving spread "
    "systematically shifts return, so c_r=0 orthogonality FAILS at any meaningful "
    "base_scale (base_scale=1.0: diff ~+85 vs 2sem~15). Shrinking base_scale only "
    "hides it by shrinking the confounder to nothing. Needs a distribution-preserving "
    "construction (median-split / half-distribution reflection in pre-squash space) — "
    "design approval pending. Remove this xfail when that lands.",
)
def test_b4_continuous_return_equivalence_cr0():
    basic = _pendulum_returns("basic", 0.0, 0.0)
    conf = _pendulum_returns("conf", 1.0, base_scale=1.0)
    diff = abs(conf.mean() - basic.mean())
    sem = np.sqrt(basic.var() + conf.var()) / np.sqrt(len(basic))
    assert diff < 2.0 * sem, (basic.mean(), conf.mean(), diff, sem)

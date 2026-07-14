"""PR 1 — marginally-matched action-dependent confounder (behavior-policy core).

The confounder is a binary PARTITION SWAP applied on top of the SHARED pi_basic
(fixed epsilon / noise scale), unified across discrete (cells {a_good},{a_bad}) and
continuous (median split), distribution-preserving in both.

  T1  marginal matching: E_U[pi_b(.|s,U)] == pi_basic(.|s) EXACTLY (incl. p=0.9);
  T2  U->A edge live and monotone in sigma (0 at sigma=0);
  T3  c_r (U->R) decoupled from sigma -> reward shift on a_bad|U=1 invariant;
  T4  intervened == (online AND agent's own action) -> mean ~= 1-sigma online, 0 offline;
  T5  continuous partition swap is distribution-preserving and needs a fixed noise scale;
  B1  shared origin: basic (pi_basic) == confounded at sigma=0;
  B2  default pi_basic_epsilon retains preference (not uniform random);
  B4' per-state action-dist equivalence (isolated from state-visitation via fixed
      states + resample-U): discrete |dP| and continuous KS ~ MC floor; a reflection
      FAILS the continuous KS (the point);
  B4'' coverage comparability: confounding keeps coverage ~ basic across sigma while
      bias degrades it with beta (the empirical orthogonality claim).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium.spaces import Discrete
from src.envs.offline.generate import build_rollout_env
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper
from src.rl.base import ActionOutput
from src.rl.off_policy.identification import Proximal
from src.rl.policies.behavior_policy import (
    MarginallyMatchedConfoundedBehaviorPolicy,
    PiBasicBehaviorPolicy,
    SkewBehaviorPolicy,
)

CPU = torch.device("cpu")
A_GOOD, A_BAD = 0, 1
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N = 40000  # MC batch; error ~ sqrt(.25/N) ~ 0.0025 << the 0.02 tolerances


def _ks(x, y):
    """Two-sample KS statistic (max |CDF_x - CDF_y|)."""
    a = np.concatenate([x, y])
    a.sort()
    cx = np.searchsorted(np.sort(x), a, side="right") / len(x)
    cy = np.searchsorted(np.sort(y), a, side="right") / len(y)
    return float(np.max(np.abs(cx - cy)))


class _ProbAgent:
    """Discrete agent exposing a fixed pi_basic(a_bad|s) = p over two actions."""

    def __init__(self, p: float) -> None:
        self.p = float(p)

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0 - self.p, self.p]]).repeat(obs.shape[0], 1)


class _MeanAgent:
    """Continuous agent emitting a fixed mean action (the pi_basic mean mu)."""

    def __init__(self, mean: float, dim: int = 1) -> None:
        self.mean = float(mean)
        self.dim = dim

    def act(self, obs: torch.Tensor, *a, **k) -> ActionOutput:
        return ActionOutput(action=torch.full((obs.shape[0], self.dim), self.mean))


class _UEnv:
    """Minimal confounder-env stand-in: exposes ``current_u`` (resampled to
    Bernoulli(0.5) per round) and receives ``current_h`` from the continuous policy."""

    def __init__(self, n: int) -> None:
        self.n_envs = n
        self.device = CPU
        self.current_u = torch.zeros(n)
        self.current_h = None

    def draw_u(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.current_u = torch.bernoulli(torch.full((self.n_envs,), 0.5), generator=g)


class _FixedDQN:
    """DQN-like base: a_bad is greedy; ANNEALED epsilon is tiny (must NOT be read)."""

    epsilon = 0.001

    def q_network(self, obs):
        return torch.tensor([[0.0, 1.0]]).repeat(obs.shape[0], 1)  # argmax = a_bad


class _AngleQ:
    """State-DEPENDENT CartPole base: prefer the action reducing the pole angle."""

    def q_network(self, obs):
        return torch.stack([-obs[:, 2], obs[:, 2]], dim=-1)


class _PendCtrl:
    """Deterministic Pendulum base controller (a non-trivial continuous mu(s))."""

    def act(self, obs, *a, **k):
        return ActionOutput(action=(2.0 * obs[:, 1] - 0.5 * obs[:, 2]).unsqueeze(-1))


def _policy(p, sigma, env, *, is_online):
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
# T1 — marginal matching (EXACT), including p=0.9.
# ---------------------------------------------------------------------------
def test_t1_marginal_matches_pi_basic_across_sigma_and_p():
    obs = torch.zeros(N, 2)
    for p in (0.1, 0.5, 0.9):
        for sigma in SIGMAS:
            torch.manual_seed(0)
            env = _UEnv(N)
            env.draw_u(seed=1)
            a = _policy(p, sigma, env, is_online=False).act(obs).action
            tv = abs((a == A_BAD).float().mean().item() - p)
            assert tv < 0.02, f"p={p} sigma={sigma}: TV={tv:.4f}"


# ---------------------------------------------------------------------------
# T2 — U->A edge: ~0 at sigma=0, monotone increasing in sigma.
# ---------------------------------------------------------------------------
def _corr_abad_u(p, sigma):
    torch.manual_seed(0)
    env = _UEnv(N)
    env.draw_u(seed=2)
    a = _policy(p, sigma, env, is_online=False).act(torch.zeros(N, 2)).action
    ab = (a == A_BAD).float().numpy()
    u = env.current_u.numpy()
    if ab.std() < 1e-8 or u.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(ab, u)[0, 1])


def test_t2_u_to_a_edge_live_and_monotone():
    corrs = [_corr_abad_u(0.5, s) for s in SIGMAS]
    assert abs(corrs[0]) < 0.02, corrs
    for lo, hi in zip(corrs, corrs[1:]):
        assert hi > lo - 1e-3, corrs
    assert corrs[-1] > 0.2, corrs[-1]


# ---------------------------------------------------------------------------
# T3 — c_r (U->R) decoupled from sigma: reward shift on a_bad|U=1 invariant.
# ---------------------------------------------------------------------------
class _ZeroRewardEnv:
    def __init__(self, n=4):
        self.n_envs = n
        self.device = CPU

    def reset(self, seed=None):
        return torch.zeros(self.n_envs, 2), {}

    def step(self, action):
        z = torch.zeros(self.n_envs, dtype=torch.bool)
        return torch.zeros(self.n_envs, 2), torch.zeros(self.n_envs), z, z, {}


def test_t3_c_r_invariant_across_sigma():
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
    assert all(abs(s - 1.0) < 1e-9 for s in shifts), shifts

    from src.envs.registry import register_default_env_wrappers

    register_default_env_wrappers()
    for sigma in SIGMAS:
        env_ag = build_rollout_env(
            "CartPole-v1", 1, CPU, 0, "bias_confounded_action", sigma, c_r=1.0
        )
        assert abs(float(env_ag.c_r) - 1.0) < 1e-9, (sigma, env_ag.c_r)
        env_add = build_rollout_env("CartPole-v1", 1, CPU, 0, "bias_confounded", sigma)
        assert abs(float(env_add.c_r) - sigma) < 1e-9


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
        assert abs(online.intervened.float().mean().item() - (1.0 - sigma)) < 0.02
        offline = _policy(0.5, sigma, env, is_online=False).act(obs)
        assert offline.intervened.float().mean().item() == 0.0


# ---------------------------------------------------------------------------
# T5 — continuous partition swap: distribution-preserving marginal, and DEGENERATE
# without a fixed noise scale (the continuous analogue of eps -> 0 => p in {0,1}).
# ---------------------------------------------------------------------------
def test_t5_continuous_partition_swap_marginal_and_needs_scale():
    import gymnasium as gym

    space = gym.make("Pendulum-v1").action_space
    obs = torch.zeros(N, 1)
    mu = 0.4
    # Basic pi_basic samples: mu + s*z. Confounded marginal (over U) must match it.
    torch.manual_seed(0)
    basic_a = (mu + 0.5 * torch.randn(N, 1)).reshape(-1).numpy()
    for sigma in (0.5, 1.0):
        torch.manual_seed(0)
        env = _UEnv(N)
        env.draw_u(seed=4)
        pol = MarginallyMatchedConfoundedBehaviorPolicy(
            _MeanAgent(mu),
            "continuous",
            space,
            env,
            strength=sigma,
            continuous_noise_scale=0.5,
        )
        a = pol.act(obs).action.reshape(-1).numpy()
        # distribution-preserving: KS(confounded marginal, pi_basic) near the MC floor.
        assert _ks(a, basic_a) < 0.03, (sigma, _ks(a, basic_a))
        assert (env.current_h is not None) and env.current_h.shape[0] == N

    # noise_scale -> 0 gives a DEGENERATE partition: all actions collapse onto mu.
    torch.manual_seed(0)
    env = _UEnv(N)
    env.draw_u(seed=4)
    deg = (
        MarginallyMatchedConfoundedBehaviorPolicy(
            _MeanAgent(mu),
            "continuous",
            space,
            env,
            strength=1.0,
            continuous_noise_scale=0.0,
        )
        .act(obs)
        .action
    )
    assert deg.sub(mu).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# The confounding-signal magnitude and pi_basic entropy.
# ---------------------------------------------------------------------------
def test_corr_matches_sigma_sqrt_p_1_minus_p():
    obs = torch.zeros(N, 2)
    for p in (0.1, 0.3, 0.5):
        for sigma in (0.25, 0.5, 1.0):
            torch.manual_seed(0)
            env = _UEnv(N)
            env.draw_u(seed=5)
            a = _policy(p, sigma, env, is_online=False).act(obs).action
            corr = float(
                np.corrcoef((a == A_BAD).float().numpy(), env.current_u.numpy())[0, 1]
            )
            assert abs(corr - sigma * np.sqrt(p * (1.0 - p))) < 0.015, (p, sigma, corr)


def test_dqn_uses_fixed_pi_basic_epsilon_not_decaying():
    obs = torch.zeros(N, 2)
    torch.manual_seed(0)
    env = _UEnv(N)
    env.draw_u(seed=7)
    pol = MarginallyMatchedConfoundedBehaviorPolicy(
        _FixedDQN(),
        "discrete",
        Discrete(2),
        env,
        strength=1.0,
        a_bad=A_BAD,
        a_good=A_GOOD,  # default pi_basic_epsilon = 0.5 -> p(a_bad) = 0.75
    )
    assert abs((pol.act(obs).action == A_BAD).float().mean().item() - 0.75) < 0.02


def test_one_hot_fallback_raises():
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
# B1 — SHARED ORIGIN: basic (pi_basic) == confounded at sigma=0 (same fixed epsilon).
# ---------------------------------------------------------------------------
def test_b1_shared_origin_basic_equals_confounded_sigma0():
    obs = torch.zeros(N, 2)
    agent = _FixedDQN()
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
            strength=0.0,
            pi_basic_epsilon=0.5,
            a_bad=A_BAD,
            a_good=A_GOOD,
        )
        .act(obs)
        .action
    )
    p_basic = (a_basic == A_BAD).float().mean().item()
    p_conf = (a_conf == A_BAD).float().mean().item()
    assert abs(p_basic - p_conf) < 0.02, (p_basic, p_conf)
    assert abs(p_basic - 0.75) < 0.02


# ---------------------------------------------------------------------------
# B2 — default pi_basic_epsilon (0.5) retains PREFERENCE (not uniform random).
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

    assert abs(p_at(0.5) - 0.75) < 0.02  # default: real preference
    assert abs(p_at(1.0) - 0.5) < 0.02  # eps=1.0 IS the uniform random tier


# ---------------------------------------------------------------------------
# B4' — PER-STATE action-dist equivalence, isolated from state-visitation: evaluate
# both policies at FIXED states with fresh per-draw U (so E_U[pi_b(a|s,U)] =
# pi_basic(a|s) holds; U-selection on visitation is not conflated). A reflection
# FAILS the continuous KS — that is the point.
# ---------------------------------------------------------------------------
def _sample_states(env_name, pol_fn, n):
    import gymnasium as gym

    penv = gym.make(env_name)
    env_u = _UEnv(1)
    pol = pol_fn(env_u)
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(0)
    S, ep = [], 0
    while len(S) < n:
        obs, _ = penv.reset(seed=ep)
        ep += 1
        env_u.current_u = torch.bernoulli(torch.tensor([0.5]), generator=g)
        for _ in range(200):
            S.append(np.array(obs, dtype=np.float32))
            o = torch.as_tensor(obs).reshape(1, -1)
            a = pol.act(o).action
            step_a = (
                a.reshape(-1).numpy() if env_name == "Pendulum-v1" else int(a.item())
            )
            obs, _, t, tr, _ = penv.step(step_a)
            if t or tr:
                break
    penv.close()
    return np.array(S[:n])


def _eval_at_states(pol_fn, states, K):
    """Actions [M, K]: each of M states replicated K times, fresh U per draw."""
    M = len(states)
    env_u = _UEnv(M * K)
    pol = pol_fn(env_u)
    torch.manual_seed(0)
    obs = torch.as_tensor(np.repeat(states, K, axis=0))
    env_u.current_u = torch.bernoulli(torch.full((M * K,), 0.5))
    return pol.act(obs).action.reshape(M, K, -1)


def test_b4prime_discrete_per_state_action_dist():
    basic = lambda e: PiBasicBehaviorPolicy(
        _AngleQ(), "discrete", Discrete(2), epsilon=0.5
    )
    S = _sample_states("CartPole-v1", basic, 300)
    ab = _eval_at_states(basic, S, 800)[:, :, 0].float().mean(1).numpy()
    for sigma in (0.5, 1.0):
        conf = lambda e, s=sigma: MarginallyMatchedConfoundedBehaviorPolicy(
            _AngleQ(), "discrete", Discrete(2), e, strength=s, pi_basic_epsilon=0.5
        )
        ac = _eval_at_states(conf, S, 800)[:, :, 0].float().mean(1).numpy()
        d = np.abs(ab - ac)
        assert d.mean() < 0.03, (sigma, d.mean(), d.max())  # ~ MC floor


def _basic_cont(env):
    ctrl = _PendCtrl()

    class _B:
        def act(self, obs):
            return ActionOutput(
                action=ctrl.act(obs).action + 0.5 * torch.randn(obs.shape[0], 1)
            )

    return _B()


def _reflection(env, sigma, scale=0.5):
    ctrl = _PendCtrl()

    class _R:
        def act(self, obs):
            mu = ctrl.act(obs).action
            u = env.current_u.reshape(-1)
            coin = torch.rand(obs.shape[0]) < sigma
            a = torch.where(
                coin.unsqueeze(-1), mu + scale * (2 * u - 1).unsqueeze(-1), mu
            )
            return ActionOutput(action=a)

    return _R()


def test_b4prime_continuous_per_state_ks_partition_passes_reflection_fails():
    import gymnasium as gym

    space = gym.make("Pendulum-v1").action_space
    S = _sample_states("Pendulum-v1", _basic_cont, 100)
    ab = _eval_at_states(_basic_cont, S, 500)[:, :, 0].numpy()
    for sigma in (0.5, 1.0):
        part = lambda e, s=sigma: MarginallyMatchedConfoundedBehaviorPolicy(
            _PendCtrl(), "continuous", space, e, strength=s, continuous_noise_scale=0.5
        )
        ap = _eval_at_states(part, S, 500)[:, :, 0].numpy()
        ar = _eval_at_states(lambda e, s=sigma: _reflection(e, s), S, 500)[
            :, :, 0
        ].numpy()
        ks_part = np.mean([_ks(ab[m], ap[m]) for m in range(len(S))])
        ks_refl = np.mean([_ks(ab[m], ar[m]) for m in range(len(S))])
        # partition preserves the per-state action distribution (KS ~ MC floor);
        # the mean-preserving reflection does NOT (much larger KS).
        assert ks_part < 0.12, (sigma, ks_part)
        assert ks_refl > 0.20 and ks_refl > 2.0 * ks_part, (sigma, ks_part, ks_refl)


# ---------------------------------------------------------------------------
# B4'' — COVERAGE comparability (empirical orthogonality): confounding keeps
# action-coverage ~ basic across sigma; bias degrades it with beta. Wires the
# previously-dead Proximal.statistical_diagnostic (action_overlap).
# ---------------------------------------------------------------------------
def _seq_batch(make_pol, B=150):
    """B natural episodes -> episode-grouped (B, T). reward_sum varies by length
    (non-degenerate reward-median strata); pad rewards with 0 and pad actions by
    resampling each row's real actions so padding never distorts coverage."""
    import gymnasium as gym

    penv = gym.make("CartPole-v1")
    env_u = _UEnv(1)
    pol = make_pol(env_u)
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(0)
    R, A = [], []
    for ep in range(B):
        obs, _ = penv.reset(seed=ep)
        env_u.current_u = torch.bernoulli(torch.tensor([0.5]), generator=g)
        rs, as_ = [], []
        for _ in range(500):
            o = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            a = int(pol.act(o).action.item())
            obs, r, t, tr, _ = penv.step(a)
            rs.append(r)
            as_.append(a)
            if t or tr:
                break
        R.append(rs)
        A.append(as_)
    penv.close()
    T = max(len(r) for r in R)
    Rp, Ap = [], []
    for rs, as_ in zip(R, A):
        pad = T - len(rs)
        idx = torch.randint(0, len(as_), (pad,)).tolist()
        Rp.append(rs + [0.0] * pad)
        Ap.append(as_ + [as_[i] for i in idx])
    return {"rewards": torch.tensor(Rp), "actions": torch.tensor(Ap)}


def test_b4primeprime_coverage_confounding_comparable_bias_degrades():
    prox = Proximal()
    basic = lambda e: PiBasicBehaviorPolicy(
        _AngleQ(), "discrete", Discrete(2), epsilon=0.5
    )
    cov_basic = prox.statistical_diagnostic(_seq_batch(basic))["action_overlap"]
    assert cov_basic > 0.2, cov_basic

    # confounding: coverage stays comparable to basic across sigma.
    for sigma in (0.25, 0.5, 1.0):
        conf = lambda e, s=sigma: MarginallyMatchedConfoundedBehaviorPolicy(
            _AngleQ(), "discrete", Discrete(2), e, strength=s, pi_basic_epsilon=0.5
        )
        cov = prox.statistical_diagnostic(_seq_batch(conf))["action_overlap"]
        assert 0.85 < cov / cov_basic < 1.15, (sigma, cov, cov_basic)

    # bias (bias_skew ON TOP of pi_basic): coverage degrades monotonically with beta.
    covs = []
    for beta in (0.0, 0.3, 0.6, 0.9):
        biased = lambda e, bb=beta: SkewBehaviorPolicy(
            PiBasicBehaviorPolicy(_AngleQ(), "discrete", Discrete(2), epsilon=0.5),
            "discrete",
            Discrete(2),
            p=bb,
            preferred=A_BAD,
        )
        covs.append(prox.statistical_diagnostic(_seq_batch(biased))["action_overlap"])
    for lo, hi in zip(covs, covs[1:]):
        assert hi < lo + 0.02, covs  # monotone non-increasing in beta
    assert covs[-1] < 0.15 and covs[-1] < 0.4 * cov_basic, covs  # strong degradation

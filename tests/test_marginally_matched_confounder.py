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
import torch
from gymnasium.spaces import Box, Discrete
from src.envs.offline.generate import build_rollout_env
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper
from src.rl.base import ActionOutput
from src.rl.policies.behavior_policy import MarginallyMatchedConfoundedBehaviorPolicy

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
# T5 — continuous mean-preserving construction (perturbations cancel under U).
# ---------------------------------------------------------------------------
def test_t5_continuous_mean_preserving():
    obs = torch.zeros(N, 2)
    mean0 = 0.3
    for sigma in SIGMAS:
        torch.manual_seed(0)
        env = _UEnv(N)
        env.draw_u(seed=4)
        pol = MarginallyMatchedConfoundedBehaviorPolicy(
            _MeanAgent(mean0),
            "continuous",
            Box(low=-5.0, high=5.0, shape=(2,)),
            env,
            strength=sigma,
            base_scale=1.0,
            is_online=True,
        )
        out = pol.act(obs)
        # marginal mean preserved: E_U[a] == a0 for every sigma (perturbations cancel).
        assert abs(out.action.mean().item() - mean0) < 0.02, (sigma, out.action.mean())
        assert abs(out.intervened.float().mean().item() - (1.0 - sigma)) < 0.02

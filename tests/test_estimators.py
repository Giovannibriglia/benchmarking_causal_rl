"""Estimator + gap + fixture sanity tests (ported/adapted from the recovered
``tests/causal_metrics`` and ``tests/envs`` suites, Phase-3 gate (a))."""

from __future__ import annotations

import torch
from src.eval.estimators import dice_chi2, mmd_gauss, plugin_tv
from src.eval.gap import compute_gap, divergence_from_probs, gap_from_samples
from tests.fixtures.block_mdp import BlockMDPEnv
from tests.fixtures.causal_base import CausalEnv
from tests.fixtures.sepsis import SepsisCausalEnv

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# estimators (ported tests)
# ---------------------------------------------------------------------------


def test_mmd_identical_samples_is_near_zero():
    torch.manual_seed(0)
    x = torch.randn(128, 4)
    assert mmd_gauss(x, x.clone()).item() <= 1e-6


def test_mmd_separated_samples_positive():
    torch.manual_seed(0)
    x = torch.randn(128, 4)
    y = torch.randn(128, 4) + 3.0
    assert mmd_gauss(x, y).item() > 0.1


def test_plugin_tv_bounds():
    p = torch.tensor([[1.0, 0.0]])
    q = torch.tensor([[0.0, 1.0]])
    assert abs(plugin_tv(p, q).item() - 1.0) < 1e-6
    assert plugin_tv(p, p).item() < 1e-8


def test_dice_chi2_zero_when_policies_match():
    torch.manual_seed(0)
    B, A = 64, 3
    logits = torch.randn(B, A)
    logp = torch.log_softmax(logits, dim=-1)
    actions = torch.multinomial(logp.exp(), 1).squeeze(-1)
    behavior_logprob = logp.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    val = dice_chi2(torch.zeros(B, 2), actions, behavior_logprob, logp)
    assert val.item() < 1e-10


# ---------------------------------------------------------------------------
# gap
# ---------------------------------------------------------------------------


def test_gap_zero_when_distributions_match():
    p = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
    for name in ("tv", "kl", "chi2", "sup", "js_normalized"):
        assert divergence_from_probs(p, p.clone(), name).item() < 1e-6, name


def test_gap_positive_when_confounded_fixture():
    """On a confounded block-MDP, observed and do reward distributions differ."""
    env = BlockMDPEnv(
        env_id="fixture-block-cell7",
        n_envs=16,
        device=DEVICE,
        seed=3,
        cell=7,
        d=3,
        alpha=2.0,
        sigma2=0.0,
    )
    env.reset(seed=3)
    actions = torch.zeros(16, dtype=torch.long)
    p_obs = env.observed_reward_distribution(actions)
    p_do = env.do_reward(actions)
    res = compute_gap(p_obs, p_do, "tv")
    assert res.delta > 0.01
    assert res.is_oracle


def test_gap_zero_when_unconfounded_fixture():
    env = BlockMDPEnv(
        env_id="fixture-block-cell3",
        n_envs=8,
        device=DEVICE,
        seed=3,
        cell=3,
        d=3,
        alpha=0.0,
        sigma2=0.0,
    )
    env.reset(seed=3)
    actions = torch.zeros(8, dtype=torch.long)
    res = compute_gap(
        env.observed_reward_distribution(actions), env.do_reward(actions), "tv"
    )
    assert res.delta < 1e-6


def test_gap_from_samples_runs():
    torch.manual_seed(0)
    r1 = torch.randn(200)
    res = gap_from_samples(r1, r1 + 2.0)
    assert res.delta > 0.0 and res.divergence == "mmd"


# ---------------------------------------------------------------------------
# fixtures (ported env contract tests)
# ---------------------------------------------------------------------------


def _exercise(env: CausalEnv):
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, torch.Tensor)
    latent = env.latent_state()
    assert latent.shape[0] == env.n_envs
    action = torch.zeros(env.n_envs, dtype=torch.long, device=env.device)
    p_do = env.do_reward(action)
    assert torch.allclose(p_do.sum(dim=-1), torch.ones(env.n_envs), atol=1e-5)
    nxt = env.do_transition(action)
    assert torch.allclose(nxt.sum(dim=-1), torch.ones(env.n_envs), atol=1e-4)
    obs2, reward, term, trunc, info = env.step(action)
    assert reward.shape == (env.n_envs,)
    assert "latent" in info


def test_block_mdp_contract():
    _exercise(
        BlockMDPEnv(
            env_id="fixture-block-cell3", n_envs=4, device=DEVICE, seed=0, cell=3, d=3
        )
    )


def test_sepsis_contract_and_known_propensities():
    env = SepsisCausalEnv(
        env_id="fixture-sepsis-cell3", n_envs=4, device=DEVICE, seed=0, cell=3
    )
    _exercise(env)
    env.reset(seed=0)
    _, _, _, _, info = env.step(torch.zeros(4, dtype=torch.long))
    # cell 3: behavior policy known -> exact propensities logged in info
    assert "behavior_logprob" in info
    assert torch.isfinite(info["behavior_logprob"]).all()

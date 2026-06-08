"""Confounding machinery tests (Phase 4): ConfoundedEnv, biased explorer,
and the assert_confounded gate with its NEGATIVE CONTROL."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from src.causal.confounding import (
    assert_confounded,
    ConfoundedEnv,
    ConfoundedExplorer,
    ConfoundingGateError,
)
from src.data.experience_source import OfflineDatasetSource

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# ConfoundedEnv
# ---------------------------------------------------------------------------


def test_confounded_env_u_fixed_per_episode_and_not_in_obs():
    env = ConfoundedEnv(gym.make("CartPole-v1"), delta=0.5, seed=0)
    obs, info = env.reset(seed=0)
    assert "confounder_u" in info and info["confounder_u"] in (-1.0, 1.0)
    assert obs.shape == (4,), "U must never be written into observations"
    u0 = info["confounder_u"]
    for _ in range(10):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert info["confounder_u"] == u0, "U is fixed within an episode"
        if term or trunc:
            break
    env.close()


def test_confounded_env_reward_shift():
    env = ConfoundedEnv(gym.make("CartPole-v1"), delta=0.5, seed=0)
    env.reset(seed=0)
    u = env.current_u
    _, r, _, _, _ = env.step(0)
    assert abs(r - (1.0 + 0.5 * u)) < 1e-9, "r' = r + delta*U*psi, psi=1"
    env.close()


def test_confounder_u_varies_across_episodes():
    env = ConfoundedEnv(gym.make("CartPole-v1"), delta=0.5, seed=3)
    us = set()
    for ep in range(20):
        _, info = env.reset(seed=ep)
        us.add(float(info["confounder_u"]))
    assert us == {-1.0, 1.0}
    env.close()


# ---------------------------------------------------------------------------
# ConfoundedExplorer
# ---------------------------------------------------------------------------


def _logits_fn(obs):
    return torch.stack([obs[:, 0], -obs[:, 0]], dim=-1)  # prefers a0 iff obs0>0


def test_confounded_explorer_exact_biased_propensities():
    torch.manual_seed(0)
    explorer = ConfoundedExplorer(_logits_fn, beta=1.5)
    obs = torch.ones(512, 2)  # a* = 0 everywhere
    u_pos = torch.ones(512)
    a_pos, logp_pos = explorer.select_action(obs, latent=u_pos)
    # exact propensity: softmax([1+1.5, -1]) for U=+1
    expected = torch.log_softmax(torch.tensor([2.5, -1.0]), dim=-1)
    for a in (0, 1):
        got = logp_pos[a_pos == a]
        if len(got):
            assert torch.allclose(got, expected[a].expand_as(got), atol=1e-5)
    # U=+1 amplifies a*; U=-1 suppresses it
    a_neg, _ = explorer.select_action(obs, latent=-torch.ones(512))
    assert (a_pos == 0).float().mean() > (a_neg == 0).float().mean() + 0.2


def test_confounded_explorer_requires_latent():
    explorer = ConfoundedExplorer(_logits_fn, beta=1.0)
    with pytest.raises(ValueError, match="latent U"):
        explorer.select_action(torch.ones(4, 2))


# ---------------------------------------------------------------------------
# assert_confounded gate (synthetic world, hermetic)
# ---------------------------------------------------------------------------


def _synthetic_confounded_source(
    beta: float, delta: float, n_episodes: int = 1000, horizon: int = 8, seed: int = 0
) -> OfflineDatasetSource:
    """Tiny 2-action world: base logits from obs, U biases action AND reward."""
    g = torch.Generator().manual_seed(seed)
    episodes = []
    for _ in range(n_episodes):
        u = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
        obs_list, act, rew, logp = [], [], [], []
        x = torch.randn(2, generator=g)
        for _t in range(horizon):
            logits = torch.stack([x[0], -x[0]])
            a_star = int(logits.argmax())
            biased = logits.clone()
            biased[a_star] += beta * u
            lp = torch.log_softmax(biased, dim=-1)
            a = int(torch.multinomial(lp.exp(), 1, generator=g))
            base_r = float(x[1] * 0.1) + (0.3 if a == a_star else 0.0)
            # U enters the reward BOTH additively and through an action
            # interaction (psi(s,a) != const): the biased policy picks a*
            # exactly when it pays more, creating the spurious A-R link that
            # breaks backdoor adjustment.
            r = base_r + delta * u + delta * u * (1.0 if a == a_star else -1.0)
            obs_list.append(x.clone())
            act.append(a)
            rew.append(r)
            logp.append(float(lp[a]))
            x = torch.randn(2, generator=g)
        obs_list.append(x.clone())
        episodes.append(
            {
                "obs": torch.stack(obs_list),
                "actions": torch.tensor(act, dtype=torch.long),
                "rewards": torch.tensor(rew),
                "terminations": torch.zeros(horizon, dtype=torch.bool),
                "truncations": torch.tensor(
                    [False] * (horizon - 1) + [True], dtype=torch.bool
                ),
                "behavior_logprob": torch.tensor(logp),
                "confounder_u": torch.full((horizon,), u),
            }
        )
    return OfflineDatasetSource(episodes, DEVICE, behavior_policy="known")


def test_gate_passes_on_confounded_data():
    src = _synthetic_confounded_source(beta=1.5, delta=0.5)
    report = assert_confounded(src)
    assert report.passed
    assert report.action_u_zscore > 3.0
    assert report.reward_u_zscore > 3.0


def test_gate_fails_on_neutered_control():
    """beta = delta = 0: labeled confounded but functionally unconfounded."""
    src = _synthetic_confounded_source(beta=0.0, delta=0.0)
    with pytest.raises(ConfoundingGateError, match="functionally unconfounded"):
        assert_confounded(src)


def test_gate_fails_on_reward_only_pathway():
    """U -> reward alone (beta=0) must fail condition (ii)."""
    src = _synthetic_confounded_source(beta=0.0, delta=0.8)
    with pytest.raises(ConfoundingGateError):
        assert_confounded(src)


def test_gate_requires_propensities():
    src = _synthetic_confounded_source(beta=1.5, delta=0.5, n_episodes=50)
    unknown = OfflineDatasetSource(src.episodes, DEVICE, behavior_policy="unknown")
    with pytest.raises(ConfoundingGateError, match="LOGGED propensities"):
        assert_confounded(unknown)


def test_learner_batches_never_contain_u():
    src = _synthetic_confounded_source(beta=1.5, delta=0.5, n_episodes=20)
    batch = src.sample(64)
    assert "confounder_u" not in batch, "U is gate-only, never a learner input"
    assert set(batch.keys()) <= {
        "obs",
        "actions",
        "rewards",
        "next_obs",
        "dones",
        "behavior_logprob",
    }


def test_psi_hook():
    env = ConfoundedEnv(gym.make("CartPole-v1"), delta=1.0, seed=0)
    assert env._psi(np.zeros(4), 0) == 1.0
    env.close()


# ---------------------------------------------------------------------------
# Continuous-anchor gate (Phase-6C Option A): synthetic, hermetic
# ---------------------------------------------------------------------------


def _synthetic_continuous_confounded(
    gamma: float, delta: float, n_episodes: int = 300, horizon: int = 30, seed: int = 0
):
    """2-dim continuous world: U biases the action mean AND shifts reward."""
    from src.data.experience_source import OfflineDatasetSource

    g = torch.Generator().manual_seed(seed)
    sigma = 0.3
    episodes = []
    for _ in range(n_episodes):
        u = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
        obs_l, act_l, rew_l, logp_l = [], [], [], []
        x = torch.randn(2, generator=g)
        for _t in range(horizon):
            mean = torch.tanh(x) + gamma * u * torch.tensor([0.7, -0.7])
            a = mean + sigma * torch.randn(2, generator=g)
            logp = torch.distributions.Normal(mean, sigma).log_prob(a).sum()
            r = float(x.sum() * 0.1) + delta * u
            obs_l.append(x.clone())
            act_l.append(a)
            rew_l.append(r)
            logp_l.append(float(logp))
            x = torch.randn(2, generator=g)
        obs_l.append(x.clone())
        episodes.append(
            {
                "obs": torch.stack(obs_l),
                "actions": torch.stack(act_l),
                "rewards": torch.tensor(rew_l),
                "terminations": torch.zeros(horizon, dtype=torch.bool),
                "truncations": torch.tensor(
                    [False] * (horizon - 1) + [True], dtype=torch.bool
                ),
                "behavior_logprob": torch.tensor(logp_l),
                "confounder_u": torch.full((horizon,), u),
            }
        )
    return OfflineDatasetSource(episodes, DEVICE, behavior_policy="known")


def test_continuous_gate_passes_on_confounded():
    src = _synthetic_continuous_confounded(gamma=1.0, delta=1.5)
    report = assert_confounded(src)
    assert report.passed
    assert report.action_u_zscore > 3.0  # (ii) holds
    assert report.reward_u_zscore > 3.0  # (iii) holds


def test_continuous_gate_fails_on_neutered():
    src = _synthetic_continuous_confounded(gamma=0.0, delta=0.0)
    with pytest.raises(ConfoundingGateError, match="functionally unconfounded"):
        assert_confounded(src)


def test_continuous_gate_condition_i_is_diagnostic_only():
    """gamma>0, delta>0 but a near-degenerate condition (i): still PASSES on
    (ii)+(iii) - condition (i) is diagnostic on continuous, not pass/fail."""
    src = _synthetic_continuous_confounded(gamma=1.0, delta=1.5)
    report = assert_confounded(src)
    # passing despite condition (i) being unreliable at horizon is the point
    assert report.passed

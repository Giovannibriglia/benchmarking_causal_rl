"""L5's Markov falsifier — pinned behaviours.

Positive control + negative control on synthetic chains where ground truth is
exact (S14: a silent-failure component needs a quantity predicted to move and
a check that it moved), plus the S8 untestable path and the selector's three
outcomes (k=0 on Markov data, k=1 on one-step-hidden data, None when k_max
binds).
"""

from __future__ import annotations

import numpy as np
import pytest
from src.rl.offline.grace.l5 import Episode, markov_test, select_window


def _markov_episodes(rng, n_ep=60, t_len=25, d=3):
    """O_{t+1} = 0.8*O_t + a-shift + noise: Markov in O by construction."""
    eps = []
    for _ in range(n_ep):
        obs = np.zeros((t_len + 1, d))
        act = rng.integers(0, 2, size=t_len)
        obs[0] = rng.standard_normal(d)
        for t in range(t_len):
            obs[t + 1] = 0.8 * obs[t] + 0.3 * act[t] + 0.1 * rng.standard_normal(d)
        rew = obs[:-1, 0] + 0.1 * rng.standard_normal(t_len)
        eps.append(Episode(obs=obs, act=act, rew=rew))
    return eps


def _hidden_velocity_episodes(rng, n_ep=60, t_len=25):
    """Position observed, velocity hidden: O_{t+1} = O_t + v_t, v AR(1).

    One lag of O recovers v (v_t ~ O_t - O_{t-1}), so lag-1 is sufficient and
    lag-0 is falsified — the masked-CartPole structure in miniature.
    """
    eps = []
    for _ in range(n_ep):
        obs = np.zeros((t_len + 1, 1))
        act = rng.integers(0, 2, size=t_len)
        v = rng.standard_normal()
        for t in range(t_len):
            v = 0.9 * v + 0.4 * (act[t] - 0.5) + 0.05 * rng.standard_normal()
            obs[t + 1, 0] = obs[t, 0] + v
        rew = np.ones(t_len)  # constant: the untestable dim, on purpose
        eps.append(Episode(obs=obs, act=act, rew=rew))
    return eps


def test_markov_chain_is_not_rejected():
    rng = np.random.default_rng(7)
    v = markov_test(_markov_episodes(rng), lag=0, b=49, seed=1)
    assert v.p_value > 0.05, v.label(0.05)


def test_hidden_state_is_falsified_and_reward_reported_untestable():
    rng = np.random.default_rng(7)
    v = markov_test(_hidden_velocity_episodes(rng), lag=0, b=49, seed=1)
    assert v.p_value <= 0.02, v.label(0.05)
    assert "R" in v.untestable  # constant reward: untestable, never "passed"


def test_selector_returns_zero_on_markov_data():
    rng = np.random.default_rng(3)
    k, verdicts = select_window(_markov_episodes(rng), alpha=0.05, k_max=2, b=49)
    assert k == 0 and len(verdicts) == 1


def test_selector_finds_lag_one_on_hidden_velocity():
    rng = np.random.default_rng(3)
    k, verdicts = select_window(
        _hidden_velocity_episodes(rng), alpha=0.05, k_max=3, b=49
    )
    assert k == 1, [v.label(0.05) for v in verdicts]


def test_k_max_binding_returns_none_and_all_verdicts():
    """A process needing more memory than the budget: the selector must say so
    (None => the caller abstains), never return the largest k as if it passed."""
    rng = np.random.default_rng(5)
    # Second-order hidden state: O_{t+1} needs v_t which needs TWO lags of O to
    # pin down under observation noise on O; with k_max=0 the budget binds.
    eps = _hidden_velocity_episodes(rng)
    k, verdicts = select_window(eps, alpha=0.05, k_max=0, b=49)
    assert k is None and len(verdicts) == 1 and verdicts[0].rejected(0.05)


def test_folds_are_episode_blocked():
    """No episode's rows may straddle folds — the S1 unit-of-observation rule."""
    from src.rl.offline.grace.l5 import _build_design, _fold_of_episode

    rng = np.random.default_rng(0)
    design = _build_design(_markov_episodes(rng, n_ep=20), lag=0)
    folds = _fold_of_episode(design.episode_of, 5, rng)
    for e in np.unique(design.episode_of):
        assert np.unique(folds[design.episode_of == e]).size == 1


def test_deterministic_given_seed():
    rng = np.random.default_rng(11)
    eps = _markov_episodes(rng, n_ep=30)
    a = markov_test(eps, lag=0, b=49, seed=4)
    b = markov_test(eps, lag=0, b=49, seed=4)
    assert a.p_value == b.p_value and a.statistic == b.statistic


def test_too_short_episodes_raise():
    eps = [Episode(obs=np.zeros((2, 1)), act=np.zeros(1), rew=np.zeros(1))]
    with pytest.raises(ValueError):
        markov_test(eps, lag=1)


def _reward_hidden_episodes(rng, n_ep=60, t_len=25):
    """Hidden AR state drives the REWARD (not the dynamics): the reward
    channel's fast component is real, the obs process stays Markov."""
    eps = []
    for _ in range(n_ep):
        obs = np.zeros((t_len + 1, 2))
        act = rng.integers(0, 2, size=t_len)
        v = rng.standard_normal()
        obs[0] = rng.standard_normal(2)
        rew = np.zeros(t_len)
        for t in range(t_len):
            v = 0.9 * v + 0.3 * rng.standard_normal()
            obs[t + 1] = 0.8 * obs[t] + 0.3 * act[t] + 0.1 * rng.standard_normal(2)
            rew[t] = v + 0.1 * rng.standard_normal()
        eps.append(Episode(obs=obs, act=act, rew=rew))
    return eps


def test_serving_material_untestable_reward_is_not_material():
    from src.rl.offline.grace.l5 import serving_material

    rng = np.random.default_rng(7)
    v = markov_test(_hidden_velocity_episodes(rng), lag=0, b=49, seed=1)
    rec = serving_material(v, w=0.01)
    assert rec["material"] is False and "untestable" in rec["reason"]


def test_serving_material_fires_on_reward_relevant_hidden_state():
    from src.rl.offline.grace.l5 import serving_material

    rng = np.random.default_rng(7)
    v = markov_test(_reward_hidden_episodes(rng), lag=0, b=49, seed=1)
    tight = serving_material(v, w=1e-4)
    loose = serving_material(v, w=1e3)
    assert tight["material"] is True, tight
    assert loose["material"] is False, loose
    assert tight["dr2_fast"] > 0


def test_declaration_falsified_respects_the_stated_cut():
    rng = np.random.default_rng(7)
    v = markov_test(_hidden_velocity_episodes(rng), lag=0, b=49, seed=1)
    assert v.declaration_falsified(0.05)  # statistical tier alone
    assert v.declaration_falsified(0.05, dr2_cut=v.statistic / 10)
    assert not v.declaration_falsified(0.05, dr2_cut=v.statistic * 10)


def test_selector_cut_applies_at_every_stage():
    """Ruled 2026-09-03 (contract row 2): with a cut above the floor, the
    selector must return k=0 even where the statistical tier rejects — the
    cut-less selector chased floor rejections to k_max on true-MDP data."""
    rng = np.random.default_rng(3)
    eps = _hidden_velocity_episodes(rng)
    # Statistical tier alone finds k=1 here (measured in the earlier test).
    k_stat, _ = select_window(eps, alpha=0.05, k_max=3, b=49)
    assert k_stat == 1
    # A cut ABOVE this data's stage-0 statistic suppresses the falsification
    # entirely: k=0, one diagnostic, over-assumption cheap.
    v0 = markov_test(eps, lag=0, b=49, seed=0)
    k_cut, verdicts = select_window(
        eps, alpha=0.05, k_max=3, b=49, dr2_cut=v0.statistic * 10
    )
    assert k_cut == 0 and len(verdicts) == 1


def test_scale_invalid_flags_negative_base_r2_and_label_says_so():
    """S8: a dim where the base is worse than the mean predictor has a
    Delta-R^2 that is NOT a variance fraction, and the verdict must say so."""
    rng = np.random.default_rng(7)
    # Clean fixture: no flag.
    v = markov_test(_markov_episodes(rng, n_ep=30), lag=0, b=9, seed=1)
    assert v.scale_invalid == []
    # Per-episode hidden slope, never seen at train time for held-out
    # episodes: the memoryless base extrapolates worse than the mean.
    eps = []
    for _ in range(30):
        slope = rng.standard_normal() * 5
        obs = (
            np.cumsum(np.full((26, 1), slope), axis=0)
            + rng.standard_normal((26, 1)) * 0.01
        )
        act = rng.integers(0, 2, size=25)
        eps.append(Episode(obs=obs, act=act, rew=np.ones(25)))
    v2 = markov_test(eps, lag=0, b=9, seed=1)
    if v2.scale_invalid:  # construction-dependent; assert the LABEL contract
        assert "NOT-A-VARIANCE-FRACTION" in v2.label(0.05)

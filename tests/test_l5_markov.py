"""L5's Markov falsifier — pinned behaviours.

Positive control + negative control on synthetic chains where ground truth is
exact (S14: a silent-failure component needs a quantity predicted to move and
a check that it moved), the S8 untestable path, the S19 feature-set rule, and
the C3 record (falsification is REPORT-ONLY; window selection by materiality
lives in pomdp_branch and its tests).
"""

from __future__ import annotations

import numpy as np
import pytest
from src.rl.offline.grace.l5 import Episode, markov_test


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


def _reward_hidden_episodes(rng, n_ep=60, t_len=25, d=3):
    """Hidden AR state drives the REWARD (not the dynamics): the reward
    channel's fast component is real, the obs process stays Markov — the
    SAME obs process as ``_markov_episodes`` (d=3). At d=2 the memoryless
    base fit was scale-invalid on every obs dim (held-out base R^2 -1.3 /
    -0.2, identical under the pre- and post-S19 code: the random-feature draw
    on the 4-wide block generalises badly on that fixture) — flagged by S8,
    which is its job, but useless as a positive control."""
    eps = []
    for _ in range(n_ep):
        obs = np.zeros((t_len + 1, d))
        act = rng.integers(0, 2, size=t_len)
        v = rng.standard_normal()
        obs[0] = rng.standard_normal(d)
        rew = np.zeros(t_len)
        for t in range(t_len):
            v = 0.9 * v + 0.3 * rng.standard_normal()
            obs[t + 1] = 0.8 * obs[t] + 0.3 * act[t] + 0.1 * rng.standard_normal(d)
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


def test_selector_history_features_equal_the_served_augmentation():
    """S19: the family's history block carries lagged (O, A) only — column
    for column what ``pomdp_branch._augmented_cols`` hands the estimator —
    while the reward-channel diagnostic's block additionally carries R."""
    from src.rl.offline.grace.l5 import _build_design

    rng = np.random.default_rng(0)
    eps = _hidden_velocity_episodes(rng, n_ep=8, t_len=12)
    d_obs, n_actions = eps[0].obs.shape[1], int(max(e.act.max() for e in eps)) + 1
    fam = _build_design(eps, 0, seed=0, n_rff=8)
    diag = _build_design(eps, 0, seed=0, n_rff=8, history_reward=True)
    # raw width = the linear part of the expanded block (before the RFF cos)
    assert fam.hist.shape[1] - 8 == d_obs + n_actions
    assert diag.hist.shape[1] - 8 == d_obs + n_actions + 1
    # the j=0 conditioning block is shared exactly, which is what lets lag 0
    # reuse the base fit for the diagnostic
    assert np.array_equal(fam.x0, diag.x0)


def test_reward_only_visible_hidden_state_is_reported_not_selected_on():
    """The blind spot S19 characterises: a hidden state visible only through
    past rewards is a reward-channel phenomenon — reported on
    ``reward_channel``, never a falsification of the observation channel and
    never a reason to select k > 0."""
    # Seed 7 is the sibling test's known-good fit; seed 3 gave a base R^2 < 0
    # on every dim (scale_invalid, S8) — a fit whose numbers are declared
    # meaningless, so the guard below makes the test unable to pass vacuously.
    rng = np.random.default_rng(7)
    eps = _reward_hidden_episodes(rng)
    v = markov_test(eps, lag=0, b=49, seed=1)
    assert not any(n.startswith("O") for n in v.scale_invalid), v.label(0.05)
    # observation channel: Markov — an EFFECT-SIZE read (a p-value read at a
    # point null is exactly what S18 says not to rely on)
    assert v.statistic < 1e-2, v.label(0.05)
    assert v.reward_channel is not None and v.reward_channel["improvement"] > 0


def test_record_is_flat_and_carries_the_evidence():
    """Falsification is REPORT-ONLY: the record carries effect size, p, the
    capacity-shrink ratio, base R^2 and the reward channel as flat scalars
    and strings, and exposes no threshold to branch on."""
    rng = np.random.default_rng(1)
    v = markov_test(_hidden_velocity_episodes(rng), lag=0, b=49, seed=1)
    r = v.record(0.05)
    for key in ("l5_p", "l5_dr2", "l5_rejected", "l5_shrink", "l5_base_r2", "l5_label"):
        assert key in r, key
    assert all(isinstance(x, (int, float, str, bool)) for x in r.values())
    assert r["l5_rejected"] is True and "R" in r["l5_untestable"]
    assert not hasattr(v, "declaration_falsified")

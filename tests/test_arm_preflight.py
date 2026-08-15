"""Preflight for the GRACE v2 arms — validated against GROUND TRUTH.

Direction, stated because it is easy to invert: these check the GENERATOR
against the logged U and the declared parameters, exactly as the existing
confounding-signature gates do. They never use GRACE's estimator or L5's
tests. L5 is validated against the generator afterwards, never the reverse.
"""

from __future__ import annotations

import numpy as np
import torch
from src.envs.offline.arm_preflight import check_drift, check_instrument, check_proxies
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper


class _FakeVecEnv:
    """Two-dimensional state, discrete actions, fixed-length episodes."""

    def __init__(self, n: int = 1, seed: int = 0) -> None:
        self.n_envs = n
        self.device = torch.device("cpu")
        self._g = torch.Generator().manual_seed(seed)
        self._t = 0

    def reset(self, seed=None):
        self._t = 0
        return torch.rand(self.n_envs, 2, generator=self._g), {}

    def step(self, action):
        self._t += 1
        obs = torch.rand(self.n_envs, 2, generator=self._g)
        return (
            obs,
            obs[:, 0].clone(),
            torch.zeros(self.n_envs, dtype=torch.bool),
            torch.full((self.n_envs,), self._t % 12 == 0),
            {},
        )


def _rollout(strength=None, instrument=None, drift=0.0, steps=24000, seed=0):
    env = ConfoundedCollectionWrapper(
        _FakeVecEnv(seed=seed),
        c_a=1.0,
        c_r=1.0,
        seed=seed,
        confounder_kind="action_gated",
        a_bad=1,
        proxy_strength=strength,
        instrument_strength=instrument,
        u_drift=drift,
    )
    obs, _ = env.reset()
    out = {k: [] for k in ("Z", "W", "I", "U", "S", "A", "R", "E")}
    episodes, current, ep = [], [], 0
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        u = float(env.current_u[0])
        p = 0.8 if u > 0.5 else 0.2
        if env.current_i is not None:
            p = min(
                max(p + (0.15 if float(env.current_i[0]) > 0.5 else -0.15), 0.02), 0.98
            )
        a = int(torch.rand(1, generator=g) < p)
        out["S"].append(obs[0].tolist())
        out["U"].append(u)
        out["A"].append(a)
        out["E"].append(ep)
        current.append(u)
        if env.current_z is not None:
            out["Z"].append(float(env.current_z[0]))
            out["W"].append(float(env.current_w[0]))
        if env.current_i is not None:
            out["I"].append(float(env.current_i[0]))
        obs, r, term, trunc, _ = env.step(torch.tensor([a]))
        out["R"].append(float(r[0]))
        if bool(term[0] or trunc[0]):
            episodes.append(current)
            current, ep = [], ep + 1
            obs, _ = env.reset()
    arrays = {k: np.array(v) for k, v in out.items()}
    arrays["episodes"] = episodes
    return arrays


def test_proxies_are_covariate_free_and_excluded():
    """The covariate-free property is what makes D-D's measurement matrices
    GLOBAL, which is what pins the latent's labelling globally. Proxy noise
    that scaled with the state would silently make them covariate-conditional
    and collapse D-D into D-B's situation with no error raised."""
    d = _rollout(strength=1.5)
    rep = check_proxies(
        z=d["Z"],
        w=d["W"],
        u=d["U"],
        state=d["S"],
        action=d["A"],
        reward=d["R"],
        episode_ids=d["E"],
    )
    assert rep.covariate_free, rep.summary()
    assert rep.max_abs_corr_proxy_state < 0.05
    assert rep.exclusions_hold, rep.reasons
    assert abs(rep.corr_z_u) > 0.5 and abs(rep.corr_w_u) > 0.5  # actually informative


def test_kruskal_check_detects_an_uninformative_view():
    """The check exists to catch a generated proxy with k-rank 1 -- which would
    mean D-D is NOT identified either. At zero signal it must FAIL, or it is
    not measuring anything."""
    dead = _rollout(strength=0.0)
    rep = check_proxies(
        z=dead["Z"],
        w=dead["W"],
        u=dead["U"],
        state=dead["S"],
        action=dead["A"],
        reward=dead["R"],
        episode_ids=dead["E"],
    )
    assert rep.k_ranks["Z"] == 1 and rep.k_ranks["W"] == 1, rep.summary()
    assert not rep.kruskal_ok

    live = _rollout(strength=1.5)
    rep2 = check_proxies(
        z=live["Z"],
        w=live["W"],
        u=live["U"],
        state=live["S"],
        action=live["A"],
        reward=live["R"],
        episode_ids=live["E"],
    )
    assert rep2.k_ranks == {"Z": 2, "W": 2, "R": 2}
    assert rep2.kruskal_ok


def test_kruskal_margin_is_monotone_in_the_signal_knob():
    """R4 sweeps this knob; the margin must move with it rather than sitting at
    whatever the first implementation produced."""
    margins = []
    for s in (0.25, 1.0, 2.0):
        d = _rollout(strength=s, steps=18000)
        rep = check_proxies(
            z=d["Z"],
            w=d["W"],
            u=d["U"],
            state=d["S"],
            action=d["A"],
            reward=d["R"],
            episode_ids=d["E"],
        )
        margins.append(rep.condition_numbers["Z"])
    assert margins == sorted(margins), margins


def test_instrument_is_exogenous_relevant_and_excluded():
    """A leaking instrument would invalidate the Balke-Pearl anchor -- L4's
    only exact reference -- while still looking plausible."""
    d = _rollout(instrument=0.5)
    rep = check_instrument(i=d["I"], u=d["U"], action=d["A"], reward=d["R"])
    assert rep.independent_of_u and rep.relevant and rep.exclusion_holds, rep.summary()
    assert abs(rep.corr_i_u) < 0.05
    assert abs(rep.corr_i_reward_given_action_and_u) < 0.05


def test_drift_matches_the_declared_rho_and_zero_is_static():
    """The rho-sweep in V-E only means something if rho does what it claims."""
    for rho in (0.0, 0.1, 0.25, 0.5):
        d = _rollout(drift=rho, steps=18000)
        rep = check_drift(u_by_episode=d["episodes"], rho=rho)
        assert rep.matches, rep.summary()
    static = _rollout(drift=0.0, steps=6000)
    rep = check_drift(u_by_episode=static["episodes"], rho=0.0)
    assert rep.realised_autocorr > 0.99  # episode-static, i.e. D-B


def test_existing_arms_are_byte_unchanged_when_the_new_features_are_off():
    """Every new feature defaults off and draws from a dedicated generator, so
    an arm that does not enable them consumes exactly the RNG it always did."""
    base = _rollout(steps=3000, seed=7)
    again = _rollout(steps=3000, seed=7)
    assert np.array_equal(base["U"], again["U"])
    assert np.array_equal(base["A"], again["A"])
    assert np.array_equal(base["R"], again["R"])
    # Proxies never enter the policy or the reward, so switching them on must
    # leave the U, action and reward streams bit-for-bit alone.
    with_proxies = _rollout(strength=1.5, steps=3000, seed=7)
    assert np.array_equal(base["U"], with_proxies["U"]), "U stream perturbed"
    assert np.array_equal(base["A"], with_proxies["A"]), "action stream perturbed"
    assert np.array_equal(base["R"], with_proxies["R"]), "reward stream perturbed"

    # The instrument is DIFFERENT and must not be asserted the same way: it
    # exists to move the action, so actions and (action-gated) rewards SHOULD
    # change -- an instrument that left them alone would be irrelevant, which
    # check_instrument explicitly rejects. What must still hold is that it does
    # not disturb the latent: I is drawn from a dedicated generator and is
    # independent of U by construction.
    with_instrument = _rollout(instrument=0.5, steps=3000, seed=7)
    assert np.array_equal(base["U"], with_instrument["U"]), "U stream perturbed"
    assert not np.array_equal(
        base["A"], with_instrument["A"]
    ), "the instrument did not move the action -- it would be irrelevant"

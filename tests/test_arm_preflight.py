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
    assert rep.exclusions_hold, rep.reasons
    # The verdict rests on the z against the episode-level null, NOT on the raw
    # correlation. On the REAL generator the marginal corr(proxy, S) reaches
    # 0.226 at this strength -- U drives the action, the action drives the next
    # state, so a proxy of U is marginally correlated with the state BY DESIGN.
    # Asserting a small marginal here would re-enshrine the bug that finding
    # exposed; the conditional statement is what covariate-freeness means.
    assert rep.null_p["proxy_vs_state_given_u"] >= 0.01, rep.summary()
    assert rep.null_p["z_vs_w_given_u"] >= 0.01
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
    rep = check_instrument(
        i=d["I"], u=d["U"], action=d["A"], reward=d["R"], episode_ids=d["E"]
    )
    assert rep.independent_of_u and rep.relevant, rep.summary()
    assert rep.exclusion_holds or not rep.exclusion_testable, rep.reasons
    # Exogeneity is "inside the null", relevance is "outside it" -- an
    # instrument that fails to move A is useless, so for relevance a null result
    # IS the failure. Judging both by one magnitude tolerance got this backwards.
    assert rep.null_p["i_vs_u"] >= 0.01, rep.summary()
    assert rep.null_p["i_vs_a"] < 0.01, rep.summary()


def test_the_exclusion_check_rejects_a_real_leak():
    """R2. A check that never rejects proves nothing, so the passing result is
    only worth something once the failing one is demonstrated. Measured on the
    real generator under the stochastic gate: residual var(R | A, U) = 0.101,
    the clean instrument passes at z = 0.94, and an injected I -> R leak of just
    0.05 is caught at z = 8.0. Under the OLD deterministic gate the residual
    variance was exactly 0, so none of these leaks were detectable at all."""
    n, block = 6000, 10
    rng = np.random.default_rng(1)
    ep = np.repeat(np.arange(n // block), block)
    u = np.repeat(rng.integers(0, 2, n // block).astype(float), block)
    i = np.repeat(rng.integers(0, 2, n // block).astype(float), block)
    a = (rng.random(n) < (0.5 + 0.2 * (i - 0.5))).astype(float)
    # Stochastic given (A, U) -- U shifts the PROBABILITY of the bonus, so R
    # stays binary and the residual variance is real.
    q = np.where(u > 0.5, 0.8, 0.2)
    clean = 1.0 + a * (rng.random(n) < q).astype(float)

    ok = check_instrument(i=i, u=u, action=a, reward=clean, episode_ids=ep)
    assert ok.exclusion_testable and ok.exclusion_holds, ok.summary()

    leaked = clean + 0.05 * i
    bad = check_instrument(i=i, u=u, action=a, reward=leaked, episode_ids=ep)
    assert bad.exclusion_testable
    assert not bad.exclusion_holds, "a leaking instrument must be caught"
    assert bad.null_p["i_vs_r_given_a_u"] < 0.01, bad.summary()


def test_exclusion_reports_when_it_cannot_be_tested_rather_than_passing():
    """On an env whose reward is a DETERMINISTIC function of (A, U) -- CartPole
    with the action gate, r = 1 + c_r*U*1[a=a_bad] -- residualising on (A, U)
    leaves no variance, so the exclusion statistic is identically zero for the
    observed data and every permutation. That is a measurement of nothing, and
    it must not read as a verified pass."""
    n = 4000
    rng = np.random.default_rng(0)
    ep = np.repeat(np.arange(n // 10), 10)
    u = np.repeat(rng.integers(0, 2, n // 10).astype(float), 10)
    i = np.repeat(rng.integers(0, 2, n // 10).astype(float), 10)
    a = (rng.random(n) < (0.5 + 0.2 * (i - 0.5))).astype(float)
    r = 1.0 + u * a  # deterministic given (A, U)
    rep = check_instrument(i=i, u=u, action=a, reward=r, episode_ids=ep)
    assert not rep.exclusion_testable, rep.summary()
    assert not rep.exclusion_holds, "an untestable exclusion must not claim a pass"
    assert any("NOT TESTABLE" in s for s in rep.reasons), rep.reasons


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


# --------------------------------------------------------------------------
# S1b — granularity. These are the regression tests for the bug the existing
# harness above CANNOT see: _FakeVecEnv ends every episode at a fixed 12 steps,
# so length carries no information and length-weighting has nothing to bite on.
# That is exactly rule S5 (a synthetic harness whose marginals happen to equal
# its conditionals passes checks the real generator fails), so the fixtures here
# make episode length an OUTCOME of the latent, as it is in a real env.
# --------------------------------------------------------------------------


def _length_coupled_arm(n_ep=600, seed=0, leak=0.0):
    """Episodes whose LENGTH is an OUTCOME of behaviour, as it is in a real env.

    **EPISODE LENGTH IS A COLLIDER, and that -- not "length correlates with
    behaviour" -- is the mechanism behind S1b.** Everything driving the action
    drives how long the episode survives, so ``L`` is a common descendant:
    ``I -> A -> L <- A <- U``. Pooling an episode-constant quantity over
    transitions weights each episode by its own ``L``, which is a form of
    conditioning on that collider -- and conditioning on a collider manufactures
    dependence between its causes. It follows that the bias is NOT uniform: a
    passive proxy that drives nothing is barely touched, while an instrument,
    whose whole job is to move the action, is hit hardest.

    The bias is largest when ``U`` and ``I`` INTERACT in the action law, because
    then the surviving episodes are systematically the ones where the two
    disagree. Measured on the real D-E arm, mean length by ``(U, I)`` cell:
    19.5 / 59.0 / 67.4 / 15.4 -- a clean disagree-lives-longer pattern, and the
    reason pooled ``corr(I, U)`` reached **-0.590** on an instrument drawn from
    its own Bernoulli. This fixture reproduces it (13 / 42 / 42 / 13, pooled
    ``corr(I, U) = -0.53`` against **-0.003** per episode) by making the
    per-step hazard a function of the realised action mix: an extreme mix
    destabilises the plant, a balanced one survives. ``L`` is therefore a
    descendant of ``A`` alone, never of ``U`` or ``I`` directly.

    The other fixture in this file (``_FakeVecEnv``) cannot show any of this: it
    truncates every episode at a fixed 12 steps, so ``L`` has no variance and
    the collider cannot open. Rule S5 exactly -- a synthetic harness whose
    marginals happen to equal its conditionals passes what the real generator
    fails.

    ``leak`` injects a genuine proxy->state association so the check can be
    shown to still reject; at ``leak = 0`` every conditional independence holds
    by construction and any rejection is a false alarm.
    """
    rng = np.random.default_rng(seed)
    u_ep = rng.integers(0, 2, n_ep).astype(float)
    i_ep = rng.integers(0, 2, n_ep).astype(float)  # never reads U
    z_ep = 1.5 * u_ep + rng.normal(0, 1, n_ep)  # passive: drives nothing
    w_ep = 1.5 * u_ep + rng.normal(0, 1, n_ep)

    ep, u, z, w, i, s, a, r = [], [], [], [], [], [], [], []
    for e in range(n_ep):
        p = min(
            max((0.8 if u_ep[e] > 0.5 else 0.2) + 0.3 * (i_ep[e] - 0.5), 0.02), 0.98
        )
        acts: list[float] = []
        n_bad = 0.0
        while len(acts) < 400:
            act = float(rng.random() < p)
            acts.append(act)
            n_bad += act
            if len(acts) >= 10:
                frac = n_bad / len(acts)
                if rng.random() < 0.008 + 1.15 * (frac - 0.5) ** 2:
                    break
        acts_arr = np.asarray(acts)
        t = acts_arr.size
        ep.extend([e] * t)
        u.extend([u_ep[e]] * t)
        z.extend([z_ep[e]] * t)
        w.extend([w_ep[e]] * t)
        i.extend([i_ep[e]] * t)
        a.extend(acts_arr.tolist())
        # FOUR state dimensions, as CartPole has: the covariate-free check is
        # then a family of dims x proxies = 8 statistics whose MAXIMUM is the
        # verdict, which is what makes S3's null-of-the-max necessary. Judged
        # per-test it runs ~8x the intended false-alarm rate. The state reads the
        # action, never the proxies -- unless `leak` is on.
        s.extend(
            (
                rng.normal(0, 1, (t, 4)) + 0.5 * acts_arr[:, None] + leak * z_ep[e]
            ).tolist()
        )
        r.extend((1.0 + u_ep[e] * acts_arr + rng.normal(0, 0.3, t)).tolist())
    return {
        "E": np.array(ep),
        "U": np.array(u),
        "Z": np.array(z),
        "W": np.array(w),
        "I": np.array(i),
        "S": np.array(s),
        "A": np.array(a),
        "R": np.array(r),
    }


def test_length_coupling_does_not_manufacture_a_proxy_state_association():
    """S1b. The proxies here are drawn from their own generator and the state
    provably never reads them, so every rejection is a false alarm. What makes
    it fire is that U sets episode LENGTH: pooled over transitions, each episode
    enters weighted by its own length, and since length is a function of U the
    statistic acquires a U-driven dependence that the episode-permuted null
    cannot reproduce. The fix is granularity, not a wider null."""
    d = _length_coupled_arm()
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
    assert rep.exclusions_hold, rep.reasons
    assert rep.n_episodes == 600, rep.summary()


def test_the_covariate_free_check_still_rejects_a_real_leak():
    """R2 again: the passing result above is worth nothing until the failing one
    is demonstrated on the SAME fixture. A check made granularity-correct could
    just as easily have been made blind."""
    d = _length_coupled_arm(leak=0.6)
    rep = check_proxies(
        z=d["Z"],
        w=d["W"],
        u=d["U"],
        state=d["S"],
        action=d["A"],
        reward=d["R"],
        episode_ids=d["E"],
    )
    assert not rep.covariate_free, rep.summary()
    assert any("covariate-free" in s for s in rep.reasons), rep.reasons


def test_length_coupling_does_not_make_an_exogenous_instrument_look_endogenous():
    """The measured case: corr(I, U) = -0.590 pooled over transitions against
    -0.034 with one row per episode, on an instrument drawn from its own
    Bernoulli. Eleven of V-B's D-E rows failed exogeneity this way."""
    d = _length_coupled_arm()
    rep = check_instrument(
        i=d["I"], u=d["U"], action=d["A"], reward=d["R"], episode_ids=d["E"]
    )
    assert rep.independent_of_u, rep.summary()
    assert rep.relevant, rep.summary()
    assert abs(rep.corr_i_u) < 0.15, rep.summary()


def test_an_episode_constant_reduction_refuses_a_per_step_quantity():
    """The mirror-image mistake: handing a genuinely per-step series to the
    episode-constant reduction would silently keep one arbitrary step's value
    and read as a clean measurement. It has to raise."""
    import pytest

    d = _length_coupled_arm(n_ep=30)
    drifting_u = d["U"] + (np.arange(d["U"].size) % 2)  # varies within episodes
    with pytest.raises(ValueError, match="varies WITHIN an episode"):
        check_proxies(
            z=d["Z"],
            w=d["W"],
            u=drifting_u,
            state=d["S"],
            action=d["A"],
            reward=d["R"],
            episode_ids=d["E"],
        )


def test_drift_measures_its_own_length_weighting_exemption():
    """D-B' keeps a transition-level statistic, and the exemption rests on rho
    being HOMOGENEOUS across episodes. That is now measured rather than
    asserted: with one declared rho the short- and long-episode halves must
    agree, and a heterogeneous-rho variant must be caught."""
    rng = np.random.default_rng(0)

    def chain(rho, t):
        u = [float(rng.integers(0, 2))]
        for _ in range(t - 1):
            u.append(1.0 - u[-1] if rng.random() < rho else u[-1])
        return u

    lengths = [20] * 200 + [80] * 200
    homogeneous = [chain(0.1, t) for t in lengths]
    rep = check_drift(u_by_episode=homogeneous, rho=0.1)
    assert rep.matches, rep.summary()
    assert rep.length_weighting_inert, rep.summary()

    # The variant that voids the exemption: rho depends on episode length, so
    # the pooled autocorrelation drifts toward whatever the LONG episodes have.
    heterogeneous = [chain(0.05 if t == 20 else 0.35, t) for t in lengths]
    bad = check_drift(u_by_episode=heterogeneous, rho=0.1)
    assert not bad.length_weighting_inert, bad.summary()


def test_no_false_alarms_across_seeds_on_a_provably_clean_arm():
    """The false-alarm RATE, not one draw of it. V-B's 130-dataset run turned
    four D-D rows down for a proxy-state association at 3.1-3.7 null SDs -- all
    of them just over a per-test 3-sd line applied to the MAXIMUM of
    dims x proxies = 8 statistics, which is S3's failure mode exactly. On this
    fixture the old per-test maximum reached 2.77 null SDs on a generator whose
    proxies provably never read the state; the margin was thin by construction,
    not by accident.

    Six independent seeds, every conditional independence true by construction,
    so every rejection here is a false alarm."""
    for seed in range(6):
        d = _length_coupled_arm(seed=seed)
        rep = check_proxies(
            z=d["Z"],
            w=d["W"],
            u=d["U"],
            state=d["S"],
            action=d["A"],
            reward=d["R"],
            episode_ids=d["E"],
        )
        assert rep.covariate_free, f"seed {seed}: {rep.summary()}"
        assert rep.exclusions_hold, f"seed {seed}: {rep.reasons}"
        inst = check_instrument(
            i=d["I"], u=d["U"], action=d["A"], reward=d["R"], episode_ids=d["E"]
        )
        assert inst.independent_of_u, f"seed {seed}: {inst.summary()}"
        assert inst.relevant, f"seed {seed}: {inst.summary()}"

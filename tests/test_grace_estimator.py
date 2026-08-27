"""L3 — the latent-class estimator, checked against GROUND TRUTH.

Direction as everywhere else: the estimator is validated against a fixture whose
latent is KNOWN, never against another estimator or against L5.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator


def _fixture(n_ep=300, T=12, seed=0, proxy=True):
    """Episode-static binary U driving BOTH channels: P(A|S,U) and P(R|S,A,U)."""
    g = torch.Generator().manual_seed(seed)
    u_ep = (torch.rand(n_ep, generator=g) < 0.5).long()
    S, A, R, E, Z = [], [], [], [], []
    for e in range(n_ep):
        u = int(u_ep[e])
        S.append(torch.randn(T, 2, generator=g))
        a = (torch.rand(T, generator=g) < (0.75 if u else 0.25)).long()
        A.append(a)
        R.append(1.0 + 1.5 * u * a.float() + 0.2 * torch.randn(T, generator=g))
        Z.append(
            torch.full((T,), (2.0 * u - 1.0) * 1.5 + float(torch.randn(1, generator=g)))
        )
        E.append(torch.full((T,), e, dtype=torch.long))
    kw = {"proxy": {"Z": torch.cat(Z)}} if proxy else {}
    return u_ep, EpisodeData(
        state=torch.cat(S),
        action=torch.cat(A),
        reward=torch.cat(R),
        episode_ids=torch.cat(E),
        **kw,
    )


def _accuracy(hard, truth):
    """Label-swap invariant: a mixture is identified only up to a permutation."""
    return max(
        float((hard == truth).float().mean()), float((hard != truth).float().mean())
    )


def test_em_recovers_a_known_episode_static_latent():
    u_ep, data = _fixture()
    est = LatentClassEstimator(state_dim=2, n_actions=2, proxy_names=("Z",), seed=0)
    fit = est.fit(data, max_iter=10, epochs=30)
    assert _accuracy(fit.hard_assignment(), u_ep) > 0.9, fit.separability()
    assert fit.separability() > 0.8
    assert 0.35 < float(fit.prior[0]) < 0.65, fit.prior


def test_responsibilities_are_per_episode_not_per_transition():
    """The load-bearing structural claim. One responsibility per episode: a
    per-transition posterior would treat an episode's rows as independent draws
    of U, inflating the effective sample size by the episode length."""
    _, data = _fixture(n_ep=40, T=7)
    est = LatentClassEstimator(state_dim=2, n_actions=2, proxy_names=("Z",), seed=0)
    fit = est.fit(data, max_iter=2, epochs=5)
    assert fit.responsibilities.shape == (40, 2)
    # ...and broadcasting back must give every row of an episode the same weight.
    rows = data.broadcast(fit.responsibilities)
    assert rows.shape == (data.n, 2)
    assert torch.allclose(rows[:7], rows[0].expand(7, 2))


def test_a_mechanism_that_cannot_take_weights_is_refused():
    """N3. The M-step is inherently weighted, so a mechanism that ignores
    weights would bias strata toward each other silently — costing L5 detection
    power in the quiet direction. It must fail fast instead."""
    with pytest.raises(ValueError, match="weighted fit is exact"):
        LatentClassEstimator(state_dim=2, n_actions=2, reward_mechanism="kde")


def test_monotonicity_is_reported_not_assumed():
    """Our M-step is a PARTIAL (SGD) maximisation, so this is generalized EM and
    the textbook monotonicity guarantee does not hold. The property exists so a
    non-monotone run is visible rather than hidden behind a plausible final
    value."""
    _, data = _fixture(n_ep=120, T=8)
    est = LatentClassEstimator(state_dim=2, n_actions=2, proxy_names=("Z",), seed=0)
    fit = est.fit(data, max_iter=4, epochs=10)
    assert isinstance(fit.monotone, bool)
    # History = the initial E-step entry + one entry per ACCEPTED step. The
    # old equality len == n_iter held BY ACCIDENT of restart-EM: that fixture
    # always ended on a rejected pass (exhaustion), so the initial entry
    # balanced the one non-appending pass. Under GEM (warm_start default,
    # 2026-08-21) every pass can accept and the honest bookkeeping bound is:
    assert 1 <= len(fit.log_likelihood) <= fit.n_iter + 1


def test_episode_data_refuses_misaligned_arrays():
    with pytest.raises(ValueError, match="transition-aligned"):
        EpisodeData(
            state=torch.zeros(10, 2),
            action=torch.zeros(9, dtype=torch.long),
            reward=torch.zeros(10),
            episode_ids=torch.zeros(10, dtype=torch.long),
        )


def _fitted(n_ep=200, T=10, seed=0, fit_seeds=(0, 1, 2)):
    """Best-of-seeds BY LIKELIHOOD, because a single short-T fit is a lottery.

    Measured on the real CartPole arm at T = 16 with 10 paired seeds: 2-3 in 10
    land in a bad basin whatever the temperature, and the two arms fail on the
    SAME seeds. A single-seed assertion at short T therefore has a ~20-30%
    chance of testing the basin draw rather than the estimand -- and this test
    was one, which is why it started failing when an unrelated change to the
    anneal schedule perturbed the trajectory.

    Selection is by LIKELIHOOD, never by the quantity under test: picking the
    fit that gives the expected do-effect would make the assertion vacuous.
    """
    u_ep, data = _fixture(n_ep=n_ep, T=T, seed=seed, proxy=False)
    best = None
    for fs in fit_seeds:
        est = LatentClassEstimator(state_dim=2, n_actions=2, u_card=2, seed=fs)
        fit = est.fit(data, max_iter=6, epochs=40)
        if best is None or fit.final_ll > best[1].final_ll:
            best = (est, fit)
    return best[0], best[1], data


def test_interventional_reward_recovers_the_known_do_effect():
    """Ground truth on the fixture is r = 1 + 1.5*u*a, so under do(a) the
    U-marginal effect is E[R|do(1)] - E[R|do(0)] = 1.5 * P(U=1) = 0.75."""
    est, fit, _ = _fitted()
    s = torch.zeros(2)
    v0 = est.interventional_reward(s, 0, fit, n_samples=1024)
    v1 = est.interventional_reward(s, 1, fit, n_samples=1024)
    assert abs(float(v0) - 1.0) < 0.15, float(v0)
    assert 0.4 < float(v1) - float(v0) < 1.1, (float(v0), float(v1))
    # C3: every estimate carries its conditions -- now including the two that
    # the T = 500 failure showed were missing. ``sep=`` was the old label; the
    # per-step separation replaces it as the correctness diagnostic and the
    # saturated-at-init flag rides alongside, because a fit frozen at its
    # initialisation must say so wherever its number goes.
    assert isinstance(v0.monotone, bool)
    assert "sep/step=" in v0.label(), v0.label()
    assert isinstance(v0.saturated_at_init, bool)
    assert 0.0 <= v0.initial_saturation <= 1.0


def test_the_target_path_is_differentiable():
    """THE contract. sample(do=) is the only differentiable interventional path;
    query/query_batch are non-differentiable BY DESIGN. Routing a target through
    the fast one would not raise -- it would return a value with no gradient,
    presenting downstream as a model that will not train. That silent failure is
    exactly what this test exists to prevent, so it pins the gradient rather
    than merely the value."""
    est, fit, _ = _fitted(n_ep=80, T=8)
    est.model.zero_grad(set_to_none=True)
    v = est.interventional_reward(torch.zeros(2), 1, fit, n_samples=256)
    assert v.value.requires_grad, "sample(do=) target lost its gradient"
    v.value.backward()
    total = sum(
        float(p.grad.norm()) for p in est.model.parameters() if p.grad is not None
    )
    assert total > 0.0, "no gradient reached the model parameters"


def test_the_readonly_sweep_is_not_differentiable_and_says_so():
    """The counterpart contract: interventional_sweep is for L4's bound
    evaluation and read-only quantities. It must NOT quietly serve as a target.
    The substantive contract is that NO GRADIENT REACHES THE MODEL through it,
    which is what would silently break an optimiser fed this value by mistake.

    Note what is NOT asserted: the amendment notes that inference-mode tensors
    raise when used in a differentiable op, but this method does arithmetic on
    the engine's output before returning, and the result is an ordinary
    gradient-free tensor rather than an inference-mode one. So the loud failure
    does not survive to the caller here -- only the missing gradient does, which
    is precisely why it is worth pinning."""
    est, fit, _ = _fitted(n_ep=80, T=8)
    states = torch.zeros(4, 2)
    est.model.zero_grad(set_to_none=True)
    est_out = est.interventional_sweep(states, [0, 1, 0, 1], fit)
    out = est_out.value
    assert out.shape == (4,)
    assert not out.requires_grad, "the read-only sweep must not carry a gradient"
    probe = (out.sum() * torch.ones(1, requires_grad=True)).sum()
    probe.backward()
    assert all(
        p.grad is None or float(p.grad.norm()) == 0.0 for p in est.model.parameters()
    ), "a gradient reached the model through the read-only sweep"


def test_both_interventional_paths_agree():
    """The fast read-only path and the differentiable one must not drift: they
    answer the same question by different machinery, so a disagreement means one
    of them is wrong. Measured 0.998/1.812 (sample) against 0.999/1.808
    (query_batch) for do(a=0)/do(a=1) on the fixture."""
    est, fit, _ = _fitted()
    s = torch.zeros(2)
    sweep = est.interventional_sweep(torch.zeros(2, 2), [0, 1], fit)
    for i, a in enumerate((0, 1)):
        looped = float(est.interventional_reward(s, a, fit, n_samples=2048))
        assert abs(looped - float(sweep.value[i])) < 0.1, (a, looped)


def test_the_sweep_is_per_row_and_refuses_a_mismatch():
    est, fit, _ = _fitted(n_ep=60, T=6)
    with pytest.raises(ValueError, match="per-row"):
        est.interventional_sweep(torch.zeros(4, 2), [0, 1], fit)


def test_the_snapshot_is_not_aliased_to_the_live_parameters():
    """The guard's load-bearing detail. state_dict() returns tensors SHARING
    STORAGE with the live parameters, so an in-place step mutates the snapshot
    too and a restore silently reinstates the already-stepped values -- the
    backtrack would accept every step it meant to reject, with nothing raising.
    Verified on this vendored copy: LinearGaussian's _bias read -0.012977, then
    -0.017710 through a naive snapshot after a step, while a deepcopy still read
    -0.012977."""
    est, _, _ = _fitted(n_ep=40, T=6)
    deep = est._snapshot()
    naive = est.model.state_dict()
    key = next(k for k in deep if deep[k].numel() and deep[k].is_floating_point())
    before = float(deep[key].reshape(-1)[0])
    with torch.no_grad():
        for p in est.model.parameters():
            p.add_(torch.ones_like(p))
    assert float(deep[key].reshape(-1)[0]) == before, "deepcopy snapshot was mutated"
    # ...and the naive one demonstrably is, which is why deepcopy is required.
    assert float(naive[key].reshape(-1)[0]) != before

    est._restore(deep)
    live = est.model.state_dict()
    assert abs(float(live[key].reshape(-1)[0]) - before) < 1e-9, "restore failed"


def test_em_is_monotone_under_the_guard():
    """GEM requires only that the M-step INCREASE the objective, so rejecting
    decreases restores the guarantee outright -- and with it the convergence
    test, which cannot distinguish convergence from oscillation if the
    likelihood may fall."""
    _, data = _fixture(n_ep=200, T=10, proxy=True)
    est = LatentClassEstimator(state_dim=2, n_actions=2, proxy_names=("Z",), seed=0)
    fit = est.fit(data, max_iter=8, epochs=30, lr=1e-2)
    assert fit.monotone, fit.log_likelihood
    # The complete end-state set: converged, stationary (including the
    # fixed-point grant from the 2026-08-21 binding audit), a genuinely
    # exhausted line search, or the iteration budget -- the old enumeration
    # (converged/exhausted/n_iter == max_iter) predates both the stationary
    # route and the anneal prefix, so it named outcomes rather than the
    # invariant.
    assert fit.finished or fit.backtrack_exhausted or fit.tau1_budget_bound, fit


def test_an_exhausted_backtrack_budget_stops_rather_than_proceeding():
    """If the budget is spent and the objective still falls, the fit must stop on
    the last good parameters and SAY so, never continue on a decrease."""
    _, data = _fixture(n_ep=60, T=6, proxy=True)
    est = LatentClassEstimator(state_dim=2, n_actions=2, proxy_names=("Z",), seed=0)
    # A wildly oversized step makes every M-step overshoot.
    fit = est.fit(data, max_iter=4, epochs=3, lr=5.0, max_backtracks=1)
    assert fit.monotone, "a decrease survived the guard"
    if fit.backtrack_exhausted:
        assert not fit.converged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_default_proxy_init_runs_on_the_data_device():
    """The proxy init is the production DEFAULT and had never run on GPU.

    Every tensor the initialiser builds must live on the data's device. The
    random branch always did -- it ends in ``.to(device)`` -- which is exactly
    why the omission on the proxy branch stayed hidden: the fallback path was
    device-correct, so nothing in the test suite exercised the default one
    anywhere but CPU. It raises rather than degrading, but only at the first
    line that happens to mix the two."""
    u_ep, data = _fixture(n_ep=40, T=8)
    data = EpisodeData(
        state=data.state.cuda(),
        action=data.action.cuda(),
        reward=data.reward.cuda(),
        episode_ids=data.episode_ids.cuda(),
        proxy={k: v.cuda() for k, v in data.proxy.items()},
    )
    est = LatentClassEstimator(
        state_dim=2, n_actions=2, proxy_names=("Z",), device="cuda", seed=0
    )
    fit = est.fit(data, max_iter=2, epochs=5, init="proxy")
    assert fit.responsibilities.is_cuda
    assert fit.separability() > 0.5


# --------------------------------------------------------------------------
# EM SATURATION ON LONG EPISODES — the failure that made this the critical path.
#
# The E-step sums per-row log-likelihoods over the episode, so a between-class
# difference of d nats per step becomes T*d nats per episode. At T = 500 the
# softmax over classes is a step function: responsibilities are 0/1 after the
# FIRST E-step, and EM is frozen in whatever basin its initialisation picked.
#
# Measured on the real D-D Acrobot arm before the fix: 6 of 6 fits at chance
# recovery (0.53-0.59), every one reporting separability 1.0000. At T = 18-38
# the identical code recovered 0.997-1.000. The grid below is chosen from where
# it actually breaks.
# --------------------------------------------------------------------------


def _long_episode_fixture(n_ep=120, T=500, seed=0):
    """The same generative structure as ``_fixture``, at a chosen episode length.

    U drives BOTH channels, so the latent is identified at any T -- which is the
    point: nothing about the IDENTIFICATION problem changes with episode length,
    only the optimiser's ability to solve it.
    """
    g = torch.Generator().manual_seed(seed)
    u_ep = (torch.rand(n_ep, generator=g) < 0.5).long()
    S, A, R, E = [], [], [], []
    for e in range(n_ep):
        u = int(u_ep[e])
        S.append(torch.randn(T, 2, generator=g))
        a = (torch.rand(T, generator=g) < (0.75 if u else 0.25)).long()
        A.append(a)
        R.append(1.0 + 1.5 * u * a.float() + 0.2 * torch.randn(T, generator=g))
        E.append(torch.full((T,), e, dtype=torch.long))
    return u_ep, EpisodeData(
        state=torch.cat(S),
        action=torch.cat(A),
        reward=torch.cat(R),
        episode_ids=torch.cat(E),
    )


def test_saturation_is_detected_and_reported_at_long_episode_length():
    """(a) The detector. A fit frozen at its initialisation must SAY so.

    Without this the pathology is invisible: the frozen fit reports
    ``separability = 1.0000``, which reads as the cleanest possible result.

    The assertion is on the ORDERING and on the reporting flag, not on a
    magnitude. An earlier version asserted ``> 0.99`` on the strength of two
    observations and the claim that the statistic is bimodal; it measures 0.867
    here, so the statistic is graded and the magnitude is not something to
    hard-code.
    """
    _, short = _long_episode_fixture(n_ep=60, T=12, seed=0)
    _, long_ = _long_episode_fixture(n_ep=60, T=500, seed=0)

    est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    f_short = est.fit(short, max_iter=4, epochs=10, init="random", temperature=1.0)
    est2 = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    f_long = est2.fit(long_, max_iter=4, epochs=10, init="random", temperature=1.0)

    assert f_long.initial_saturation > f_short.initial_saturation, (
        f_short.initial_saturation,
        f_long.initial_saturation,
    )
    assert not f_short.saturated_at_init, f_short.initial_saturation
    assert f_long.saturated_at_init, f_long.initial_saturation
    # And it travels with the number (C3), which is the whole point.
    out = f_long.estimate(torch.tensor(1.0))
    assert out.saturated_at_init
    assert "SATURATED-AT-INIT" in out.label(), out.label()


def test_separability_cannot_distinguish_a_confident_wrong_fit():
    """(c) Why ``separability`` had to be replaced as the correctness diagnostic.

    It is a function of the RESPONSIBILITIES ALONE, so it measures how confident
    the posterior is and nothing about whether the posterior is right. Hand it
    hard 0/1 responsibilities that are completely wrong and it still reads
    1.0000 -- which is exactly what happened on real data: D-D Acrobot at
    T = 500 reported separability 1.0000 at 0.53 recovery, in six fits out of
    six.

    An earlier version of this test asserted that ``separation_per_step`` is
    invariant to episode length. It is not, and should not be: it measures how
    separated the fitted classes are, so a poorly-fitted short-episode run reads
    low for a good reason. What it does NOT do is read perfect for a wrong fit,
    because it is computed from the model's per-step log-likelihoods rather than
    from the posterior's own confidence.
    """
    _, data = _long_episode_fixture(n_ep=40, T=20, seed=0)
    est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    fit = est.fit(data, max_iter=3, epochs=10, init="random")

    n_ep = fit.responsibilities.shape[0]
    # Hard, and deliberately the WRONG way round for half the episodes.
    wrong = torch.zeros(n_ep, 2)
    wrong[: n_ep // 2, 0] = 1.0
    wrong[n_ep // 2 :, 1] = 1.0
    confident_wrong = replace(fit, responsibilities=wrong)

    assert confident_wrong.separability() > 0.999, confident_wrong.separability()
    # ...and the label makes the confidence look like a result, which is the trap.
    assert "sep(telemetry)=1.00" in confident_wrong.estimate(torch.tensor(0.0)).label()
    # The replacement is a different quantity entirely: it does not read the
    # responsibilities, so hand-forging them cannot move it.
    assert confident_wrong.separation_per_step == fit.separation_per_step


def test_a_temperature_change_does_not_exhaust_the_backtrack_guard():
    """(b), the way this fix could have failed SILENTLY.

    The guard compares the objective before and after the M-step. At a
    temperature change those are DIFFERENT FUNCTIONS -- an objective at tau = 500
    against one at tau = 63 is not a comparison -- so the first annealing step
    read as a catastrophic decrease, the backtrack budget was exhausted on
    iteration 1, and the fit stopped before ever reaching tau = 1. Annealing
    present in the code and absent in effect, reported as a normal fit.

    Measured before the fix: 3 backtracks at iteration 1, tau stuck at 63.0.
    """
    _, data = _long_episode_fixture(n_ep=40, T=500, seed=0)
    est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    fit = est.fit(data, max_iter=6, epochs=10, init="random")
    assert fit.n_anneal > 0, fit.temperatures
    assert fit.temperatures[0] > 1.0, fit.temperatures
    assert fit.reached_tau_one, fit.temperatures
    assert fit.n_iter > fit.n_anneal, (fit.n_iter, fit.n_anneal)


def test_a_fit_that_stops_while_tempered_says_so():
    """A fit that stops mid-anneal is not an estimate of anything -- its last
    objective is a smoothed surrogate, not the likelihood -- and it reads exactly
    like a converged fit unless asked. It happens at small iteration budgets,
    where the backtrack guard exhausts before the schedule flattens."""
    _, data = _long_episode_fixture(n_ep=40, T=500, seed=0)
    est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    # n_anneal deliberately longer than the fit can complete.
    fit = est.fit(data, max_iter=1, epochs=5, init="random", n_anneal=40)
    if not fit.reached_tau_one:
        out = fit.estimate(torch.tensor(1.0))
        assert not out.reached_tau_one
        assert "STOPPED-WHILE-TEMPERED" in out.label(), out.label()


def test_the_monotone_guard_reads_the_objective_the_m_step_maximises():
    """(b), the part that would silently undo the fix.

    During annealing the M-step maximises the TEMPERED objective, and the true
    likelihood may legitimately fall while it does. A guard that judged those
    steps by the untempered likelihood would reject exactly the moves the anneal
    exists to make -- annealing would be present in the code and absent in
    effect. ``monotone`` is therefore evaluated only over the tau = 1 phase.
    """
    _, data = _long_episode_fixture(n_ep=60, T=500, seed=0)
    est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    fit = est.fit(data, max_iter=6, epochs=10, init="random")
    assert fit.n_anneal > 0, "episodic data must trigger the anneal"
    assert fit.temperatures[0] > 1.0
    # monotone judges the tau = 1 tail only; the annealed prefix is excluded by
    # design, so a legitimate anneal cannot be reported as non-monotone.
    assert isinstance(fit.monotone, bool)
    assert fit.n_anneal == sum(1 for t in fit.temperatures if t > 1.0)


def test_the_reward_mechanism_type_is_resolved_from_the_data():
    """R IS NOT CONTINUOUS ON THESE ARMS, and fitting an MDN to it is a
    modelling error whose symptom is the ``min_scale`` floor.

    Every arm in this benchmark has a reward that is deterministic given
    ``(S, A, U)`` -- CartPole pays 1 (+ the gated bonus), Acrobot -1 (+ the
    bonus) -- so R has finite support and no conditional noise. Measured on the
    real CartPole D-D arm: support exactly ``{1.0, 2.0}``, 2 distinct values
    over 5315 rows, and the MDN drove its scale onto the floor across every row.

    That is not cosmetic. Both calibration layers are likelihood-based, so a
    reward log-density pinned at an arbitrary floor puts ``min_scale`` inside
    L4's compatible set and L5's likelihood ratios.

    The type is DERIVED, not assumed: a finite-support variable's support does
    not grow when you look at more of it. The test covers both directions,
    because a rule that only ever answers one way is not a rule.
    """
    g = torch.Generator().manual_seed(0)
    n_ep, T = 40, 10
    ep = torch.arange(n_ep).repeat_interleave(T)
    state = torch.randn(n_ep * T, 2, generator=g)
    action = (torch.rand(n_ep * T, generator=g) < 0.5).long()

    def make(reward):
        return EpisodeData(state=state, action=action, reward=reward, episode_ids=ep)

    # Finite support -> categorical.
    discrete = make(1.0 + action.float())
    est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    est.fit(discrete, max_iter=1, epochs=3, init="random")
    assert (
        est.resolved_reward_mechanism == "categorical[2]"
    ), est.resolved_reward_mechanism
    assert est._reward_levels.tolist() == [1.0, 2.0]

    # Genuinely continuous -> MDN, so the rule is not just "always discrete".
    cont = make(1.0 + action.float() + 0.3 * torch.randn(n_ep * T, generator=g))
    est2 = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    est2.fit(cont, max_iter=1, epochs=3, init="random")
    assert est2.resolved_reward_mechanism == "mdn", est2.resolved_reward_mechanism

    # And the resolved type rides on every number (C3): a likelihood read
    # without knowing which mechanism produced it is not comparable to one read
    # the other way.
    out = est.fit(discrete, max_iter=1, epochs=3, init="random").estimate(
        torch.tensor(0.0)
    )
    assert "R=categorical[2]" in out.label(), out.label()


def test_a_discrete_reward_is_decoded_back_into_reward_units():
    """The interventional paths must return E[R], not E[class index].

    With a categorical R the model emits indices; averaging them and reporting
    the result would be a plausible number in the wrong units. Ground truth on
    this fixture is r = 10 + 10*a, so do(a=1) - do(a=0) = 10 -- a scale that no
    index average could produce by accident.
    """
    g = torch.Generator().manual_seed(0)
    n_ep, T = 60, 10
    ep = torch.arange(n_ep).repeat_interleave(T)
    action = (torch.rand(n_ep * T, generator=g) < 0.5).long()
    data = EpisodeData(
        state=torch.randn(n_ep * T, 2, generator=g),
        action=action,
        reward=10.0 + 10.0 * action.float(),
        episode_ids=ep,
    )
    est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    fit = est.fit(data, max_iter=3, epochs=20, init="random")
    assert est.resolved_reward_mechanism == "categorical[2]"
    v0 = float(est.interventional_sweep(torch.zeros(4, 2), [0] * 4, fit).value.mean())
    v1 = float(est.interventional_sweep(torch.zeros(4, 2), [1] * 4, fit).value.mean())
    # In reward units both sit in [10, 20]; as indices they would sit in [0, 1].
    assert 9.0 < v0 < 21.0, v0
    assert 9.0 < v1 < 21.0, v1
    assert v1 > v0, (v0, v1)


def test_lr_at_theta_hat_is_zero():
    """The V4 walk regression, pinned as an INVARIANT: LR(theta-hat) == 0.

    L4's LR constraint and ``fit.final_ll`` must be the SAME functional. V4's
    entire bounds block failed because a second implementation that "mirrored"
    e_step did not mirror it (per-row U term + S marginal), putting
    LR(theta-hat) = 70,686 against c = 821 on the frozen row -- every walk
    started far outside its own region and the fallback returned width 0.
    Both paths now call ``_episode_log_liks`` (one construction site); this
    test pins the invariant rather than the implementation, and would have
    caught the bug in one line. WITH proxies, deliberately: the proxy channel
    is where a second divergence would land. Also asserts the constraint
    still carries gradients (the other way this could silently break).
    """
    import copy

    from src.rl.offline.grace.l4 import _observed_ll_differentiable

    _, data = _fixture(n_ep=60, T=6)
    est = LatentClassEstimator(state_dim=2, n_actions=2, proxy_names=("Z",), seed=0)
    fit = est.fit(data, max_iter=4, epochs=30, init="proxy")

    model_c = copy.deepcopy(est.model)
    for p in model_c.parameters():
        p.requires_grad_(True)
    prior_logits = torch.nn.Parameter(torch.log(fit.prior.detach().clamp_min(1e-8)))
    ll = _observed_ll_differentiable(model_c, prior_logits, est, data)
    lr = 2.0 * (float(fit.final_ll) - float(ll.detach()))
    assert abs(lr) < 1e-2, f"LR(theta-hat) = {lr}, invariant says 0"
    # the constraint must remain differentiable in the clone's parameters
    ll.backward()
    grads = [p.grad for p in model_c.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    assert prior_logits.grad is not None


def test_replicate_pinning_prevents_mechanism_class_flip():
    """The symmetry rule applied to the model CLASS (the V4 s1 cluster).

    Re-resolving the reward type per replicate let a resample that dropped a
    rare level fit a DIFFERENT mechanism than the observed fit. The scenario
    below is the exact replicate case: the observed data resolves
    categorical[2]; a resample missing the rare level would resolve dirac
    unpinned. Pinned, the replicate keeps the observed class.
    """
    g = torch.Generator().manual_seed(0)
    n_ep, T = 20, 6
    ep = torch.arange(n_ep).repeat_interleave(T)
    action = (torch.rand(n_ep * T, generator=g) < 0.5).long()
    reward = torch.ones(n_ep * T)
    # the rare level, in an episode the stratified half INCLUDES so the
    # observed data resolves categorical (mirrors the criterion's own split)
    rare_ep = int(torch.randperm(n_ep, generator=torch.Generator().manual_seed(0))[0])
    reward[rare_ep * T] = 2.0
    observed = EpisodeData(
        state=torch.randn(n_ep * T, 2, generator=g),
        action=action,
        reward=reward,
        episode_ids=ep,
    )
    src = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    src._resolve_reward_type(observed)
    assert src.resolved_reward_mechanism == "categorical[2]"

    keep = ep != rare_ep  # the rare-level episode was not drawn
    resample = EpisodeData(
        state=observed.state[keep],
        action=observed.action[keep],
        reward=observed.reward[keep],
        episode_ids=observed.episode_ids[keep],
    )
    control = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    control._resolve_reward_type(resample)
    assert control.resolved_reward_mechanism.startswith("dirac"), (
        "the control must flip, or this test is vacuous: "
        f"{control.resolved_reward_mechanism}"
    )

    pinned = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    pinned.pin_reward_resolution(src)
    pinned._resolve_reward_type(resample)
    assert pinned.resolved_reward_mechanism == "categorical[2]"
    assert pinned._reward_levels.tolist() == src._reward_levels.tolist()

    unfitted = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
    try:
        LatentClassEstimator(state_dim=2, n_actions=2, seed=1).pin_reward_resolution(
            unfitted
        )
        raise AssertionError("pinning from an unresolved source must raise")
    except ValueError:
        pass


def test_reward_resolution_half_is_stratified_by_episode_not_row_order():
    """The half-sample keys on episode membership, not data order.

    Pre-fix, the half was the FIRST n//2 rows, so a two-valued reward whose
    second level sat only in the LAST episode always read "support grows" and
    resolved continuous -- deterministically wrong on finite support. Post-fix
    the half is a fixed-seed random half of episodes: a rare level in an
    IN-HALF episode resolves categorical no matter where that episode sits in
    data order. (A rare level in an out-of-half episode still resolves
    continuous -- the criterion's inherent coin-flip, erring in the documented
    direction -- which is what replicate PINNING neutralises where it
    matters.)
    """
    g = torch.Generator().manual_seed(0)
    n_ep, T = 20, 6
    ep = torch.arange(n_ep).repeat_interleave(T)

    # which episodes the fixed-seed split puts in the half, per the criterion
    perm = torch.randperm(n_ep, generator=torch.Generator().manual_seed(0))
    in_half = int(perm[0])

    def resolve(rare_ep):
        reward = torch.ones(n_ep * T)
        reward[rare_ep * T] = 2.0
        est = LatentClassEstimator(state_dim=2, n_actions=2, seed=0)
        est._resolve_reward_type(
            EpisodeData(
                state=torch.randn(n_ep * T, 2, generator=g),
                action=(torch.rand(n_ep * T, generator=g) < 0.5).long(),
                reward=reward,
                episode_ids=ep,
            )
        )
        return est.resolved_reward_mechanism

    # rare level in an in-half episode -> categorical, wherever it sits in
    # data order (the last episode included, which the old row-half ALWAYS
    # misread whenever it was not literally in the first half of rows)
    assert resolve(in_half) == "categorical[2]"
    out_half = int(perm[-1])
    assert resolve(out_half) == "mdn"  # the documented conservative direction


def test_episode_constant_channels_enter_once_per_episode():
    """S1c pinned at the likelihood: lengthening an episode must not
    re-multiply its proxy draw.

    The proxies are drawn ONCE per episode from p(.|U); summing them per row
    implements p(Z|U)^T -- a model the declared diagram does not describe
    (A1), and measured at 78-90 nats against R's 39.7 before the fix. The
    test doubles every episode's rows, which doubles the per-step channels
    (A, R) exactly and must leave the proxy term untouched: ll(2T) = 2*S + P,
    ll(T) = S + P, so 2*ll(T) - ll(2T) == P. Pre-fix that identity fails by
    a factor of T.
    """
    _, data = _fixture(n_ep=40, T=5)
    est = LatentClassEstimator(state_dim=2, n_actions=2, proxy_names=("Z",), seed=0)
    fit = est.fit(data, max_iter=3, epochs=20, init="proxy")

    dup = EpisodeData(  # every episode's rows doubled, same proxy draw
        state=data.state.repeat_interleave(2, dim=0),
        action=data.action.repeat_interleave(2),
        reward=data.reward.repeat_interleave(2),
        episode_ids=data.episode_ids.repeat_interleave(2),
        proxy={k: v.repeat_interleave(2) for k, v in data.proxy.items()},
    )
    ll1 = est._episode_log_liks(data)
    ll2 = est._episode_log_liks(dup)
    implied_proxy = 2.0 * ll1 - ll2  # == P if and only if P entered once

    # P read directly off the mechanism, one row per episode
    with torch.no_grad():
        cols = []
        for k in range(est.u_card):
            u_k = torch.full((data.n,), k, dtype=torch.long)
            per_node = est.model.log_prob(est._frame(data, u_k), per_node=True)
            cols.append(data.first_rows(per_node["Z"].reshape(-1)))
        direct_proxy = torch.stack(cols, dim=1)

    assert torch.allclose(implied_proxy, direct_proxy, atol=1e-2), (
        implied_proxy[:3],
        direct_proxy[:3],
    )
    # and the per-step channels DID double, or the test proves nothing
    assert not torch.allclose(ll1, ll2, atol=1.0), (ll1[:3], ll2[:3])


def test_a_within_episode_varying_proxy_is_refused():
    """The guard behind the reduction: a per-step proxy must not be routed
    through the episode-constant channel (D-B's lagged construction is a
    different channel, and taking row 0 of it would be silently wrong)."""
    g = torch.Generator().manual_seed(0)
    n_ep, T = 10, 4
    ep = torch.arange(n_ep).repeat_interleave(T)
    kw = dict(
        state=torch.randn(n_ep * T, 2, generator=g),
        action=(torch.rand(n_ep * T, generator=g) < 0.5).long(),
        reward=torch.randn(n_ep * T, generator=g),
        episode_ids=ep,
    )
    EpisodeData(proxy={"Z": torch.arange(n_ep).float().repeat_interleave(T)}, **kw)
    try:
        EpisodeData(proxy={"Z": torch.arange(n_ep * T).float()}, **kw)
        raise AssertionError("a within-episode-varying proxy must be refused")
    except ValueError as e:
        assert "WITHIN an episode" in str(e)

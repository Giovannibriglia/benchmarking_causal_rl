"""L3 — the latent-class estimator, checked against GROUND TRUTH.

Direction as everywhere else: the estimator is validated against a fixture whose
latent is KNOWN, never against another estimator or against L5.
"""

from __future__ import annotations

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
    assert len(fit.log_likelihood) == fit.n_iter


def test_episode_data_refuses_misaligned_arrays():
    with pytest.raises(ValueError, match="transition-aligned"):
        EpisodeData(
            state=torch.zeros(10, 2),
            action=torch.zeros(9, dtype=torch.long),
            reward=torch.zeros(10),
            episode_ids=torch.zeros(10, dtype=torch.long),
        )


def _fitted(n_ep=200, T=10, seed=0):
    u_ep, data = _fixture(n_ep=n_ep, T=T, seed=seed, proxy=False)
    est = LatentClassEstimator(state_dim=2, n_actions=2, u_card=2, seed=seed)
    fit = est.fit(data, max_iter=6, epochs=40)
    return est, fit, data


def test_interventional_reward_recovers_the_known_do_effect():
    """Ground truth on the fixture is r = 1 + 1.5*u*a, so under do(a) the
    U-marginal effect is E[R|do(1)] - E[R|do(0)] = 1.5 * P(U=1) = 0.75."""
    est, fit, _ = _fitted()
    s = torch.zeros(2)
    v0 = est.interventional_reward(s, 0, fit, n_samples=1024)
    v1 = est.interventional_reward(s, 1, fit, n_samples=1024)
    assert abs(float(v0) - 1.0) < 0.15, float(v0)
    assert 0.4 < float(v1) - float(v0) < 1.1, (float(v0), float(v1))
    # C3: every estimate carries its conditions.
    assert isinstance(v0.monotone, bool) and "sep=" in v0.label()


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
    assert fit.converged or fit.backtrack_exhausted or fit.n_iter == 8


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

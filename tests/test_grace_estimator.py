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

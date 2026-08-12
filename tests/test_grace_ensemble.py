"""§D.6 — ensemble plumbing: the K-member bootstrap yields a nonzero, finite
interval on confounded data and a near-zero interval on clean deterministic
data."""

from __future__ import annotations

import torch
from src.rl.offline.grace import cell_graph, GraceMachinery, GraceOptions
from tests._grace_test_utils import DEV, FakeSeqBuffer, make_confounded_episodes


def _width(confounded: bool, c_r: float, state_reward: bool) -> float:
    eps, _ = make_confounded_episodes(
        n_ep=250,
        t_len=12,
        confounded=confounded,
        c_r=c_r,
        seed=6,
        state_reward=state_reward,
    )
    m = GraceMachinery(
        cell_graph("mdp", "template"),
        GraceOptions(n_bins=3, em_iters=6, ensemble_k=5, router=False),
        n_actions=2,
        device=DEV,
        gamma=0.5,
    )
    m.fit_from_buffer(FakeSeqBuffer(eps))
    assert torch.isfinite(m.q_lo).all() and torch.isfinite(m.q_hi).all()
    assert bool((m.q_hi >= m.q_lo - 1e-6).all())
    visited = m.visited_s.to(m.q_hi.device)
    return float((m.q_hi - m.q_lo)[visited].mean())


def test_interval_wider_under_confounding_near_zero_on_clean_data():
    # Clean rung: HOMOGENEOUS rewards (no state term, no U term) — the
    # bootstrap members then agree up to count noise and the interval must be
    # near-zero. Confounded rung: real latent heterogeneity -> nonzero width.
    w_conf = _width(confounded=True, c_r=1.0, state_reward=True)
    w_clean = _width(confounded=False, c_r=0.0, state_reward=False)
    assert w_conf > 0.02, w_conf
    assert w_clean < 1e-3, w_clean
    assert w_conf > 10.0 * (w_clean + 1e-6), (w_conf, w_clean)

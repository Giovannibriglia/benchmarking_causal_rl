"""fix/confounded-wrapper-rng (issue #36) — isolated-RNG reproducibility.

ConfoundedCollectionWrapper used to sample its latent U on the GLOBAL torch RNG,
so a freshly generated confounded dataset's gate-test signature depended on
cumulative process RNG state. The fix gives the wrapper an isolated per-instance
torch.Generator, seeded at construction (the offline GENERATE path threads
generate_offline_dataset's seed into it).

These are WRAPPER-LEVEL tests of the U stream — exactly issue #36's acceptance
criterion ("two wrappers with the same seed produce byte-identical rollouts
regardless of intervening torch.rand()"). They deliberately do NOT assert on a
freshly *generated dataset's* gate metadata: at σ<1 the confounded ACTION (the
agent's action + the mixture coin in ConfoundedBehaviorPolicy) also draws on the
global stream — outside this wrapper's scope — and at σ=1 the marginal is
degenerate (action == U, reward == base + U ⇒ Corr ≡ 1 for any seed). Isolating
U here is the property the wrapper fix actually owns.
"""

from __future__ import annotations

import torch
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper


class _ToyVecEnv:
    """Minimal vector env exposing only what the wrapper touches. ``step`` reports
    all sub-envs done, so each step (like each reset) drives a fresh U sample."""

    def __init__(self, n_envs=8):
        self.n_envs = n_envs
        self.device = torch.device("cpu")

    def reset(self, seed=None):
        return torch.zeros(self.n_envs, 2), {}

    def step(self, action):
        n = self.n_envs
        return (
            torch.zeros(n, 2),
            torch.zeros(n),
            torch.zeros(n, dtype=torch.bool),
            torch.ones(n, dtype=torch.bool),  # all done -> U resamples
            {},
        )


def _u_sequence(seed, k=25, u_dist="bernoulli"):
    """The sequence of latent U draws: the construction draw plus one per reset."""
    w = ConfoundedCollectionWrapper(
        _ToyVecEnv(), c_a=1.0, c_r=1.0, u_dist=u_dist, seed=seed
    )
    seq = [w.current_u.clone()]
    for _ in range(k):
        _, info = w.reset()
        seq.append(info["confounder_u"].clone())
    return torch.stack(seq)


def test_wrapper_u_reproducible_same_seed():
    """Same seed -> identical U stream, even with the global RNG perturbed between
    the two constructions. The core property the isolated generator enables."""
    torch.manual_seed(123)
    s1 = _u_sequence(7)
    torch.rand(100)  # perturb the global stream between constructions
    s2 = _u_sequence(7)
    assert torch.equal(s1, s2)

    # Same guarantee for the normal-U variant (covers both _sample_u branches).
    torch.manual_seed(5)
    n1 = _u_sequence(7, u_dist="normal")
    torch.rand(100)
    n2 = _u_sequence(7, u_dist="normal")
    assert torch.equal(n1, n2)


def test_wrapper_u_differs_by_seed():
    """Different seeds -> different U streams, confirming the seed is actually
    consumed (not a no-op param)."""
    assert not torch.equal(_u_sequence(1), _u_sequence(2))
    assert not torch.equal(
        _u_sequence(1, u_dist="normal"), _u_sequence(2, u_dist="normal")
    )


def test_wrapper_seed_none_keeps_global_stream():
    """Back-compat: seed=None leaves U on the GLOBAL stream (the runner's online
    A1 path, which must stay byte-identical / run-seed-reproducible). So it is
    governed by the global seed, not isolated from it."""
    torch.manual_seed(42)
    g1 = _u_sequence(None)
    torch.manual_seed(42)
    g2 = _u_sequence(None)
    assert torch.equal(g1, g2)  # reproducible via the global seed (pre-#36 behavior)

    torch.manual_seed(43)
    g3 = _u_sequence(None)
    assert not torch.equal(g1, g3)  # and the global seed governs it

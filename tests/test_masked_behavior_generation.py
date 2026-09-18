"""Masked-behaviour generation — the information-set pieces, pinned.

Finding 1 (2026-09-02): masking at LOAD leaves the logged actions dependent
on hidden state (S->A); the true-POMDP row needs the behavior policy to ACT
on the masked view (O->A). These tests pin the wrapper's view semantics, the
mirrored absence of diagnostic reads, and the S6 identity suffix.
"""

from __future__ import annotations

import numpy as np
import torch
from src.envs.offline.generate import _MaskedViewPolicy, dataset_name


class _Inner:
    def __init__(self):
        self.seen = None
        self.a_bad = 1

    def act(self, obs):
        self.seen = obs
        return "acted"

    def action_probs(self, obs):
        return obs  # echo, so the test sees the view it received


class _InnerNoProbs:
    def act(self, obs):
        return "acted"


def test_view_deletes_masked_columns_numpy_and_torch():
    p = _MaskedViewPolicy(_Inner(), (1, 3))
    p.act(np.arange(8.0).reshape(2, 4))
    assert p._inner.seen.shape == (2, 2)
    assert np.allclose(p._inner.seen, [[0.0, 2.0], [4.0, 6.0]])
    p.act(torch.arange(8.0).reshape(2, 4))
    assert tuple(p._inner.seen.shape) == (2, 2)


def test_prob_reads_are_forwarded_through_the_view_and_absence_mirrored():
    p = _MaskedViewPolicy(_Inner(), (1,))
    out = p.action_probs(np.array([[1.0, 2.0, 3.0]]))
    assert out.shape == (1, 2)  # the masked view, not the full obs
    assert p.a_bad == 1  # plain attributes delegate
    # The rollout loops probe with getattr(..., None): an inner without the
    # method must read as ABSENT, never as a wrapper that returns None.
    q = _MaskedViewPolicy(_InnerNoProbs(), (1,))
    assert getattr(q, "action_probs", None) is None
    assert getattr(q, "_base_action_probs", None) is None


def test_dataset_name_carries_the_mask_identity():
    base = dataset_name("CartPole-v1", "medium", "bias_confounded_action", 0.25)
    masked = dataset_name(
        "CartPole-v1",
        "medium",
        "bias_confounded_action",
        0.25,
        behavior_mask_indices=(1, 3),
    )
    assert base != masked and masked.endswith("-om13-v0")
    # inert when off (S6: the historical ids are untouched)
    assert base == dataset_name("CartPole-v1", "medium", "bias_confounded_action", 0.25)


def test_build_generator_agent_is_masked_dim_when_behaviour_is_masked(monkeypatch):
    """The generator trained on the masked view must be REBUILT masked-dim,
    or its checkpoint cannot load (first execution, 2026-09-03: state-dict
    size mismatch [64, 2] vs [64, 4]). Pinned without training: the tier
    'random' path skips training and returns the built agent."""
    import src.benchmarking.registry as reg_mod
    from src.benchmarking.registry import register_default_algorithms
    from src.envs.offline import generate as g

    register_default_algorithms()
    seen = {}
    real_get = reg_mod.registry.get

    class _Entry:
        def __init__(self, entry):
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def builder(self, **kw):
            seen.update(kw)
            return self._entry.builder(**kw)

    monkeypatch.setattr(reg_mod.registry, "get", lambda algo: _Entry(real_get(algo)))
    g.build_generator_agent("CartPole-v1", "dqn", "random", seed=0, device="cpu")
    full = seen["obs_dim"]
    g.build_generator_agent(
        "CartPole-v1",
        "dqn",
        "random",
        seed=0,
        device="cpu",
        behavior_mask_indices=(1, 3),
    )
    assert seen["obs_dim"] == full - 2 and seen["obs_shape"] == (full - 2,)

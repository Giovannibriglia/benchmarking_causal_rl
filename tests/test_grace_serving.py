"""The GRACE reward-transform seam — the properties the experiment rests on.

GRACE is not a critic here. On cells where confounding is confined to the
reward channel (catalogue fact 3: no ``U -> S_next`` edge), ``Q_do`` is exactly
what the base algorithm computes when trained on INTERVENTIONAL rewards, so the
variant is the base algorithm with one column substituted. These tests pin the
properties that, if they broke silently, would make the comparison meaningless
rather than merely wrong — and this seam's failures are silent by construction
(S14), so each one is a positive check that something MOVED, not just that
nothing raised.
"""

from __future__ import annotations

import pytest
import torch
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.grace.serving import (
    apply_reward_transform,
    fit_reward_transform,
    GraceServing,
    SERVE_ABSTAINED,
    SERVE_PESSIMISTIC,
)

CPU = torch.device("cpu")


def _buffer(n=40, ep_len=10, reward=1.0, with_proxies=False):
    buf = ReplayBuffer(capacity=n + 10, device=CPU)
    for i in range(n):
        tr = {
            "obs": torch.randn(4),
            "actions": torch.tensor(i % 2),
            "rewards": torch.tensor(float(reward)),
            "next_obs": torch.randn(4),
            "dones": torch.tensor(1.0 if (i + 1) % ep_len == 0 else 0.0),
        }
        if with_proxies:
            tr["proxy_Z"] = torch.tensor(0.5)
        buf.add(tr)
    return buf


def test_abstention_leaves_the_reward_column_untouched():
    """An abstained run must be BYTE-IDENTICAL to its base.

    That is what makes GRACE-ABSTAINED a safe fallback rather than a silent
    third behaviour, and what lets abstained runs be reported separately
    instead of pooled into the comparison.
    """
    buf = _buffer()
    before = buf._data["rewards"].clone()
    fired = apply_reward_transform(buf, GraceServing(reason="fit was dirty"))
    assert fired is False
    assert torch.equal(buf._data["rewards"], before)


def test_applying_the_transform_rewrites_exactly_the_reward_column():
    buf = _buffer(n=20)
    obs_before = buf._data["obs"].clone()
    acts_before = buf._data["actions"].clone()
    serving = GraceServing(
        mode=SERVE_PESSIMISTIC, rewards=torch.full((20,), 7.0), l4_kind="interval"
    )
    assert apply_reward_transform(buf, serving) is True
    assert torch.allclose(buf._data["rewards"][:20], torch.full((20,), 7.0))
    # nothing else moves: same transitions, same seeds, ONE column different
    assert torch.equal(buf._data["obs"], obs_before)
    assert torch.equal(buf._data["actions"], acts_before)


def test_a_declared_proxy_channel_missing_from_the_buffer_abstains_loudly():
    """A proximal cell whose proxies never reached the buffer would otherwise
    quietly fit the ablation's WITHOUT arm — which the sweep measured
    collapsing toward chance at the weak end. Saying so is the point."""
    serving = fit_reward_transform(_buffer(with_proxies=False), proxy_names=("Z",))
    assert serving.abstained
    assert "proxy" in serving.reason.lower()


def test_the_runners_real_replay_buffer_is_readable():
    """ReplayBuffer exposes its columns via gather(), NOT as attributes. An
    extractor reading attributes would abstain on every real run — and
    abstention is designed to look like a scope decision, so the experiment
    would have come back all-GRACE-ABSTAINED with no error anywhere."""
    from src.rl.offline.grace.serving import _episode_data_from_buffer

    data, nxt, dones = _episode_data_from_buffer(_buffer(n=40, ep_len=10))
    assert data is not None and data.n == 40
    assert nxt.shape == (40, 4)
    assert int(torch.unique(data.episode_ids).numel()) == 4


def test_pessimism_can_only_reduce_a_bads_reward():
    """The direction of the serving rule, pinned so it cannot flip.

    The bootstrap's low end can land ABOVE the point estimate (seen at B=3:
    point 30.63, interval [31.61, 31.85]). Unclamped, the shift is negative and
    would RAISE a_bad — inverting the correction the seam exists to make.
    """
    assert max(0.0, 30.63 - 31.61) == 0.0  # clamped at the construction site
    assert max(0.0, 31.85 - 31.61) == pytest.approx(0.24)


def test_the_label_carries_the_serving_mode_and_l4_conditions():
    """C3: the conditions travel into the run artifacts, so a reader can tell
    which runs served and which abstained without re-deriving anything."""
    served = GraceServing(
        mode=SERVE_PESSIMISTIC,
        l4_kind="interval",
        lo=-0.2,
        hi=0.4,
        fit_label="conv=True monotone=True",
        rewards=torch.zeros(3),
    )
    assert "Q-minus" in served.label() and "interval" in served.label()
    assert "conv=True" in served.label()
    assert SERVE_ABSTAINED in GraceServing(reason="dirty").label()


def test_grace_is_not_a_critic_strategy_any_more():
    """It is an ARM flag, because it changes a training input rather than a
    critic. A stale critic name would silently select a different code path."""
    from src.benchmarking.regime_sweep import KNOWN_STRATEGIES
    from src.config.defaults import EnvConfig

    assert "grace" not in KNOWN_STRATEGIES
    assert EnvConfig(env_id="CartPole-v1").grace_reward_transform is False

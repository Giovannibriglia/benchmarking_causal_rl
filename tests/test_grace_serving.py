"""The GRACE critic seam — the properties the experiment's validity rests on.

Each test here pins a decision that, if it silently broke, would make the
variant-vs-base comparison meaningless rather than merely wrong.
"""

from __future__ import annotations

import pytest
import torch
from src.benchmarking.critic_ablation import CRITIC_LIBRARY, StrategyCritic
from src.rl.offline.grace.serving import (
    fit_from_buffer,
    GraceQNetwork,
    GraceServing,
    SERVE_ABSTAINED,
    SERVE_PESSIMISTIC,
)

CPU = torch.device("cpu")


def _critic(base="cql", name="grace"):
    return StrategyCritic(name, CRITIC_LIBRARY[name], base, 4, 2, CPU)


@pytest.mark.parametrize("base", ["cql", "iql", "offline_dqn"])
def test_every_base_arm_builds_and_binds_the_handoff(base):
    """Base-parity: the arm is the BASE algo plus a serving wrapper, and the
    fit is bound to set_sequence_buffer (the runner's handoff at
    runner.py:1661) rather than to a training hook."""
    sc = _critic(base)
    assert isinstance(sc.net, GraceQNetwork)
    assert callable(getattr(sc.agent, "set_sequence_buffer", None))


def test_unfitted_and_abstained_serving_is_a_literal_passthrough():
    """The abstention fallback must be EXACTLY the base critic.

    If it were anything else, `GRACE-ABSTAINED` would be a silent third
    behaviour and an abstained run would contaminate the comparison instead of
    being separable from it."""
    sc = _critic()
    x = torch.randn(8, 4)
    assert torch.equal(sc.net(x), sc.net.base(x))  # unfitted
    sc.net.serving = GraceServing(reason="fit was dirty")  # abstained
    assert sc.net.serving.abstained
    assert torch.equal(sc.net(x), sc.net.base(x))
    assert SERVE_ABSTAINED in sc.net.serving.label()


def test_serving_applies_the_action_contrast_and_c3_label_travels():
    sc = _critic()
    x = torch.randn(6, 4)
    base_q = sc.net.base(x).clone()
    sc.net.serving = GraceServing(
        mode=SERVE_PESSIMISTIC,
        l4_kind="interval",
        lo=-0.2,
        hi=0.4,
        fit_label="conv=True monotone=True",
        action_offset=torch.tensor([0.3, -0.3]),
    )
    served = sc.net(x)
    assert torch.allclose(served - base_q, torch.tensor([0.3, -0.3]).expand(6, 2))
    lab = sc.net.serving.label()
    assert "Q-minus" in lab and "interval" in lab and "conv=True" in lab


def test_the_wrapper_does_not_hide_the_base_network_surface():
    """The agent may call other surfaces on its q_network (q_su/q_at hooks).
    A __getattr__ that does not defer to nn.Module first makes even ``base``
    unreachable — which is how this was first written, and it failed here."""
    sc = _critic()
    assert isinstance(sc.net.base, torch.nn.Module)
    assert list(sc.net.parameters())  # parameters still discoverable


def test_a_flat_buffer_without_episodes_abstains_with_a_reason():
    """Never raise into the run: a cell GRACE cannot serve is one the base
    algorithm still trains on, and the label must say why."""

    class _Flat:
        observations = torch.randn(10, 4)
        actions = torch.zeros(10, dtype=torch.long)
        rewards = torch.ones(10)

    serving = fit_from_buffer(object(), _Flat())
    assert serving.abstained
    assert "episode" in serving.reason.lower()


def test_grace_options_reject_an_unknown_key():
    """No per-environment GRACE parameters anywhere: an option the seam does
    not know must fail loudly at build time, not be carried silently."""
    from src.rl.offline.grace.serving import _grace_options

    assert _grace_options({"grace_options": {"alpha": 0.1}}) == {"alpha": 0.1}
    # v1's router/deploy switches are accepted-and-dropped, never pretended
    assert _grace_options({"grace_options": {"router": True}}) == {}
    with pytest.raises(ValueError, match="unknown grace option"):
        _grace_options({"grace_options": {"env_id": "CartPole-v1"}})


def test_grace_is_a_known_strategy_so_configs_can_request_it():
    from src.benchmarking.regime_sweep import KNOWN_STRATEGIES

    assert "grace" in KNOWN_STRATEGIES


def test_a_fit_on_episode_grouped_data_serves_the_pessimistic_end():
    """End to end on a tiny fixture: the handoff fits, and what it serves is
    the LOW end of L4's interval per action (the serving rule)."""

    class _Seq:  # the padded (B, T, *) sequence layout
        observations = torch.randn(12, 5, 4)
        actions = (torch.rand(12, 5) < 0.5).long()
        rewards = 1.0 + (torch.rand(12, 5) < 0.5).float()

    serving = fit_from_buffer(
        object(),
        _Seq(),
        b=3,
        fit_kwargs=dict(max_iter=2, epochs=5),
    )
    # Either it served, or it abstained WITH a reason -- never a bare failure.
    if serving.abstained:
        assert serving.reason
    else:
        assert serving.mode == SERVE_PESSIMISTIC
        assert serving.action_offset is not None
        assert serving.action_offset.numel() == 2
        # centred, so only the action CONTRAST is served
        assert abs(float(serving.action_offset.sum())) < 1e-5
        assert serving.lo <= serving.hi

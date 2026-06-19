"""feat/off-policy-recurrent-integration — independent target-net hidden state.

The recurrent learn paths forward the target net over next_obs with its OWN
zero-init hidden state (separate forward call, never reusing the online net's
state). This pins that the two nets are independent: with diverged weights the
same input yields different outputs/hidden states.
"""

from __future__ import annotations

import torch
from src.benchmarking.registry import register_default_algorithms, registry

register_default_algorithms()
DEV = torch.device("cpu")


def test_dqn_target_net_forwards_independently():
    _, dqn = registry.get("dqn").builder(
        obs_dim=3,
        action_dim=2,
        action_type="discrete",
        device=DEV,
        action_space=None,
        obs_shape=(3,),
        critic_network="lstm",
    )
    # At init the target is a copy of online -> identical outputs (shared init).
    seq = torch.randn(2, 5, 3)
    with torch.no_grad():
        online0, _ = dqn.q_network(seq)
        target0, _ = dqn.target_network(seq)
    assert torch.allclose(online0, target0)

    # Diverge the target's weights; the same sequence now yields different
    # outputs -> the target builds its OWN hidden state from its OWN params,
    # independent of the online net's forward.
    with torch.no_grad():
        for p in dqn.target_network.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        online1, _ = dqn.q_network(seq)
        target1, _ = dqn.target_network(seq)
    assert not torch.allclose(online1, target1)
    # Online output unchanged by perturbing the target (no shared state).
    assert torch.allclose(online0, online1)

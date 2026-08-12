"""§D.4 — filter correctness: the CBN filter matches the closed-form forward
algorithm on a 2-state toy HMM, and the MDP U-belief update matches Bayes."""

from __future__ import annotations

import torch
from src.rl.offline.grace.cbn import CBNParams
from src.rl.offline.grace.filter import BeliefFilter
from tests._grace_test_utils import DEV


def _hmm_params():
    p0 = torch.tensor([0.6, 0.4], device=DEV)
    trans = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]], device=DEV).permute(1, 0, 2)
    # ^ (S=2, A=1, S'=2): from s=0 -> [0.7, 0.3]; from s=1 -> [0.2, 0.8]
    emis = torch.tensor([[0.9, 0.1], [0.3, 0.7]], device=DEV)  # (S, O)
    return CBNParams(
        p_u=torch.ones(1, device=DEV),
        pb_a=torch.full((1, 2, 1), 1.0, device=DEV),
        p_r=torch.full((1, 2, 1, 1), 1.0, device=DEV),
        p0=p0,
        trans=trans,
        p_done=torch.zeros(2, 1, device=DEV),
        emis=emis,
    )


def test_pomdp_filter_matches_forward_algorithm():
    p = _hmm_params()
    f = BeliefFilter(p, n_classes=1, pomdp=True, device=DEV)
    obs_seq = [0, 1, 1, 0]

    # Closed-form forward algorithm (normalized alphas).
    alpha = p.p0 * p.emis[:, obs_seq[0]]
    alpha = alpha / alpha.sum()
    closed = [alpha.clone()]
    for o in obs_seq[1:]:
        alpha = (alpha @ p.trans[:, 0]) * p.emis[:, o]
        alpha = alpha / alpha.sum()
        closed.append(alpha.clone())

    # The CBN filter: reset -> condition on o_0, then predict+correct per step.
    b = f.condition_obs(f.reset(1), torch.tensor([obs_seq[0]], device=DEV))
    assert torch.allclose(b.reshape(2), closed[0], atol=1e-6)
    for i, o in enumerate(obs_seq[1:]):
        b = f.step(
            b,
            torch.tensor([obs_seq[i]], device=DEV),
            torch.tensor([0], device=DEV),
            reward_class=None,
            next_obs_symbol=torch.tensor([o], device=DEV),
        )
        assert torch.allclose(b.reshape(2), closed[i + 1], atol=1e-6), i


def test_mdp_u_belief_update_matches_bayes():
    # Single state, 1 action, 2 reward classes; P(rc=1 | u=1) = 0.9,
    # P(rc=1 | u=0) = 0.2. One observed rc=1 update from the 0.5 prior:
    # posterior = 0.9 / (0.9 + 0.2) = 9/11.
    p = CBNParams(
        p_u=torch.tensor([0.5, 0.5], device=DEV),
        pb_a=torch.full((2, 1, 1), 1.0, device=DEV),
        p_r=torch.tensor([[[[0.8, 0.2]]], [[[0.1, 0.9]]]], device=DEV),
        p0=torch.ones(1, device=DEV),
        trans=torch.ones(1, 1, 1, device=DEV),
        p_done=torch.zeros(1, 1, device=DEV),
    )
    f = BeliefFilter(p, n_classes=2, pomdp=False, device=DEV)
    b = f.reset(1)
    assert torch.allclose(b, torch.tensor([[0.5, 0.5]], device=DEV))
    b = f.step(
        b,
        torch.tensor([0], device=DEV),
        torch.tensor([0], device=DEV),
        reward_class=torch.tensor([1], device=DEV),
    )
    assert torch.allclose(
        b, torch.tensor([[2.0 / 11.0, 9.0 / 11.0]], device=DEV), atol=1e-6
    )
    # Entropy telemetry is normalized to [0, 1].
    ent = f.entropy(b)
    assert 0.0 < float(ent) < 1.0

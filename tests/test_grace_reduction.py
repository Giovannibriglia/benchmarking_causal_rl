"""§D.1 — the reduction / no-harm guarantee (B6) + the A1 alias.

With U disabled and full observability, GRACE's interventional Bellman target
must equal the standard empirical Bellman target up to sampling noise; and the
grace training path is parameter-identical to the observational floor
(``grace_obs_only`` is the floor by construction — the A1 test-verified
alias, replacing a burned sweep arm).
"""

from __future__ import annotations

import torch
from src.rl.nets.mlp import MLP
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.grace import (
    build_grace_dqn,
    cell_graph,
    GraceMachinery,
    GraceOptions,
)
from tests._grace_test_utils import DEV, FakeSeqBuffer, make_confounded_episodes

_GAMMA = 0.9


def test_reduction_interventional_equals_bellman_targets():
    # U-LESS declared graph (the biased arm has no U node) + unconfounded data
    # with c_r = 0: full observability, no latent -> the CBN is the empirical
    # MDP and its q_do backup must match model-free Bellman targets.
    eps, _ = make_confounded_episodes(
        n_ep=500, t_len=15, confounded=False, c_r=0.0, seed=3
    )
    m = GraceMachinery(
        cell_graph("mdp", "biased"),
        GraceOptions(n_bins=3, alpha=0.1, ensemble_k=0, interval=False, router=False),
        n_actions=2,
        device=DEV,
        gamma=_GAMMA,
    )
    m.fit_from_buffer(FakeSeqBuffer(eps))
    p = m.cbn.params
    assert p.p_u.numel() == 1  # U disabled: a single degenerate stratum

    # A FIXED value function (arbitrary, shared by both target computations).
    gq = torch.Generator().manual_seed(0)
    q_fixed = torch.rand(m.cbn.n_s, m.cbn.n_a, generator=gq).to(DEV)
    v_fixed = q_fixed.max(dim=1).values

    # GRACE target (the single target-computation boundary's one-step form):
    # y_cbn(s,a) = E[R|s,a] + gamma * (1 - p_done) * E_{s'}[v(s')].
    r_mean = (p.p_r * m.cbn.class_values.view(1, 1, 1, -1)).sum(-1)[0]
    ev = torch.einsum("saj,j->sa", p.trans, v_fixed)
    y_cbn = r_mean + _GAMMA * (1.0 - p.p_done) * ev

    # Baseline empirical Bellman targets on the same fixed batch, averaged
    # per (s, a) bin: mean_i[r_i + gamma * (1 - done_i) * v(s'_i)].
    sums = torch.zeros(m.cbn.n_s * m.cbn.n_a, device=DEV)
    counts = torch.zeros(m.cbn.n_s * m.cbn.n_a, device=DEV)
    for ep in eps:
        for tr in ep:
            s = int(m.discretizer.encode(tr["obs"].unsqueeze(0).to(DEV)))
            sn = int(m.discretizer.encode(tr["next_obs"].unsqueeze(0).to(DEV)))
            a = int(tr["actions"])
            done = float(tr["dones"])
            y = float(tr["rewards"]) + _GAMMA * (1.0 - done) * float(v_fixed[sn])
            sums[s * 2 + a] += y
            counts[s * 2 + a] += 1.0
    have = counts >= 30  # enough samples for the noise bound
    y_emp = (sums / counts.clamp_min(1.0)).view(m.cbn.n_s, m.cbn.n_a)
    diff = (y_cbn - y_emp).reshape(-1)[have]
    assert bool(have.any())
    assert float(diff.abs().max()) < 0.08, float(diff.abs().max())


def test_grace_training_path_is_the_observational_floor():
    """A1 (grace_obs_only alias) + base-parity: identical RNG draws, identical
    parameters, and — while the router serves obs / is unfitted — identical
    forward outputs to the observational floor."""
    torch.manual_seed(7)
    q_floor = MLP(4, 2).to(DEV)
    _tgt_floor = MLP(4, 2).to(DEV)
    _agent_floor = DQN(q_floor, _tgt_floor, ReplayBuffer(1000, DEV), device=DEV)

    torch.manual_seed(7)
    net, agent = build_grace_dqn(
        obs_dim=4, action_dim=2, device=DEV, action_type="discrete"
    )
    # Parameter identity with the floor (same classes, same draw order).
    floor_sd = q_floor.state_dict()
    grace_sd = net.obs_net.state_dict()
    assert set(floor_sd) == set(grace_sd)
    for k in floor_sd:
        assert torch.equal(floor_sd[k], grace_sd[k]), k

    x = torch.randn(16, 4, device=DEV)
    # Unfitted machinery -> forward IS the floor.
    assert torch.equal(net(x), q_floor(x))
    # Fitted but UNCALIBRATED router -> serves obs -> still the floor.
    eps, _ = make_confounded_episodes(n_ep=60, t_len=8, seed=1)
    agent.set_sequence_buffer(FakeSeqBuffer(eps))
    assert agent._grace_machinery.ready
    assert agent._grace_machinery.verdict.label == "uncalibrated"
    assert torch.allclose(net(x), q_floor(x))
    # The strategy hook trains through q_obs — the trainable head, never a
    # routed table.
    assert torch.equal(agent._strategy.critic_value(net, x, {}), net.obs_net(x))

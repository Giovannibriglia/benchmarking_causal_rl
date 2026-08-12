"""§D.2 — mutilation correctness on an analytic 3-node SCM, + the A7
shared-parameter assertions.

SCM (single state; SIX steps per episode — the static per-episode U needs
repeated within-episode measurements to be identified, the proximal insight;
a one-step design leaves the latent-class mixture unidentified):
U ~ Bernoulli(0.5); per step P(a=1|u) = 0.8 if u else 0.2; r = a * (1 + u).
Analytic per-step values:

    E[R | do(a=1)] = E_u[1 + u]            = 1.5   (mutilated / interventional)
    E[R | a=1]     = 1 + P(u=1|a=1) = 1.8          (observational, confounded)
    bias           = P(u=1|a=1) - P(u=1)   = 0.3
"""

from __future__ import annotations

import torch
from src.rl.offline.grace import cell_graph, GraceMachinery, GraceOptions
from tests._grace_test_utils import DEV, FakeSeqBuffer

_N = 1500
_T = 6


def _episodes(seed: int = 0, n: int = _N, c_r_mult: int = 1):
    g = torch.Generator().manual_seed(seed)
    eps = []
    for _ in range(n):
        u = int(torch.randint(0, 2, (1,), generator=g))
        ep = []
        for t in range(_T):
            a = int(torch.rand(1, generator=g) < (0.8 if u else 0.2))
            r = float(a * (1 + c_r_mult * u))
            ep.append(
                {
                    "obs": torch.tensor([0.5, 0.5]),
                    "actions": torch.tensor(a),
                    "rewards": torch.tensor(r),
                    "next_obs": torch.tensor([0.5, 0.5]),
                    "dones": torch.tensor(float(t == _T - 1)),
                    "confounder_u": torch.tensor(float("nan")),  # R5 poison
                }
            )
        eps.append(ep)
    return eps


def _fit_machinery():
    m = GraceMachinery(
        cell_graph("mdp", "template"),
        GraceOptions(
            n_bins=2,
            em_iters=25,
            alpha=0.05,  # near-MLE: the analytic comparison needs low smoothing
            ensemble_k=0,
            interval=False,
            router=False,
        ),
        n_actions=2,
        device=DEV,
        gamma=0.0,  # one-step: q_do == E[R | s, a, u]
    )
    m.fit_from_buffer(FakeSeqBuffer(_episodes()))
    return m


def test_interventional_vs_observational_vs_analytic():
    m = _fit_machinery()
    p = m.cbn.params
    q_do = m.cbn.q_do(gamma=0.0)  # (n_u, n_s, n_a) = E[R | s, a, u]
    s0 = int(m.discretizer.encode(torch.tensor([[0.5, 0.5]], device=DEV))[0])

    # Interventional (mutilated): E[R | do(a=1)] = sum_u P(u) E[R | a=1, u].
    do_val = float((p.p_u.view(-1, 1, 1) * q_do).sum(0)[s0, 1])
    assert abs(do_val - 1.5) < 0.06, do_val

    # Observational (full graph): E[R | a=1] = sum_u P(u | a=1) E[R | a=1, u].
    w = p.p_u * p.pb_a[:, s0, 1]
    w = w / w.sum()
    obs_val = float((w * q_do[:, s0, 1]).sum())
    assert abs(obs_val - 1.8) < 0.06, obs_val

    # The two differ by exactly the known confounding bias.
    assert abs((obs_val - do_val) - 0.3) < 0.08


def test_nbn_mirror_queries_and_shared_mechanisms():
    m = _fit_machinery()
    net_obs, net_mut = m.cbn.build_nbn()
    vals = m.codec.class_values.cpu()

    def _expect_r(net, a: int) -> float:
        pr = net.query(["R"], {"A": torch.tensor(a)}).cpu()
        return float((pr * vals).sum())

    # Mutilated net: A is a root -> conditioning IS do(). Exact VE.
    assert abs(_expect_r(net_mut, 1) - 1.5) < 0.06
    # Observational net: conditioning on A drags U along the backdoor.
    assert abs(_expect_r(net_obs, 1) - 1.8) < 0.06

    # A7: every non-A mechanism is the SAME module object in both channels,
    # backed by the same storage.
    for node in ("R", "U", "S", "S_next", "O"):
        m_obs = net_obs.mechanisms[node]
        m_mut = net_mut.mechanisms[node]
        assert m_obs is m_mut, node
        assert m_obs._logits.data_ptr() == m_mut._logits.data_ptr(), node
    assert net_obs.mechanisms["A"] is not net_mut.mechanisms["A"]

    # A7: a change through the shared module reaches BOTH channels' queries
    # (after the mandatory VE cache invalidation — the caches are instance-
    # level and keyed by id(model), so a stale answer is the failure mode).
    shared_r = net_obs.mechanisms["R"]
    shared_r._logits = shared_r._logits.flip(-1)  # scramble the R CPT
    m.cbn.invalidate_nbn_cache()
    assert abs(_expect_r(net_mut, 1) - 1.5) > 0.2
    assert abs(_expect_r(net_obs, 1) - 1.8) > 0.2


def test_refit_propagates_to_both_channels_and_invalidates_cache(monkeypatch):
    m = _fit_machinery()
    net_obs, net_mut = m.cbn.build_nbn()
    vals = m.codec.class_values.cpu()
    before = float((net_mut.query(["R"], {"A": torch.tensor(1)}).cpu() * vals).sum())
    # A7 spy: with a LIVE engine (the query above created it), an in-place CPT
    # change followed by the machinery's invalidation must call the engine's
    # instance-level invalidate_cache — a stale id()-keyed cache is the trap.
    calls = {"n": 0}
    from nbn import TensorVariableElimination

    original = TensorVariableElimination.invalidate_cache

    def _spy(self, *a, **k):
        calls["n"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(TensorVariableElimination, "invalidate_cache", _spy)
    shared_r = net_obs.mechanisms["R"]
    shared_r._logits = shared_r._logits.flip(-1)
    m.cbn.invalidate_nbn_cache()
    assert calls["n"] >= 1
    changed = float((net_mut.query(["R"], {"A": torch.tensor(1)}).cpu() * vals).sum())
    assert abs(changed - before) > 0.2  # a stale cache would return `before`
    shared_r._logits = shared_r._logits.flip(-1)  # restore
    m.cbn.invalidate_nbn_cache()
    # Refit on a DIFFERENT SCM (c_r doubled: r = a * (1 + 2u)) and rebuild:
    # both channels' queries must move consistently.
    m.fit_from_buffer(FakeSeqBuffer(_episodes(seed=9, c_r_mult=2)))
    net_obs2, net_mut2 = m.cbn.build_nbn()
    vals2 = m.codec.class_values.cpu()
    after_mut = float(
        (net_mut2.query(["R"], {"A": torch.tensor(1)}).cpu() * vals2).sum()
    )
    after_obs = float(
        (net_obs2.query(["R"], {"A": torch.tensor(1)}).cpu() * vals2).sum()
    )
    assert abs(after_mut - 2.0) < 0.1  # E[R|do(a=1)] = E_u[1 + 2u] = 2.0
    assert abs(after_obs - 2.6) < 0.1  # 1 + 2 * P(u=1|a=1) = 2.6
    assert abs(after_mut - before) > 0.3

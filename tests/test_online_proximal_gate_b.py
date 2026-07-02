"""Gate B — ONLINE proximal deconfounding (fixed confounded behavior policy).

The online-collected analog of cell-7's fixed-behavior offline proximal setting.
A rolling episode-grouped buffer turns over; the E-step is refreshed over completed
episodes on a cadence. The correctness item this gates is LABEL PERSISTENCE: the
existing offline canonicalization ("higher-reward stratum = U=1" + the no-spread ->
prior fallback) must be re-applied on EVERY refresh so the U=1 labeling cannot flip
across turnover — the failure the Gate-A addendum exposed.

Estimator math (run_em/m_step body) is unchanged; refresh() only rebuilds the cached
batch from the current buffer and re-runs the SAME warm-start + run_em.
"""

from __future__ import annotations

import warnings

import torch
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer
from src.rl.offline.proximal import ProximalEM

warnings.filterwarnings("ignore")
_CPU = torch.device("cpu")
OBS_DIM, ACT_DIM, EP_LEN = 4, 2, 12


def _gen_episode(theta, c_r, c_a):
    """One episode under a FIXED confounded behavior policy: action conditions on the
    per-episode U (NOT on any learner -> stationary), reward = r_clean(s,a) + c_r*U +
    noise, U never entering obs. Returns (transition-dicts, true_U)."""
    u = float(torch.bernoulli(torch.tensor(0.5)))
    ep = []
    for _ in range(EP_LEN):
        obs = torch.randn(OBS_DIM)
        a = (
            int(round(u))
            if torch.rand(1).item() < c_a
            else int(torch.randint(0, ACT_DIM, (1,)))
        )
        r = float(obs @ theta[a]) + c_r * u + 0.1 * float(torch.randn(1))
        ep.append(
            {
                "obs": obs,
                "actions": torch.tensor(a),
                "rewards": torch.tensor(r),
                "next_obs": obs,
                "dones": torch.tensor(0.0),
            }
        )
    return ep, u


def _fill(buf, n, theta, c_r, c_a):
    us = []
    for _ in range(n):
        ep, u = _gen_episode(theta, c_r, c_a)
        for tr in ep:
            buf.add(env_id=0, transition=tr)
        buf.mark_episode_end(env_id=0)
        us.append(u)
    return us


def _signed_corr(r_tau, true_u):
    a = torch.as_tensor(r_tau, dtype=torch.float32)
    b = torch.as_tensor(true_u, dtype=torch.float32)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _drive_refreshes(c_r, c_a, n_eps=30, refreshes=6, seed=0):
    """Fill, hand off, then turn the buffer over `refreshes` times through the REAL
    ProximalEM.refresh(). Returns per-refresh (signed_corr(r_tau,U), delta, median)."""
    torch.manual_seed(seed)
    theta = 0.5 * torch.randn(ACT_DIM, OBS_DIM)
    buf = SequenceReplayBuffer(capacity=n_eps * EP_LEN, device=_CPU)  # holds ~n_eps eps
    true_us = _fill(buf, n_eps, theta, c_r, c_a)
    em = ProximalEM(OBS_DIM, ACT_DIM, _CPU)
    em.set_sequence_buffer(buf)  # first fill

    out = []
    for _ in range(refreshes):
        true_us += _fill(buf, n_eps, theta, c_r, c_a)  # full turnover (evicts oldest)
        em.refresh()  # <-- the Gate-B path: rebuild view + re-canonicalize
        eps = list(buf.iter_episodes())
        r_tau = [float(e[0]["r_tau"]) for e in eps]
        U = true_us[-len(eps) :]
        out.append(
            {
                "corr": _signed_corr(r_tau, U),
                "delta": float(em.rm.delta),
                "median": float(torch.tensor(r_tau).median()),
            }
        )
    return out


# --------------------------------------------------------------------------
# GATE 1 — LABEL CONSISTENCY across turnover (the gate that failed before).
# --------------------------------------------------------------------------
def test_gate1_label_consistency_across_refreshes_sigma1():
    """sigma=1: across >=5 buffer refreshes, SIGNED corr(r_tau, true U) holds ONE
    sign and |corr|->1, and delta's sign is consistent (softplus>=0 structurally).
    No label flip re-entering through buffer churn."""
    traj = _drive_refreshes(c_r=1.0, c_a=1.0, refreshes=6)
    corrs = [r["corr"] for r in traj]
    deltas = [r["delta"] for r in traj]
    assert len(corrs) >= 5
    # |corr| -> 1 on every refresh (recovery holds through turnover).
    assert all(abs(c) > 0.9 for c in corrs), corrs
    # CONSISTENT sign — all positive or all negative, never flipping between refreshes.
    assert all(c > 0.9 for c in corrs) or all(c < -0.9 for c in corrs), corrs
    # delta sign consistent (non-negative by construction — the -delta basin is unreachable).
    assert all(d >= 0.0 for d in deltas), deltas
    # delta stays a real, non-degenerate shift the whole time (strata separated).
    assert all(d > 0.1 for d in deltas), deltas


# --------------------------------------------------------------------------
# GATE 2 — sigma=0 collapse STILL holds across refreshes (no manufactured split).
# --------------------------------------------------------------------------
def test_gate2_sigma0_collapse_across_refreshes():
    """sigma=0: the no-spread fallback must fire on EVERY refresh — r_tau -> prior
    (~0.5) with no spurious U-correlation, and delta collapses (monotone). The
    per-refresh canonicalization must not manufacture a split where there is none."""
    traj = _drive_refreshes(c_r=0.0, c_a=0.0, refreshes=6)
    medians = [r["median"] for r in traj]
    deltas = [r["delta"] for r in traj]
    # r_tau -> prior on every refresh (posterior reverts, no split).
    assert all(abs(m - 0.5) < 0.15 for m in medians), medians
    # delta is monotone non-increasing (collapsing toward 0 — the L2 safety proof).
    assert all(a >= b - 1e-6 for a, b in zip(deltas, deltas[1:])), deltas
    # and it is actually shrinking, not stuck.
    assert deltas[-1] < deltas[0], deltas


# --------------------------------------------------------------------------
# GATE 3 — FIVE-KEYS online: the learner reads ZERO realized U (infers only).
# --------------------------------------------------------------------------
def test_gate3_five_keys_online_spec_and_collection():
    """The online proximal spec never loads U, and the online grouped collection
    stores only the five base keys — the realized U is never available to the
    learner; the E-step infers it."""
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    spec = registry.get("online_dqn_proximal")
    assert spec.kind == "off_policy"
    assert spec.data_regime == "online"
    assert spec.needs_episode_grouping is True
    assert spec.requires_confounder_u is False  # never reads the realized U

    # The store carries the 5 base keys and NEVER the realized U (confounder_u): that
    # is the five-keys invariant. (r_tau, the INFERRED latent the E-step samples, is
    # not the realized U and may be present — offline stores it too.)
    torch.manual_seed(0)
    theta = 0.5 * torch.randn(ACT_DIM, OBS_DIM)
    buf = SequenceReplayBuffer(capacity=10_000, device=_CPU)
    _fill(buf, 5, theta, c_r=1.0, c_a=1.0)
    tr = next(buf.iter_episodes())[0]
    assert {"obs", "actions", "rewards", "next_obs", "dones"} <= set(tr.keys())
    assert "confounder_u" not in tr  # realized U is never stored -> the learner infers

    # proximal source never READS the realized U either (grep the read-patterns).
    from pathlib import Path

    src = Path("src/rl/offline/proximal.py").read_text()
    assert 'infos["confounder_u"]' not in src and "load_u=True" not in src


# --------------------------------------------------------------------------
# GATE 4 — OFFLINE proximal path untouched: m_step default stays on bare run_em.
# --------------------------------------------------------------------------
def test_gate4_offline_default_is_byte_frozen_run_em():
    """A freshly built ProximalEM is OFFLINE by default (online=False): its m_step
    cadence must call the byte-frozen run_em over the once-cached static batch, NOT
    refresh. This is what keeps the offline cell-7/8 goldens bitwise. Online mode is
    opt-in, set only by the online-grouped loop."""
    em = ProximalEM(OBS_DIM, ACT_DIM, _CPU)
    assert em.online is False  # offline by default

    # Assert the cadence branch calls run_em (offline) — never refresh — when offline,
    # and refresh (online) when online. Spy without touching estimator math.
    calls = {"run_em": 0, "refresh": 0}
    em.run_em = lambda: calls.__setitem__("run_em", calls["run_em"] + 1)  # type: ignore
    em.refresh = lambda: calls.__setitem__("refresh", calls["refresh"] + 1)  # type: ignore

    window = {
        "r_tau": torch.full((2, 3), 0.5),
        "obs": torch.zeros(2, 3, OBS_DIM),
        "actions": torch.zeros(2, 3, dtype=torch.long),
        "rewards": torch.zeros(2, 3),
        "next_obs": torch.zeros(2, 3, OBS_DIM),
        "dones": torch.zeros(2, 3),
    }
    em._updates = em.estep_interval - 1  # next m_step hits the cadence
    em.m_step(window, base_learn=lambda b: {"loss": 0.0})
    assert calls == {"run_em": 1, "refresh": 0}  # OFFLINE default -> run_em only

    em.online = True
    em._updates = em.estep_interval - 1
    em.m_step(window, base_learn=lambda b: {"loss": 0.0})
    assert calls == {"run_em": 1, "refresh": 1}  # ONLINE -> refresh

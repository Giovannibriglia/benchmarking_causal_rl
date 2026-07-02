"""Gate B — ACTION-DEPENDENT confounder cell: r += c_r*U*1[a=a_bad].

Unlike the additive cells 7/8 (nuisance U, argmax-preserving, no return gap), this
confounder inflates ONE suboptimal action's apparent reward -> the confounded floor is
lured into a_bad and LOSES eval RETURN; deconfounding recovers it. Deploy is the U=0
REFERENCE STRATUM (the clean world), not the prior-marginal (which bakes in +0.5*c_r on
a_bad and stays wrong at strong c_r).

Gates: (1) return gap floor_RET < proximal_RET ~ oracle_RET at c_r=1.0 AND c_r>=1.5;
(2) deploy proof — marginal keeps a_bad at strong c_r while U=0 recovers a_good;
(3) sparse-signal identifiability through the real builder; (4) five-keys; (5) additive
byte-frozen is covered by the existing cell-7/8 + golden suites.
"""

from __future__ import annotations

import warnings

import torch
from src.rl.nets.mlp import MLP
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer
from src.rl.offline.oracle_u import build_oracle_u_dqn
from src.rl.offline.proximal import build_proximal_action_dqn

warnings.filterwarnings("ignore")
_CPU = torch.device("cpu")
OBS_DIM, ACT_DIM, A_BAD, A_GOOD = 2, 2, 1, 0
R_CLEAN = torch.tensor([1.0, 0.3])  # a_good=0 truly best; a_bad=1 confounding-inflated


# ==========================================================================
# GATE — wrapper: action_gated shifts only a_bad; additive line byte-frozen.
# ==========================================================================
class _FakeVecEnv:
    def __init__(self, n=3):
        self.n_envs = n
        self.device = _CPU

    def step(self, action):
        r = torch.zeros(self.n_envs)  # clean reward 0 -> the shift is the whole signal
        d = torch.zeros(self.n_envs, dtype=torch.bool)
        return torch.zeros(self.n_envs, OBS_DIM), r, d, d, {}

    def reset(self, seed=None):
        return torch.zeros(self.n_envs, OBS_DIM), {}


def test_gate_wrapper_action_gated_vs_additive():
    from src.envs.wrappers.confounded import ConfoundedCollectionWrapper

    # action_gated: reward shifted ONLY where action == a_bad.
    w = ConfoundedCollectionWrapper(
        _FakeVecEnv(), c_a=1.0, c_r=1.0, confounder_kind="action_gated", a_bad=A_BAD
    )
    w.current_u = torch.ones(3)  # U=1 all envs
    actions = torch.tensor([A_GOOD, A_BAD, A_GOOD])
    _, reward, *_ = w.step(actions)
    assert torch.allclose(
        reward, torch.tensor([0.0, 1.0, 0.0])
    ), reward  # only a_bad shifted

    # additive (default): reward shifted on EVERY env regardless of action (frozen line).
    w2 = ConfoundedCollectionWrapper(
        _FakeVecEnv(), c_a=1.0, c_r=1.0
    )  # default additive
    w2.current_u = torch.ones(3)
    _, reward2, *_ = w2.step(actions)
    assert torch.allclose(reward2, torch.tensor([1.0, 1.0, 1.0])), reward2
    # invalid kind rejected
    import pytest

    with pytest.raises(ValueError):
        ConfoundedCollectionWrapper(_FakeVecEnv(), confounder_kind="bogus")


# ==========================================================================
# GATE 3 — sparse-signal identifiability THROUGH THE REAL BUILDER.
# ==========================================================================
def _action_gated_episodes(
    c_r, n_eps=50, ep_len=12, p_bad_u1=0.75, p_bad_u0=0.25, seed=1
):
    """Confounded episodes for the E-step: U per episode, behavior takes a_bad more when
    U=1, reward = obs@theta[a] + c_r*U*1[a=a_bad] + noise. Returns (list-of-episodes, U).
    """
    g = torch.Generator().manual_seed(seed)
    theta = 0.5 * torch.randn(ACT_DIM, OBS_DIM, generator=g)
    eps, us = [], []
    for _ in range(n_eps):
        u = float(torch.bernoulli(torch.tensor(0.5), generator=g))
        p_bad = p_bad_u1 if u == 1.0 else p_bad_u0
        ep = []
        for _ in range(ep_len):
            obs = torch.randn(OBS_DIM, generator=g)
            a = A_BAD if torch.rand(1, generator=g).item() < p_bad else A_GOOD
            r = (
                float(obs @ theta[a])
                + c_r * u * (1.0 if a == A_BAD else 0.0)
                + 0.1 * float(torch.randn(1, generator=g))
            )
            ep.append(
                {
                    "obs": obs,
                    "actions": torch.tensor(a),
                    "rewards": torch.tensor(r),
                    "next_obs": obs,
                    "dones": torch.tensor(0.0),
                }
            )
        eps.append(ep)
        us.append(u)
    return eps, torch.tensor(us)


def _run_em_via_builder(c_r, **kw):
    """Drive the REAL build_proximal_action_dqn agent's ProximalEM over action-gated
    data; return (r_tau per episode, delta[a], true U)."""
    eps, us = _action_gated_episodes(c_r, **kw)
    _, agent = build_proximal_action_dqn(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, device=_CPU, action_type="discrete"
    )
    buf = SequenceReplayBuffer(capacity=100_000, device=_CPU)
    for ep in eps:
        for tr in ep:
            buf.add(env_id=0, transition=tr)
        buf.mark_episode_end(env_id=0)
    agent.set_sequence_buffer(buf)  # first EM (warm-start + run_em)
    for _ in range(40):  # let L2 concentrate delta on a_bad (shrink the unused action)
        agent._proximal_em.run_em()
    r_tau = torch.tensor([float(e[0]["r_tau"]) for e in buf.iter_episodes()])
    return r_tau, agent._proximal_em.rm.delta.detach(), us


def _corr(a, b):
    a, b = a.float(), b.float()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def test_gate3_sparse_identifiability_through_builder():
    # sigma=1: recovery + delta localizes to a_bad (delta[a_bad] >> delta[a_good]).
    r1, d1, u1 = _run_em_via_builder(c_r=1.0)
    assert _corr(r1, u1) > 0.9, _corr(r1, u1)
    assert float(d1[A_BAD]) > 2.0 * float(d1[A_GOOD]), d1  # signal localized to a_bad
    # sigma=0: collapse — r_tau -> prior, delta vector -> ~0 (no manufactured split).
    r0, d0, _ = _run_em_via_builder(c_r=0.0)
    assert abs(float(r0.median()) - 0.5) < 0.15, r0.median()
    assert float(d0.max()) < 0.6, d0  # both actions' delta shrinking, no split


# ==========================================================================
# GATES 1 & 2 — return gap + deploy-convention proof, real critics (bandit, gamma=0).
# ==========================================================================
def _bandit_episodes(c_r, n_eps=300, ep_len=10, p_bad_u1=0.8, p_bad_u0=0.2, seed=3):
    g = torch.Generator().manual_seed(seed)
    obs = torch.ones(OBS_DIM)  # stateless -> Q depends only on (a[,u])
    eps = []
    for _ in range(n_eps):
        u = float(torch.bernoulli(torch.tensor(0.5), generator=g))
        p_bad = p_bad_u1 if u == 1.0 else p_bad_u0
        ep = []
        for _ in range(ep_len):
            a = A_BAD if torch.rand(1, generator=g).item() < p_bad else A_GOOD
            r = (
                float(R_CLEAN[a])
                + c_r * u * (1.0 if a == A_BAD else 0.0)
                + 0.1 * float(torch.randn(1, generator=g))
            )
            ep.append(
                {
                    "obs": obs.clone(),
                    "actions": torch.tensor(a),
                    "rewards": torch.tensor(r),
                    "next_obs": obs.clone(),
                    "dones": torch.tensor(0.0),
                    "confounder_u": torch.tensor(u),
                }
            )
        eps.append(ep)
    return eps


def _train_flat(agent, eps, steps, with_u):
    agent.gamma = 0.0  # bandit -> target = reward (no bootstrap)
    for ep in eps:
        for tr in ep:
            t = (
                dict(tr)
                if with_u
                else {k: v for k, v in tr.items() if k != "confounder_u"}
            )
            agent.buffer.add(t)
    for _ in range(steps):
        agent.update(agent.buffer.sample(256))


def _train_proximal(agent, eps, steps):
    agent.gamma = 0.0
    buf = SequenceReplayBuffer(capacity=200_000, device=_CPU)
    for ep in eps:
        for tr in ep:  # five keys only — the estimator INFERS U (no confounder_u)
            buf.add(
                env_id=0,
                transition={k: v for k, v in tr.items() if k != "confounder_u"},
            )
        buf.mark_episode_end(env_id=0)
    agent.set_sequence_buffer(buf)
    for _ in range(steps):
        agent.update(buf.sample_sequences(128, 8))


def _returns_at(c_r):
    torch.manual_seed(0)
    eps = _bandit_episodes(c_r)
    obs1 = torch.ones(1, OBS_DIM)
    ret = lambda q: float(R_CLEAN[int(q.argmax())])  # noqa: E731

    # FLOOR — plain observational DQN Q(s,a) = E[r|s,a] (no U).
    floor = DQN(
        MLP(OBS_DIM, ACT_DIM),
        MLP(OBS_DIM, ACT_DIM),
        ReplayBuffer(200_000, _CPU),
        device=_CPU,
    )
    _train_flat(floor, eps, 500, with_u=False)
    floor_q = floor.q_network(obs1)[0]

    # ORACLE — reads realized U; deploy marginal vs U=0 read off the SAME critic.
    oq, oracle = build_oracle_u_dqn(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, device=_CPU, action_type="discrete"
    )
    _train_flat(oracle, eps, 500, with_u=True)

    # PROXIMAL (action-conditional + U=0 deploy) — infers U, five keys only.
    pq, prox = build_proximal_action_dqn(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, device=_CPU, action_type="discrete"
    )
    _train_proximal(prox, eps, 600)

    with torch.no_grad():
        return {
            "floor": ret(floor_q),
            "oracle_marg": ret(oq.forward(obs1)[0]),
            "oracle_U0": ret(oq.q_at(obs1, 0.0)[0]),
            "prox_marg": ret(0.5 * (pq.q_at(obs1, 0.0) + pq.q_at(obs1, 1.0))[0]),
            "prox_U0": ret(
                pq.forward(obs1)[0]
            ),  # UReferenceStratumQ.forward == q_at(0)
        }


def test_gate1_return_gap_and_gate2_deploy_proof():
    good, bad = float(R_CLEAN[A_GOOD]), float(R_CLEAN[A_BAD])
    r10 = _returns_at(c_r=1.0)
    r18 = _returns_at(c_r=1.8)  # strong: 0.5*c_r=0.9 > clean_gap 0.7 -> marginal fails

    # GATE 1 — return gap at BOTH c_r; U=0-deploy proximal recovers a_good, ~ oracle.
    for tag, r in (("c_r=1.0", r10), ("c_r=1.8", r18)):
        assert r["floor"] == bad, (tag, r)  # floor lured into a_bad
        assert r["prox_U0"] == good, (tag, r)  # proximal recovers
        assert r["oracle_U0"] == good, (tag, r)  # oracle recovers
        assert r["floor"] < r["prox_U0"], (tag, r)  # the headline ordering

    # GATE 2 — deploy proof: at STRONG c_r the prior-MARGINAL keeps a_bad (known-wrong)
    # while the U=0 reference stratum recovers a_good. This is the evidence U=0 is
    # correct, not merely chosen.
    assert r18["oracle_marg"] == bad, r18  # marginal bakes in +0.5*c_r -> still a_bad
    assert r18["prox_marg"] == bad, r18
    assert r18["oracle_U0"] == good and r18["prox_U0"] == good, r18


# ==========================================================================
# GATE 4 — five-keys: the action-conditional path reads ZERO realized U.
# ==========================================================================
def test_gate4_five_keys():
    from pathlib import Path

    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    spec = registry.get("offline_dqn_proximal_action")
    assert spec.requires_confounder_u is False
    assert spec.needs_episode_grouping is True and spec.data_regime == "offline"
    # Five-keys READ patterns: never pulls the dataset's stored/realized U nor asks the
    # loader for it. (batch["confounder_u"] DOES appear — but only as the E-step's
    # INFERRED sample that m_step WRITES and the Proximal hook reads back, not a
    # dataset read; so it is not a five-keys violation.)
    src = Path("src/rl/offline/proximal.py").read_text()
    assert 'infos["confounder_u"]' not in src  # never reads the dataset's stored U
    assert "load_u=True" not in src  # never asks the loader for the realized U
    # the U=0 deploy is a fixed-u forward (q_at(obs, 0.0)), not a realized-U read:
    assert "q_at(obs, 0.0)" in src

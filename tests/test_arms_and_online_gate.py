"""PR 3 — the three arms (basic/biased/confounded) + the online intervened gate.

H1  basic arm collects with pi_basic at the FIXED epsilon and does NOT drift as the
    learner's epsilon anneals.
H2  the shared-pi_basic_epsilon assertion fires (validator + runner online guard).
H3  biased arm: coverage (action_overlap) degrades monotonically with beta, logged.
H4  confounded arm: coverage stays within +/-15% of basic across the sigma sweep.
H5  ONLINE: mean(intervened) ~= 1 - sigma at every checkpoint (fails on master).
H6  the intervened flag survives the GROUPED and RECURRENT collectors.
"""

from __future__ import annotations

import csv
import warnings

import minari
import numpy as np
import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.envs.offline.generate import generate_offline_dataset
from src.envs.registry import register_default_env_wrappers
from src.rl.policies.behavior_policy import (
    AgentBehaviorPolicy,
    assert_shared_pi_basic_epsilon,
    PiBasicBehaviorPolicy,
)

warnings.filterwarnings("ignore")
CPU = torch.device("cpu")


def _register():
    register_default_algorithms()
    register_default_env_wrappers()


def _runner(tmp_path, env_cfg, algo="dqn", n_episodes=1, extra_train=None):
    _register()
    train_cfg = TrainingConfig(
        n_episodes=n_episodes,
        n_checkpoints=2,
        device="cpu",
        algorithm=algo,
        **(extra_train or {}),
    )
    return BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(tmp_path / "run"), timestamp="t"),
        registry.get(algo),
    )


# ---------------------------------------------------------------------------
# H1 — basic arm uses pi_basic at the FIXED epsilon; no drift when the learner anneals.
# ---------------------------------------------------------------------------
def test_h1_basic_arm_uses_pi_basic_fixed_epsilon(tmp_path):
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=4,
        seed=0,
        behavior_policy="pi_basic",
        pi_basic_epsilon=0.5,
    )
    runner = _runner(tmp_path, env_cfg)
    assert isinstance(runner.collection_policy, PiBasicBehaviorPolicy)
    assert runner.collection_policy.epsilon == 0.5

    # Anneal the LEARNER's epsilon far below the fixed pi_basic epsilon.
    runner.agent.epsilon = 0.01
    obs = torch.zeros(1, 4).repeat(8000, 1)  # one state, replicated
    torch.manual_seed(0)
    a_basic = runner.collection_policy.act(obs).action
    freq_basic = torch.bincount(a_basic, minlength=2).float() / len(a_basic)
    # pi_basic explores at eps=0.5 -> greedy action ~0.75, NOT the annealed ~0.995.
    assert 0.70 < freq_basic.max().item() < 0.80, freq_basic
    # the annealed agent policy IS near-greedy -> proves they differ (no drift).
    a_agent = AgentBehaviorPolicy(runner.agent).act(obs).action
    freq_agent = torch.bincount(a_agent.long().reshape(-1), minlength=2).float()
    assert (freq_agent / freq_agent.sum()).max().item() > 0.95
    runner.train_env.close()
    runner.eval_env.close()


# ---------------------------------------------------------------------------
# H2 — the shared-epsilon assertion fires (validator + runner online guard).
# ---------------------------------------------------------------------------
def test_h2_shared_epsilon_validator_fires_on_mismatch():
    # a well-formed cell passes
    assert_shared_pi_basic_epsilon(
        [("pi_basic", 0.5), ("biased", 0.5), ("bias_confounded_action", 0.5)]
    )
    # a mismatched cell raises
    with pytest.raises(ValueError, match="share pi_basic_epsilon"):
        assert_shared_pi_basic_epsilon(
            [("pi_basic", 0.5), ("biased", 0.3), ("bias_confounded_action", 0.5)]
        )
    # a missing (None) arm epsilon raises
    with pytest.raises(ValueError, match="explicitly"):
        assert_shared_pi_basic_epsilon([("pi_basic", None)])


def test_h2_runner_online_arm_requires_explicit_epsilon(tmp_path):
    # online arm run WITHOUT pi_basic_epsilon -> the runner refuses to build.
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=4,
        seed=0,
        behavior_policy="pi_basic",
        pi_basic_epsilon=None,
    )
    with pytest.raises(ValueError, match="must set pi_basic_epsilon"):
        _runner(tmp_path, env_cfg)


# ---------------------------------------------------------------------------
# Offline generation + training helpers (for H3, H4).
# ---------------------------------------------------------------------------
def _gen(dataset_id, behavior_policy, strength, *, n_eps=120):
    try:
        minari.delete_dataset(dataset_id)
    except Exception:
        pass
    return generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy=behavior_policy,
        behavior_strength=strength,
        pi_basic_epsilon=0.5,
        rollout_episodes=n_eps,
        seed=0,
        dataset_id=dataset_id,
    )


def _train_and_read_coverage(tmp_path, dataset_id, behavior_policy, strength):
    _register()
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=2,
        seed=0,
        offline_dataset=dataset_id,
        behavior_policy=behavior_policy,
        behavior_strength=strength,
        pi_basic_epsilon=0.5,
    )
    run_dir = tmp_path / "run"
    BenchmarkRunner(
        env_cfg,
        TrainingConfig(n_episodes=1, n_checkpoints=2, device="cpu", algorithm="cql"),
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("cql"),
    ).run()
    with open(run_dir / "arm_diagnostics.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows, "arm_diagnostics.csv must be non-empty for an arm run"
    # reward-independent coverage (also confirms statistical_diagnostic wired the
    # separability/action_overlap columns).
    assert rows[-1]["action_overlap"] != "" and rows[-1]["separability"] != ""
    return float(rows[-1]["action_coverage"])


# ---------------------------------------------------------------------------
# H3 — biased coverage degrades monotonically with beta, and is logged.
# ---------------------------------------------------------------------------
def test_h3_biased_coverage_degrades_with_beta(tmp_path):
    covs = []
    for beta in (0.0, 0.4, 0.8):
        did = f"test/pr3-biased-b{int(beta * 100)}-v0"
        _gen(did, "biased", beta)
        try:
            covs.append(_train_and_read_coverage(tmp_path, did, "biased", beta))
        finally:
            minari.delete_dataset(did)
    for lo, hi in zip(covs, covs[1:]):
        assert hi < lo + 0.02, covs  # monotone non-increasing
    assert covs[-1] < 0.5 * covs[0], covs  # strong degradation at high beta


# ---------------------------------------------------------------------------
# H4 — confounded coverage within +/-15% of basic across the sigma sweep, the
# empirical orthogonality claim. Tested at the POLICY level with a SHARED base agent
# and the REWARD-INDEPENDENT action_coverage (the runner logs this same metric):
#   * production generation would build a DIFFERENT fresh agent per arm (random tier),
#     so basic/confounded must share ONE checkpoint -> that comparison is deferred to
#     PR 5's shared-checkpoint sweep (see report);
#   * statistical_diagnostic's action_overlap uses reward-median strata, which the
#     c_r*U reward shift contaminates -> NOT sigma-invariant. action_coverage is.
# ---------------------------------------------------------------------------
class _AngleQ:
    def q_network(self, obs):
        return torch.stack([-obs[:, 2], obs[:, 2]], dim=-1)


class _UEnv:
    def __init__(self, n):
        self.n_envs = n
        self.current_u = torch.zeros(n)
        self.current_h = None
        self.device = CPU


def _rollout_action_coverage(pol_fn, n_eps=250):
    import gymnasium as gym
    from src.benchmarking.registry import register_default_algorithms  # noqa: F401

    penv = gym.make("CartPole-v1")
    env = _UEnv(1)
    pol = pol_fn(env)
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(0)
    acts = []
    for ep in range(n_eps):
        obs, _ = penv.reset(seed=ep)
        env.current_u = torch.bernoulli(torch.tensor([0.5]), generator=g)
        for _ in range(200):
            o = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            a = int(pol.act(o).action.item())
            acts.append(a)
            obs, r, t, tr, _ = penv.step(a)
            if t or tr:
                break
    penv.close()
    c = np.bincount(np.asarray(acts), minlength=2).astype(float)
    return float((c / c.sum()).min())


def test_h4_confounded_coverage_matches_basic():
    from gymnasium.spaces import Discrete
    from src.rl.policies.behavior_policy import (
        build_collection_policy,
        MarginallyMatchedConfoundedBehaviorPolicy,
    )

    agent = _AngleQ()  # ONE shared base policy across all arms
    cov_basic = _rollout_action_coverage(
        lambda e: PiBasicBehaviorPolicy(agent, "discrete", Discrete(2), epsilon=0.5)
    )
    assert cov_basic > 0.3, cov_basic
    for sigma in (0.25, 0.5, 1.0):
        cov = _rollout_action_coverage(
            lambda e, s=sigma: MarginallyMatchedConfoundedBehaviorPolicy(
                agent, "discrete", Discrete(2), e, strength=s, pi_basic_epsilon=0.5
            )
        )
        assert abs(cov - cov_basic) / cov_basic < 0.15, (sigma, cov, cov_basic)
    # contrast: the biased arm DOES degrade coverage (not orthogonal to beta).
    cov_biased = _rollout_action_coverage(
        lambda e: build_collection_policy(
            "biased", agent, "discrete", Discrete(2), strength=0.8, pi_basic_epsilon=0.5
        )
    )
    assert cov_biased < 0.5 * cov_basic, (cov_biased, cov_basic)


# ---------------------------------------------------------------------------
# H5 — ONLINE: mean(intervened) ~= 1 - sigma at every checkpoint (fails on master:
# arm_diagnostics.csv / intervened_mean do not exist there).
# ---------------------------------------------------------------------------
def test_h5_online_intervened_fraction(tmp_path):
    _register()
    for sigma in (0.25, 0.5, 1.0):
        env_cfg = EnvConfig(
            env_id="CartPole-v1",
            n_train_envs=4,
            n_eval_envs=2,
            rollout_len=64,
            seed=0,
            behavior_policy="bias_confounded_action",
            behavior_strength=sigma,
            pi_basic_epsilon=0.5,
        )
        run_dir = tmp_path / f"run_s{int(sigma * 100)}"
        BenchmarkRunner(
            env_cfg,
            TrainingConfig(
                n_episodes=6, n_checkpoints=3, device="cpu", algorithm="dqn"
            ),
            RunConfig(run_dir=str(run_dir), timestamp="t"),
            registry.get("dqn"),
        ).run()  # the in-training gate raises if mean(intervened) != 1-sigma
        with open(run_dir / "arm_diagnostics.csv") as f:
            rows = list(csv.DictReader(f))
        ivs = [float(r["intervened_mean"]) for r in rows if r["intervened_mean"] != ""]
        assert ivs, "intervened_mean must be logged online"
        for iv in ivs:
            assert abs(iv - (1.0 - sigma)) < 0.15, (sigma, iv)


# ---------------------------------------------------------------------------
# H6 — the intervened flag survives the GROUPED and RECURRENT collectors.
# ---------------------------------------------------------------------------
class _FakeVecEnv:
    def __init__(self, n=2, obs_dim=3):
        self.n_envs = n
        self.device = CPU
        self._obs_dim = obs_dim
        self._t = 0

    def reset(self, seed=None):
        self._t = 0
        return torch.zeros(self.n_envs, self._obs_dim), {}

    def step(self, action):
        self._t += 1
        term = torch.zeros(self.n_envs, dtype=torch.bool)
        trunc = torch.full((self.n_envs,), self._t % 5 == 0)
        return (
            torch.zeros(self.n_envs, self._obs_dim),
            torch.ones(self.n_envs),
            term,
            trunc,
            {},
        )


class _InterveningPolicy:
    """Emits intervened per step (mimics the confounded arm) for collector plumbing."""

    def act(self, obs, *a, **k):
        from src.rl.base import ActionOutput

        n = obs.shape[0]
        return ActionOutput(
            action=torch.zeros(n, dtype=torch.long),
            intervened=torch.ones(n, dtype=torch.bool),
        )


def test_h6_intervened_survives_grouped_and_recurrent_collectors():
    from src.data.experience_source import OnlineSource
    from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

    # GROUPED
    env = _FakeVecEnv()
    src = OnlineSource(env, CPU)
    buf = SequenceReplayBuffer(capacity=100_000, device=CPU)
    obs, _ = env.reset()

    class _EM:
        prior_p = 0.5

    class _Agent:
        _proximal_em = _EM()

        def set_sequence_buffer(self, b):
            pass

        def update(self, batch):
            return {}

    out = src.collect_off_policy_grouped(
        _Agent(),
        buf,
        obs,
        collection_policy=_InterveningPolicy(),
        n_steps=40,
        n_envs=2,
        warmup=1,
        batch_size=2,
        seq_len=4,
    )
    assert len(out) == 4  # (obs, metrics, handed_off, last_batch)
    batch = buf.sample_sequences(2, 4)
    assert "intervened" in batch and batch["intervened"].float().mean().item() == 1.0

    # RECURRENT
    env2 = _FakeVecEnv()
    src2 = OnlineSource(env2, CPU)
    buf2 = SequenceReplayBuffer(capacity=100_000, device=CPU)
    obs2, _ = env2.reset()

    class _RecAgent:
        def act(self, obs, state=None):
            from src.rl.base import ActionOutput

            return ActionOutput(
                action=torch.zeros(obs.shape[0], dtype=torch.long), state=state
            )

        def update(self, batch):
            return {}

    src2.collect_off_policy_recurrent(
        _RecAgent(),
        buf2,
        obs2,
        collection_policy=_InterveningPolicy(),
        n_steps=40,
        n_envs=2,
        warmup=1,
        batch_size=2,
        seq_len=4,
    )
    batch2 = buf2.sample_sequences(2, 4)
    assert "intervened" in batch2 and batch2["intervened"].float().mean().item() == 1.0

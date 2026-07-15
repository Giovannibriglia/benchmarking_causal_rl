"""PR 3 — the three arms (basic/biased/confounded) + the online intervened gate.

H1  basic arm collects with pi_basic at the FIXED epsilon and does NOT drift as the
    learner's epsilon anneals.
H2  the shared-pi_basic_epsilon assertion fires (validator + runner online guard).
H3  biased arm: state-conditional coverage degrades monotonically with beta, logged.
H4  confounded arm: state-conditional coverage EQUALS basic EXACTLY (a marginal-matching
    implementation check), biased degrades monotonically in beta.
H5  ONLINE: mean(intervened) ~= 1 - sigma at every checkpoint (fails on master).
H6  the intervened flag survives the GROUPED and RECURRENT collectors.
"""

from __future__ import annotations

import csv
import warnings

import minari
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
# H4 — the STATE-CONDITIONAL coverage mean_s[min_a p_b(a|s)]. By the marginal-matching
# theorem the confounded arm's U-marginalized p_b == pi_basic, so coverage(confounded)
# == coverage(basic) EXACTLY (a check on the IMPLEMENTATION, not a 15% band) at every
# sigma; the biased arm degrades it monotonically in beta. Shared base agent + FIXED
# states isolate the metric from state-visitation and reward confounding.
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


def _state_conditional_coverage(pol, states):
    with torch.no_grad():
        return float(pol.action_probs(states).min(dim=-1).values.mean().item())


def test_h4_confounded_coverage_equals_basic_exactly():
    from gymnasium.spaces import Discrete
    from src.rl.policies.behavior_policy import (
        build_collection_policy,
        MarginallyMatchedConfoundedBehaviorPolicy,
    )

    agent = _AngleQ()  # ONE shared base policy across all arms
    torch.manual_seed(0)
    states = torch.randn(5000, 4)  # fixed reference states (varied pole angle)
    cov_basic = _state_conditional_coverage(
        PiBasicBehaviorPolicy(agent, "discrete", Discrete(2), epsilon=0.5), states
    )
    assert cov_basic > 0.1, cov_basic
    env = _UEnv(1)
    for sigma in (0.25, 0.5, 1.0):
        conf = MarginallyMatchedConfoundedBehaviorPolicy(
            agent, "discrete", Discrete(2), env, strength=sigma, pi_basic_epsilon=0.5
        )
        # EXACT equality: confounded action_probs IS pi_basic (marginal matching). If
        # this ever differs, the marginal matching is broken.
        assert _state_conditional_coverage(conf, states) == cov_basic, sigma
    # the biased arm degrades coverage MONOTONICALLY in beta.
    covs = [
        _state_conditional_coverage(
            build_collection_policy(
                "biased",
                agent,
                "discrete",
                Discrete(2),
                strength=b,
                pi_basic_epsilon=0.5,
            ),
            states,
        )
        for b in (0.25, 0.5, 0.75)
    ]
    assert covs[0] < cov_basic, (covs, cov_basic)
    for lo, hi in zip(covs, covs[1:]):
        assert hi < lo, covs  # strictly monotone decreasing


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

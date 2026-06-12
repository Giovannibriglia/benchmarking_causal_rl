"""A1 collection behavior policies — the three pure action-selection policies.

Each correctness bar exercises the policy's DISTINGUISHING behavior, failing on
its named bug:
  * anti_reward returns the argmin-Q action (NOT argmax) under a known critic;
  * bias_skew's empirical action distribution matches p;
  * bias_suboptimal's agent-vs-base fraction matches beta.

The off-policy golden bitwise-safety (default "agent" path untouched) is covered
by the regression suite, not here; this file pins the opt-in policies' behavior.
"""

from __future__ import annotations

import numpy as np
import torch
from src.rl.base import ActionOutput
from src.rl.policies.behavior_policy import (
    AgentBehaviorPolicy,
    AntiRewardBehaviorPolicy,
    build_collection_policy,
    SkewBehaviorPolicy,
    SuboptimalBehaviorPolicy,
)


class _DiscreteSpace:
    def __init__(self, n):
        self.n = n


class _BoxSpace:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _FixedAgent:
    """Agent whose act() returns a fixed discrete action for every env."""

    def __init__(self, action_idx):
        self.action_idx = action_idx

    def act(self, obs, *a, **k):
        b = obs.shape[0]
        return ActionOutput(action=torch.full((b,), self.action_idx, dtype=torch.long))


# --------------------------------------------------------------------------
# Bar 1 — anti-reward returns argmin-Q, not argmax
# --------------------------------------------------------------------------
class _StubDQN:
    def __init__(self, q):
        self._q = q

    def q_network(self, obs):
        return self._q


def test_anti_reward_discrete_returns_argmin_not_argmax():
    # argmin index is 2 (value 0.0); argmax index is 1 (value 5.0).
    q = torch.tensor([[1.0, 5.0, 0.0, 3.0]])
    pol = AntiRewardBehaviorPolicy(
        _StubDQN(q), "discrete", _DiscreteSpace(4), epsilon=0.0
    )
    action = pol.act(torch.zeros(1, 3)).action
    assert action.item() == 2  # argmin, NOT 1 (argmax)


class _StubDDPGCritic:
    """Q(s,a) = a (single action dim) -> argmin picks the smallest action."""

    def critic(self, x):
        return x[:, -1:]


def test_anti_reward_continuous_picks_min_q_candidate():
    pol = AntiRewardBehaviorPolicy(
        _StubDDPGCritic(), "continuous", _BoxSpace([-1.0], [1.0]), n_candidates=3
    )
    obs = torch.zeros(1, 3)
    cands = torch.tensor([[[0.5]], [[-0.9]], [[0.1]]])  # [K=3, B=1, A=1]
    chosen = pol._argmin_over_candidates(obs, cands)
    assert torch.allclose(chosen, torch.tensor([[-0.9]]))  # lowest Q == lowest a


class _StubSAC:
    """Twin-Q on the DE-SCALED action: q1=q2=sum(a/scale). Confirms anti_reward
    applies SAC's action_scale before the critic."""

    action_scale = 2.0

    def q1(self, x):
        return x[:, -1:]

    def q2(self, x):
        return x[:, -1:]


def test_anti_reward_sac_uses_descaled_action():
    pol = AntiRewardBehaviorPolicy(
        _StubSAC(), "continuous", _BoxSpace([-2.0], [2.0]), n_candidates=2
    )
    obs = torch.zeros(1, 3)
    cands = torch.tensor([[[2.0]], [[-2.0]]])  # stored (scaled) actions
    chosen = pol._argmin_over_candidates(obs, cands)
    assert torch.allclose(chosen, torch.tensor([[-2.0]]))  # min descaled Q


# --------------------------------------------------------------------------
# Bar 2 — action-skew empirical distribution matches p
# --------------------------------------------------------------------------
def test_bias_skew_distribution_matches_p():
    agent = _FixedAgent(1)  # agent always picks action 1; preferred is 0
    space = _DiscreteSpace(4)
    obs = torch.zeros(8000, 3)

    # p = 1 -> always preferred (0); p = 0 -> always the agent (1).
    assert (
        SkewBehaviorPolicy(agent, "discrete", space, p=1.0).act(obs).action == 0
    ).all()
    assert (
        SkewBehaviorPolicy(agent, "discrete", space, p=0.0).act(obs).action == 1
    ).all()

    torch.manual_seed(0)
    a = SkewBehaviorPolicy(agent, "discrete", space, p=0.3).act(obs).action
    frac_pref = (a == 0).float().mean().item()
    assert abs(frac_pref - 0.3) < 0.03


def test_bias_skew_continuous_prefers_low_bound():
    agent = _FixedAgentBox(high=2.0)
    pol = SkewBehaviorPolicy(agent, "continuous", _BoxSpace([-2.0], [2.0]), p=1.0)
    action = pol.act(torch.zeros(5, 3)).action
    assert torch.allclose(action, torch.full((5, 1), -2.0))  # the low extreme


class _FixedAgentBox:
    def __init__(self, high):
        self.high = high

    def act(self, obs, *a, **k):
        b = obs.shape[0]
        return ActionOutput(action=torch.full((b, 1), self.high))


# --------------------------------------------------------------------------
# Bar 3 — suboptimality mixture mixes at beta
# --------------------------------------------------------------------------
def test_bias_suboptimal_fraction_matches_beta():
    n = 50
    agent = _FixedAgent(7)  # agent always picks 7; base is uniform over 50
    space = _DiscreteSpace(n)
    obs = torch.zeros(8000, 3)

    # beta = 1 -> all agent (7).
    assert (
        SuboptimalBehaviorPolicy(agent, "discrete", space, beta=1.0).act(obs).action
        == 7
    ).all()

    # beta = 0 -> all uniform base; action 7 appears only ~1/n (NOT all agent).
    torch.manual_seed(0)
    a0 = SuboptimalBehaviorPolicy(agent, "discrete", space, beta=0.0).act(obs).action
    assert (a0 == 7).float().mean().item() < 0.1

    # beta = 0.5 -> agent-fraction ~= beta + (1-beta)/n (base collisions).
    torch.manual_seed(0)
    a = SuboptimalBehaviorPolicy(agent, "discrete", space, beta=0.5).act(obs).action
    expected = 0.5 + 0.5 / n
    assert abs((a == 7).float().mean().item() - expected) < 0.03


# --------------------------------------------------------------------------
# Factory + default byte-identity
# --------------------------------------------------------------------------
def test_factory_agent_returns_default_policy():
    agent = _FixedAgent(0)
    pol = build_collection_policy("agent", agent, "discrete", _DiscreteSpace(2))
    assert isinstance(pol, AgentBehaviorPolicy) and pol.agent is agent


def test_factory_strength_maps_to_primary_param():
    agent = _FixedAgent(0)
    anti = build_collection_policy(
        "anti_reward", agent, "discrete", _DiscreteSpace(2), strength=0.25
    )
    skew = build_collection_policy(
        "bias_skew", agent, "discrete", _DiscreteSpace(2), strength=0.8
    )
    sub = build_collection_policy(
        "bias_suboptimal", agent, "discrete", _DiscreteSpace(2), strength=0.4
    )
    assert anti.epsilon == 0.25 and skew.p == 0.8 and sub.beta == 0.4


# --------------------------------------------------------------------------
# Runner integration — opt-in selection wires the policy into collection
# --------------------------------------------------------------------------
def test_runner_selects_opt_in_collection_policy(tmp_path):
    import csv

    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner, EVAL_COLUMNS, TRAIN_COLUMNS
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.config.device import detect_device
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=5,
        seed=0,
        behavior_policy="bias_suboptimal",
        behavior_strength=0.5,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,
        n_checkpoints=1,
        device=str(detect_device()),
        algorithm="dqn",
        aggregation="mean",
    )
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("dqn"),
    )
    # The opt-in policy is wired into the collection seam, not the default.
    assert isinstance(runner.collection_policy, SuboptimalBehaviorPolicy)
    assert runner.collection_policy.beta == 0.5
    runner.run()

    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS

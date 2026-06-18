"""feat/on-policy-behavior-policy-validation — reject on-policy algos under an
action-bias-only behavior_policy.

Behavior policies have two mechanism classes:
- action-bias-only (curiosity, anti_reward, bias_skew, bias_suboptimal): only
  consulted in the off-policy collection branch; on-policy algos (ppo, a2c, …)
  ignore them entirely, so listing one is a no-op redundant run.
- reward-perturbing (bias_confounded): also wraps train_env, so on-policy algos
  genuinely respond (affects_on_policy() == True).

_validate_algos_against_behavior_policy enforces this at YAML load time. The
on-policy set is derived from the algorithm registry (AlgorithmSpec.kind), so we
register the defaults first.
"""

from __future__ import annotations

import pytest
from src.benchmarking.registry import register_default_algorithms
from src.benchmarking.runner import _validate_algos_against_behavior_policy
from src.rl.policies.behavior_policy import (
    AntiRewardBehaviorPolicy,
    BehaviorPolicy,
    ConfoundedBehaviorPolicy,
    CuriosityBehaviorPolicy,
)

register_default_algorithms()  # populate the registry so kind lookup works


def test_action_bias_only_behavior_rejects_on_policy_algo():
    """behavior_policy=curiosity with ppo in algos must fail to load."""
    with pytest.raises(ValueError) as ei:
        _validate_algos_against_behavior_policy(["dqn", "ppo"], "curiosity")
    msg = str(ei.value)
    assert "curiosity" in msg and "ppo" in msg
    assert "on-policy" in msg or "action-bias-only" in msg


def test_action_bias_only_behavior_accepts_off_policy_only_algo_list():
    """behavior_policy=curiosity with algos=[dqn] loads successfully."""
    _validate_algos_against_behavior_policy(["dqn"], "curiosity")  # no raise


def test_anti_reward_also_rejects_on_policy_algo():
    """anti_reward is action-bias-only too — sac+ppo (continuous arm) is rejected."""
    with pytest.raises(ValueError):
        _validate_algos_against_behavior_policy(["sac", "ppo"], "anti_reward")


def test_confounded_behavior_accepts_on_policy_algo():
    """bias_confounded affects on-policy algos via its reward wrapper, so
    algos=[ppo, dqn] is valid (mirrors the Cell 7/8 online σ-sweep YAMLs)."""
    _validate_algos_against_behavior_policy(
        ["ppo", "dqn"], "bias_confounded"
    )  # no raise


def test_agent_behavior_accepts_any_algo_list():
    """behavior_policy=agent is the unconfounded baseline; any algo list loads."""
    _validate_algos_against_behavior_policy(["ppo", "dqn", "sac"], "agent")  # no raise


def test_affects_on_policy_flags():
    """The class-level flag is the source of truth the validator consults."""
    assert BehaviorPolicy.affects_on_policy() is False
    assert CuriosityBehaviorPolicy.affects_on_policy() is False
    assert AntiRewardBehaviorPolicy.affects_on_policy() is False
    assert ConfoundedBehaviorPolicy.affects_on_policy() is True


def test_validator_honors_injected_on_policy_set():
    """With an explicit on_policy_algos set, the registry is not consulted —
    a2c (registry-on-policy) is ignored when not in the injected set."""
    # a2c omitted from the injected set => treated as off-policy here => no raise.
    _validate_algos_against_behavior_policy(
        ["a2c"], "curiosity", on_policy_algos={"ppo"}
    )
    # ppo in the injected set => rejected.
    with pytest.raises(ValueError):
        _validate_algos_against_behavior_policy(
            ["ppo"], "curiosity", on_policy_algos={"ppo"}
        )

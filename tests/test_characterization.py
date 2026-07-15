"""Characterization tests: freeze the observable contracts of master.

These tests encode the §2 invariants of the causal-RL refactor:
CSV schemas, CLI flag names/defaults, registry contents, checkpoint spacing.
They must stay green, unmodified, through every phase.
"""

from __future__ import annotations

import re
import sys

import pytest
from tests.conftest import GOLDEN_DIR, REPO_ROOT

# ---------------------------------------------------------------------------
# CSV schemas
# ---------------------------------------------------------------------------

EXPECTED_TRAIN_COLUMNS = [
    "episode",
    "algorithm",
    "environment",
    "train_return_mean",
    "train_return_std",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "kl",
    "critic_loss",
    "actor_loss",
    "q_loss",
]

EXPECTED_EVAL_COLUMNS = [
    "episode",
    "algorithm",
    "environment",
    "eval_return_mean",
    "eval_return_std",
]

EXPECTED_CRITIC_ABLATION_COLUMNS = [
    "episode",
    "algorithm",
    "environment",
    "critic",
    "train_loss",
    "advantage_mean",
    "explained_variance",
    "pearson",
    "spearman",
    "mutual_information",
    "kl",
    "js_normalized",
    "mse",
    "td_error_mean",
    "real_reward_mean",
    "pred_reward_mean",
    "reward_explained_variance",
    "reward_pearson",
    "reward_spearman",
    "reward_mutual_information",
    "reward_kl",
    "reward_js_normalized",
    "reward_mse",
    "reward_error_mean",
]


def test_train_csv_schema_frozen():
    from src.benchmarking.runner import TRAIN_COLUMNS

    assert TRAIN_COLUMNS == EXPECTED_TRAIN_COLUMNS


def test_eval_csv_schema_frozen():
    from src.benchmarking.runner import EVAL_COLUMNS

    assert EVAL_COLUMNS == EXPECTED_EVAL_COLUMNS


def test_critic_ablation_csv_schema_frozen():
    from src.benchmarking.critic_ablation import CRITIC_ABLATION_COLUMNS

    assert CRITIC_ABLATION_COLUMNS == EXPECTED_CRITIC_ABLATION_COLUMNS


# ---------------------------------------------------------------------------
# CLI flags and defaults
# ---------------------------------------------------------------------------


def test_cli_defaults_frozen(monkeypatch):
    import main as main_mod

    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = main_mod.parse_args()

    assert args.mode == "benchmark"
    assert args.ablation is False
    assert args.envs is None
    assert args.algos is None
    assert args.env_set is None
    assert args.env_wrapper == "auto"
    assert args.env_entry_point is None
    assert args.env_kwargs is None
    assert args.n_train_envs == 16
    assert args.n_eval_envs == 16
    assert args.rollout_len == 1024
    assert args.n_episodes == 250
    assert args.n_checkpoints == 25
    assert args.seed == 42
    assert args.reproduce is None
    assert args.deterministic is False
    assert args.aggregation == "iqm"
    assert args.ablation_critics is None
    assert args.ablation_lr == 3e-4
    assert args.ablation_hidden_dims == "64,64"
    assert args.ablation_bins == 32


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------


def test_default_algorithms_registered():
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    expected_kinds = {
        "vanilla": "on_policy",
        "a2c": "on_policy",
        "ppo": "on_policy",
        "trpo": "on_policy",
        "dqn": "off_policy",
        "ddpg": "off_policy",
    }
    for name, kind in expected_kinds.items():
        spec = registry.get(name)
        assert spec.kind == kind, name


def test_env_sets_frozen_names():
    from src.benchmarking.registry import ENV_SETS, expand_env_set

    for name in ("gymnasium", "mujoco", "gymnasium-robotics"):
        assert name in ENV_SETS
        assert len(expand_env_set(name)) > 0
    assert "CartPole-v1" in expand_env_set("gymnasium")
    assert "HalfCheetah-v5" in expand_env_set("mujoco")


# ---------------------------------------------------------------------------
# Checkpoint spacing
# ---------------------------------------------------------------------------


def test_checkpoint_episodes_contract():
    from src.config.defaults import TrainingConfig

    eps = TrainingConfig(n_episodes=250, n_checkpoints=25).checkpoint_episodes()
    assert eps[0] == 0 and eps[-1] == 249 and len(eps) == 25

    # clamping: count in [2, n_episodes]
    eps = TrainingConfig(n_episodes=5, n_checkpoints=25).checkpoint_episodes()
    assert eps == [0, 1, 2, 3, 4]
    eps = TrainingConfig(n_episodes=5, n_checkpoints=1).checkpoint_episodes()
    assert eps == [0, 4]


# ---------------------------------------------------------------------------
# Dependency pins (Phase-3 gate): d3rlpy declares gymnasium==1.0.0 but runs
# on 1.2.3; this guard catches any accidental downgrade (golden numbers and
# the autoreset fix depend on gymnasium >=1.1 semantics).
# ---------------------------------------------------------------------------


def test_gymnasium_version_pinned():
    import gymnasium

    assert gymnasium.__version__ == "1.2.3", (
        "gymnasium was downgraded (probably by a d3rlpy reinstall); "
        "restore with: pip install gymnasium==1.2.3"
    )


# ---------------------------------------------------------------------------
# Grep snapshots (§3.3 / §8): seeding call sites and critic_ablation refs
# ---------------------------------------------------------------------------

_SEED_PATTERN = re.compile(r"manual_seed|np\.random\.seed|default_rng|seed\(")
_ABLATION_PATTERN = re.compile(r"critic_ablation")


def _grep_py_files(pattern: re.Pattern) -> list[str]:
    """Content snapshot: ``<file>:<matched line>`` with NO line number. Keying on
    line numbers made this snapshot drift on every PR that merely inserted a line
    anywhere above a match in main.py/runner.py; the anti-drift signal we actually
    want is which matching LINES exist, not where. Sorted-unique = a content set."""
    files = sorted((REPO_ROOT / "src").rglob("*.py")) + [REPO_ROOT / "main.py"]
    hits: set[str] = set()
    for path in files:
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT)
        for line in path.read_text(encoding="utf-8").splitlines():
            if pattern.search(line):
                hits.add(f"{rel}:{line}")
    return sorted(hits)


@pytest.mark.parametrize(
    "pattern,golden_name",
    [
        (_SEED_PATTERN, "seeding_call_sites.txt"),
        (_ABLATION_PATTERN, "critic_ablation_refs.txt"),
    ],
    ids=["seeding-call-sites", "critic-ablation-refs"],
)
def test_grep_snapshot_unchanged(pattern, golden_name):
    golden = (GOLDEN_DIR / golden_name).read_text(encoding="utf-8").splitlines()
    current = _grep_py_files(pattern)
    # Content-set comparison (line numbers stripped): stable under line insertions,
    # catches only genuine addition/removal of a matching call site.
    assert set(current) == set(golden), (
        f"{golden_name} drifted (a matching call site was ADDED or REMOVED, not "
        "merely moved). If intentional, regenerate the snapshot and note it."
    )

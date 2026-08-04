"""Phase-2 subprocess supervisor for the (regime × L-sweep) driver.

Three properties, matching the approved design:

S1  max_workers==1 is the SERIAL in-process path: its leaf set is byte-identical to a
    direct run_cell (same 19 leaves at the same parameter paths). The parallel
    machinery must not engage at 1.
S2  max_workers==2 on a 2-group cell (1 env × 2 seeds) runs both groups to completion
    with per-group Minari stores that are created, used, and cleaned up — and no
    namespace crash (the whole reason for the per-group store). The default ~/.minari
    store is never touched (isolation), and the failure summary is empty. Offline
    groups are point-grain: one `generate` task + one `train` task per sweep point,
    each with its own log.
S3  a deliberately failing group (a bogus env id, which fails at the generate task's
    generator build; its train tasks are dep-skipped) is recorded in failed_groups
    while the OTHER group still completes — a non-zero overall exit, never a silent
    drop.

The parallel tests are minutes-scale (each group runs a full 7-point smoke across
its task set) so they carry the `slow` marker. They run on the detected device (``_DEV``) — the same
device the proven M2 smoke uses; the σ-arm confounding gate is device-sensitive at
the tiny budget, so matching that configuration keeps them stable.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import minari
import pytest
from src.benchmarking.regime_sweep import run_cell
from src.benchmarking.sweep_supervisor import run_sweep
from src.config.device import detect_device

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
_OFFLINE_MDP = (
    _REPO / "reproducibility" / "rl_regimes" / "offline_mdp" / "critic_ablation.yaml"
)
_DEV = str(detect_device())
# Mirrors regime_sweep._SMOKE_BUDGET (rollout_episodes=40 keeps the σ=1.0 gate stable).
_TINY = {
    "n_episodes": 1,
    "n_checkpoints": 2,
    "n_train_envs": 2,
    "n_eval_envs": 2,
    "rollout_len": 2,
    "rollout_episodes": 40,
    # small offline budget (else the merge with _base inherits 50_000 offline_grad_steps).
    "offline_grad_steps": 4,
}
# 19 leaves per (env, seed) group for offline_mdp with ONE algo:
#   basic(4) + biased(1×3) + confounded(4×3).
_LEAVES_PER_GROUP = 19


def _purge(prefix: str) -> None:
    for d in list(minari.list_local_datasets()):
        if str(d).startswith(prefix):
            try:
                minari.delete_dataset(d)
            except Exception:
                pass


def _rel_leaves(leaves, root: Path):
    return {str(Path(p).resolve().relative_to(root.resolve())) for p in leaves}


# --------------------------------------------------------------------------- #
# S1 — max_workers==1 is byte-identical to a direct run_cell                     #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_s1_serial_path_matches_direct_run_cell(tmp_path):
    _purge("s1direct/")
    _purge("s1serial/")
    root_direct = tmp_path / "direct"
    direct = run_cell(
        _OFFLINE_MDP,
        results_root=str(root_direct),
        dataset_prefix="s1direct",
        envs=["CartPole-v1"],
        algos=["cql"],
        seeds=[0],
        budget_overrides=_TINY,
        device=_DEV,
    )

    root_serial = tmp_path / "serial"
    result = run_sweep(
        _OFFLINE_MDP,
        results_root=str(root_serial),
        dataset_prefix="s1serial",
        envs=["CartPole-v1"],
        algos=["cql"],
        seeds=[0],
        budget_overrides=_TINY,
        device=_DEV,
        max_workers=1,  # the serial in-process path — must equal run_cell exactly
    )

    # No subprocess machinery at 1: single group, no per-worker log.
    assert len(result.groups) == 1
    assert result.groups[0].log_path is None
    assert result.ok and result.failed_groups == []
    # The leaf SET is identical, parameter-path for parameter-path.
    assert _rel_leaves(result.leaves, root_serial) == _rel_leaves(direct, root_direct)
    assert len(result.leaves) == _LEAVES_PER_GROUP == len(direct)
    _purge("s1direct/")
    _purge("s1serial/")


# --------------------------------------------------------------------------- #
# S2 — max_workers==2 runs both groups; stores isolated + cleaned; no crash      #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_s2_parallel_two_groups_isolated_and_clean(tmp_path):
    _purge("s2par/")
    root = tmp_path / "results"
    stores = tmp_path / "stores"
    logs = tmp_path / "logs"

    result = run_sweep(
        _OFFLINE_MDP,
        results_root=str(root),
        dataset_prefix="s2par",
        envs=["CartPole-v1"],
        algos=["cql"],
        seeds=[0, 1],  # two groups -> two workers
        device=_DEV,
        max_workers=2,
        smoke=True,  # subprocess budget rides --smoke
        scratch_root=str(stores),
        log_dir=str(logs),
    )

    # Both groups completed, none failed.
    assert result.ok, [g.reason for g in result.failed_groups]
    assert result.failed_groups == []
    assert len(result.groups) == 2
    for g in result.groups:
        assert g.ok and g.returncode == 0
        assert g.expected_leaf_count == _LEAVES_PER_GROUP
        assert len(g.leaves) == _LEAVES_PER_GROUP
    assert len(result.leaves) == 2 * _LEAVES_PER_GROUP

    # Per-task logs kept separate: one generate log + one per-point train log
    # per group (offline_mdp has 7 sweep points).
    for seed in (0, 1):
        assert (logs / f"group_CartPole-v1_seed{seed}_gen.log").exists()
        point_logs = list(logs.glob(f"group_CartPole-v1_seed{seed}_b*.log"))
        assert len(point_logs) == 7, sorted(p.name for p in point_logs)

    # Per-group stores were created + CLEANED (no leftover worker_ dirs under scratch).
    leftover = [p.name for p in stores.iterdir() if p.name.startswith("worker_")]
    assert leftover == [], leftover

    # Isolation: the default ~/.minari store never saw these datasets (they lived and
    # died inside each worker's MINARI_DATASETS_PATH) -> no shared-namespace write.
    assert not any(str(d).startswith("s2par/") for d in minari.list_local_datasets())
    _purge("s2par/")


# --------------------------------------------------------------------------- #
# S3 — a failing group is surfaced, the other group survives, exit is non-zero  #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_s3_failing_group_recorded_other_survives(tmp_path):
    _purge("s3par/")
    root = tmp_path / "results"
    stores = tmp_path / "stores"
    logs = tmp_path / "logs"

    result = run_sweep(
        _OFFLINE_MDP,
        results_root=str(root),
        dataset_prefix="s3par",
        # one real env (succeeds) + one bogus env (its generate task fails at
        # generator build; its per-point train tasks are dependency-skipped).
        envs=["CartPole-v1", "NotARealEnvXYZ-v0"],
        algos=["cql"],
        seeds=[0],
        device=_DEV,
        max_workers=2,
        smoke=True,
        scratch_root=str(stores),
        log_dir=str(logs),
    )

    # Overall failure -> the CLI exits non-zero.
    assert not result.ok
    assert len(result.groups) == 2

    failed = result.failed_groups
    assert len(failed) == 1
    assert failed[0].env == "NotARealEnvXYZ-v0"
    assert failed[0].returncode != 0
    assert "exited" in failed[0].reason
    # The failure log is captured and non-empty (the traceback landed there).
    assert failed[0].log_path is not None and failed[0].log_path.exists()
    assert failed[0].log_path.stat().st_size > 0

    # The GOOD group was NOT dropped: it completed with its full leaf set.
    good = [g for g in result.groups if g.env == "CartPole-v1"]
    assert len(good) == 1 and good[0].ok
    assert len(good[0].leaves) == _LEAVES_PER_GROUP
    assert len(result.leaves) == _LEAVES_PER_GROUP  # only the good group's leaves

    # Stores cleaned even for the crashed worker.
    leftover = [p.name for p in stores.iterdir() if p.name.startswith("worker_")]
    assert leftover == [], leftover
    _purge("s3par/")

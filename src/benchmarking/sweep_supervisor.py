"""Subprocess supervisor for the (regime × L-sweep) driver.

The parallel grain is the (env, seed) GROUP, never a single sweep point. One
subprocess owns ONE group's full L (all 7 points + training) start-to-finish, so
the shared-generator invariant (run_cell's ``for env: for seed:`` iteration) lives
entirely inside a group and different groups share nothing. There are ``E*S`` such
groups for ``max_workers`` slots to churn.

``max_workers == 1`` is the serial path: the supervisor calls ``run_cell`` IN-PROCESS
(no subprocess, no env-var, no log machinery) so it is BYTE-IDENTICAL to the
pre-supervisor behaviour — same leaves, same order. The subprocess pool engages only
at ``max_workers >= 2``.

Isolation (offline): each worker gets its OWN Minari store via
``MINARI_DATASETS_PATH=<scratch>/worker_<env>_<seed>``, deleted when the group
finishes. This kills the store-level namespace-metadata TOCTOU (all groups of a
regime share the id-namespace ``sweep/{regime}``, whose ``namespace_metadata.json``
would otherwise be a concurrent write) AND the full-store ``list_local_datasets``
scan the load path runs. Dataset content is unchanged (ids are identical; only the
store root differs), so the datasets a worker writes are read back by that same
worker's training step through the same env var.

Driver-agnostic: ``_supervise`` knows only (a) an opaque list of groups, (b) a
per-group ``build_command`` -> argv, (c) a per-group ``prepare_group`` -> (env
overrides, cleanup), and (d) a per-group ``verify_group`` -> result. The offline
driver (``run_sweep`` below) supplies the Minari env hook + leaf-count verifier. A
future ONLINE driver reuses ``_supervise`` verbatim with an empty env hook (online
has no Minari store) and its own single-group entry point — see the ONLINE PLUG-IN
note on ``_supervise``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.benchmarking.regime_sweep import (
    _safe,
    arm_label,
    critics_for_arm,
    load_sweep_spec,
    results_leaf,
    run_cell,
    sweep_points,
)

# Repo root: this file is <root>/src/benchmarking/sweep_supervisor.py, so parents[2]
# is the root a ``python -m src.benchmarking.regime_sweep`` child must run from.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The run-dir file set every leaf must hold to count as complete (mirrors the shared
# files _run_point copies into each per-critic leaf). A dir missing any of these is a
# half-written / truncated leaf and does NOT count toward the group's leaf total.
_LEAF_MARKER_FILES = ("config.yaml", "critic_ablation_metrics.csv")


# --------------------------------------------------------------------------- #
# Result types                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class GroupResult:
    """The outcome of one (env, seed) group."""

    env: str
    seed: int
    returncode: int
    ok: bool
    reason: str  # "" when ok; else why it failed (crash / short leaf count)
    log_path: Optional[Path]
    leaves: List[Path] = field(default_factory=list)
    expected_leaf_count: int = 0

    @property
    def group(self) -> Tuple[str, int]:
        return (self.env, self.seed)


@dataclass
class SweepResult:
    """Aggregate outcome of a sweep run (serial or parallel)."""

    leaves: List[Path]
    groups: List[GroupResult]

    @property
    def failed_groups(self) -> List[GroupResult]:
        return [g for g in self.groups if not g.ok]

    @property
    def ok(self) -> bool:
        return not self.failed_groups


# --------------------------------------------------------------------------- #
# Leaf accounting (the truncation check)                                        #
# --------------------------------------------------------------------------- #
def _expected_leaves(
    spec, algos: Sequence[str], env: str, seed: int, results_root: str | Path
) -> List[Path]:
    """The exact leaf paths a completed (env, seed) group must produce — one per
    (sweep point x algo x that arm's critic). Critic sets vary by arm (biased runs
    observational-only), so this is a SUM over points, not a flat product; that makes
    it the authoritative expected count for the truncation check."""
    out: List[Path] = []
    for beta, sigma in sweep_points():
        arm = arm_label(beta, sigma)
        for algo in algos:
            for critic in critics_for_arm(arm):
                out.append(
                    results_leaf(
                        results_root, spec.regime, beta, sigma, env, algo, critic, seed
                    )
                )
    return out


def _leaf_complete(leaf: Path) -> bool:
    return leaf.is_dir() and all((leaf / f).exists() for f in _LEAF_MARKER_FILES)


# --------------------------------------------------------------------------- #
# The generic pool (driver-agnostic)                                            #
# --------------------------------------------------------------------------- #
def _supervise(
    groups: Sequence[object],
    *,
    build_command: Callable[[object], List[str]],
    prepare_group: Callable[[object], Tuple[Dict[str, str], Callable[[], None]]],
    verify_group: Callable[[object, int, Optional[Path]], GroupResult],
    log_dir: Path,
    log_name: Callable[[object], str],
    max_workers: int,
    poll_interval: float = 0.1,
) -> List[GroupResult]:
    """Keep ``max_workers`` subprocesses alive, refilling as they finish.

    ONLINE PLUG-IN: this function is intentionally offline-agnostic. The future online
    driver calls it with the SAME shape — its own ``build_command`` (an online
    single-group entry point), an empty-env ``prepare_group`` (online has no Minari
    store, so env overrides = {} and cleanup = no-op), and its own ``verify_group``.
    The pool / refill / failure / per-worker-log machinery below is reused verbatim.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    pending: List[object] = list(groups)
    # popen -> (group, log file handle, cleanup callable, log path)
    running: Dict[subprocess.Popen, Tuple[object, object, Callable[[], None], Path]] = (
        {}
    )
    results: List[GroupResult] = []

    def _launch(group: object) -> None:
        env_overrides, cleanup = prepare_group(group)
        log_path = log_dir / f"group_{log_name(group)}.log"
        logf = open(log_path, "w")
        env = {**os.environ, **env_overrides}
        proc = subprocess.Popen(
            build_command(group),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        running[proc] = (group, logf, cleanup, log_path)

    while pending or running:
        while pending and len(running) < max_workers:
            _launch(pending.pop(0))
        done = [p for p in running if p.poll() is not None]
        if not done:
            time.sleep(poll_interval)
            continue
        for proc in done:
            group, logf, cleanup, log_path = running.pop(proc)
            logf.close()
            try:
                cleanup()  # tear the per-worker store down regardless of outcome
            except Exception:
                pass
            results.append(verify_group(group, int(proc.returncode), log_path))
    return results


# --------------------------------------------------------------------------- #
# Offline driver                                                                #
# --------------------------------------------------------------------------- #
def run_sweep(
    sweep_yaml: str | Path,
    *,
    results_root: str | Path = "results",
    dataset_prefix: str = "sweep",
    device: str | None = None,
    envs: Sequence[str] | None = None,
    algos: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    budget_overrides: Dict[str, int] | None = None,
    max_workers: int | None = None,
    smoke: bool = False,
    log_dir: str | Path | None = None,
    scratch_root: str | Path | None = None,
) -> SweepResult:
    """Run a cell's (env, seed) groups, serial or ``max_workers``-wide.

    ``max_workers``: None -> read from the spec (``_base/parallel.yaml``, default 1);
    an int overrides it (the CLI ``--max-workers`` path). ``max_workers == 1`` runs
    ``run_cell`` in-process (byte-identical to the pre-supervisor serial path).
    ``>= 2`` fans the groups across subprocesses, each an isolated single-group
    ``run_cell`` (``--envs <env> --seeds <seed> --max-workers 1``).

    ``budget_overrides`` applies to the in-process (max_workers==1) path. Under the
    subprocess path, budgets ride the child's YAML + ``smoke`` flag (arbitrary
    per-key overrides are not forwardable over the CLI); pass ``smoke=True`` for a
    tiny-budget parallel run.
    """
    spec = load_sweep_spec(sweep_yaml)
    if spec.data_regime != "offline":
        # run_cell would raise the same NotImplementedError; surface it early and with
        # the same message so the online path is an explicit not-yet-here, not a crash.
        raise NotImplementedError(
            f"run_sweep drives the OFFLINE path; regime '{spec.regime}' is online. "
            "The online driver will reuse _supervise with an empty Minari env hook."
        )

    run_envs = list(envs) if envs is not None else spec.envs
    run_algos = list(algos) if algos is not None else spec.algos
    run_seeds = [int(s) for s in (seeds if seeds is not None else spec.seeds)]
    eff_workers = int(spec.max_workers if max_workers is None else max_workers)

    # --- serial in-process path: BYTE-IDENTICAL to today's run_cell ----------
    if eff_workers <= 1:
        leaves = run_cell(
            sweep_yaml,
            results_root=results_root,
            dataset_prefix=dataset_prefix,
            device=device,
            envs=run_envs,
            algos=run_algos,
            seeds=run_seeds,
            budget_overrides=budget_overrides,
        )
        groups = []
        for e in run_envs:
            for s in run_seeds:
                expected = _expected_leaves(spec, run_algos, e, s, results_root)
                groups.append(
                    GroupResult(
                        env=e,
                        seed=s,
                        returncode=0,
                        ok=True,
                        reason="",
                        log_path=None,
                        leaves=[p for p in expected if _leaf_complete(p)],
                        expected_leaf_count=len(expected),
                    )
                )
        return SweepResult(leaves=list(leaves), groups=groups)

    # --- parallel subprocess path (max_workers >= 2) -------------------------
    scratch = (
        Path(scratch_root)
        if scratch_root
        else Path(tempfile.mkdtemp(prefix="sweep_stores_"))
    )
    scratch.mkdir(parents=True, exist_ok=True)
    logs = Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="sweep_logs_"))

    groups: List[Tuple[str, int]] = [(e, s) for e in run_envs for s in run_seeds]

    def _group_tag(group: object) -> str:
        env, seed = group  # type: ignore[misc]
        return f"{_safe(env)}_seed{seed}"

    def _build_command(group: object) -> List[str]:
        env, seed = group  # type: ignore[misc]
        cmd = [
            sys.executable,
            "-m",
            "src.benchmarking.regime_sweep",
            str(sweep_yaml),
            "--envs",
            env,
            "--seeds",
            str(seed),
            "--results-root",
            str(results_root),
            "--dataset-prefix",
            str(dataset_prefix),
            # Force the child onto the serial in-process run_cell path for its ONE
            # group — without this it would re-read max_workers>=2 from the YAML and
            # recursively spawn a pool.
            "--max-workers",
            "1",
        ]
        if run_algos:
            cmd += ["--algos", *run_algos]
        if device:
            cmd += ["--device", str(device)]
        if smoke:
            # --smoke sets the tiny budget in the child; the explicit --results-root /
            # --dataset-prefix above still win (they are not None), so results land in
            # OUR tree, not results_smoke/.
            cmd += ["--smoke"]
        return cmd

    def _prepare_group(
        group: object,
    ) -> Tuple[Dict[str, str], Callable[[], None]]:
        # The ONLY offline-specific bit: a per-worker Minari store. The online driver
        # supplies ({}, no-op) here instead.
        store = scratch / f"worker_{_group_tag(group)}"
        store.mkdir(parents=True, exist_ok=True)

        def _cleanup() -> None:
            shutil.rmtree(store, ignore_errors=True)

        return {"MINARI_DATASETS_PATH": str(store)}, _cleanup

    def _verify_group(
        group: object, returncode: int, log_path: Optional[Path]
    ) -> GroupResult:
        env, seed = group  # type: ignore[misc]
        expected = _expected_leaves(spec, run_algos, env, seed, results_root)
        present = [p for p in expected if _leaf_complete(p)]
        if returncode != 0:
            ok, reason = False, f"subprocess exited {returncode}"
        elif len(present) != len(expected):
            # A clean exit that still dropped leaves is a SILENT truncation — the
            # check that catches it. Treat as a failure, never as done.
            ok, reason = (
                False,
                f"leaf count {len(present)} != expected {len(expected)}",
            )
        else:
            ok, reason = True, ""
        return GroupResult(
            env=env,
            seed=seed,
            returncode=returncode,
            ok=ok,
            reason=reason,
            log_path=log_path,
            leaves=present,
            expected_leaf_count=len(expected),
        )

    group_results = _supervise(
        groups,
        build_command=_build_command,
        prepare_group=_prepare_group,
        verify_group=_verify_group,
        log_dir=logs,
        log_name=_group_tag,
        max_workers=eff_workers,
    )
    # Deterministic order (env, then seed) regardless of finish order.
    group_results.sort(key=lambda g: (g.env, g.seed))
    all_leaves = [leaf for g in group_results if g.ok for leaf in g.leaves]
    return SweepResult(leaves=all_leaves, groups=group_results)


def format_summary(result: SweepResult) -> str:
    """A one-block human summary for the CLI (goes to the supervisor's own stream)."""
    lines = [
        f"[sweep_supervisor] {len(result.groups)} group(s), "
        f"{len(result.leaves)} leaf(s) written, "
        f"{len(result.failed_groups)} failed."
    ]
    for g in result.failed_groups:
        loc = f" (log: {g.log_path})" if g.log_path else ""
        lines.append(f"  FAILED {g.env} seed{g.seed}: {g.reason}{loc}")
    return "\n".join(lines)

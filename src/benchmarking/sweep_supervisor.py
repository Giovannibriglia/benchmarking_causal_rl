"""Subprocess supervisor for the (regime × L-sweep) driver.

The parallel grain (offline) is the (env, seed, sweep-point) TASK: per (env, seed)
group one ``generate`` task builds the shared generator, every point dataset and
the M1 hash gate in ONE process (the shared-generator invariant is untouched — it
never spans processes), then one ``train`` task per sweep point trains that point's
algos/critics, gated on its group's ``generate``. Training is numerics-identical
to the monolithic child because both ends re-seed (``BenchmarkRunner.run`` and the
per-point rollout seeding). The finer grain load-balances the pool: a cheap env's
group no longer strands its worker while an expensive one grinds, and there are
``E*S*(1+P)`` tasks instead of ``E*S`` groups for ``max_workers`` slots to churn.
ONLINE cells keep the whole-group task (no datasets, nothing to phase-split).

``max_workers == 1`` is the serial path: the supervisor calls ``run_cell`` IN-PROCESS
(no subprocess, no env-var, no log machinery) so it is BYTE-IDENTICAL to the
pre-supervisor behaviour — same leaves, same order. The subprocess pool engages only
at ``max_workers >= 2``.

Isolation (offline): each GROUP gets its OWN Minari store via
``MINARI_DATASETS_PATH=<scratch>/worker_<env>_<seed>``, written only by the group's
single ``generate`` task, read by its ``train`` tasks (concurrent readers of a
static store — no writer left, so no race), and deleted when the group's LAST task
finishes (refcounted). This kills the store-level namespace-metadata TOCTOU (all
groups of a regime share the id-namespace ``sweep/{regime}``, whose
``namespace_metadata.json`` would otherwise be a concurrent write) AND the
full-store ``list_local_datasets`` scan the load path runs. Dataset content is
unchanged (ids are identical; only the store root differs).

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
import re
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
    classical_results_leaf,
    load_sweep_spec,
    param_dirname,
    parse_algo_entry,
    results_leaf,
    run_cell,
)

# Repo root: this file is <root>/src/benchmarking/sweep_supervisor.py, so parents[2]
# is the root a ``python -m src.benchmarking.regime_sweep`` child must run from.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _leaf_marker_files(spec) -> Tuple[str, ...]:
    """The run-dir file set every leaf must hold to count as complete. A dir
    missing any of these is a half-written / truncated leaf and does NOT count
    toward the group's leaf total. The OFFLINE critic ablation slices a
    critic_ablation_metrics.csv into every leaf; classical and online-ablation
    leaves are ordinary benchmark run dirs (eval_metrics.csv is their tail file)."""
    if spec.simulation == "critic_ablation" and spec.data_regime == "offline":
        return ("config.yaml", "critic_ablation_metrics.csv")
    return ("config.yaml", "eval_metrics.csv")


# --------------------------------------------------------------------------- #
# Result types                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Task:
    """One schedulable unit: a whole group (online), or a group's ``generate`` /
    per-point ``train`` slice (offline)."""

    env: str
    seed: int
    kind: str  # "group" | "generate" | "train"
    point: Optional[str] = None  # param dirname, kind == "train" only

    @property
    def group(self) -> Tuple[str, int]:
        return (self.env, self.seed)


@dataclass
class GroupResult:
    """The outcome of one (env, seed) group (or, internally, one task of it)."""

    env: str
    seed: int
    returncode: int
    ok: bool
    reason: str  # "" when ok; else why it failed (crash / short leaf count)
    log_path: Optional[Path]
    leaves: List[Path] = field(default_factory=list)
    expected_leaf_count: int = 0
    label: str = ""  # task tag ("gen" / point dirname) on internal task results

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
    spec,
    algos: Sequence[str],
    env: str,
    seed: int,
    results_root: str | Path,
    *,
    point: Optional[str] = None,
) -> List[Path]:
    """The exact leaf paths a completed (env, seed) group must produce —
    restricted to one sweep point when ``point`` (a param dirname) is given.

    critic_ablation: one per (sweep point x algo x that arm's critic) — critic
    sets vary by arm (biased runs observational-only), so this is a SUM over
    points, not a flat product. classical: one per (sweep point x algo), under
    the ``{regime}/classical/`` subtree. Either way this is the authoritative
    expected count for the truncation check."""
    out: List[Path] = []
    for beta, sigma in spec.points():
        if point is not None and param_dirname(beta, sigma) != point:
            continue
        arm = arm_label(beta, sigma)
        for algo in algos:
            # the {algo} path segment is the entry's algo_id (verbatim for the
            # explicit name__actor__critic form) — must match the drivers.
            _, _, _, algo_id = parse_algo_entry(algo, spec.observability)
            if spec.simulation == "classical":
                out.append(
                    classical_results_leaf(
                        results_root, spec.regime, beta, sigma, env, algo_id, seed
                    )
                )
            else:
                for critic in spec.critics_for(arm):
                    out.append(
                        results_leaf(
                            results_root,
                            spec.regime,
                            beta,
                            sigma,
                            env,
                            algo_id,
                            critic,
                            seed,
                        )
                    )
    return out


def _leaf_complete(leaf: Path, marker_files: Sequence[str]) -> bool:
    return leaf.is_dir() and all((leaf / f).exists() for f in marker_files)


# --------------------------------------------------------------------------- #
# Live multi-bar mirror (interactive terminals only)                            #
# --------------------------------------------------------------------------- #
# A tqdm-rendered segment ("37%|###  | 18500/50000 [...]"). The children's own
# tqdm output lands in their per-group log; the parent MIRRORS the latest such
# segment per worker into a fixed terminal slot — multi-process tqdm is not a
# thing (each worker is its own OS process; the terminal has one cursor), so
# the parent is the single writer that multiplexes the bars.
_BAR_SEG = re.compile(r"\d+%\|")


def _last_bar_line(log_path: Path, tail_bytes: int = 8192) -> Optional[str]:
    """The child's CURRENT progress line: last tqdm-looking segment in the log
    tail (tqdm redraws with \\r, plain prints end with \\n — split on both and
    take the last bar-shaped piece; fall back to the last non-empty line)."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    segs = [s.strip() for s in re.split(r"[\r\n]", chunk) if s.strip()]
    for s in reversed(segs):
        if _BAR_SEG.search(s):
            return s
    return segs[-1] if segs else None


class _LiveBars:
    """Fixed per-worker-slot terminal lines mirroring the children's own tqdm
    bars. Active only on an interactive stdout; everywhere else (CI, piped,
    tests) every method is a no-op and the plain launched/finished prints keep
    the historical line-per-event behavior."""

    def __init__(self, n_slots: int, enabled: bool) -> None:
        self._bars: list = []
        self._by_group: Dict[str, int] = {}  # group tag -> slot
        self._free: List[int] = list(range(n_slots))
        self._last_refresh = 0.0
        self.enabled = enabled and sys.stdout.isatty()
        if not self.enabled:
            return
        try:
            from tqdm import tqdm
        except ImportError:
            self.enabled = False
            return
        self._tqdm = tqdm
        self._width = shutil.get_terminal_size().columns
        self._bars = [
            tqdm(
                total=1,
                position=i,
                bar_format="{desc}",
                leave=False,
                dynamic_ncols=True,
            )
            for i in range(n_slots)
        ]
        for i, b in enumerate(self._bars):
            b.set_description_str(f"[worker {i}] idle")

    def print(self, msg: str) -> None:
        """A persistent line that does not tear the bars."""
        if self.enabled:
            self._tqdm.write(msg)
        else:
            print(msg, flush=True)

    def attach(self, tag: str) -> None:
        if self.enabled and self._free:
            self._by_group[tag] = self._free.pop(0)

    def detach(self, tag: str) -> None:
        if not self.enabled:
            return
        slot = self._by_group.pop(tag, None)
        if slot is not None:
            self._bars[slot].set_description_str(f"[worker {slot}] idle")
            self._free.insert(0, slot)

    def refresh(self, running_logs: Dict[str, Path], min_interval: float = 0.5):
        if not self.enabled:
            return
        now = time.time()
        if now - self._last_refresh < min_interval:
            return
        self._last_refresh = now
        for tag, log_path in running_logs.items():
            slot = self._by_group.get(tag)
            if slot is None:
                continue
            line = _last_bar_line(log_path) or "starting..."
            text = f"[{tag}] {line}"
            self._bars[slot].set_description_str(text[: max(self._width - 2, 40)])

    def close(self) -> None:
        for b in self._bars:
            b.close()


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
    dependencies: Optional[Callable[[object], Sequence[object]]] = None,
    skip_group: Optional[Callable[[object, GroupResult], GroupResult]] = None,
) -> List[GroupResult]:
    """Keep ``max_workers`` subprocesses alive, refilling as they finish.

    ``dependencies(task)`` (optional) returns the tasks that must finish OK before
    ``task`` may launch (matched by object identity). A task whose dependency
    FAILED is never launched: it is resolved through ``skip_group(task,
    failed_dep_result)`` so the driver can synthesize its failure result AND run
    its bookkeeping (store refcounts). With no ``dependencies`` this is the
    historical FIFO pool.

    ONLINE PLUG-IN: this function is intentionally offline-agnostic. The online
    driver calls it with the SAME shape — its own ``build_command``, an empty-env
    ``prepare_group`` (online has no Minari store), and its own ``verify_group``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    total = len(groups)
    # Surface the log dir UP FRONT: the workers' stdout/stderr (tqdm included)
    # stream into per-task files there — without this line the only mention of
    # a log path was in the FAILED summary, leaving a healthy multi-hour run
    # with no visible progress at all.
    print(
        f"[sweep_supervisor] {total} task(s), {max_workers} worker(s); "
        f"per-task logs: {log_dir}/group_*.log  "
        f"— full detail with: tail -f {log_dir}/group_*.log",
        flush=True,
    )
    # Cap each child's intra-op threads: N children each defaulting to ALL cores
    # oversubscribes the CPU max_workers-fold (context-switch thrash, slower
    # per-worker throughput). An explicit user export still wins (we only fill
    # vars absent from the parent environment). GPU numerics untouched.
    thread_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    threads_per_worker = str(max(1, (os.cpu_count() or max_workers) // max_workers))
    thread_env = {v: threads_per_worker for v in thread_vars if v not in os.environ}
    # On an interactive terminal, MIRROR each worker's current progress line
    # (its own tqdm bar, or the generation-phase print) into a fixed slot below.
    bars = _LiveBars(max_workers, enabled=True)
    pending: List[object] = list(groups)
    # popen -> (task, log file handle, cleanup callable, log path)
    running: Dict[subprocess.Popen, Tuple[object, object, Callable[[], None], Path]] = (
        {}
    )
    results: List[GroupResult] = []
    finished: Dict[int, GroupResult] = {}  # id(task) -> its result

    def _dep_block(task: object) -> Optional[GroupResult] | str:
        """None = ready; "wait" = a dep still pending/running; a GroupResult =
        that dep FAILED (the task must be skipped)."""
        if dependencies is None:
            return None
        for dep in dependencies(task):
            res = finished.get(id(dep))
            if res is None:
                return "wait"
            if not res.ok:
                return res
        return None

    def _launch(task: object) -> None:
        env_overrides, cleanup = prepare_group(task)
        log_path = log_dir / f"group_{log_name(task)}.log"
        logf = open(log_path, "w")
        # PYTHONUNBUFFERED: a redirected child block-buffers stdout, so its log
        # file stays EMPTY for minutes/hours; unbuffered makes the per-task log
        # tail-able in real time (numerics untouched — output buffering only).
        env = {**os.environ, **thread_env, **env_overrides, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            build_command(task),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        running[proc] = (task, logf, cleanup, log_path)
        bars.attach(log_name(task))
        bars.print(f"[sweep_supervisor] launched {log_name(task)}")

    def _schedule() -> None:
        """Drop dep-failed tasks, then fill free slots with ready tasks (FIFO
        among ready). Loops until a full pass changes nothing."""
        progressed = True
        while progressed:
            progressed = False
            for task in list(pending):
                block = _dep_block(task)
                if isinstance(block, GroupResult):
                    pending.remove(task)
                    res = (
                        skip_group(task, block)
                        if skip_group
                        else verify_group(task, -1, block.log_path)
                    )
                    finished[id(task)] = res
                    results.append(res)
                    bars.print(
                        f"[sweep_supervisor] skipped {log_name(task)}: {res.reason} "
                        f"[{len(results)}/{total} tasks done]"
                    )
                    progressed = True
                elif block is None and len(running) < max_workers:
                    pending.remove(task)
                    _launch(task)
                    progressed = True
                    if len(running) >= max_workers:
                        break

    try:
        while pending or running:
            _schedule()
            bars.refresh({log_name(g): lp for (g, _, _, lp) in running.values()})
            done = [p for p in running if p.poll() is not None]
            if not done:
                time.sleep(poll_interval)
                continue
            for proc in done:
                task, logf, cleanup, log_path = running.pop(proc)
                logf.close()
                try:
                    cleanup()  # per-task bookkeeping regardless of outcome
                except Exception:
                    pass
                res = verify_group(task, int(proc.returncode), log_path)
                finished[id(task)] = res
                results.append(res)
                bars.detach(log_name(task))
                state = "ok" if res.ok else f"FAILED ({res.reason})"
                bars.print(
                    f"[sweep_supervisor] finished {log_name(task)}: {state} "
                    f"[{len(res.leaves)}/{res.expected_leaf_count} leaves; "
                    f"{len(results)}/{total} tasks done]"
                )
    finally:
        bars.close()
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
    ``>= 2`` fans TASKS across subprocesses: offline, one ``--phase generate``
    child per (env, seed) group then one ``--phase train --points <p>`` child per
    sweep point (dependency-gated on its group's generate); online, one
    whole-group child. Every child is a serial ``run_cell``
    (``--envs <env> --seeds <seed> --max-workers 1``).

    ``budget_overrides`` applies to the in-process (max_workers==1) path. Under the
    subprocess path, budgets ride the child's YAML + ``smoke`` flag (arbitrary
    per-key overrides are not forwardable over the CLI); pass ``smoke=True`` for a
    tiny-budget parallel run.
    """
    spec = load_sweep_spec(sweep_yaml)
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
        markers = _leaf_marker_files(spec)
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
                        leaves=[p for p in expected if _leaf_complete(p, markers)],
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

    # Task graph. Offline: per group ONE "generate" (shared generator + all
    # datasets + M1, single process) then one "train" per sweep point, gated on
    # the group's generate. All generates are queued FIRST so the pool fills with
    # dataset builds up front and train tasks stream in behind them. Online: one
    # whole-group task, no deps (no datasets to phase-split).
    point_names = [param_dirname(b, s) for (b, s) in spec.points()]
    group_keys: List[Tuple[str, int]] = [
        (e, int(s)) for e in run_envs for s in run_seeds
    ]
    tasks: List[_Task] = []
    task_deps: Dict[int, List[_Task]] = {}
    group_tasks: Dict[Tuple[str, int], List[_Task]] = {g: [] for g in group_keys}
    if spec.data_regime == "offline":
        gens = {g: _Task(g[0], g[1], "generate") for g in group_keys}
        tasks.extend(gens.values())
        for g in group_keys:
            group_tasks[g].append(gens[g])
        for g in group_keys:
            for p in point_names:
                t = _Task(g[0], g[1], "train", p)
                task_deps[id(t)] = [gens[g]]
                tasks.append(t)
                group_tasks[g].append(t)
    else:
        for g in group_keys:
            t = _Task(g[0], g[1], "group")
            tasks.append(t)
            group_tasks[g].append(t)

    def _task_tag(task: object) -> str:
        t: _Task = task  # type: ignore[assignment]
        base = f"{_safe(t.env)}_seed{t.seed}"
        if t.kind == "generate":
            return f"{base}_gen"
        if t.kind == "train":
            return f"{base}_{t.point}"
        return base

    # Per-group store lifecycle: the store outlives the generate task (its train
    # tasks read it), so teardown is REFCOUNTED — every task of the group, run or
    # skipped, releases once; the last release removes the store.
    remaining: Dict[Tuple[str, int], int] = {
        g: len(ts) for g, ts in group_tasks.items()
    }

    def _store_path(group: Tuple[str, int]) -> Path:
        return scratch / f"worker_{_safe(group[0])}_seed{group[1]}"

    def _release(group: Tuple[str, int]) -> None:
        remaining[group] -= 1
        if remaining[group] <= 0:
            shutil.rmtree(_store_path(group), ignore_errors=True)

    def _build_command(task: object) -> List[str]:
        t: _Task = task  # type: ignore[assignment]
        cmd = [
            sys.executable,
            "-m",
            "src.benchmarking.regime_sweep",
            str(sweep_yaml),
            "--envs",
            t.env,
            "--seeds",
            str(t.seed),
            "--results-root",
            str(results_root),
            "--dataset-prefix",
            str(dataset_prefix),
            # Force the child onto the serial in-process run_cell path for its ONE
            # task — without this it would re-read max_workers>=2 from the YAML and
            # recursively spawn a pool.
            "--max-workers",
            "1",
        ]
        if t.kind == "generate":
            cmd += ["--phase", "generate"]
        elif t.kind == "train":
            cmd += ["--phase", "train", "--points", str(t.point)]
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
        task: object,
    ) -> Tuple[Dict[str, str], Callable[[], None]]:
        t: _Task = task  # type: ignore[assignment]
        # The ONLY offline-specific bit: a per-group Minari store (kills the
        # store-level namespace TOCTOU). generate writes it; train tasks read it
        # (static by then — the single writer has exited). Online cells have no
        # Minari store, so no env override; the release keeps refcounts symmetric
        # (rmtree of a never-created path is a no-op).
        if spec.data_regime != "offline":
            return {}, (lambda: _release(t.group))
        store = _store_path(t.group)
        store.mkdir(parents=True, exist_ok=True)
        return {"MINARI_DATASETS_PATH": str(store)}, (lambda: _release(t.group))

    def _task_label(t: _Task) -> str:
        return "gen" if t.kind == "generate" else (t.point or "group")

    def _verify_task(
        task: object, returncode: int, log_path: Optional[Path]
    ) -> GroupResult:
        t: _Task = task  # type: ignore[assignment]
        # generate produces datasets (in the worker store), not leaves — its
        # expected leaf set is empty and its verdict is the exit code alone.
        if t.kind == "generate":
            expected: List[Path] = []
        else:
            expected = _expected_leaves(
                spec, run_algos, t.env, t.seed, results_root, point=t.point
            )
        markers = _leaf_marker_files(spec)
        present = [p for p in expected if _leaf_complete(p, markers)]
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
            env=t.env,
            seed=t.seed,
            returncode=returncode,
            ok=ok,
            reason=reason,
            log_path=log_path,
            leaves=present,
            expected_leaf_count=len(expected),
            label=_task_label(t),
        )

    def _skip_task(task: object, dep_result: GroupResult) -> GroupResult:
        t: _Task = task  # type: ignore[assignment]
        _release(t.group)  # never launched -> its cleanup never runs; release here
        expected = _expected_leaves(
            spec, run_algos, t.env, t.seed, results_root, point=t.point
        )
        return GroupResult(
            env=t.env,
            seed=t.seed,
            returncode=dep_result.returncode,
            ok=False,
            reason=f"skipped: group generate failed ({dep_result.reason})",
            log_path=dep_result.log_path,
            leaves=[],
            expected_leaf_count=len(expected),
            label=_task_label(t),
        )

    task_results = _supervise(
        tasks,
        build_command=_build_command,
        prepare_group=_prepare_group,
        verify_group=_verify_task,
        log_dir=logs,
        log_name=_task_tag,
        max_workers=eff_workers,
        dependencies=(lambda t: task_deps.get(id(t), ())),
        skip_group=_skip_task,
    )

    # Aggregate task results back to the public per-(env, seed) GroupResult —
    # the reporting contract (format_summary, callers) is group-level.
    group_results: List[GroupResult] = []
    for g in group_keys:
        rs = [r for r in task_results if (r.env, r.seed) == g]
        failed = [r for r in rs if not r.ok]
        first_fail = failed[0] if failed else None
        group_results.append(
            GroupResult(
                env=g[0],
                seed=g[1],
                returncode=(first_fail.returncode if first_fail else 0),
                ok=not failed,
                reason=(
                    f"[{first_fail.label}] {first_fail.reason}" if first_fail else ""
                ),
                log_path=(
                    first_fail.log_path
                    if first_fail
                    else (rs[-1].log_path if rs else None)
                ),
                leaves=[leaf for r in rs for leaf in r.leaves],
                expected_leaf_count=sum(r.expected_leaf_count for r in rs),
            )
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

"""Offline-budget calibration PROBE — measurement only (branch feat/offline-budget-probe).

DIAGNOSES whether the production offline budget drives the offline critics into
DIVERGENCE. Production trains n_episodes=250 x rollout_len=1024 = 256_000 gradient
steps against rollout_episodes=30 (~750 CartPole transitions) — those budget numbers
(n_train_envs=16, rollout_len=1024) are ON-POLICY vectorized-rollout parameters reused
verbatim as offline gradient steps. This probe changes NO source (runner / gates /
critics / budgets.yaml untouched): it drives the SHIPPED BenchmarkRunner at the sigma=0
BASIC point and reads the per-checkpoint CSVs it already writes.

GRID (steps x dataset size interact — swept together): rollout_episodes in {30, 300} x
seeds {0,1,2} = 6 runs. Each is trained ONCE to 256k grad steps with n_checkpoints=25;
the step-curve is READ OFF those 25 checkpoints (never a separate training per step).
Critics: observational, proximal, oracle_u ONLY (no sensitivity, no bare-DQN arm).

MEASURED per (dataset_size, seed, critic) at every checkpoint:
  1. value_mse_to_oracle  (critic_ablation_metrics.csv)
  2. apparent_q_mean      (critic_ablation_metrics.csv) — the ABSOLUTE anchor; the
     theoretical bound is r_max/(1-gamma). CartPole base r=1 but the sigma=0 basic
     confounder (bias_confounded_action, c_r=1.0) shifts reward by U*1[a=a_bad], so the
     dataset r_max is ~2 -> bound ~2/(1-gamma). Both are drawn/reported; Q far above the
     bound is divergence regardless of the MSE.
  3. eval_return          (eval_metrics.csv) — the base actor on CLEAN CartPole (the
     runner's eval env is unconfounded), max 500. Ground-truth "does the policy work".

Modes:
  --run-one --rollout-episodes RE --seed S --out DIR   one training run -> CSVs in DIR
  --orchestrate --out-root ROOT [--max-workers N]      pool the 6 runs (isolated Minari
                                                       stores + per-run logs), then plot
  --plot --out-root ROOT                               (re)build the two figures + table

Tiny-budget smoke (validate plumbing in ~1 min) via env overrides:
  NE / RL / NC / NEV / RE_GRID / SEEDS, e.g.
  NE=4 RL=8 NC=4 NEV=4 RE_GRID=8 SEEDS=0 python tools/probe_offline_budget.py \
      --run-one --rollout-episodes 8 --seed 0 --out /tmp/probe_smoke/ds8_seed0 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Config (env-overridable so a tiny smoke can validate the plumbing fast)       #
# --------------------------------------------------------------------------- #
ENV = os.environ.get("ENV", "CartPole-v1")
BASE_ALGO = os.environ.get("ALGO", "cql")  # the critic-ablation base class + base actor
GEN_ALGO = "dqn"  # discrete generator for the shared pi_basic
GAMMA = float(os.environ.get("GAMMA", "0.99"))
EPISODE_CAP = int(os.environ.get("EPISODE_CAP", "500"))  # CartPole-v1 truncation
CRITICS = ["observational", "proximal", "oracle_u"]

N_EPISODES = int(os.environ.get("NE", "250"))
ROLLOUT_LEN = int(os.environ.get("RL", "1024"))  # = offline grad steps / epoch
N_CHECKPOINTS = int(os.environ.get("NC", "25"))
N_EVAL_ENVS = int(os.environ.get("NEV", "16"))
RE_GRID = [int(x) for x in os.environ.get("RE_GRID", "30,300").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]

# Table read-out targets (grad steps); the nearest checkpoint to each is reported.
TABLE_TARGETS = [5_000, 20_000, 50_000, 100_000, 256_000]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def q_bound(r_max: float) -> Tuple[float, float]:
    """(infinite-horizon, finite-horizon) discounted-return bound for reward r_max."""
    inf = r_max / (1.0 - GAMMA)
    fin = r_max * (1.0 - GAMMA**EPISODE_CAP) / (1.0 - GAMMA)
    return inf, fin


# --------------------------------------------------------------------------- #
# One training run: drive the SHIPPED runner at the sigma=0 basic point         #
# --------------------------------------------------------------------------- #
def run_one(rollout_episodes: int, seed: int, out_dir: Path, device: str) -> int:
    import minari
    from src.benchmarking.critic_ablation import CriticAblationConfig
    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.config.seeding import set_seed
    from src.envs.offline.generate import (
        build_generator_agent,
        generate_offline_dataset,
    )
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONE shared pi_basic per (env, seed), exactly as run_cell builds it.
    agent, _ = build_generator_agent(ENV, GEN_ALGO, "random", seed=seed, device=device)
    did = f"probe/{ENV.split('-v')[0].lower()}-re{rollout_episodes}-seed{seed}-v0"
    try:
        minari.delete_dataset(did)
    except Exception:
        pass
    set_seed(seed, deterministic=True)
    ds = generate_offline_dataset(
        env_id=ENV,
        generator_algo=GEN_ALGO,
        tier="random",
        behavior_policy="bias_confounded_action",  # the sigma=0 basic construction
        behavior_strength=0.0,
        confounder_c_r=1.0,
        pi_basic_epsilon=0.5,
        rollout_episodes=rollout_episodes,
        seed=seed,
        dataset_id=did,
        agent=agent,
        device=device,
    )
    ds.storage.update_metadata({"behavior_strength_sigma": 0.0})

    # r_max actually present in the dataset (base CartPole 1 + confounder shift).
    r_max, n_transitions = _dataset_reward_stats(ds)

    env_cfg = EnvConfig(
        env_id=ENV,
        n_train_envs=2,  # offline: unused for data, cheap to construct
        n_eval_envs=N_EVAL_ENVS,
        rollout_len=ROLLOUT_LEN,
        seed=seed,
        offline_dataset=did,
        behavior_policy="bias_confounded_action",
        behavior_strength=0.0,
        pi_basic_epsilon=0.5,
        confounder_c_r=1.0,
        mask_indices=None,
    )
    train_cfg = TrainingConfig(
        n_episodes=N_EPISODES,
        n_checkpoints=N_CHECKPOINTS,
        deterministic=True,
        device=device,
        algorithm=BASE_ALGO,
        aggregation="iqm",
        critic_network="mlp",
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(out_dir), timestamp="probe"),
        registry.get(BASE_ALGO),
        critic_ablation_cfg=CriticAblationConfig(critics=list(CRITICS)),
    ).run()

    inf_bound, fin_bound = q_bound(r_max)
    (out_dir / "probe_meta.json").write_text(
        json.dumps(
            {
                "rollout_episodes": rollout_episodes,
                "seed": seed,
                "base_algo": BASE_ALGO,
                "gamma": GAMMA,
                "rollout_len": ROLLOUT_LEN,
                "n_episodes": N_EPISODES,
                "n_checkpoints": N_CHECKPOINTS,
                "n_transitions": n_transitions,
                "r_max": r_max,
                "q_bound_infinite": inf_bound,
                "q_bound_finite": fin_bound,
            },
            indent=2,
        )
    )
    try:
        minari.delete_dataset(did)
    except Exception:
        pass
    print(
        f"[probe] done RE={rollout_episodes} seed={seed}: {n_transitions} transitions, "
        f"r_max={r_max:.3f}, Q_bound(inf)={inf_bound:.1f} -> {out_dir}"
    )
    return 0


def _dataset_reward_stats(ds) -> Tuple[float, int]:
    r_max, n = 1.0, 0
    try:
        for ep in ds.iterate_episodes():
            r = ep.rewards
            n += int(len(r))
            m = float(r.max()) if len(r) else 1.0
            r_max = max(r_max, m)
    except Exception:
        # Fallback: the known basic-confounder construction (base 1 + c_r=1.0 shift).
        r_max = 2.0
    return r_max, n


# --------------------------------------------------------------------------- #
# Orchestration: pool the 6 runs through the merged supervisor's generic pool    #
# --------------------------------------------------------------------------- #
def orchestrate(out_root: Path, max_workers: int, device: str) -> int:
    import shutil

    from src.benchmarking.sweep_supervisor import _supervise, GroupResult

    out_root.mkdir(parents=True, exist_ok=True)
    stores = out_root / "_stores"
    logs = out_root / "_logs"
    groups: List[Tuple[int, int]] = [(re, s) for re in RE_GRID for s in SEEDS]

    def _tag(g) -> str:
        re, s = g
        return f"re{re}_seed{s}"

    def _run_dir(g) -> Path:
        re, s = g
        return out_root / f"ds{re}_seed{s}"

    def _build_command(g) -> List[str]:
        re, s = g
        return [
            sys.executable,
            "-m",
            "tools.probe_offline_budget",
            "--run-one",
            "--rollout-episodes",
            str(re),
            "--seed",
            str(s),
            "--out",
            str(_run_dir(g)),
            "--device",
            device,
        ]

    def _prepare_group(g):
        store = stores / f"worker_{_tag(g)}"
        store.mkdir(parents=True, exist_ok=True)
        return {"MINARI_DATASETS_PATH": str(store)}, lambda: shutil.rmtree(
            store, ignore_errors=True
        )

    def _verify_group(g, returncode, log_path) -> GroupResult:
        re, s = g
        rd = _run_dir(g)
        have = all(
            (rd / f).exists()
            for f in ("critic_ablation_metrics.csv", "eval_metrics.csv")
        )
        ok = returncode == 0 and have
        reason = (
            ""
            if ok
            else (f"exit {returncode}" if returncode else "missing output CSVs")
        )
        return GroupResult(
            env=f"re{re}",
            seed=s,
            returncode=returncode,
            ok=ok,
            reason=reason,
            log_path=log_path,
            leaves=[rd] if ok else [],
            expected_leaf_count=1,
        )

    print(
        f"[probe] orchestrating {len(groups)} runs (RE={RE_GRID} x seeds={SEEDS}) "
        f"max_workers={max_workers} device={device} -> {out_root}"
    )
    results = _supervise(
        groups,
        build_command=_build_command,
        prepare_group=_prepare_group,
        verify_group=_verify_group,
        log_dir=logs,
        log_name=_tag,
        max_workers=max_workers,
    )
    results.sort(key=lambda r: (r.env, r.seed))
    failed = [r for r in results if not r.ok]
    for r in results:
        status = "OK  " if r.ok else f"FAIL({r.reason})"
        print(f"  {status} {r.env} seed{r.seed}  log={r.log_path}")
    # Build figures + table from whatever completed (do not lose partial work).
    try:
        plot(out_root)
    except Exception as e:  # plotting must never mask a training failure
        print(f"[probe] plotting failed: {e}")
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
# Read the per-checkpoint CSVs                                                   #
# --------------------------------------------------------------------------- #
def _read_run(run_dir: Path):
    """-> (meta dict, {critic: {grad_steps: (mse, q)}}, {grad_steps: eval_return})."""
    meta = json.loads((run_dir / "probe_meta.json").read_text())
    rl = int(meta["rollout_len"])
    crit: Dict[str, Dict[int, Tuple[float, float]]] = {c: {} for c in CRITICS}
    with (run_dir / "critic_ablation_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            c = row.get("critic")
            if c not in crit:
                continue
            step = (int(row["episode"]) + 1) * rl
            crit[c][step] = (
                _f(row.get("value_mse_to_oracle")),
                _f(row.get("apparent_q_mean")),
            )
    ev: Dict[int, float] = {}
    ep_path = run_dir / "eval_metrics.csv"
    if ep_path.exists():
        with ep_path.open() as f:
            for row in csv.DictReader(f):
                ev[(int(row["episode"]) + 1) * rl] = _f(row.get("eval_return_mean"))
    return meta, crit, ev


def _f(v):
    if v in ("", None):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _agg(vals: List[float]) -> Tuple[float, float]:
    xs = [v for v in vals if v == v]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = (
        math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    )
    return m, sd


def _collect(out_root: Path):
    """Aggregate all present runs over seeds. -> (per_re, steps, bounds)."""
    runs: Dict[int, List[Tuple[dict, dict, dict]]] = {re: [] for re in RE_GRID}
    for re in RE_GRID:
        for s in SEEDS:
            rd = out_root / f"ds{re}_seed{s}"
            if (rd / "probe_meta.json").exists() and (
                rd / "critic_ablation_metrics.csv"
            ).exists():
                runs[re].append(_read_run(rd))
    steps = sorted({(i + 1) * ROLLOUT_LEN for i in _checkpoint_indices()})
    return runs, steps


def _checkpoint_indices() -> List[int]:
    from src.config.defaults import TrainingConfig

    return TrainingConfig(
        n_episodes=N_EPISODES, n_checkpoints=N_CHECKPOINTS
    ).checkpoint_episodes()


# --------------------------------------------------------------------------- #
# Figures + table                                                               #
# --------------------------------------------------------------------------- #
def plot(out_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, steps = _collect(out_root)
    present = {re: len(v) for re, v in runs.items()}
    if not any(present.values()):
        print("[probe] no completed runs to plot yet.")
        return
    r_max = max((m["r_max"] for v in runs.values() for (m, _, _) in v), default=2.0)
    inf_bound, fin_bound = q_bound(r_max)

    # colors: dataset size -> hue family; critic -> line style
    re_color = {30: "#1f77b4", 300: "#d62728"}
    crit_style = {"observational": "-", "proximal": "--", "oracle_u": ":"}

    def series(re, critic, idx):
        """(steps, mean, sd) over seeds for metric idx (0=mse,1=q)."""
        xs, ms, sds = [], [], []
        for st in steps:
            vals = [
                crit[critic][st][idx]
                for (_, crit, _) in runs[re]
                if st in crit.get(critic, {})
            ]
            if vals:
                m, sd = _agg(vals)
                xs.append(st)
                ms.append(m)
                sds.append(sd)
        return xs, ms, sds

    # ---- Figure 1: value_mse_to_oracle vs grad steps (obs + prox; oracle_u == 0) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for re in RE_GRID:
        if not runs[re]:
            continue
        for critic in ("observational", "proximal"):
            xs, ms, sds = series(re, critic, 0)
            if not xs:
                continue
            ax.plot(
                xs,
                ms,
                crit_style[critic],
                color=re_color.get(re, "#555"),
                label=f"RE={re} {critic}",
                linewidth=1.8,
            )
            lo = [max(m - sd, 1e-6) for m, sd in zip(ms, sds)]
            hi = [m + sd for m, sd in zip(ms, sds)]
            ax.fill_between(xs, lo, hi, color=re_color.get(re, "#555"), alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offline gradient steps (log)")
    ax.set_ylabel("value_mse_to_oracle (mean +/- seed sd, log)")
    ax.set_title(
        f"{ENV} sigma=0 basic — obs/prox MSE-to-oracle vs offline budget "
        f"(base={BASE_ALGO})\noracle_u MSE == 0 by construction (omitted)"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    f1 = out_root / "fig_mse_vs_steps.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # ---- Figure 2: apparent_q_mean vs grad steps, with the Q bound ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for re in RE_GRID:
        if not runs[re]:
            continue
        for critic in CRITICS:
            xs, ms, sds = series(re, critic, 1)
            if not xs:
                continue
            ax.plot(
                xs,
                ms,
                crit_style[critic],
                color=re_color.get(re, "#555"),
                label=f"RE={re} {critic}",
                linewidth=1.8,
            )
            ax.fill_between(
                xs,
                [m - sd for m, sd in zip(ms, sds)],
                [m + sd for m, sd in zip(ms, sds)],
                color=re_color.get(re, "#555"),
                alpha=0.12,
            )
    ax.axhline(
        inf_bound,
        color="black",
        linestyle="-",
        linewidth=1.3,
        label=f"Q bound r_max/(1-g) = {inf_bound:.0f}  (r_max={r_max:.1f})",
    )
    ax.axhline(
        100.0,
        color="gray",
        linestyle="-.",
        linewidth=1.0,
        label="Q bound if r=1 -> 100",
    )
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("offline gradient steps (log)")
    ax.set_ylabel("apparent_q_mean (mean +/- seed sd, symlog)")
    ax.set_title(
        f"{ENV} sigma=0 basic — apparent Q vs offline budget (base={BASE_ALGO}); "
        "Q above the bound = divergence"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    f2 = out_root / "fig_q_vs_steps.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    _write_table(out_root, runs, steps, r_max, inf_bound, present)
    print(f"[probe] wrote {f1.name}, {f2.name}, table.csv, table.md to {out_root}")


def _nearest(steps: List[int], target: int) -> int:
    return min(steps, key=lambda s: abs(s - target))


def _write_table(out_root, runs, steps, r_max, inf_bound, present):
    cols = [
        "target_step",
        "grad_steps",
        "rollout_episodes",
        "obs_mse_mean",
        "obs_mse_sd",
        "prox_mse_mean",
        "prox_mse_sd",
        "obs_q_mean",
        "prox_q_mean",
        "oracle_q_mean",
        "eval_return_mean",
        "eval_return_sd",
        "n_seeds",
    ]

    def cell(re, critic, st, idx):
        return _agg(
            [
                crit[critic][st][idx]
                for (_, crit, _) in runs[re]
                if st in crit.get(critic, {})
            ]
        )

    rows = []
    for target in TABLE_TARGETS:
        st = _nearest(steps, target)
        for re in RE_GRID:
            if not runs[re]:
                continue
            om, osd = cell(re, "observational", st, 0)
            pm, psd = cell(re, "proximal", st, 0)
            oq, _ = cell(re, "observational", st, 1)
            pq, _ = cell(re, "proximal", st, 1)
            orq, _ = cell(re, "oracle_u", st, 1)
            evs = [ev[st] for (_, _, ev) in runs[re] if st in ev]
            evm, evsd = _agg(evs)
            rows.append(
                {
                    "target_step": target,
                    "grad_steps": st,
                    "rollout_episodes": re,
                    "obs_mse_mean": round(om, 4),
                    "obs_mse_sd": round(osd, 4),
                    "prox_mse_mean": round(pm, 4),
                    "prox_mse_sd": round(psd, 4),
                    "obs_q_mean": round(oq, 3),
                    "prox_q_mean": round(pq, 3),
                    "oracle_q_mean": round(orq, 3),
                    "eval_return_mean": round(evm, 2),
                    "eval_return_sd": round(evsd, 2),
                    "n_seeds": len(runs[re]),
                }
            )
    with (out_root / "table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # human-readable markdown
    lines = [
        f"# Offline-budget probe — {ENV} sigma=0 basic, base={BASE_ALGO}",
        f"Q bound r_max/(1-gamma) = {inf_bound:.1f} (r_max={r_max:.1f}); r=1 bound = 100.",
        "Runs present: "
        + ", ".join(f"RE{re}={present[re]}/{len(SEEDS)}" for re in RE_GRID),
        "",
        "| target | steps | RE | obs_mse | prox_mse | obs_Q | prox_Q | oracle_Q | eval_ret | n |",
        "|--|--|--|--|--|--|--|--|--|--|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target_step']//1000}k | {r['grad_steps']} | {r['rollout_episodes']} "
            f"| {r['obs_mse_mean']}±{r['obs_mse_sd']} | {r['prox_mse_mean']}±{r['prox_mse_sd']} "
            f"| {r['obs_q_mean']} | {r['prox_q_mean']} | {r['oracle_q_mean']} "
            f"| {r['eval_return_mean']}±{r['eval_return_sd']} | {r['n_seeds']} |"
        )
    (out_root / "table.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-one", action="store_true")
    ap.add_argument("--orchestrate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rollout-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-root", default="docs/offline_budget_probe/runs")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.run_one:
        out = Path(a.out or f"probe_out/ds{a.rollout_episodes}_seed{a.seed}")
        return run_one(a.rollout_episodes, a.seed, out, a.device)
    if a.orchestrate:
        return orchestrate(Path(a.out_root), a.max_workers, a.device)
    if a.plot:
        plot(Path(a.out_root))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

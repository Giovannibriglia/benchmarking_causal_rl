"""Offline-budget calibration PROBE v2 — measurement only (feat/offline-budget-probe-v2).

v1 minimised value_mse_to_oracle and landed on 1024 steps — but there obs_Q=6.70,
prox_Q=7.01, oracle_Q=6.69 while the MC anchor (below) is ~8.7: the critics were at
BEHAVIOUR scale (untrained value), not converged. MSE was tiny because three
near-initialisation heads trivially agree, and the metric references the ORACLE, which
moves with the budget — so it cannot tell "both accurate" from "both untrained".

v2 adds an ABSOLUTE, budget-independent anchor and reports scale, not just agreement.

WHAT apparent_q_mean AVERAGES (read from critic_ablation.py, NOT assumed):
  set_sequence_buffer caches a fixed eval set = EVERY dataset transition (obs, a_data)
  from seq_buffer.iter_episodes(), deterministically subsampled to 4000 via
  torch.linspace(0, n-1, 4000) when n>4000. checkpoint_rows_strategy then computes, per
  critic, q_c = predict_q_adj(obs_e).gather(1, act_e) — i.e. Q at the DATASET ACTION —
  and apparent_q_mean = q_c.mean(); value_mse_to_oracle = mean((q_c - q_oracle)^2) over
  the same set (q_oracle = oracle_u's Q_adj at the same (obs, a_data)).

THE ANCHOR (matched to that exactly): MC return-to-go on the SAME eval set.
  Reconstruct the SAME buffer (fill_sequence_buffer_from_minari, cap 1M, load_u=True),
  iterate iter_episodes in insertion order, and for each episode's rewards r_0..r_{T-1}:
      G_t = sum_{k=0}^{T-1-t} gamma^k r_{t+k}
  Flatten in the same order, apply the SAME linspace(0,n-1,4000) subsample, take the
  mean. This is the OFF-POLICY (behaviour) value at (s, a_data) over the dataset dist.
  CAVEAT (reported, not hidden): pi_basic is RANDOM-tier (mean episode ~13.7 steps), so
  the anchor is the BEHAVIOUR value (~8.7), NOT the improved-policy value the conservative
  critic targets (~100, the discounted value of the learned policy that eval shows
  surviving ~1020 steps). So critic_Q/anchor ~= 1 marks BEHAVIOUR-scale = UNDER-trained;
  the improved-policy scale is ~10-12x the anchor. We therefore ALSO overlay an
  eval-derived improved-value reference V_improved ~= (1-gamma^L)/(1-gamma), L=eval_return
  (clean env, undiscounted length), so the plots show the climb behaviour->improved.

GRID: rollout_episodes {300, 1000, 3000} x seeds {0,1,2} = 9 runs; each trained ONCE to
256k grad steps (n_checkpoints=25). Training cost is O(1) in dataset size (buffer
sampling), so larger RE costs only generation. critics: observational, proximal, oracle_u.

Modes: --run-one / --orchestrate (pools via the merged supervisor's _supervise,
max_workers 3) / --plot. Budget env-overridable (NE/RL/NC/NEV/RE_GRID/SEEDS) for smoke.
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

ENV = os.environ.get("ENV", "CartPole-v1")
BASE_ALGO = os.environ.get("ALGO", "cql")
GEN_ALGO = "dqn"
GAMMA = float(os.environ.get("GAMMA", "0.99"))
EPISODE_CAP = int(os.environ.get("EPISODE_CAP", "500"))
CRITICS = ["observational", "proximal", "oracle_u"]

N_EPISODES = int(os.environ.get("NE", "250"))
ROLLOUT_LEN = int(os.environ.get("RL", "1024"))
N_CHECKPOINTS = int(os.environ.get("NC", "25"))
N_EVAL_ENVS = int(os.environ.get("NEV", "16"))
RE_GRID = [int(x) for x in os.environ.get("RE_GRID", "300,1000,3000").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
EVAL_SUBSAMPLE = 4000  # must match critic_ablation.set_sequence_buffer

TABLE_TARGETS = [5_000, 20_000, 50_000, 100_000, 256_000]
RE_COLOR = {30: "#1f77b4", 300: "#1f77b4", 1000: "#2ca02c", 3000: "#d62728"}
CRIT_STYLE = {"observational": "-", "proximal": "--", "oracle_u": ":"}


def v_improved(eval_return: float) -> float:
    """Discounted value of a policy whose (clean) episode length ~= eval_return."""
    L = max(1.0, float(eval_return))
    return (1.0 - GAMMA**L) / (1.0 - GAMMA)


# --------------------------------------------------------------------------- #
# MC anchor — matched to apparent_q_mean's eval set                             #
# --------------------------------------------------------------------------- #
def mc_anchor_from_dataset(dataset_id: str, device: str) -> Tuple[float, float, int]:
    """(anchor over the 4000-subsample, anchor over the full set, n_transitions)."""
    import torch
    from src.envs.offline.minari_loader import fill_sequence_buffer_from_minari
    from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

    buf = SequenceReplayBuffer(capacity=1_000_000, device=device)
    fill_sequence_buffer_from_minari(dataset_id, buf, device, load_u=True)
    g_all: List[float] = []
    for ep in buf.iter_episodes():  # insertion order == set_sequence_buffer's order
        rews = [float(tr["rewards"]) for tr in ep]
        acc, G = 0.0, [0.0] * len(rews)
        for t in range(len(rews) - 1, -1, -1):
            acc = rews[t] + GAMMA * acc
            G[t] = acc
        g_all.extend(G)
    n = len(g_all)
    g = torch.tensor(g_all, dtype=torch.float64)
    if n > EVAL_SUBSAMPLE:  # SAME deterministic subsample as the eval set
        idx = torch.linspace(0, n - 1, EVAL_SUBSAMPLE).long()
        sub = g[idx]
    else:
        sub = g
    return float(sub.mean().item()), float(g.mean().item()), n


# --------------------------------------------------------------------------- #
# One training run                                                              #
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

    agent, _ = build_generator_agent(ENV, GEN_ALGO, "random", seed=seed, device=device)
    did = f"probev2/{ENV.split('-v')[0].lower()}-re{rollout_episodes}-seed{seed}-v0"
    try:
        minari.delete_dataset(did)
    except Exception:
        pass
    set_seed(seed, deterministic=True)
    ds = generate_offline_dataset(
        env_id=ENV,
        generator_algo=GEN_ALGO,
        tier="random",
        behavior_policy="bias_confounded_action",
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

    # ABSOLUTE anchor, computed from the dataset BEFORE training (training-independent).
    anchor_sub, anchor_full, n_trans = mc_anchor_from_dataset(did, device)

    env_cfg = EnvConfig(
        env_id=ENV,
        n_train_envs=2,
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
        RunConfig(run_dir=str(out_dir), timestamp="probev2"),
        registry.get(BASE_ALGO),
        critic_ablation_cfg=CriticAblationConfig(critics=list(CRITICS)),
    ).run()

    (out_dir / "probe_meta.json").write_text(
        json.dumps(
            {
                "rollout_episodes": rollout_episodes,
                "seed": seed,
                "base_algo": BASE_ALGO,
                "gamma": GAMMA,
                "rollout_len": ROLLOUT_LEN,
                "n_episodes": N_EPISODES,
                "n_transitions": n_trans,
                "mc_anchor": anchor_sub,
                "mc_anchor_full": anchor_full,
            },
            indent=2,
        )
    )
    try:
        minari.delete_dataset(did)
    except Exception:
        pass
    print(
        f"[probe2] done RE={rollout_episodes} seed={seed}: {n_trans} trans, "
        f"MC_anchor={anchor_sub:.3f} -> {out_dir}"
    )
    return 0


# --------------------------------------------------------------------------- #
# Orchestration (reuses the merged supervisor pool)                             #
# --------------------------------------------------------------------------- #
def orchestrate(out_root: Path, max_workers: int, device: str) -> int:
    import shutil

    from src.benchmarking.sweep_supervisor import _supervise, GroupResult

    out_root = out_root.resolve()  # absolute: relative MINARI_DATASETS_PATH breaks HDF5
    out_root.mkdir(parents=True, exist_ok=True)
    stores, logs = out_root / "_stores", out_root / "_logs"
    groups: List[Tuple[int, int]] = [(re, s) for re in RE_GRID for s in SEEDS]

    def _tag(g):
        return f"re{g[0]}_seed{g[1]}"

    def _run_dir(g):
        return out_root / f"ds{g[0]}_seed{g[1]}"

    def _cmd(g):
        return [
            sys.executable,
            "-m",
            "tools.probe_offline_budget_v2",
            "--run-one",
            "--rollout-episodes",
            str(g[0]),
            "--seed",
            str(g[1]),
            "--out",
            str(_run_dir(g)),
            "--device",
            device,
        ]

    def _prep(g):
        store = stores / f"worker_{_tag(g)}"
        store.mkdir(parents=True, exist_ok=True)
        return {"MINARI_DATASETS_PATH": str(store)}, lambda: shutil.rmtree(
            store, ignore_errors=True
        )

    def _verify(g, rc, log_path):
        rd = _run_dir(g)
        ok = rc == 0 and all(
            (rd / f).exists()
            for f in (
                "critic_ablation_metrics.csv",
                "eval_metrics.csv",
                "probe_meta.json",
            )
        )
        return GroupResult(
            env=f"re{g[0]}",
            seed=g[1],
            returncode=rc,
            ok=ok,
            reason="" if ok else (f"exit {rc}" if rc else "missing outputs"),
            log_path=log_path,
            leaves=[rd] if ok else [],
            expected_leaf_count=1,
        )

    print(
        f"[probe2] orchestrating {len(groups)} runs (RE={RE_GRID} x seeds={SEEDS}) "
        f"max_workers={max_workers} device={device} -> {out_root}"
    )
    results = _supervise(
        groups,
        build_command=_cmd,
        prepare_group=_prep,
        verify_group=_verify,
        log_dir=logs,
        log_name=_tag,
        max_workers=max_workers,
    )
    results.sort(key=lambda r: (r.env, r.seed))
    for r in results:
        print(f"  {'OK  ' if r.ok else f'FAIL({r.reason})'} {r.env} seed{r.seed}")
    try:
        plot(out_root)
    except Exception as e:
        print(f"[probe2] plotting failed: {e}")
    return 0 if all(r.ok for r in results) else 1


# --------------------------------------------------------------------------- #
# Read + aggregate                                                              #
# --------------------------------------------------------------------------- #
def _f(v):
    if v in ("", None):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _read_run(run_dir: Path):
    meta = json.loads((run_dir / "probe_meta.json").read_text())
    rl = int(meta["rollout_len"])
    # {critic: {steps: {mse, q, rel_err, ratio}}}
    crit: Dict[str, Dict[int, Dict[str, float]]] = {c: {} for c in CRITICS}
    anchor = float(meta["mc_anchor"])
    with (run_dir / "critic_ablation_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            c = row.get("critic")
            if c not in crit:
                continue
            step = (int(row["episode"]) + 1) * rl
            q = _f(row.get("apparent_q_mean"))
            mse = _f(row.get("value_mse_to_oracle"))
            rel = (
                math.sqrt(mse) / abs(q)
                if (mse == mse and q and abs(q) > 1e-9)
                else float("nan")
            )
            crit[c][step] = {
                "mse": mse,
                "q": q,
                "rel_err": rel,
                "ratio": (q / anchor) if anchor else float("nan"),
            }
    ev: Dict[int, float] = {}
    if (run_dir / "eval_metrics.csv").exists():
        with (run_dir / "eval_metrics.csv").open() as f:
            for row in csv.DictReader(f):
                ev[(int(row["episode"]) + 1) * rl] = _f(row.get("eval_return_mean"))
    return meta, crit, ev, anchor


def _collect(out_root: Path):
    runs: Dict[int, List] = {re: [] for re in RE_GRID}
    for re in RE_GRID:
        for s in SEEDS:
            rd = out_root / f"ds{re}_seed{s}"
            if (rd / "probe_meta.json").exists() and (
                rd / "critic_ablation_metrics.csv"
            ).exists():
                runs[re].append(_read_run(rd))
    steps = sorted({(i + 1) * ROLLOUT_LEN for i in _ckpt_idx()})
    return runs, steps


def _ckpt_idx():
    from src.config.defaults import TrainingConfig

    return TrainingConfig(
        n_episodes=N_EPISODES, n_checkpoints=N_CHECKPOINTS
    ).checkpoint_episodes()


def _agg(vals):
    xs = [v for v in vals if v == v]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = (
        math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    )
    return m, sd


# --------------------------------------------------------------------------- #
# Figures + tables                                                              #
# --------------------------------------------------------------------------- #
def plot(out_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, steps = _collect(out_root)
    if not any(runs.values()):
        print("[probe2] no completed runs to plot.")
        return
    anchors = {
        re: _agg([m["mc_anchor"] for (m, _, _, _) in runs[re]])
        for re in RE_GRID
        if runs[re]
    }
    # eval-derived improved-value proxy (mean over seeds of the last-checkpoint eval).
    vimp = {}
    for re in RE_GRID:
        if not runs[re]:
            continue
        last = max(steps)
        vimp[re] = _agg(
            [v_improved(ev.get(last, float("nan"))) for (_, _, ev, _) in runs[re]]
        )[0]

    def series(re, critic, key):
        xs, ms, sds = [], [], []
        for st in steps:
            vals = [
                c[critic][st][key]
                for (_, c, _, _) in runs[re]
                if st in c.get(critic, {})
            ]
            if vals:
                m, sd = _agg(vals)
                xs.append(st)
                ms.append(m)
                sds.append(sd)
        return xs, ms, sds

    # ---- Fig 1: rel_err = sqrt(mse)/|Q| ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for re in RE_GRID:
        if not runs[re]:
            continue
        for critic in ("observational", "proximal"):
            xs, ms, sds = series(re, critic, "rel_err")
            if xs:
                ax.plot(
                    xs,
                    ms,
                    CRIT_STYLE[critic],
                    color=RE_COLOR.get(re),
                    lw=1.8,
                    label=f"RE={re} {critic}",
                )
                ax.fill_between(
                    xs,
                    [max(m - s, 1e-4) for m, s in zip(ms, sds)],
                    [m + s for m, s in zip(ms, sds)],
                    color=RE_COLOR.get(re),
                    alpha=0.12,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offline gradient steps (log)")
    ax.set_ylabel("rel_err = sqrt(mse)/|Q| (log)")
    ax.set_title(
        f"{ENV} sigma=0 — RELATIVE error vs offline budget (base={BASE_ALGO})\n"
        "oracle_u rel_err == 0 (self-reference); obs/prox shown"
    )
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "fig_relerr_vs_steps.png", dpi=130)
    plt.close(fig)

    # ---- Fig 2: critic_Q / MC_anchor ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for re in RE_GRID:
        if not runs[re]:
            continue
        for critic in CRITICS:
            xs, ms, sds = series(re, critic, "ratio")
            if xs:
                ax.plot(
                    xs,
                    ms,
                    CRIT_STYLE[critic],
                    color=RE_COLOR.get(re),
                    lw=1.8,
                    label=f"RE={re} {critic}",
                )
                ax.fill_between(
                    xs,
                    [m - s for m, s in zip(ms, sds)],
                    [m + s for m, s in zip(ms, sds)],
                    color=RE_COLOR.get(re),
                    alpha=0.10,
                )
    ax.axhline(1.0, color="black", lw=1.3, label="ratio = 1 (behaviour/anchor scale)")
    if vimp and anchors:
        imp_ratio = _agg([vimp[re] / anchors[re][0] for re in vimp])[0]
        ax.axhline(
            imp_ratio,
            color="gray",
            ls="-.",
            lw=1.1,
            label=f"improved-policy scale ~= {imp_ratio:.1f}x anchor (eval-derived)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offline gradient steps (log)")
    ax.set_ylabel("apparent_Q / MC_anchor (log)")
    ax.set_title(
        f"{ENV} sigma=0 — critic Q / MC behaviour-anchor (base={BASE_ALGO})\n"
        "ratio~1 = behaviour scale (UNDER-trained); the target is the improved scale"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "fig_ratio_vs_steps.png", dpi=130)
    plt.close(fig)

    # ---- Fig 3: MC anchor stability across RE + eval_return ----
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.5))
    res = [re for re in RE_GRID if runs[re]]
    axl.bar(
        [str(r) for r in res],
        [anchors[r][0] for r in res],
        yerr=[anchors[r][1] for r in res],
        color="#8888cc",
        capsize=4,
    )
    for r in res:
        for m, _, _, _ in runs[r]:
            axl.plot(str(r), m["mc_anchor"], "k.", ms=8)
    axl.set_xlabel("rollout_episodes")
    axl.set_ylabel("MC anchor")
    axl.set_title("MC anchor stability across RE (should be flat)")
    for re in res:
        xs, ms, sds = [], [], []
        for st in steps:
            vals = [ev[st] for (_, _, ev, _) in runs[re] if st in ev]
            if vals:
                m, sd = _agg(vals)
                xs.append(st)
                ms.append(m)
                sds.append(sd)
        axr.plot(xs, ms, color=RE_COLOR.get(re), lw=1.8, label=f"RE={re}")
    axr.set_xscale("log")
    axr.set_xlabel("offline gradient steps (log)")
    axr.set_ylabel("eval_return (clean env)")
    axr.set_title("eval_return — SATURATED, no discriminating signal")
    axr.legend(fontsize=8)
    axr.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "fig_anchor_eval.png", dpi=130)
    plt.close(fig)

    _tables(out_root, runs, steps, anchors, vimp)
    print(
        f"[probe2] wrote 3 figures + table.md/csv + proximal_per_seed.md to {out_root}"
    )


def _nearest(steps, target):
    return min(steps, key=lambda s: abs(s - target))


def _tables(out_root, runs, steps, anchors, vimp):
    def agg_at(re, critic, st, key):
        return _agg(
            [c[critic][st][key] for (_, c, _, _) in runs[re] if st in c.get(critic, {})]
        )

    rows = []
    for target in TABLE_TARGETS:
        st = _nearest(steps, target)
        for re in RE_GRID:
            if not runs[re]:
                continue
            evm, _ = _agg([ev[st] for (_, _, ev, _) in runs[re] if st in ev])
            rows.append(
                {
                    "target": f"{target // 1000}k",
                    "steps": st,
                    "RE": re,
                    "anchor": round(anchors[re][0], 2),
                    "obs_relerr": round(
                        agg_at(re, "observational", st, "rel_err")[0], 3
                    ),
                    "prox_relerr": round(agg_at(re, "proximal", st, "rel_err")[0], 3),
                    "obs_ratio": round(agg_at(re, "observational", st, "ratio")[0], 2),
                    "prox_ratio": round(agg_at(re, "proximal", st, "ratio")[0], 2),
                    "oracle_ratio": round(agg_at(re, "oracle_u", st, "ratio")[0], 2),
                    "eval_ret": round(evm, 1),
                    "n": len(runs[re]),
                }
            )
    with (out_root / "table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [
        f"# Offline-budget probe v2 — {ENV} sigma=0, base={BASE_ALGO}",
        "MC behaviour-anchor (per RE): "
        + ", ".join(f"RE{re}={anchors[re][0]:.2f}" for re in anchors)
        + f".  eval-derived improved scale ~= {_agg([vimp[re] / anchors[re][0] for re in vimp])[0]:.1f}x anchor.",
        "ratio=apparent_Q/anchor (1=behaviour scale=under-trained; ~10-12=improved scale; >>that=diverged)",
        "",
        "| target | steps | RE | anchor | obs_relerr | prox_relerr | obs_ratio | prox_ratio | oracle_ratio | eval_ret | n |",
        "|--|--|--|--|--|--|--|--|--|--|--|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['steps']} | {r['RE']} | {r['anchor']} | {r['obs_relerr']} "
            f"| {r['prox_relerr']} | {r['obs_ratio']} | {r['prox_ratio']} | {r['oracle_ratio']} "
            f"| {r['eval_ret']} | {r['n']} |"
        )
    (out_root / "table.md").write_text("\n".join(lines) + "\n")

    # proximal per-seed at 256k (the v1 anomaly: RE=300 gave sd>mean, one seed exploded)
    st = max(steps)
    pl = [
        "# proximal per-seed at 256k (chase the v1 instability)",
        "",
        "| RE | seed | prox_Q | prox_mse | prox_relerr | prox_ratio |",
        "|--|--|--|--|--|--|",
    ]
    for re in RE_GRID:
        for m, c, _, anc in runs.get(re, []):
            d = c["proximal"].get(st, {})
            pl.append(
                f"| {re} | {m['seed']} | {d.get('q', float('nan')):.2f} | {d.get('mse', float('nan')):.1f} "
                f"| {d.get('rel_err', float('nan')):.3f} | {d.get('ratio', float('nan')):.2f} |"
            )
    (out_root / "proximal_per_seed.md").write_text("\n".join(pl) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-one", action="store_true")
    ap.add_argument("--orchestrate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rollout-episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-root", default="docs/offline_budget_probe/runs_v2")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.run_one:
        out = Path(a.out or f"probe_out_v2/ds{a.rollout_episodes}_seed{a.seed}")
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

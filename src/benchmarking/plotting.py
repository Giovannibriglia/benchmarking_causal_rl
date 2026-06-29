from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.benchmarking.table_formatting import (
    best_indices_per_column,
    detect_sweep_families,
    family_label,
    format_cell,
    metric_direction,
    metric_label,
    strength_to_float_label,
)

# Global color map for consistent algorithm colors across plots
COLOR_MAP: Dict[str, tuple] = {}


def get_algo_color(algo: str, palette: str = "Set1"):
    """Return consistent color for each algo using a matplotlib palette (default Set1)."""
    import matplotlib

    if algo not in COLOR_MAP:
        idx = len(COLOR_MAP) % 10  # cycle palette if more than 10 algos
        COLOR_MAP[algo] = matplotlib.colormaps[palette](idx)
    return COLOR_MAP[algo]


def short_algo_label(algo: str) -> str:
    """Legend label for an algorithm identifier. The canonical on-policy id
    ``name__actor__critic`` renders compactly as ``name (actor/critic)``; bare
    names are returned unchanged. Note historical CSVs predating per-component
    ids carry a bare ``ppo`` (equivalent to ``ppo__mlp__mlp``); they render as
    ``ppo`` and are not auto-merged with new ids."""
    name = str(algo)
    # Oracle-U deconfounding variants are an oracle REFERENCE (they read the true
    # U), so brand them as such in every legend; the underlying base renders
    # normally. Only strings ending in the suffix are affected -> no impact on
    # any non-oracle algo label.
    if name.endswith("_oracle_u"):
        return f"{short_algo_label(name[: -len('_oracle_u')])} (oracle-U, true U)"
    parts = name.split("__")
    if len(parts) == 3:
        return f"{parts[0]} ({parts[1]}/{parts[2]})"
    return name


NON_METRIC_COLS = {
    "algorithm",
    "environment",
    "critic",
    "seed",
    "episode",
    "episode_idx",
    "checkpoint",
    "checkpoint_idx",
    "step",
    "frames",
    "time",
    "wall_time",
}


def load_run(run_name: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir = Path("runs") / run_name
    cfg_path = run_dir / "config.yaml"
    train_path = run_dir / "train_metrics.csv"
    eval_path = run_dir / "eval_metrics.csv"
    critic_path = run_dir / "critic_ablation_metrics.csv"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    if not train_path.exists() and not eval_path.exists() and not critic_path.exists():
        raise FileNotFoundError(
            "No metrics CSVs found (train_metrics.csv, eval_metrics.csv, or critic_ablation_metrics.csv)"
        )

    with cfg_path.open("r") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    critic_df = pd.read_csv(critic_path) if critic_path.exists() else pd.DataFrame()
    return config, train_df, eval_df, critic_df


def _get_rollout_len(cfg: dict) -> int:
    # Handle both nested and flat config snapshots
    if isinstance(cfg, dict):
        if "env" in cfg and isinstance(cfg["env"], dict):
            if "rollout_len" in cfg["env"]:
                return int(cfg["env"]["rollout_len"])
            if "rollout-len" in cfg["env"]:
                return int(cfg["env"]["rollout-len"])
        if "rollout_len" in cfg:
            return int(cfg["rollout_len"])
        if "rollout-len" in cfg:
            return int(cfg["rollout-len"])
    print("[warn] rollout_len not found in config; defaulting frames=x")
    return 1


def _resolve_x_label(x_axis: str, train_df: pd.DataFrame) -> str:
    """X-axis label, offline-aware.

    Offline runs (the PR3 signature: ``train_return_mean`` is all-empty because
    offline learners have gradient EPOCHS, not env episodes) plot the epoch index
    on the ``episode`` axis, so the label reads "Training epochs", not "Episodes".
    Online runs are unchanged. ``frames`` is regime-agnostic. Missing
    train_metrics.csv (empty df) falls back to the online label without crashing.
    """
    if x_axis == "frames":
        return "Frames"
    is_offline = (
        not train_df.empty
        and "train_return_mean" in train_df.columns
        and train_df["train_return_mean"].isna().all()
    )
    return "Training epochs" if is_offline else "Episodes"


def _per_context_table(sub: pd.DataFrame, env: str, algo: str, outdir: Path) -> None:
    """One LaTeX table per (env, algo): the per-context return distribution at the
    final checkpoint (columns from EVAL_PER_CONTEXT_COLUMNS)."""
    ep_max = sub["episode"].max()
    fin = sub[sub["episode"] == ep_max].sort_values("context_bin")
    if fin.empty:
        return
    header = [
        "\\textbf{bin}",
        "\\textbf{low}",
        "\\textbf{high}",
        "\\textbf{n}",
        "\\textbf{IQM}",
        "\\textbf{IQR-STD}",
    ]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{per-context return ({_escape_latex(env)}, "
        f"{_escape_latex(algo)}, final checkpoint)}}",
        f"\\label{{tab:per_context_{algo}_{env.replace('/', '-')}}}",
        "\\begin{tabular}{rrrrrr}",
        " & ".join(header) + " \\\\",
        "\\hline\\hline",
    ]
    for _, r in fin.iterrows():
        lines.append(
            " & ".join(
                [
                    str(int(r["context_bin"])),
                    f"{float(r['context_value_low']):.3f}",
                    f"{float(r['context_value_high']):.3f}",
                    str(int(r["n_episodes_in_bin"])),
                    f"{float(r['return_iqm']):.3f}",
                    f"{float(r['return_iqr_std']):.3f}",
                ]
            )
            + " \\\\"
        )
    lines += ["\\end{tabular}", "\\end{table}"]
    table_dir = outdir / "tables" / "per_context"
    _ensure_dir(table_dir)
    with (table_dir / f"{algo}_{env.replace('/', '-')}.tex").open("w") as f:
        f.write("\n".join(lines))


def render_eval_per_context(
    run_dir: Path,
    outdir: Path,
    x_axis: str,
    aggregation: str,
    formats: Iterable[str],
    x_label: str,
    rollout_len: int,
) -> None:
    """Render the Cell-2 per-context return bands from ``eval_per_context.csv``.

    One figure per (algorithm, environment): one line+band per ``context_bin``
    (center = ``return_iqm``, shaded spread = ``return_iqr_std``) over the
    checkpoint axis. The CSV is pre-aggregated by the runner, so ``aggregation``
    is informational here, not computational. Gated on file existence — absent =>
    a no-op with a log line (no fallback; this schema has no sensible substitute).

    No cross-env "overall" figure: bins are per-checkpoint empirical ranges that
    differ by env (each env masks different velocity components on different
    scales), so there is no common bin axis to aggregate over. Per-env figures
    are the Cell-2 deliverable.
    """
    import matplotlib

    from src.benchmarking.runner import EVAL_PER_CONTEXT_COLUMNS

    csv_path = Path(run_dir) / "eval_per_context.csv"
    if not csv_path.exists():
        print("[info] No per-context data for this run; skipping per_context split.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[info] eval_per_context.csv is empty; skipping per_context split.")
        return
    missing = [c for c in EVAL_PER_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        print(f"[warn] eval_per_context.csv missing columns {missing}; skipping.")
        return

    df["x"] = df["episode"] * rollout_len if x_axis == "frames" else df["episode"]
    n_bins = int(df["context_bin"].max()) + 1
    cmap = matplotlib.colormaps["viridis"]

    for (env, algo), sub in df.groupby(["environment", "algorithm"]):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
        for b in sorted(sub["context_bin"].unique()):
            bsub = sub[sub["context_bin"] == b].sort_values("x")
            x = bsub["x"].to_numpy()
            y = bsub["return_iqm"].to_numpy()
            spread = bsub["return_iqr_std"].fillna(0).to_numpy()
            color = cmap(int(b) / max(1, n_bins - 1))
            ax.plot(x, y, label=f"bin {int(b)}", color=color, linewidth=1.0)
            if len(bsub) >= 2:
                ax.fill_between(x, y - spread, y + spread, color=color, alpha=0.15)
        ax.set_title(f"per-context return - {env} - {algo}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("return (IQM ± IQR-STD)")
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title="context_bin",
        )
        fig.tight_layout()
        subdir = outdir / "plots" / "per_context" / "per_env"
        _ensure_dir(subdir)
        fname = subdir / f"{algo}_{env.replace('/', '-')}"
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)
        _per_context_table(sub, env, algo, outdir)


# ---------------------------------------------------------------------------
# Cell 7/8 value-trace renderers (offline_value_trace.csv) + the per-context
# final-checkpoint snapshot (eval_per_context.csv). All gated on file
# existence; non-relevant runs are unaffected.
# ---------------------------------------------------------------------------
def _sigma_from_run(run_dir: Path) -> Optional[float]:
    """The confounding strength σ for a run, or None.

    Reads it from the run-dir name first (``confounded_sigma_050_*`` -> 0.50),
    then falls back to the σ-encoded id in config.yaml's offline_dataset map
    (``...-sigma050-v0`` -> 0.50). config.yaml does not store behavior_strength
    directly, so these two encodings are the source of truth.
    """
    run_dir = Path(run_dir)
    m = re.search(r"sigma_?(\d{3})", run_dir.name)
    if m:
        return int(m.group(1)) / 100.0
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            ds = (cfg.get("env", {}) or {}).get("offline_dataset")
            ids = list(ds.values()) if isinstance(ds, dict) else ([ds] if ds else [])
            for did in ids:
                mm = re.search(r"-sigma(\d{3})-", str(did))
                if mm:
                    return int(mm.group(1)) / 100.0
    return None


def _final_true_return(ev: pd.DataFrame, env: str, algo: str) -> float:
    """Final-checkpoint true eval return for (env, algo) from eval_metrics, else NaN."""
    if ev.empty or "eval_return_mean" not in ev.columns:
        return float("nan")
    esub = ev[(ev["environment"] == env) & (ev["algorithm"] == algo)]
    if esub.empty:
        return float("nan")
    return float(esub.loc[esub["episode"].idxmax()]["eval_return_mean"])


def _value_trace_table(
    vt_sub: pd.DataFrame, ev: pd.DataFrame, env: str, algo: str, outdir: Path
) -> None:
    """Per-(env, algo) table: final apparent Q, final true return, their gap."""
    vt_sub = vt_sub.sort_values("epoch")
    final_apparent = float(vt_sub.iloc[-1]["apparent_value_iqm"])
    final_true = _final_true_return(ev, env, algo)
    gap = final_apparent - final_true if final_true == final_true else float("nan")

    def fmt(x):
        return f"{x:.3f}" if x == x else "--"

    header = [
        "\\textbf{final apparent Q}",
        "\\textbf{final true return}",
        "\\textbf{gap (app-true)}",
        "\\textbf{n epochs}",
    ]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{value trace ({_escape_latex(env)}, {_escape_latex(algo)}, "
        "final checkpoint)}}",
        f"\\label{{tab:value_trace_{algo}_{env.replace('/', '-')}}}",
        "\\begin{tabular}{rrrr}",
        " & ".join(header) + " \\\\",
        "\\hline\\hline",
        " & ".join([fmt(final_apparent), fmt(final_true), fmt(gap), str(len(vt_sub))])
        + " \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ]
    table_dir = outdir / "tables" / "value_trace" / "per_config"
    _ensure_dir(table_dir)
    (table_dir / f"{algo}_{env.replace('/', '-')}.tex").write_text("\n".join(lines))


def render_value_trace_per_config(
    run_dir: Path,
    outdir: Path,
    x_axis: str,
    formats: Iterable[str],
    x_label: str,
    rollout_len: int,
) -> None:
    """Per-config two-curve overlay (Cell 7/8): apparent Q (solid) vs true eval
    return (dashed) over training epochs, one figure per (env, algo). The σ-wedge
    is the gap between the curves. apparent Q is pre-aggregated by the runner;
    true return is joined from eval_metrics.csv on (epoch == episode, algo, env).
    Gated on offline_value_trace.csv existence — absent => skip with a log line.
    """
    from src.benchmarking.runner import OFFLINE_VALUE_TRACE_COLUMNS

    csv_path = Path(run_dir) / "offline_value_trace.csv"
    if not csv_path.exists():
        print("[info] No value-trace data for this run; skipping value_trace split.")
        return
    vt = pd.read_csv(csv_path)
    if vt.empty:
        print("[info] offline_value_trace.csv is empty; skipping value_trace split.")
        return
    missing = [c for c in OFFLINE_VALUE_TRACE_COLUMNS if c not in vt.columns]
    if missing:
        print(f"[warn] offline_value_trace.csv missing columns {missing}; skipping.")
        return

    eval_path = Path(run_dir) / "eval_metrics.csv"
    ev = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    sigma = _sigma_from_run(run_dir)

    # Floor-vs-ceiling pairing (intra-run): when a base algo and its *_oracle_u
    # sibling both appear in THIS run's value trace, render ONE overlaid figure
    # for the pair (base solid, oracle Q_adj dashed, u=0 anchor dotted) instead of
    # two separate per-config figures. Purely additive: with no pair present
    # `consumed` is empty and the per-config loop below is byte-identical to
    # before. Tables stay per-algo (written inside the pair helper for paired
    # algos, in the loop for the rest).
    algos_present = set(vt["algorithm"].unique())
    consumed: set[str] = set()
    for base in sorted(b for b in algos_present if f"{b}_oracle_u" in algos_present):
        variant = f"{base}_oracle_u"
        _render_value_trace_pair(
            vt, ev, base, variant, x_axis, formats, x_label, rollout_len, sigma, outdir
        )
        consumed.update({base, variant})

    for (env, algo), sub in vt.groupby(["environment", "algorithm"]):
        if algo in consumed:
            continue  # rendered as a floor-vs-ceiling overlay above
        sub = sub.sort_values("epoch")
        x = sub["epoch"] * rollout_len if x_axis == "frames" else sub["epoch"]
        # Twin-row layout (recon §6): apparent Q (discounted, small) and true
        # return (undiscounted, large) live on a shared axis pin the smaller
        # curve to the x-axis on solved envs. Split into two rows, each on its
        # own y-axis scale, sharing the training-epoch x-axis.
        fig, (ax_app, ax_tr) = plt.subplots(2, 1, sharex=True, figsize=(6, 5), dpi=500)
        ap = sub["apparent_value_iqm"].to_numpy()
        ap_s = sub["apparent_value_iqr_std"].fillna(0).to_numpy()
        c_ap = get_algo_color(f"{algo}:apparent")
        ax_app.plot(x, ap, label="apparent Q", color=c_ap, linewidth=1.2)
        if len(sub) >= 2:
            ax_app.fill_between(x, ap - ap_s, ap + ap_s, color=c_ap, alpha=0.15)

        true_plotted = False
        if not ev.empty and {"episode", "eval_return_mean"}.issubset(ev.columns):
            esub = ev[(ev["environment"] == env) & (ev["algorithm"] == algo)]
            if not esub.empty:
                merged = sub.merge(
                    esub, left_on="epoch", right_on="episode", how="left"
                )
                xt = (
                    merged["epoch"] * rollout_len
                    if x_axis == "frames"
                    else merged["epoch"]
                )
                tr = merged["eval_return_mean"].to_numpy()
                c_tr = get_algo_color(f"{algo}:true")
                ax_tr.plot(xt, tr, label="true return", color=c_tr, linewidth=1.2)
                if "eval_return_std" in merged and len(merged) >= 2:
                    tr_s = merged["eval_return_std"].fillna(0).to_numpy()
                    ax_tr.fill_between(xt, tr - tr_s, tr + tr_s, color=c_tr, alpha=0.10)
                true_plotted = True
        if not true_plotted:
            print(
                f"[info] No matching eval data for {env}/{algo}; plotting apparent Q only."
            )

        suptitle = f"apparent Q vs true return - {env} - {algo}"
        if sigma is not None:
            suptitle += f" (σ={sigma:g})"
        fig.suptitle(suptitle)
        ax_app.set_ylabel("apparent Q (discounted)")
        ax_tr.set_ylabel("true return (undiscounted)")
        ax_tr.set_xlabel(x_label)
        for ax in (ax_app, ax_tr):
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        subdir = outdir / "plots" / "value_trace" / "per_config"
        _ensure_dir(subdir)
        fname = subdir / f"{algo}_{env.replace('/', '-')}"
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)
        _value_trace_table(sub, ev, env, algo, outdir)


def _render_value_trace_pair(
    vt: pd.DataFrame,
    ev: pd.DataFrame,
    base: str,
    variant: str,
    x_axis: str,
    formats: Iterable[str],
    x_label: str,
    rollout_len: int,
    sigma: "float | None",
    outdir: Path,
) -> None:
    """Floor-vs-ceiling overlay for a base algo and its ``*_oracle_u`` sibling.

    One twin-row figure per env both appear in: row 0 apparent Q — base SOLID,
    oracle Q_adj DASHED (branded "oracle-U, true U"), oracle u=0 anchor DOTTED
    (when populated) — each with its IQR band; row 1 the true eval return likewise
    (base solid / oracle dashed). Both lines share the base hue (``get_algo_color``)
    so the pair reads as one family distinguished by linestyle. Per-algo value
    tables are still written (the figure is the only thing the pairing collapses).
    """
    color = get_algo_color(base)
    envs = sorted(
        set(vt[vt["algorithm"] == base]["environment"].unique())
        & set(vt[vt["algorithm"] == variant]["environment"].unique())
    )
    for env in envs:
        fig, (ax_app, ax_tr) = plt.subplots(2, 1, sharex=True, figsize=(6, 5), dpi=500)
        for algo, ls in ((base, "-"), (variant, "--")):
            sub = vt[
                (vt["environment"] == env) & (vt["algorithm"] == algo)
            ].sort_values("epoch")
            if sub.empty:
                continue
            label = short_algo_label(algo)
            x = sub["epoch"] * rollout_len if x_axis == "frames" else sub["epoch"]
            ap = sub["apparent_value_iqm"].to_numpy()
            ap_s = sub["apparent_value_iqr_std"].fillna(0).to_numpy()
            ax_app.plot(x, ap, color=color, linestyle=ls, linewidth=1.2, label=label)
            if len(sub) >= 2:
                ax_app.fill_between(x, ap - ap_s, ap + ap_s, color=color, alpha=0.12)
            # u=0 anchor (oracle variant only, when the column is present/filled).
            if (
                "apparent_value_u0_iqm" in sub.columns
                and sub["apparent_value_u0_iqm"].notna().any()
            ):
                u0 = sub["apparent_value_u0_iqm"].to_numpy()
                ax_app.plot(
                    x,
                    u0,
                    color=color,
                    linestyle=":",
                    linewidth=1.0,
                    label=f"{label} u=0 anchor",
                )
            # True return row, joined on epoch == episode (mirrors per-config).
            if not ev.empty and {"episode", "eval_return_mean"}.issubset(ev.columns):
                esub = ev[(ev["environment"] == env) & (ev["algorithm"] == algo)]
                if not esub.empty:
                    merged = sub.merge(
                        esub, left_on="epoch", right_on="episode", how="left"
                    )
                    xt = (
                        merged["epoch"] * rollout_len
                        if x_axis == "frames"
                        else merged["epoch"]
                    )
                    tr = merged["eval_return_mean"].to_numpy()
                    ax_tr.plot(
                        xt, tr, color=color, linestyle=ls, linewidth=1.2, label=label
                    )
                    if "eval_return_std" in merged and len(merged) >= 2:
                        tr_s = merged["eval_return_std"].fillna(0).to_numpy()
                        ax_tr.fill_between(
                            xt, tr - tr_s, tr + tr_s, color=color, alpha=0.10
                        )
            _value_trace_table(sub, ev, env, algo, outdir)

        suptitle = f"floor vs oracle ceiling - {env} - {base}"
        if sigma is not None:
            suptitle += f" (σ={sigma:g})"
        fig.suptitle(suptitle)
        ax_app.set_ylabel("apparent Q (discounted)")
        ax_tr.set_ylabel("true return (undiscounted)")
        ax_tr.set_xlabel(x_label)
        for ax in (ax_app, ax_tr):
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        subdir = outdir / "plots" / "value_trace" / "per_config"
        _ensure_dir(subdir)
        fname = subdir / f"{base}__vs__oracle_u_{env.replace('/', '-')}"
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)


def _collect_sigma_sweep(run_dir: Path):
    """Walk the cell directory for sibling runs of the same arm, pick the latest
    run per σ, and collect final-checkpoint (apparent Q, true return) per
    (env, algo). Returns ``(records, sigma_dirs)`` where records maps
    (env, algo) -> sorted list of (σ, apparent, true) and sigma_dirs maps σ -> dir.
    Empty records when no sibling value-trace data exists.
    """
    run_dir = Path(run_dir)
    name = run_dir.name
    if "_discrete_" in name:
        arm = "discrete"
    elif "_continuous_" in name:
        arm = "continuous"
    else:
        return {}, {}

    # Arm-tag filter (recon §1): gated runs aggregate only with gated siblings and
    # non-gated only with non-gated. Gated/non-gated runs of the same arm share the
    # arm token (e.g. ``..._discrete_gated_<ts>`` and ``..._discrete_<ts>`` both
    # match the glob) but have different env sets, so mixing them in one σ-sweep
    # panel is empirically wrong. Symmetric in both directions.
    is_gated = "_gated_" in name

    sigma_dirs: Dict[float, Path] = {}
    excluded = 0
    for d in sorted(run_dir.parent.glob(f"confounded_sigma_*_{arm}_*")):
        if not (d / "offline_value_trace.csv").exists():
            continue
        if ("_gated_" in d.name) != is_gated:
            excluded += 1
            continue
        s = _sigma_from_run(d)
        if s is None:
            continue
        if s in sigma_dirs:
            keep, drop = (
                (d, sigma_dirs[s])
                if d.name > sigma_dirs[s].name
                else (sigma_dirs[s], d)
            )
            print(
                f"[info] σ-sweep: multiple runs at σ={s:g}; using {keep.name}, "
                f"skipping {drop.name}."
            )
            sigma_dirs[s] = keep
        else:
            sigma_dirs[s] = d

    if excluded:
        tag = "gated" if is_gated else "non-gated"
        other = "non-gated" if is_gated else "gated"
        print(
            f"[info] σ-sweep: excluding {excluded} {other} siblings (current run "
            f"is {tag}; only {tag} sibling runs are aggregated)."
        )

    records: Dict[tuple, list] = {}
    for s in sorted(sigma_dirs):
        d = sigma_dirs[s]
        vt = pd.read_csv(d / "offline_value_trace.csv")
        ev = (
            pd.read_csv(d / "eval_metrics.csv")
            if (d / "eval_metrics.csv").exists()
            else pd.DataFrame()
        )
        for (env, algo), sub in vt.groupby(["environment", "algorithm"]):
            apparent = float(sub.loc[sub["epoch"].idxmax()]["apparent_value_iqm"])
            records.setdefault((env, algo), []).append(
                (s, apparent, _final_true_return(ev, env, algo))
            )
    for k in records:
        records[k] = sorted(records[k])
    return records, sigma_dirs


def _build_sigma_sweep_env_figure(env: str, algo_to_pts: Dict[str, list]):
    """Twin-row small multiples for one env: top row apparent Q vs σ, bottom row
    true return vs σ, one column per algo (recon §6). Each subplot keeps an
    independent y-axis — a shared axis compresses the small-scale curve (e.g. CQL
    apparent Q ~10 vs DQN ~600) onto the x-axis and hides the σ-wedge. Returns
    ``(fig, axes)`` with axes a 2×N array (row 0 apparent, row 1 true). σ=0.0 gets
    a dashed anchor line on every subplot; the text label only on the top-left.
    """
    algos = sorted(algo_to_pts)
    n = len(algos)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), dpi=500, squeeze=False)
    for j, algo in enumerate(algos):
        pts = sorted(algo_to_pts[algo])
        xs = [p[0] for p in pts]
        ax_app, ax_tr = axes[0][j], axes[1][j]
        ax_app.plot(
            xs,
            [p[1] for p in pts],
            marker="o",
            color=get_algo_color(f"{algo}:apparent"),
        )
        ax_tr.plot(
            xs,
            [p[2] for p in pts],
            marker="s",
            color=get_algo_color(f"{algo}:true"),
        )
        ax_app.set_title(algo)
        has_anchor = 0.0 in xs
        for ax in (ax_app, ax_tr):
            if has_anchor:
                ax.axvline(0.0, linestyle="--", color="gray", alpha=0.6, linewidth=0.8)
            if xs:
                ax.set_xticks(xs)
            ax.grid(True, alpha=0.3)
        ax_tr.set_xlabel("σ (confounding strength)")
        if has_anchor and j == 0:
            ax_app.text(
                0.0,
                1.0,
                " unconfounded anchor",
                transform=ax_app.get_xaxis_transform(),
                fontsize=7,
                va="top",
                color="gray",
            )
        if j == 0:
            ax_app.set_ylabel("apparent Q (discounted)")
            ax_tr.set_ylabel("true return (undiscounted)")
    fig.suptitle(f"σ-sweep — {env}")
    fig.tight_layout()
    return fig, axes


def _sigma_sweep_table(pts, env: str, algo: str, outdir: Path) -> None:
    def fmt(x):
        return f"{x:.3f}" if x == x else "--"

    pts = sorted(pts)
    # σ=0 anchor for scale-invariant ratios. The old `apparent - true` gap was
    # contaminated by the discounted-vs-undiscounted scale offset (recon §6:
    # CQL σ=0 gap ≈ -890 = Q≈7 minus return≈900, encoding scale not confounding).
    # apparent_rel / true_rel summarize the σ-trajectory shape honestly instead.
    anchor = next((p for p in pts if p[0] == 0.0), None)
    ap0 = anchor[1] if anchor else float("nan")
    tr0 = anchor[2] if anchor else float("nan")

    def rel(val, base):
        if base != base or abs(base) < 1e-9 or val != val:
            return float("nan")
        return val / base

    header = [
        "\\textbf{σ}",
        "\\textbf{apparent Q}",
        "\\textbf{true return}",
        "\\textbf{apparent\\_rel}",
        "\\textbf{true\\_rel}",
    ]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{σ-sweep ({_escape_latex(env)}, {_escape_latex(algo)})}}",
        f"\\label{{tab:sigma_sweep_{algo}_{env.replace('/', '-')}}}",
        "\\begin{tabular}{rrrrr}",
        " & ".join(header) + " \\\\",
        "\\hline\\hline",
    ]
    for s, ap, tr in pts:
        lines.append(
            " & ".join(
                [f"{s:g}", fmt(ap), fmt(tr), fmt(rel(ap, ap0)), fmt(rel(tr, tr0))]
            )
            + " \\\\"
        )
    lines += ["\\end{tabular}", "\\end{table}"]
    table_dir = outdir / "tables" / "value_trace" / "sigma_sweep"
    _ensure_dir(table_dir)
    (table_dir / f"{algo}_{env.replace('/', '-')}.tex").write_text("\n".join(lines))


def render_value_trace_sigma_sweep(
    run_dir: Path, outdir: Path, formats: Iterable[str]
) -> None:
    """σ-sweep paper figure (Cell 7/8): final-checkpoint apparent Q and true return
    as a function of σ across sibling runs in the same cell directory, as twin-row
    small multiples — one figure per env, 2 rows (apparent Q / true return) × N
    columns (algos), each subplot on its own y-axis. σ=0.0 is annotated as the
    unconfounded anchor. Renders partial sweeps (with a log line) and does not
    error on incomplete sweeps. No cross-env figure (envs differ in scale)."""
    records, sigma_dirs = _collect_sigma_sweep(run_dir)
    if not records:
        print("[info] value_trace σ-sweep: no sibling value-trace runs; skipping.")
        return
    if len(sigma_dirs) == 1:
        print("[info] Only 1 σ-value found in sibling runs; σ-sweep is incomplete.")

    by_env: Dict[str, Dict[str, list]] = {}
    for (env, algo), pts in records.items():
        by_env.setdefault(env, {})[algo] = pts

    subdir = outdir / "plots" / "value_trace" / "sigma_sweep"
    for env, algo_to_pts in by_env.items():
        fig, _axes = _build_sigma_sweep_env_figure(env, algo_to_pts)
        _ensure_dir(subdir)
        fname = subdir / env.replace("/", "-")
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)
        for algo, pts in algo_to_pts.items():
            _sigma_sweep_table(pts, env, algo, outdir)


def _final_eval_return(ev: pd.DataFrame, env: str, algo: str) -> tuple:
    """Final-checkpoint (eval_return_mean, eval_return_std) for (env, algo), else
    (NaN, NaN). The mean already reflects the run's --aggregation choice (mean or
    IQM) — the runner writes whichever statistic into eval_return_mean."""
    if ev.empty or "eval_return_mean" not in ev.columns:
        return float("nan"), float("nan")
    esub = ev[(ev["environment"] == env) & (ev["algorithm"] == algo)]
    if esub.empty:
        return float("nan"), float("nan")
    row = esub.loc[esub["episode"].idxmax()]
    std = (
        float(row["eval_return_std"])
        if "eval_return_std" in esub.columns
        else float("nan")
    )
    return float(row["eval_return_mean"]), std


def _collect_online_sigma_sweep(run_dir: Path):
    """Online analog of ``_collect_sigma_sweep``: walk the cell directory for
    sibling runs of the same arm, pick the latest run per σ, and collect the
    final-checkpoint eval return per (env, algo) from eval_metrics.csv.

    The online σ-sweep runs are the online variants of Cells 7/8 — their YAMLs
    (and so run dirs) carry the ``online_confounded_sigma_*`` prefix, while the
    offline Cell 7/8 runs are ``confounded_sigma_*``. This globs only the
    ``online_confounded_*`` dirs, so it never picks up the offline siblings (which
    render_value_trace_sigma_sweep handles via its own ``confounded_sigma_*``
    glob) — the two σ-sweep renderers partition the runs by prefix and never
    double-cover. The offline_value_trace.csv guard below is a belt-and-suspenders
    check on top of the prefix split. Returns ``(records, sigma_dirs)`` where
    records maps (env, algo) -> sorted list of (σ, eval_return_mean,
    eval_return_std)."""
    run_dir = Path(run_dir)
    name = run_dir.name
    if "_discrete_" in name:
        arm = "discrete"
    elif "_continuous_" in name:
        arm = "continuous"
    else:
        return {}, {}

    # Gated/non-gated runs of the same arm share the arm token but have different
    # env sets; aggregate only with same-gating siblings (mirrors the offline
    # collector's recon §1 rule).
    is_gated = "_gated_" in name

    sigma_dirs: Dict[float, Path] = {}
    excluded = 0
    for d in sorted(run_dir.parent.glob(f"online_confounded_sigma_*_{arm}_*")):
        if not (d / "eval_metrics.csv").exists():
            continue
        # An online variant should never carry a value trace; skip if it somehow
        # does (offline runs are matched by the confounded_sigma_* glob instead).
        if (d / "offline_value_trace.csv").exists():
            continue
        if ("_gated_" in d.name) != is_gated:
            excluded += 1
            continue
        s = _sigma_from_run(d)
        if s is None:
            continue
        if s in sigma_dirs:
            keep, drop = (
                (d, sigma_dirs[s])
                if d.name > sigma_dirs[s].name
                else (sigma_dirs[s], d)
            )
            print(
                f"[info] online σ-sweep: multiple runs at σ={s:g}; using {keep.name}, "
                f"skipping {drop.name}."
            )
            sigma_dirs[s] = keep
        else:
            sigma_dirs[s] = d

    if excluded:
        tag = "gated" if is_gated else "non-gated"
        other = "non-gated" if is_gated else "gated"
        print(
            f"[info] online σ-sweep: excluding {excluded} {other} siblings (current "
            f"run is {tag}; only {tag} sibling runs are aggregated)."
        )

    records: Dict[tuple, list] = {}
    for s in sorted(sigma_dirs):
        ev = pd.read_csv(sigma_dirs[s] / "eval_metrics.csv")
        if ev.empty or "algorithm" not in ev.columns:
            continue
        for (env, algo), _ in ev.groupby(["environment", "algorithm"]):
            mean, std = _final_eval_return(ev, env, algo)
            records.setdefault((env, algo), []).append((s, mean, std))
    for k in records:
        records[k] = sorted(records[k])
    return records, sigma_dirs


def _build_online_sigma_sweep_env_figure(env: str, algo_to_pts: Dict[str, list]):
    """One env, one row: eval return vs σ, one line per algorithm (algos overlaid
    on a shared axis — all curves share the eval-return scale, unlike the offline
    apparent-vs-true twin-row which needs split axes). Error bands are ±std.
    σ=0.0 gets a dashed anchor line + label. Returns ``(fig, ax)``."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
    all_sigmas: set = set()
    for algo in sorted(algo_to_pts):
        pts = sorted(algo_to_pts[algo])
        xs = [p[0] for p in pts]
        ys = np.array([p[1] for p in pts], dtype=float)
        es = np.nan_to_num(np.array([p[2] for p in pts], dtype=float))
        all_sigmas.update(xs)
        color = get_algo_color(algo)
        ax.plot(
            xs, ys, marker="o", color=color, label=short_algo_label(algo), linewidth=1.2
        )
        if len(xs) >= 2:
            ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.15)
    xs_sorted = sorted(all_sigmas)
    if 0.0 in all_sigmas:
        ax.axvline(0.0, linestyle="--", color="gray", alpha=0.6, linewidth=0.8)
        ax.text(
            0.0,
            1.0,
            " unconfounded anchor",
            transform=ax.get_xaxis_transform(),
            fontsize=7,
            va="top",
            color="gray",
        )
    if xs_sorted:
        ax.set_xticks(xs_sorted)
    ax.set_xlabel("σ (confounding strength)")
    ax.set_ylabel("eval return (undiscounted)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle(f"online σ-sweep — {env}")
    fig.tight_layout()
    return fig, ax


def render_online_sigma_sweep(
    run_dir: Path, outdir: Path, formats: Iterable[str]
) -> None:
    """Online σ-sweep paper figure (online variants of Cells 7/8): final-
    checkpoint eval return as a function of σ across sibling runs in the same cell
    directory — one figure per env, one line per algorithm, x-axis σ. The online
    analog of Cell 7/8's twin-row value-trace σ-wedge, but a single row: online
    runs produce no apparent-Q estimate (no offline learner), only true return.

    For these online variants the DQN (off-policy) line receives the full σ·U
    confounding while the PPO (on-policy) line receives only the reward half, so
    DQN diverging from PPO as σ grows is the figure's structural-confounding
    signal. Renders partial sweeps with a log line; skips cleanly when no online
    confounded siblings exist."""
    records, sigma_dirs = _collect_online_sigma_sweep(run_dir)
    if not records:
        print("[info] online σ-sweep: no online confounded sibling runs; skipping.")
        return
    if len(sigma_dirs) == 1:
        print("[info] Only 1 σ-value found in sibling runs; σ-sweep is incomplete.")

    by_env: Dict[str, Dict[str, list]] = {}
    for (env, algo), pts in records.items():
        by_env.setdefault(env, {})[algo] = pts

    subdir = outdir / "plots" / "online_sigma_sweep"
    for env, algo_to_pts in by_env.items():
        fig, _ax = _build_online_sigma_sweep_env_figure(env, algo_to_pts)
        _ensure_dir(subdir)
        fname = subdir / env.replace("/", "-")
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)


def _per_context_final_table(
    fin: pd.DataFrame, env: str, algo: str, outdir: Path
) -> None:
    header = [
        "\\textbf{bin}",
        "\\textbf{midpoint}",
        "\\textbf{n}",
        "\\textbf{IQM}",
        "\\textbf{IQR-STD}",
    ]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{final-checkpoint per-context return ({_escape_latex(env)}, "
        f"{_escape_latex(algo)})}}",
        f"\\label{{tab:per_context_final_{algo}_{env.replace('/', '-')}}}",
        "\\begin{tabular}{rrrrr}",
        " & ".join(header) + " \\\\",
        "\\hline\\hline",
    ]
    for _, r in fin.iterrows():
        lines.append(
            " & ".join(
                [
                    str(int(r["context_bin"])),
                    f"{float(r['context_midpoint']):.3f}",
                    str(int(r["n_episodes_in_bin"])),
                    f"{float(r['return_iqm']):.3f}",
                    f"{float(r['return_iqr_std']):.3f}",
                ]
            )
            + " \\\\"
        )
    lines += ["\\end{tabular}", "\\end{table}"]
    table_dir = outdir / "tables" / "per_context_final"
    _ensure_dir(table_dir)
    (table_dir / f"{algo}_{env.replace('/', '-')}.tex").write_text("\n".join(lines))


def render_eval_per_context_final(
    run_dir: Path, outdir: Path, formats: Iterable[str]
) -> None:
    """Final-checkpoint per-context snapshot (Cell 2/4): return vs context-value
    midpoint at the LAST checkpoint only, one figure per (env, algo). Unlike the
    over-training per_context view, bins are fixed at one checkpoint so the x-axis
    is a meaningful context-value space. Gated on eval_per_context.csv existence.
    """
    from src.benchmarking.runner import EVAL_PER_CONTEXT_COLUMNS

    csv_path = Path(run_dir) / "eval_per_context.csv"
    if not csv_path.exists():
        print(
            "[info] No per-context data for this run; skipping per_context_final split."
        )
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[info] eval_per_context.csv is empty; skipping per_context_final split.")
        return
    missing = [c for c in EVAL_PER_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        print(f"[warn] eval_per_context.csv missing columns {missing}; skipping.")
        return

    df["context_midpoint"] = (df["context_value_low"] + df["context_value_high"]) / 2.0
    for (env, algo), sub in df.groupby(["environment", "algorithm"]):
        ep_max = sub["episode"].max()
        fin = sub[sub["episode"] == ep_max].sort_values("context_midpoint")
        if fin.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
        x = fin["context_midpoint"].to_numpy()
        y = fin["return_iqm"].to_numpy()
        spread = fin["return_iqr_std"].fillna(0).to_numpy()
        color = get_algo_color(algo)
        ax.plot(x, y, marker="o", color=color, label=short_algo_label(algo))
        ax.fill_between(x, y - spread, y + spread, color=color, alpha=0.15)
        ax.set_title(f"final-checkpoint per-context return - {env} - {algo}")
        ax.set_xlabel("context value (bin midpoint)")
        ax.set_ylabel("return (IQM ± IQR-STD)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        fig.tight_layout()
        subdir = outdir / "plots" / "per_context_final"
        _ensure_dir(subdir)
        fname = subdir / f"{algo}_{env.replace('/', '-')}"
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)
        _per_context_final_table(fin, env, algo, outdir)


def discover_metrics(df: pd.DataFrame) -> List[Tuple[str, str, Optional[str]]]:
    """Return list of (logical_metric, mean_col, std_col_or_None)."""
    if df.empty:
        return []
    metrics: List[Tuple[str, str, Optional[str]]] = []
    cols = set(df.columns)
    paired = set()
    for col in cols:
        if col in NON_METRIC_COLS or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col.endswith("_mean"):
            base = col[: -len("_mean")]
            std_col = f"{base}_std"
            if std_col in cols and pd.api.types.is_numeric_dtype(df[std_col]):
                metrics.append((base, col, std_col))
                paired.add(col)
                paired.add(std_col)
                continue
        if col.endswith("_std"):
            base = col[: -len("_std")]
            mean_col = f"{base}_mean"
            if mean_col in cols and pd.api.types.is_numeric_dtype(df[mean_col]):
                continue  # handled when mean encountered
        if col not in paired:
            metrics.append((col, col, None))
    return metrics


def discover_critic_metrics(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    metrics: List[str] = []
    for col in df.columns:
        if col in NON_METRIC_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)
    return metrics


def _iqm(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    if values.size < 4:
        return float(np.mean(values))
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    lower = int(0.25 * (n - 1))
    upper = int(0.75 * (n - 1))
    middle = sorted_vals[lower : upper + 1]
    return float(np.mean(middle))


def _iqr_std(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    return float((q3 - q1) / 1.349)


def _collapse_envs(
    df: pd.DataFrame, group_cols: List[str], aggregation: str
) -> pd.DataFrame:
    """Collapse the dimension(s) not in ``group_cols`` by aggregating per group.

    A single flag-aware reducer for every cross-dimension collapse outside the
    cross-seed step. Two distinct uses share it:

    - Cross-environment "overall" collapse (train / eval / critic overall
      views, and the LaTeX "Overall" table row): ``group_cols`` keeps
      ``algorithm`` (and ``critic`` where present) plus ``x``; the
      ``environment`` dimension is collapsed.
    - Critic-dimension averaging (critic views): ``group_cols`` keeps
      ``environment, algorithm, x`` and the ``critic`` dimension is collapsed.

    For the critic *overall* figure the two are applied in order — average
    critics within ``(environment, algorithm, x)`` first, then collapse the
    environment dimension — so the env collapse operates on already
    critic-averaged centers.

    Under ``aggregation='iqm'`` the per-group center is the IQM of the input
    centers; under ``'mean'`` it is their arithmetic mean. The spread is the
    mean of the per-group spreads regardless of the flag, matching
    ``compute_aggregates``'s cross-seed spread convention (mean of per-seed
    stds). The center stat honors ``--aggregation``; the spread does not.
    """
    if df.empty:
        return df
    center_fn = _iqm if aggregation == "iqm" else "mean"
    return (
        df.groupby(group_cols)
        .agg({"center": center_fn, "spread": "mean"})
        .reset_index()
    )


def compute_aggregates(
    df: pd.DataFrame,
    metric_name: str,
    mean_col: str,
    std_col: Optional[str],
    x_axis: str,
    aggregation: str,
    rollout_len: int,
) -> pd.DataFrame:
    """Cross-seed aggregate per ``(environment, algorithm, x)`` -> center+spread.

    The ``center`` honors ``--aggregation``: IQM under ``iqm``, arithmetic mean
    under ``mean``.

    The ``spread`` is deliberately NOT symmetric with the center when a
    ``std_col`` is available (the train/eval case): it is the mean of the
    per-seed recorded stds, regardless of the flag. This is intentional — the
    runner records a per-seed std (within-seed return variability), which is the
    quantity the band is meant to convey, so the band reads as "typical
    within-seed spread" rather than the dispersion of the IQM estimator across
    seeds. It is *not* the IQR-std of pooled raw returns, because the raw
    returns are not retained at this stage — only the per-seed (mean, std)
    summaries are. When no ``std_col`` exists (the critic case), the spread does
    follow the flag (std under ``mean``, IQR-std under ``iqm``) to stay matched
    with the center.

    If a future change wants center+spread to share a statistical method
    (IQR-std under ``iqm``), it requires retaining raw returns or a schema
    change — see the follow-up issue forward-referenced from the PR description.
    """
    if df.empty or mean_col not in df.columns:
        return pd.DataFrame()

    use_cols = [mean_col, "algorithm", "environment"]
    if std_col:
        use_cols.append(std_col)
    df = df.dropna(subset=use_cols)
    if x_axis == "frames":
        x_col = "frames"
        if x_col not in df.columns:
            ep_col = "episode" if "episode" in df.columns else "episode_idx"
            df[x_col] = df[ep_col] * rollout_len
    else:
        x_col = "episode" if "episode" in df.columns else "episode_idx"

    def agg_func(group: pd.DataFrame):
        vals = group[mean_col].dropna().to_numpy()
        if vals.size == 0:
            return pd.Series({"center": np.nan, "spread": np.nan})
        if aggregation == "mean":
            center = float(np.mean(vals))
        else:
            center = _iqm(vals)

        if std_col:
            std_vals = group[std_col].dropna().to_numpy()
            spread = float(np.mean(std_vals)) if std_vals.size > 0 else 0.0
        else:
            if aggregation == "mean":
                spread = float(np.std(vals)) if vals.size > 1 else 0.0
            else:
                spread = _iqr_std(vals) if vals.size > 1 else 0.0
        return pd.Series({"center": center, "spread": spread})

    grouped = (
        df.groupby(["environment", "algorithm", x_col])
        .apply(agg_func)
        .reset_index()
        .rename(columns={x_col: "x"})
        .sort_values(by=["environment", "algorithm", "x"])
    )
    grouped["metric"] = metric_name
    return grouped


def compute_critic_aggregates(
    df: pd.DataFrame,
    metric_name: str,
    x_axis: str,
    aggregation: str,
    rollout_len: int,
) -> pd.DataFrame:
    if df.empty or metric_name not in df.columns:
        return pd.DataFrame()
    use_cols = [metric_name, "algorithm", "environment", "critic"]
    df = df.dropna(subset=use_cols)
    if df.empty:
        return pd.DataFrame()

    if x_axis == "frames":
        x_col = "frames"
        if x_col not in df.columns:
            ep_col = "episode" if "episode" in df.columns else "episode_idx"
            df[x_col] = df[ep_col] * rollout_len
    else:
        x_col = "episode" if "episode" in df.columns else "episode_idx"

    def agg_func(group: pd.DataFrame):
        vals = group[metric_name].dropna().to_numpy()
        if vals.size == 0:
            return pd.Series({"center": np.nan, "spread": np.nan})
        if aggregation == "mean":
            center = float(np.mean(vals))
            spread = float(np.std(vals)) if vals.size > 1 else 0.0
        else:
            center = _iqm(vals)
            spread = _iqr_std(vals) if vals.size > 1 else 0.0
        return pd.Series({"center": center, "spread": spread})

    grouped = (
        df.groupby(["environment", "algorithm", "critic", x_col])
        .apply(agg_func)
        .reset_index()
        .rename(columns={x_col: "x"})
        .sort_values(by=["environment", "algorithm", "critic", "x"])
    )
    grouped["metric"] = metric_name
    return grouped


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metric(
    aggregated: pd.DataFrame,
    metric: str,
    split: str,
    outdir: Path,
    formats: Iterable[str],
    x_label: str,
    overall: bool = False,
):
    if aggregated.empty:
        print(
            f"[warn] no data to plot for {metric} ({'overall' if overall else 'per-env'})"
        )
        return
    envs = sorted(aggregated["environment"].unique()) if not overall else ["overall"]
    for env in envs:
        data = aggregated if overall else aggregated[aggregated["environment"] == env]
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
        for algo in sorted(data["algorithm"].unique()):
            sub = data[data["algorithm"] == algo]
            if len(sub) < 1:
                continue
            x = sub["x"].to_numpy()
            y = sub["center"].to_numpy()
            spread = sub["spread"].fillna(0).to_numpy()
            color = get_algo_color(algo)
            ax.plot(x, y, label=short_algo_label(algo), color=color)
            if len(sub) >= 2:
                ax.fill_between(x, y - spread, y + spread, color=color, alpha=0.2)
        ax.set_title(f"{metric} - {env}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        split_dir = outdir / "plots" / split
        subdir = split_dir / ("overall" if overall else "per_env") / metric
        _ensure_dir(subdir)
        fname = subdir / ("overall" if overall else env)
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)


def plot_critic_metric(
    aggregated: pd.DataFrame,
    metric: str,
    outdir: Path,
    formats: Iterable[str],
    x_label: str,
    overall: bool = False,
    aggregation: str = "iqm",
    n_envs: int = 0,
):
    """Critic-ablation renderer — algos overlaid as colored lines.

    The ``critic`` dimension is averaged out *upstream* by the caller (via
    ``_collapse_envs`` over ``(environment, algorithm, x)``), so ``aggregated``
    carries one curve per algorithm, not per critic.

    per_env (``overall=False``): one figure per ``(metric, environment)``;
    ``aggregated`` is indexed by ``(environment, algorithm, x)``. Filename
    ``plots/critic/per_env/<metric>/<env>.png``.

    overall (``overall=True``): one figure per ``metric``; ``aggregated`` is the
    env-collapsed frame indexed by ``(algorithm, x)``. Filename
    ``plots/critic/overall/<metric>.png``. A subtitle records how many
    environments were averaged and under which ``--aggregation`` flag.
    """
    if aggregated.empty:
        print(
            f"[warn] no critic data to plot for {metric} ({'overall' if overall else 'per-env'})"
        )
        return

    envs = ["overall"] if overall else sorted(aggregated["environment"].unique())
    for env in envs:
        data = aggregated if overall else aggregated[aggregated["environment"] == env]
        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
        for algo in sorted(data["algorithm"].unique()):
            sub = data[data["algorithm"] == algo].sort_values("x")
            if sub.empty:
                continue
            x = sub["x"].to_numpy()
            y = sub["center"].to_numpy()
            spread = sub["spread"].fillna(0).to_numpy()
            color = get_algo_color(algo)
            ax.plot(x, y, label=short_algo_label(algo), color=color, marker="o")
            if len(sub) >= 2:
                ax.fill_between(x, y - spread, y + spread, color=color, alpha=0.2)

        title_env = "overall" if overall else env
        ax.set_title(f"{metric} - {title_env}", pad=18 if overall else None)
        if overall:
            method = "iqm" if aggregation == "iqm" else "mean"
            ax.text(
                0.5,
                1.0,
                f"averaged across {n_envs} environments ({method})",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()

        split_dir = outdir / "plots" / "critic"
        if overall:
            subdir = split_dir / "overall"
            _ensure_dir(subdir)
            fname = subdir / metric
        else:
            subdir = split_dir / "per_env" / metric
            _ensure_dir(subdir)
            fname = subdir / env.replace("/", "-")
        for fmt in formats:
            fig.savefig(fname.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)


def _escape_latex(text: str) -> str:
    return text.replace("_", "\\_")


def _standard_table_caption(run_name: Optional[str], metric: str, split: str) -> str:
    """Build a benchmark-naming caption for a standard per-run table.

    From a run name like ``rl_regimes/cell_1/online_discrete_20260618_202831``
    and metric ``eval_return`` produces
    ``Cell 1 online discrete — eval return (mean ± std)``. Falls back to the
    bare metric/split when the run name carries no recognizable cell/arm.
    """
    label = metric_label(metric)
    pieces: List[str] = []
    if run_name:
        cell_match = re.search(r"cell_(\d+)", run_name)
        if cell_match:
            pieces.append(f"Cell {cell_match.group(1)}")
        # Arm = run folder name minus the trailing _<date>_<time> stamp.
        folder = Path(run_name).name
        arm = re.sub(r"_\d{8}_\d{6}$", "", folder).replace("_", " ").strip()
        if arm:
            pieces.append(arm)
    prefix = " ".join(pieces) if pieces else f"{metric} ({split})"
    return f"{prefix} — {label} (mean ± std)"


def make_latex_table(
    df: pd.DataFrame,
    metric: str,
    split: str,
    aggregation: str,
    outdir: Path,
    precision: int = 1,
    run_name: Optional[str] = None,
):
    """Render a standard per-run table: envs (+ an Overall row) down the rows,
    algorithms across the columns. The best algorithm per env row is bolded,
    with direction (max/min) chosen per metric. Caption names the benchmark.
    """
    if df.empty:
        print(f"[warn] no data for table {metric} ({split})")
        return
    envs = sorted(df["environment"].unique())
    algos = sorted(df["algorithm"].unique())

    # Table rows are the env groups plus a trailing "Overall" group; columns are
    # algorithms. Build aligned (center, spread) matrices indexed [group, algo],
    # NaN where a cell is absent, then bold the best algo within each group.
    groups = list(envs) + ["Overall"]
    n_groups, n_algos = len(groups), len(algos)
    centers = np.full((n_groups, n_algos), np.nan)
    spreads = np.full((n_groups, n_algos), np.nan)

    for gi, env in enumerate(envs):
        env_df = df[df["environment"] == env]
        for ai, algo in enumerate(algos):
            cell_df = env_df[env_df["algorithm"] == algo]
            if cell_df.empty:
                continue
            centers[gi, ai] = cell_df.iloc[0]["center"]
            spreads[gi, ai] = cell_df.iloc[0]["spread"]

    # Overall row: cross-env collapse of centers/spreads where present. Center
    # honors --aggregation (IQM under iqm, mean under mean); spread is the mean
    # of per-env spreads regardless of flag — matching _collapse_envs and
    # compute_aggregates's cross-seed spread convention.
    for ai, algo in enumerate(algos):
        vals, sprs = [], []
        for env in envs:
            cell = df[(df["environment"] == env) & (df["algorithm"] == algo)]
            if not cell.empty:
                vals.append(cell.iloc[0]["center"])
                sprs.append(cell.iloc[0]["spread"])
        if vals:
            centers[-1, ai] = (
                _iqm(np.asarray(vals, dtype=float))
                if aggregation == "iqm"
                else float(np.mean(vals))
            )
            spreads[-1, ai] = float(np.mean(sprs))

    # Best algo per group: candidates are algos (rows of the transpose), groups
    # are the columns. best_per_group[gi] = set of best algo indices in group gi.
    direction = metric_direction(metric)
    best_per_group = best_indices_per_column(centers.T, direction)

    rows = []
    for gi, group in enumerate(groups):
        label = group if group == "Overall" else _escape_latex(group)
        row = [label]
        for ai in range(n_algos):
            row.append(
                format_cell(
                    centers[gi, ai],
                    spreads[gi, ai],
                    is_best=ai in best_per_group[gi],
                    precision=precision,
                )
            )
        rows.append(row)

    # Bold headers; escape underscores
    header = ["\\textbf{Environment}"] + [
        f"\\textbf{{{_escape_latex(a)}}}" for a in algos
    ]
    col_spec = "l" + "|c" * len(algos)
    caption = _standard_table_caption(run_name, metric, split)
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{split}_{metric}}}",
        "%\\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        " & ".join(header) + " \\\\",
        "\\hline\\hline",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_dir = outdir / "tables" / split
    _ensure_dir(table_dir)
    with (table_dir / f"{metric}.tex").open("w") as f:
        f.write("\n".join(lines))


def _final_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ep_col = "episode" if "episode" in df.columns else "episode_idx"
    # take last episode per env/algo
    max_ep = df.groupby(["environment", "algorithm"])[ep_col].transform("max")
    return df[df[ep_col] == max_ep]


def build_tables(
    df: pd.DataFrame,
    metrics: List[Tuple[str, str, Optional[str]]],
    split: str,
    aggregation: str,
    outdir: Path,
    precision: int = 1,
    run_name: Optional[str] = None,
):
    if df.empty:
        return
    df_final = _final_checkpoint(df)
    for metric_name, mean_col, std_col in metrics:
        if mean_col not in df_final.columns:
            continue
        if df_final[mean_col].dropna().empty:
            continue
        agg_rows = []
        for (env, algo), sub in df_final.groupby(["environment", "algorithm"]):
            vals_mean = sub[mean_col].dropna().to_numpy()
            if vals_mean.size == 0:
                continue
            if aggregation == "mean":
                center = float(np.mean(vals_mean))
            else:
                center = _iqm(vals_mean)

            if std_col and std_col in sub.columns:
                vals_std = sub[std_col].dropna().to_numpy()
                spread = float(np.mean(vals_std)) if vals_std.size > 0 else 0.0
            else:
                if aggregation == "mean":
                    spread = float(np.std(vals_mean)) if vals_mean.size > 1 else 0.0
                else:
                    spread = _iqr_std(vals_mean) if vals_mean.size > 1 else 0.0
            agg_rows.append(
                {
                    "environment": env,
                    "algorithm": algo,
                    "center": center,
                    "spread": spread,
                }
            )
        if not agg_rows:
            continue
        agg_df = pd.DataFrame(agg_rows)
        make_latex_table(
            agg_df,
            metric_name,
            split,
            aggregation,
            outdir,
            precision=precision,
            run_name=run_name,
        )


_RUN_STAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def _latest_run_dir(runs_cell_dir: Path, basename: str) -> Optional[Path]:
    """Return the most recent ``<basename>_<date>_<time>`` run dir under
    ``runs_cell_dir``, or None if none exist.

    Anchored on the timestamp suffix so a prefix like ``confounded_sigma_050_
    discrete`` does not accidentally match its ``..._discrete_gated`` sibling.
    """
    pattern = re.compile(rf"^{re.escape(basename)}_\d{{8}}_\d{{6}}$")
    candidates = [
        p
        for p in runs_cell_dir.glob(f"{basename}_*")
        if p.is_dir() and pattern.match(p.name)
    ]
    if not candidates:
        return None
    # Timestamp suffix sorts lexically => latest name wins.
    return max(candidates, key=lambda p: p.name)


def render_sweep_family_table(
    family_stem: str,
    members: List[Tuple[str, str]],
    runs_cell_dir: Path,
    out_cell_dir: Path,
    metric: str = "eval_return",
    aggregation: str = "iqm",
    precision: int = 1,
) -> List[Path]:
    """For a detected sweep family, emit one LaTeX table per env with strength
    values as columns, algorithms as rows, and the best algorithm per strength
    column bolded. Missing family members (YAMLs not yet run) are skipped with a
    warning rather than crashing.

    Returns the list of written ``.tex`` paths.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # (env, algo, strength) -> (center, spread). Collected across whichever
    # family members have a run on disk.
    records: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    present_strengths: List[str] = []
    missing_strengths: List[str] = []

    for strength, basename in members:
        run_dir = _latest_run_dir(runs_cell_dir, basename)
        eval_csv = run_dir / "eval_metrics.csv" if run_dir else None
        if run_dir is None or not eval_csv.exists():
            missing_strengths.append(strength)
            continue
        df = pd.read_csv(eval_csv)
        if mean_col not in df.columns or df.empty:
            missing_strengths.append(strength)
            continue
        final = _final_checkpoint(df)
        present_strengths.append(strength)
        for (env, algo), sub in final.groupby(["environment", "algorithm"]):
            vals = sub[mean_col].dropna().to_numpy()
            if vals.size == 0:
                continue
            center = float(np.mean(vals)) if aggregation == "mean" else _iqm(vals)
            if std_col in sub.columns:
                svals = sub[std_col].dropna().to_numpy()
                spread = float(np.mean(svals)) if svals.size else 0.0
            else:
                spread = float(np.std(vals)) if vals.size > 1 else 0.0
            records[(env, algo, strength)] = (center, spread)

    if missing_strengths:
        print(
            f"[warn] sweep family '{family_stem}': no run found for "
            f"strengths {sorted(missing_strengths)} — skipping those columns"
        )
    if not records:
        print(f"[warn] sweep family '{family_stem}': no run data — skipping")
        return []

    strengths = sorted(present_strengths)
    envs = sorted({env for (env, _, _) in records})
    direction = metric_direction(metric)
    written: List[Path] = []
    _ensure_dir(out_cell_dir)

    for env in envs:
        algos = sorted({a for (e, a, _) in records if e == env})
        centers = np.full((len(algos), len(strengths)), np.nan)
        spreads = np.full((len(algos), len(strengths)), np.nan)
        for ai, algo in enumerate(algos):
            for si, strength in enumerate(strengths):
                rec = records.get((env, algo, strength))
                if rec is not None:
                    centers[ai, si], spreads[ai, si] = rec

        best_per_col = best_indices_per_column(centers, direction)

        header = ["\\textbf{Algorithm}"] + [
            f"\\textbf{{{strength_to_float_label(s)}}}" for s in strengths
        ]
        col_spec = "l" + "|c" * len(strengths)
        caption = (
            f"{family_label(family_stem)} strength sweep on "
            f"{_escape_latex(env)} — {metric_label(metric)} (mean ± std)"
        )
        env_safe = env.replace("/", "-")
        lines = [
            "\\begin{table}[!t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{tab:sweep_{family_stem}_{env_safe}}}",
            "%\\footnotesize",
            f"\\begin{{tabular}}{{{col_spec}}}",
            " & ".join(header) + " \\\\",
            "\\hline\\hline",
        ]
        for ai, algo in enumerate(algos):
            row = [_escape_latex(algo)]
            for si in range(len(strengths)):
                row.append(
                    format_cell(
                        centers[ai, si],
                        spreads[ai, si],
                        is_best=ai in best_per_col[si],
                        precision=precision,
                    )
                )
            lines.append(" & ".join(row) + " \\\\")
        lines += ["\\end{tabular}", "\\end{table}"]

        out_path = out_cell_dir / f"{family_stem}_{env_safe}.tex"
        out_path.write_text("\n".join(lines))
        written.append(out_path)

    return written


def render_sweep_tables(
    run_name: str,
    base_outdir: Path,
    aggregation: str,
    precision: int = 1,
) -> List[Path]:
    """Detect sweep families for the cell that ``run_name`` belongs to and emit
    aggregated strength-sweep tables, one per (family, env).

    Sweep tables aggregate across sibling runs, so they are written to
    ``<base_outdir>/<cell_path>/sweep_tables/`` (not the per-run output dir).
    The cell's YAML directory under ``reproducibility/<cell_path>`` drives
    family detection; each member's latest run dir under ``runs/<cell_path>``
    supplies the data.
    """
    cell_path = Path(run_name).parent  # e.g. rl_regimes/cell_7
    yaml_dir = Path("reproducibility") / cell_path
    runs_cell_dir = Path("runs") / cell_path
    out_cell_dir = base_outdir / cell_path / "sweep_tables"

    if not yaml_dir.exists():
        print(f"[warn] sweep_tables: YAML dir {yaml_dir} not found — skipping")
        return []

    families = detect_sweep_families(yaml_dir)
    if not families:
        print(f"[warn] sweep_tables: no sweep families detected in {yaml_dir}")
        return []

    written: List[Path] = []
    for family_stem, members in sorted(families.items()):
        written.extend(
            render_sweep_family_table(
                family_stem,
                members,
                runs_cell_dir,
                out_cell_dir,
                aggregation=aggregation,
                precision=precision,
            )
        )
    return written


def run_plotting(
    run_name: str,
    split: str,
    x_axis: str,
    aggregation: str,
    outdir: Path,
    formats: List[str],
    precision: int = 1,
):
    config, train_df, eval_df, critic_df = load_run(run_name)
    rollout_len = _get_rollout_len(config)
    base_outdir = Path(outdir)
    outdir = base_outdir / run_name
    # Offline-aware: offline runs label the checkpoint axis "Training epochs".
    x_label = _resolve_x_label(x_axis, train_df)
    mode = (
        config.get("training", {}).get("mode", "benchmark")
        if isinstance(config, dict)
        else "benchmark"
    )

    if split in ("train", "both", "all") and not train_df.empty:
        train_metrics = discover_metrics(train_df)
        for metric_name, mean_col, std_col in train_metrics:
            agg = compute_aggregates(
                train_df,
                metric_name,
                mean_col,
                std_col,
                x_axis,
                aggregation,
                rollout_len,
            )
            if agg.empty or agg.shape[0] < 2:
                continue
            plot_metric(
                agg, metric_name, "train", outdir, formats, x_label, overall=False
            )
            # overall figure: collapse the environment dimension, flag-aware.
            overall = _collapse_envs(agg, ["algorithm", "x"], aggregation)
            overall["environment"] = "overall"
            plot_metric(
                overall, metric_name, "train", outdir, formats, x_label, overall=True
            )
        build_tables(
            train_df,
            train_metrics,
            "train",
            aggregation,
            outdir,
            precision=precision,
            run_name=run_name,
        )

    if split in ("eval", "both", "all") and not eval_df.empty:
        eval_metrics = discover_metrics(eval_df)
        for metric_name, mean_col, std_col in eval_metrics:
            agg = compute_aggregates(
                eval_df,
                metric_name,
                mean_col,
                std_col,
                x_axis,
                aggregation,
                rollout_len,
            )
            if agg.empty or agg.shape[0] < 2:
                continue
            plot_metric(
                agg, metric_name, "eval", outdir, formats, x_label, overall=False
            )
            # overall figure: collapse the environment dimension, flag-aware.
            overall = _collapse_envs(agg, ["algorithm", "x"], aggregation)
            overall["environment"] = "overall"
            plot_metric(
                overall, metric_name, "eval", outdir, formats, x_label, overall=True
            )
        build_tables(
            eval_df,
            eval_metrics,
            "eval",
            aggregation,
            outdir,
            precision=precision,
            run_name=run_name,
        )

    plot_critic = split in ("critic", "all") or (
        split == "both" and mode == "critic_ablation"
    )
    critic_source = critic_df
    if plot_critic and critic_source.empty and not train_df.empty:
        critic_source = train_df.copy()
        critic_source["critic"] = "standard"
    if plot_critic and not critic_source.empty:
        critic_metrics = discover_critic_metrics(critic_source)
        for metric_name in critic_metrics:
            agg = compute_critic_aggregates(
                critic_source,
                metric_name,
                x_axis,
                aggregation,
                rollout_len,
            )
            if agg.empty:
                continue
            # Drop the critic dimension: average critics within each
            # (environment, algorithm, x) tuple (flag-aware center, mean of
            # spreads). Algos are then overlaid as lines, one figure per
            # (metric, env).
            per_env = _collapse_envs(
                agg, ["environment", "algorithm", "x"], aggregation
            )
            plot_critic_metric(
                per_env, metric_name, outdir, formats, x_label, overall=False
            )
            # Overall: collapse the environment dimension on the already
            # critic-averaged frame, flag-aware. One figure per metric, one
            # line per algo, env-averaged.
            n_envs = per_env["environment"].nunique()
            overall = _collapse_envs(per_env, ["algorithm", "x"], aggregation)
            overall["environment"] = "overall"
            plot_critic_metric(
                overall,
                metric_name,
                outdir,
                formats,
                x_label,
                overall=True,
                aggregation=aggregation,
                n_envs=n_envs,
            )

    # Per-context (Cell 2): gated on eval_per_context.csv existence. Skips
    # silently for non-masked runs (no file). Included in the "all" umbrella.
    if split in ("per_context", "all"):
        render_eval_per_context(
            Path("runs") / run_name,
            outdir,
            x_axis,
            aggregation,
            formats,
            x_label,
            rollout_len,
        )

    # Value trace (Cells 7-8): per-config apparent-vs-true overlay + σ-sweep
    # panel. Gated on offline_value_trace.csv existence (confounded offline runs).
    if split in ("value_trace", "all"):
        render_value_trace_per_config(
            Path("runs") / run_name, outdir, x_axis, formats, x_label, rollout_len
        )
        render_value_trace_sigma_sweep(Path("runs") / run_name, outdir, formats)

    # Online σ-sweep (online variants of Cells 7/8): eval return vs σ across
    # sibling online_confounded_* runs, one line per algo. Gated on
    # eval_metrics.csv; the online_confounded_* glob partitions these from the
    # offline confounded_* siblings (value-trace renderer). Skips cleanly when
    # there are no online confounded siblings.
    if split in ("online_sigma_sweep", "all"):
        render_online_sigma_sweep(Path("runs") / run_name, outdir, formats)

    # Final-checkpoint per-context snapshot (Cells 2/4): the clean companion to
    # the over-training per_context view. Gated on eval_per_context.csv.
    if split in ("per_context_final", "all"):
        render_eval_per_context_final(Path("runs") / run_name, outdir, formats)

    # Sweep-family tables (strength dials): aggregate eval return across sibling
    # runs of the same sweep family (one table per env, strengths as columns).
    # Detected from the cell's YAML filename conventions; writes to
    # outputs/<cell_path>/sweep_tables/ (not the per-run dir). Skips cleanly when
    # no families are detected or no sibling runs exist yet.
    if split in ("sweep_tables", "all"):
        render_sweep_tables(run_name, base_outdir, aggregation, precision=precision)


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmarking runs and export tables"
    )
    parser.add_argument("--run", required=True, help="Run folder name under runs/")
    parser.add_argument(
        "--split",
        choices=[
            "train",
            "eval",
            "critic",
            "per_context",
            "per_context_final",
            "value_trace",
            "online_sigma_sweep",
            "sweep_tables",
            "both",
            "all",
        ],
        default="both",
        help="Which split(s) to plot",
    )
    parser.add_argument(
        "--x-axis",
        choices=["episodes", "frames"],
        default="frames",
        help="X-axis for plots",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "iqm"],
        default="iqm",
        help="Aggregation across seeds",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Base output directory for plots and tables",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="File formats to save",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Decimal places for table mean ± std cells",
    )
    args = parser.parse_args()

    run_plotting(
        run_name=args.run,
        split=args.split,
        x_axis=args.x_axis,
        aggregation=args.aggregation,
        outdir=Path(args.outdir),
        formats=args.formats,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()

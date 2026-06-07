"""Causal-cells figures and tables (Phase 7) — additive plot.py extensions.

Everything regenerates from CSVs alone (acceptance): the matrix summary
(``outputs/matrix_summary.csv``), the cell-2 summary, the sweep run CSVs for
the OPE table, and the cached Γ-calibration / horizon CSVs.

PLOT-DESIGN RULE (Phase-7 gate): cells are NOT presented as a monotone
difficulty axis — basics are non-monotone in cell index. Cells are grouped by
IDENTIFICATION REGIME (identified / confounded / hidden state) and captions
say so.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarking.plotting import _ensure_dir, _escape_latex
from src.eval.regret import iqm, stratified_bootstrap_iqm_ci

# identification-regime grouping (NOT cell-index order)
REGIMES: List[tuple] = [
    ("identified", [2, 3, 5]),
    ("confounded", [7]),
    ("hidden state", [4, 6, 8]),
]
CELL_NAMES = {
    2: "2: Invisible Gene",
    3: "3: Perfect Archive",
    4: "4: Burned Files",
    5: "5: Doctor's Intuition",
    6: "6: Fog of History",
    7: "7: Shadowed Vitals",
    8: "8: Dark Ages",
}
BASIC_COLOR, VARIANT_COLOR = "#d62728", "#1f77b4"


def _save(fig, outdir: Path, name: str, formats: Sequence[str]) -> None:
    _ensure_dir(outdir)
    for ext in formats:
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {outdir}/{name}.{{{','.join(formats)}}}")


def _summary_rows(matrix_csv: Path, cell2_csv: Path) -> pd.DataFrame:
    """One row per (cell, role): IQM normalized regret + CI. Cell 3 uses the
    medium tier in the money plot (the tier sweep has its own panel)."""
    m = pd.read_csv(matrix_csv)
    m = m[(m.cell != 3) | (m.tier == "medium")]
    rows = (
        m[["cell", "role", "algo", "iqm_norm_regret", "ci_low", "ci_high"]]
        .copy()
        .reset_index(drop=True)  # sparse filtered index would make .loc[len]
        # OVERWRITE rows instead of appending
    )

    c2 = pd.read_csv(cell2_csv)
    for role, grp in c2.groupby("role"):
        vals = grp.normalized_regret.to_numpy(dtype=float)
        lo, hi = stratified_bootstrap_iqm_ci(vals.reshape(1, -1))
        rows.loc[len(rows)] = [2, role, grp.algo.iloc[0], iqm(vals), lo, hi]
    return rows


def money_plot(
    matrix_csv: Path, cell2_csv: Path, outdir: Path, formats: Sequence[str]
) -> None:
    rows = _summary_rows(matrix_csv, cell2_csv)
    order = [c for _, cells in REGIMES for c in cells]
    xpos = {c: i for i, c in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    # regime bands + labels
    start = 0
    for i, (label, cells) in enumerate(REGIMES):
        end = start + len(cells)
        if i % 2 == 1:
            ax.axvspan(start - 0.5, end - 0.5, color="0.92", zorder=0)
        ax.text(
            (start + end - 1) / 2,
            1.13,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
        )
        start = end
    for role, color, off in (
        ("basic", BASIC_COLOR, -0.12),
        ("variant", VARIANT_COLOR, +0.12),
    ):
        sub = rows[rows.role == role]
        xs = [xpos[int(c)] + off for c in sub.cell]
        ys = sub.iqm_norm_regret.to_numpy(dtype=float)
        yerr = np.vstack(
            [ys - sub.ci_low.to_numpy(float), sub.ci_high.to_numpy(float) - ys]
        )
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            color=color,
            capsize=4,
            markersize=7,
            label=role,
            lw=1.5,
        )
    ax.axhline(0.0, color="0.4", ls="--", lw=1, label="cell-1 reference")
    ax.axhline(1.0, color="0.7", ls=":", lw=1, label="random floor")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CELL_NAMES[c] for c in order], rotation=20, ha="right")
    ax.set_ylabel("normalized regret (IQM, 95% CI, 5 seeds)")
    ax.set_ylim(-0.05, 1.2)
    ax.set_title(
        "Regret vs Cell-1 reference, grouped by identification regime\n"
        "(cells are NOT a monotone difficulty axis; cell 3 shown on the "
        "medium tier)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, outdir, "money_plot_regret", formats)


def tier_panel(matrix_csv: Path, outdir: Path, formats: Sequence[str]) -> None:
    m = pd.read_csv(matrix_csv)
    m = m[(m.cell == 3) & (m.tier.isin(["simple", "medium", "expert"]))]
    tiers = ["simple", "medium", "expert"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for role, color, off in (
        ("basic", BASIC_COLOR, -0.1),
        ("variant", VARIANT_COLOR, +0.1),
    ):
        sub = m[m.role == role].set_index("tier").loc[tiers]
        xs = np.arange(len(tiers)) + off
        ys = sub.iqm_norm_regret.to_numpy(float)
        yerr = np.vstack(
            [ys - sub.ci_low.to_numpy(float), sub.ci_high.to_numpy(float) - ys]
        )
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", color=color, capsize=4, label=role)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ax.set_xlabel("coverage tier (cell 3)")
    ax.set_ylabel("normalized regret (IQM, 95% CI)")
    ax.set_title(
        "Coverage axis: the expert tier INVERTS the ordering (S4)", fontsize=10
    )
    ax.legend()
    _save(fig, outdir, "cell3_tier_panel", formats)


def ope_table(run_csvs: List[Path], outdir: Path) -> None:
    """Cells x estimators LaTeX table: IQM over seeds of each estimate vs the
    true deployment J; KZ interval where reported (confounded cells)."""
    frames = [pd.read_csv(p) for p in run_csvs]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.role.isin(["basic", "variant"])]
    df = df[(df.cell != 3) | (df.tier == "medium")]

    def _fmt(v) -> str:
        return "--" if pd.isna(v) else f"{v:.0f}"

    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{OPE estimates vs true deployment $J$ (IQM over 5 seeds, "
        "CartPole anchor; cell 3 on the medium tier). KZ: Kallus--Zhou "
        "sensitivity interval at $\\Gamma{=}2$, reported for the confounded "
        "cells. Grouped by identification regime, not cell index.}",
        "\\label{tab:cells_estimators}",
        "\\begin{tabular}{llc|cccc|c}",
        "\\textbf{Cell} & \\textbf{Policy} & \\textbf{true $J$} & "
        "\\textbf{naive} & \\textbf{DM} & \\textbf{IPW} & \\textbf{DR} & "
        "\\textbf{KZ interval} \\\\",
        "\\hline\\hline",
    ]
    for _, cells in REGIMES:
        for cell in cells:
            sub = df[df.cell == cell]
            if sub.empty:
                continue
            for role in ("basic", "variant"):
                r = sub[sub.role == role]
                if r.empty:
                    continue
                agg: Dict[str, float] = {
                    k: iqm(r[k].to_numpy(float))
                    for k in ("J", "ope_naive", "ope_dm", "ope_ipw", "ope_dr")
                }
                if "ope_kz_lb" in r and r.ope_kz_lb.notna().any():
                    kz = (
                        f"$[{iqm(r.ope_kz_lb.to_numpy(float)):.0f},\\,"
                        f"{iqm(r.ope_kz_ub.to_numpy(float)):.0f}]$"
                    )
                else:
                    kz = "--"
                algo = _escape_latex(str(r.algo.iloc[0]))
                lines.append(
                    f"{cell} & {algo} & {_fmt(agg['J'])} & "
                    f"{_fmt(agg['ope_naive'])} & {_fmt(agg['ope_dm'])} & "
                    f"{_fmt(agg['ope_ipw'])} & {_fmt(agg['ope_dr'])} & {kz} \\\\"
                )
        lines.append("\\hline")
    lines[-1] = "\\hline\\hline" if lines[-1] == "\\hline" else lines[-1]
    lines += ["\\end{tabular}", "\\end{table}"]
    table_dir = outdir / "tables"
    _ensure_dir(table_dir)
    (table_dir / "cells_estimators.tex").write_text("\n".join(lines))
    print(f"wrote {table_dir}/cells_estimators.tex")


def gamma_figure(gamma_csv: Path, outdir: Path, formats: Sequence[str]) -> None:
    df = pd.read_csv(gamma_csv)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.fill_between(
        df.gamma,
        df.lower,
        df.upper,
        color=VARIANT_COLOR,
        alpha=0.25,
        label="KZ interval",
    )
    ax.plot(df.gamma, df.lower, color=VARIANT_COLOR, lw=1.5)
    ax.plot(df.gamma, df.upper, color=VARIANT_COLOR, lw=1.5)
    ax.axhline(
        float(df.true_j.iloc[0]),
        color="0.2",
        ls="--",
        lw=1.2,
        label="true deployment $J$",
    )
    ax.set_xlabel("sensitivity budget $\\Gamma$ (odds ratio)")
    ax.set_ylabel("value bound")
    ax.set_title(
        "Γ-calibration (cell 8): the interval first covers the truth\n"
        "when Γ matches the actual confounding strength",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    _save(fig, outdir, "gamma_calibration", formats)


def horizon_figure(horizon_csv: Path, outdir: Path, formats: Sequence[str]) -> None:
    df = pd.read_csv(horizon_csv)
    regimes = df.regime.tolist()
    x = np.arange(len(regimes))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for est, color, off in (
        ("ipw", "#2ca02c", -0.12),
        ("dr", "#9467bd", +0.12),
    ):
        ax.bar(
            x + off,
            (df[est] / df.true_j).to_numpy(float),
            width=0.22,
            color=color,
            label=est.upper(),
        )
    ax.axhline(1.0, color="0.2", ls="--", lw=1.2, label="truth")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_xlabel("horizon regime (cell 3, known propensities)")
    ax.set_ylabel("estimate / truth (log scale)")
    ax.set_title(
        "Horizon ablation: graphical identifiability holds everywhere,\n"
        "statistical identifiability of IPW degrades with horizon",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    _save(fig, outdir, "horizon_ablation", formats)


def online_curves(
    bench_run_dirs: List[Path],
    reference_run_dirs: List[Path],
    outdir: Path,
    formats: Sequence[str],
    rollout_len: int = 512,
    n_train_envs: int = 8,
) -> None:
    """Online cells 1/2/2fs learning curves from EXISTING benchmark CSVs:
    window eval return vs frames, IQM band over seeds. NOTE (caption): the
    window eval metric compresses differences near saturation - final
    per-episode J lives in the money plot; this figure shows training
    dynamics. Cell-1 reference runs keep only first/last checkpoints."""
    series = {
        "causal/cartpole-cell1": ("cell 1 (reference)", "0.3"),
        "causal/cartpole-cell2": ("cell 2 basic (masked)", BASIC_COLOR),
        "causal/cartpole-cell2fs": ("cell 2 variant (frame-stack)", VARIANT_COLOR),
    }
    frames_per_ep = rollout_len * n_train_envs
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for env_id, (label, color) in series.items():
        runs = reference_run_dirs if "cell1" in env_id else bench_run_dirs
        per_seed = []
        for rd in runs:
            df = pd.read_csv(Path(rd) / "eval_metrics.csv")
            df = df[df.environment == env_id]
            if not df.empty:
                per_seed.append(df.set_index("episode").eval_return_mean)
        if not per_seed:
            continue
        mat = pd.concat(per_seed, axis=1).dropna()
        x = (mat.index.to_numpy(float) + 1) * frames_per_ep
        vals = mat.to_numpy(float)
        center = np.array([iqm(v) for v in vals])
        lo = np.quantile(vals, 0.25, axis=1)
        hi = np.quantile(vals, 0.75, axis=1)
        ax.plot(x, center, color=color, label=label, marker="o", ms=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2)
    ax.set_xlabel("frames")
    ax.set_ylabel("window eval return (IQM, IQR band, 5 seeds)")
    ax.set_title(
        "Online cells: training dynamics (window eval metric;\n"
        "near-saturation differences are compressed - see money plot for J)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    _save(fig, outdir, "online_learning_curves", formats)


def curve_grid(
    curve_run_dirs: List[Path], outdir: Path, formats: Sequence[str]
) -> None:
    """Per-cell offline learning curves (J vs gradient step), regime-grouped
    like the money plot; IQM band across seeds (single seed = plain line)."""
    frames = []
    refs = {}
    for rd in curve_run_dirs:
        rd = Path(rd)
        frames.append(pd.read_csv(rd / "learning_curve.csv"))
        m = pd.read_csv(rd / "causal_cells_metrics.csv")
        cell = int(m.cell.iloc[0])
        refs[cell] = (
            iqm(m[m.role == "reference"].J.to_numpy(float)),
            float(m[m.role == "random"].J.iloc[0]),
        )
    df = pd.concat(frames, ignore_index=True)
    cells = [c for _, cs in REGIMES for c in cs if c in set(df.cell)]
    ncols = min(3, max(1, len(cells)))
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    for i, cell in enumerate(cells):
        ax = axes[i // ncols][i % ncols]
        sub = df[df.cell == cell]
        for role, color in (("basic", BASIC_COLOR), ("variant", VARIANT_COLOR)):
            r = sub[sub.role == role]
            if r.empty:
                continue
            piv = r.pivot_table(index="step", columns="seed", values="J")
            x = piv.index.to_numpy(float)
            vals = piv.to_numpy(float)
            center = np.array([iqm(v[~np.isnan(v)]) for v in vals])
            label = f"{role} ({r.algo.iloc[0]})"
            ax.plot(x, center, color=color, label=label, marker="o", ms=3)
            if vals.shape[1] > 1:
                ax.fill_between(
                    x,
                    np.nanquantile(vals, 0.25, axis=1),
                    np.nanquantile(vals, 0.75, axis=1),
                    color=color,
                    alpha=0.2,
                )
        if cell in refs:
            ax.axhline(refs[cell][0], color="0.4", ls="--", lw=1)
            ax.axhline(refs[cell][1], color="0.7", ls=":", lw=1)
        ax.set_title(CELL_NAMES.get(cell, str(cell)), fontsize=10)
        ax.set_xlabel("gradient step")
        ax.set_ylabel("J (true env)")
        ax.legend(fontsize=8)
    for j in range(len(cells), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        "Offline cells: learning curves, grouped by identification regime "
        "(dashed: cell-1 reference; dotted: random floor)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, outdir, "offline_learning_curves", formats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal-cells figures/tables (regenerate from CSVs alone)"
    )
    parser.add_argument("--matrix", default="outputs/matrix_summary.csv")
    parser.add_argument("--cell2", default="outputs/cell2_summary.csv")
    parser.add_argument("--gamma", default="outputs/gamma_calibration.csv")
    parser.add_argument("--horizon", default="outputs/horizon_ablation.csv")
    parser.add_argument(
        "--run-csvs",
        nargs="*",
        default=[],
        help="sweep run causal_cells_metrics.csv paths (for the OPE table)",
    )
    parser.add_argument("--online-bench-runs", nargs="*", default=[])
    parser.add_argument("--online-reference-runs", nargs="*", default=[])
    parser.add_argument("--curve-runs", nargs="*", default=[])
    parser.add_argument("--outdir", default="outputs/causal_cells")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    money_plot(Path(args.matrix), Path(args.cell2), outdir, args.formats)
    tier_panel(Path(args.matrix), outdir, args.formats)
    if args.run_csvs:
        ope_table([Path(p) for p in args.run_csvs], outdir)
    if Path(args.gamma).exists():
        gamma_figure(Path(args.gamma), outdir, args.formats)
    if Path(args.horizon).exists():
        horizon_figure(Path(args.horizon), outdir, args.formats)
    if args.online_bench_runs:
        online_curves(
            [Path(p) for p in args.online_bench_runs],
            [Path(p) for p in args.online_reference_runs],
            outdir,
            args.formats,
        )
    if args.curve_runs:
        curve_grid([Path(p) for p in args.curve_runs], outdir, args.formats)


if __name__ == "__main__":
    main()

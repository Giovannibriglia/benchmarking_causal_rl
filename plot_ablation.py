from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from src.algos import EMPIRICAL_CHECKS
from tqdm import tqdm

# ─────────── config ───────────
# Aggregation across environments for "overall" curves and summary plots:
#   AGG_MODE: "iqm" or "mean"
#   ERROR_MODE: "iqr" (uses IQR/2) or "std"
AGG_MODE = "iqm"  # "iqm" | "mean"
ERROR_MODE = "iqr"  # "iqr" | "std"

# Which algos and metrics to include
ALGOS: List[str] = list(EMPIRICAL_CHECKS.keys())
ALGOS.remove("ppo")
ALGOS.remove("trpo")

METRICS: List[str] = [
    "adv_var",
    "value_mse",
    "v_explained_variance",
    "v_corr",
    "v_spearman",
    "td_mean",
    "adv_mean",
    "v_mi",
    "v_wass",
    "v_kl",
    "v_js",
]

FOLDER_PATH = "runs/gymnasium_ablation_ok"
OUT_DIR = Path(f"{FOLDER_PATH}/plots")


# ─────────── utils ───────────
def percentage_improvement(baseline: np.ndarray, new: np.ndarray) -> np.ndarray:
    if baseline.shape != new.shape:
        raise ValueError("Arrays must have the same shape")
    with np.errstate(divide="ignore", invalid="ignore"):
        improvement = (new - baseline) / np.abs(baseline) * 100.0
        improvement = np.where(baseline == 0, np.nan, improvement)
    return improvement


def iqm_and_iqr(values: np.ndarray) -> Tuple[float, float]:
    """IQM (interquartile mean) and IQR/2 as robust std proxy."""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return np.nan, np.nan
    q25, q75 = np.percentile(vals, [25, 75])
    middle = vals[(vals >= q25) & (vals <= q75)]
    return float(middle.mean()), float((q75 - q25) / 2.0)


def mean_and_std(values: np.ndarray) -> Tuple[float, float]:
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def aggregate(values: np.ndarray, agg_mode: str, err_mode: str) -> Tuple[float, float]:
    """
    Aggregate a 1D array across environments.
    """
    if agg_mode == "iqm":
        center, _ = iqm_and_iqr(values)
    elif agg_mode == "mean":
        center, _ = mean_and_std(values)
    else:
        raise ValueError("agg_mode must be 'iqm' or 'mean'")

    if err_mode == "iqr":
        _, spread = iqm_and_iqr(values)
    elif err_mode == "std":
        _, spread = mean_and_std(values)
    else:
        raise ValueError("err_mode must be 'iqr' or 'std'")

    return center, spread


# ─────────── load helpers ───────────
def list_envs_from_folder(folder: str | Path) -> List[str]:
    folder = Path(folder)
    envs = []
    for p in folder.glob("*_metrics.json"):
        name = p.name.replace("_metrics.json", "")
        envs.append(name)
    envs.sort()
    return envs


def load_metric_arrays(
    folder: str | Path, env: str, algo: str, metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns baseline and causal arrays for one env/algo/metric across checkpoints.
    """
    p = Path(folder) / f"{env}_metrics.json"
    with p.open("r") as f:
        data = json.load(f)
    d = data[env][algo]
    base = np.asarray(d[metric], dtype=float)
    causal = np.asarray(d[f"causal_{metric}"], dtype=float)
    if base.shape != causal.shape:
        raise ValueError(f"Length mismatch for {env}/{algo}/{metric}")
    return base, causal


# ─────────── main aggregation ───────────
def build_delta_tensor(
    folder: str | Path,
    envs: List[str],
    algos: List[str],
    metrics: List[str],
) -> Tuple[int, Dict[str, Dict[str, np.ndarray]]]:
    """
    Returns:
      n_checkpoints,
      deltas[algo][metric] = np.ndarray of shape [n_checkpoints, n_envs]
      where each entry is %Δ (causal vs base) for a given checkpoint/env.
    """
    # Infer n_checkpoints from first env/metric
    first_env = envs[0]
    first_algo = algos[0]
    first_metric = metrics[0]
    base0, causal0 = load_metric_arrays(folder, first_env, first_algo, first_metric)
    n_checkpoints = base0.shape[0]

    # Prepare containers
    deltas: Dict[str, Dict[str, np.ndarray]] = {
        a: {m: np.full((n_checkpoints, len(envs)), np.nan) for m in metrics}
        for a in algos
    }

    # Fill per env
    for e_idx, env in enumerate(envs):
        for algo in algos:
            for m in metrics:
                base, causal = load_metric_arrays(folder, env, algo, m)
                if base.shape[0] != n_checkpoints:
                    raise ValueError(f"Inconsistent checkpoints for {env}/{algo}/{m}")
                delta = percentage_improvement(base, causal)  # [T]
                deltas[algo][m][:, e_idx] = delta

    return n_checkpoints, deltas


# ─────────── plotting ───────────
def make_colors(algos: List[str]) -> Dict[str, tuple]:
    cmap = cm.get_cmap("Set1", len(algos))
    return {algo: cmap(i) for i, algo in enumerate(algos)}


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_per_env(
    out_dir: Path,
    envs: List[str],
    metrics: List[str],
    algos: List[str],
    n_checkpoints: int,
    folder: str | Path,
    colors: Dict[str, tuple],
    agg_mode: str,
):
    """
    Per-environment %Δ curves (no shading).
    File name: {agg}_{env}_{metric}_None.pdf
    """
    steps = np.arange(n_checkpoints)
    for env in tqdm(envs, desc="Plotting per env metrics..."):
        for m in metrics:
            plt.figure(dpi=500, figsize=(5.2, 4.4))
            for algo in algos:
                base, causal = load_metric_arrays(folder, env, algo, m)
                y = percentage_improvement(base, causal)
                plt.plot(steps, y, label=algo, color=colors[algo], linewidth=1.7)

            plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
            plt.title(f"{env} — {m} (%Δ causal vs base)")
            plt.xlabel("Checkpoint")
            plt.ylabel("% Δ")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=9)
            fname = f"{agg_mode}_{env}_{m}_None.pdf"
            savefig(out_dir / fname)


def plot_overall_timeseries(
    out_dir: Path,
    envs: List[str],
    metrics: List[str],
    algos: List[str],
    n_checkpoints: int,
    deltas: Dict[str, Dict[str, np.ndarray]],
    colors: Dict[str, tuple],
    agg_mode: str,
    err_mode: str,
):
    """
    Overall (across envs) time series with chosen aggregation + shaded error.
    File name: {agg}_{all}_{metric}_{error}.pdf
    """
    steps = np.arange(n_checkpoints)
    for m in tqdm(metrics, desc="Plotting overall metrics..."):
        plt.figure(dpi=500, figsize=(5.2, 4.4))
        for algo in algos:
            mat = deltas[algo][m]  # [T, E]
            centers = np.full(n_checkpoints, np.nan)
            spreads = np.full(n_checkpoints, np.nan)
            for t in range(n_checkpoints):
                c, s = aggregate(mat[t, :], agg_mode, err_mode)
                centers[t], spreads[t] = c, s

            plt.plot(steps, centers, label=algo, color=colors[algo], linewidth=1.8)
            plt.fill_between(
                steps,
                centers - spreads,
                centers + spreads,
                alpha=0.2,
                color=colors[algo],
            )

        plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
        plt.title(f"Overall — {m} (%Δ causal vs base)")
        plt.xlabel("Checkpoint")
        ylabel = "% Δ ("
        ylabel += "IQM" if agg_mode == "iqm" else "Mean"
        ylabel += " ± "
        ylabel += "IQR/2" if err_mode == "iqr" else "Std"
        ylabel += ")"
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=9)
        fname = f"{agg_mode}_all_{m}_{err_mode}.pdf"
        savefig(out_dir / fname)


def plot_overall_errorbars(
    out_dir: Path,
    metrics: List[str],
    algos: List[str],
    n_checkpoints: int,
    deltas: Dict[str, Dict[str, np.ndarray]],
    colors: Dict[str, tuple],
    agg_mode: str,
    err_mode: str,
):
    """
    Single summary errorbar per algo & metric:
      y = mean over checkpoints of aggregated centers
      err = mean over checkpoints of aggregated spreads
    Also writes a LaTeX table with metrics as rows and algorithms as columns,
    reporting "mean ± std" (or "IQM ± IQR/2" depending on flags).
    File name (plots): {agg}_all_{metric}_{error}_summary.pdf
    File name (latex): {agg}_all_{error}_summary_table.txt
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows for the LaTeX table (one row per metric)
    latex_rows = []  # each row: [metric_label, cell_algo1, cell_algo2, ...]
    title_center = "IQM" if agg_mode == "iqm" else "Mean"
    title_error = "IQR/2" if err_mode == "iqr" else "Std"

    def _fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return r"--"
        return f"{x:.2f}"

    def _escape_latex(s: str) -> str:
        # Minimal escaping for common LaTeX specials in metric names
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("$", r"\$")
        )

    for m in tqdm(metrics, desc="Plotting error bars per metric..."):
        plt.figure(dpi=500, figsize=(5.2, 4.4))
        xs = np.arange(len(algos))
        centers_mean = []
        spreads_mean = []

        for algo in algos:
            # mat shape: [T, E]  (T checkpoints, E environments)
            mat = deltas[algo][m]
            centers = np.full(n_checkpoints, np.nan)
            spreads = np.full(n_checkpoints, np.nan)
            for t in range(n_checkpoints):
                c, s = aggregate(mat[t, :], agg_mode, err_mode)
                centers[t], spreads[t] = c, s

            centers_mean.append(np.nanmean(centers))
            spreads_mean.append(np.nanmean(spreads))

        # Plot
        for i, algo in enumerate(algos):
            plt.errorbar(
                xs[i],
                centers_mean[i],
                yerr=spreads_mean[i],
                fmt="o",
                capsize=8,
                elinewidth=2,
                markersize=6,
                color=colors[algo],
                label=algo,
            )

        plt.xticks(xs, algos)
        plt.title(f"Overall — {m} ({title_center} ± {title_error} across envs)")
        plt.ylabel("% Δ")
        plt.grid(True, alpha=0.3, axis="y")
        fname = f"{agg_mode}_all_{m}_{err_mode}_summary.pdf"
        savefig(out_dir / fname)

        # Build the LaTeX row for this metric
        row_cells = []
        for c_mean, s_mean in zip(centers_mean, spreads_mean):
            cell = f"{_fmt(c_mean)} $\\pm$ {_fmt(s_mean)}"
            row_cells.append(cell)
        latex_rows.append([_escape_latex(m), *row_cells])

    # ------- Write LaTeX table -------
    colspec = "l|" + "c" * len(algos)
    header = " & ".join(["Metric"] + [_escape_latex(a) for a in algos]) + r" \\"

    caption = (
        f"Overall per-metric summary (%$\\Delta$). "
        f"Entries show {title_center} $\\pm$ {title_error} across environments, "
        f"averaged over checkpoints."
    )
    label = f"tab:overall_{agg_mode}_{err_mode}"

    table_lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        f"\\begin{{tabular}}{{{colspec}}}",  # <-- fixed
        header,
        r"\hline",
    ]
    for row in latex_rows:
        table_lines.append(" & ".join(row) + r" \\")
    table_lines += [
        r"\hline",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
        "",
    ]

    latex_path = out_dir / f"{agg_mode}_all_{err_mode}_summary_table.txt"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))


# ─────────── run ───────────
def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover envs
    envs = list_envs_from_folder(FOLDER_PATH)
    if not envs:
        raise RuntimeError(f"No *_metrics.json found in {FOLDER_PATH}")

    # build delta tensor
    n_checkpoints, deltas = build_delta_tensor(
        folder=FOLDER_PATH,
        envs=envs,
        algos=ALGOS,
        metrics=METRICS,
    )

    colors = make_colors(ALGOS)

    # 1) per-env curves (no shading)
    plot_per_env(
        out_dir=out_dir,
        envs=envs,
        metrics=METRICS,
        algos=ALGOS,
        n_checkpoints=n_checkpoints,
        folder=FOLDER_PATH,
        colors=colors,
        agg_mode=AGG_MODE,
    )

    # 2) overall time series (across envs) with chosen aggregation + shading
    plot_overall_timeseries(
        out_dir=out_dir,
        envs=envs,
        metrics=METRICS,
        algos=ALGOS,
        n_checkpoints=n_checkpoints,
        deltas=deltas,
        colors=colors,
        agg_mode=AGG_MODE,
        err_mode=ERROR_MODE,
    )

    # 3) overall summary errorbar (one point per algo)
    plot_overall_errorbars(
        out_dir=out_dir,
        metrics=METRICS,
        algos=ALGOS,
        n_checkpoints=n_checkpoints,
        deltas=deltas,
        colors=colors,
        agg_mode=AGG_MODE,
        err_mode=ERROR_MODE,
    )


if __name__ == "__main__":
    main()

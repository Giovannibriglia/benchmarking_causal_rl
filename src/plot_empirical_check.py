# -*- coding: utf-8 -*-
# grouped_empirical_checks.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm

from src.algos import EMPIRICAL_CHECKS

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


# ─────────── categorizations ───────────
def _env_group(env_name: str, how_to_group: str = "default") -> str:
    name = env_name.lower()
    if how_to_group == "default":
        classic = ("cartpole", "mountaincar", "acrobot", "pendulum")
        if any(k in name for k in classic):
            return "classic_control"
        box2d = ("lunarlander", "bipedalwalker", "carracing")
        if any(k in name for k in box2d):
            return "box2d"
        toytext = ("frozenlake", "cliffwalking", "taxi", "blackjack")
        if any(k in name for k in toytext):
            return "toy_text"
        mujoco = (
            "ant",
            "halfcheetah",
            "hopper",
            "humanoid",
            "humanoidstandup",
            "invertedpendulum",
            "inverteddoublependulum",
            "pusher",
            "reacher",
            "swimmer",
            "walker2d",
        )
        if any(k in name for k in mujoco) or "mujoco" in name:
            return "mujoco"
        return "other"
    elif how_to_group == "hardness":
        easier = (
            "cartpole",
            "mountaincar-v0",
            "acrobot",
            "frozenlake",
            "cliffwalking",
            "taxi",
            "blackjack",
        )
        if any(k in name for k in easier):
            return "easier"
        medium = ("pendulum", "lunarlandercontinuous-v3", "reacher", "invertedpendulum")
        if any(k in name for k in medium):
            return "medium"
        hard = (
            "bipedalwalker",
            "mountaincarcontinuous",
            "pusher",
            "inverteddoublependulum",
            "ant",
            "halfcheetah",
            "hopper",
            "humanoid",
            "humanoidstandup",
            "pusher",
            "swimmer",
            "walker2d",
            "carracing",
            "lunarlander-v3",
        )
        if any(k in name for k in hard) or "mujoco" in name:
            return "hard"
        raise ValueError(f"{name} not categorized")
    else:
        raise NotImplementedError(f"{how_to_group} not implemented")


# ─────────── utils ───────────
def percentage_improvement(baseline: np.ndarray, new: np.ndarray) -> np.ndarray:
    if baseline.shape != new.shape:
        raise ValueError("Arrays must have the same shape")
    with np.errstate(divide="ignore", invalid="ignore"):
        improvement = (new - baseline) / np.abs(baseline) * 100.0
        improvement = np.where(baseline == 0, np.nan, improvement)
    return improvement


def iqm_and_iqr(values: np.ndarray) -> Tuple[float, float]:
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
    first_env = envs[0]
    first_algo = algos[0]
    first_metric = metrics[0]
    base0, causal0 = load_metric_arrays(folder, first_env, first_algo, first_metric)
    n_checkpoints = base0.shape[0]

    deltas: Dict[str, Dict[str, np.ndarray]] = {
        a: {m: np.full((n_checkpoints, len(envs)), np.nan) for m in metrics}
        for a in algos
    }

    for e_idx, env in enumerate(envs):
        for algo in algos:
            for m in metrics:
                base, causal = load_metric_arrays(folder, env, algo, m)
                if base.shape[0] != n_checkpoints:
                    raise ValueError(f"Inconsistent checkpoints for {env}/{algo}/{m}")
                delta = percentage_improvement(base, causal)  # [T]
                deltas[algo][m][:, e_idx] = delta

    return n_checkpoints, deltas


# ─────────── grouping helpers (NEW) ───────────
def env_indices_by_group(envs: List[str], how_to_group: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, e in enumerate(envs):
        g = _env_group(e, how_to_group)
        groups.setdefault(g, []).append(i)
    return groups


def subset_by_indices(mat: np.ndarray, idxs: List[int]) -> np.ndarray:
    # mat shape [T, E] -> [T, |idxs|]
    if len(idxs) == 0:
        return np.full((mat.shape[0], 0), np.nan)
    return mat[:, idxs]


# ─────────── plotting ───────────
def make_colors(algos: List[str]) -> Dict[str, tuple]:
    cmap = cm.get_cmap("Set1", len(algos))
    return {algo: cmap(i) for i, algo in enumerate(algos)}


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_per_env(
    input_path: Path,
    envs: List[str],
    metrics: List[str],
    algos: List[str],
    n_checkpoints: int,
    folder: str | Path,
    colors: Dict[str, tuple],
    agg_mode: str,
):
    out_dir = input_path / "plots_per_env"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    input_path: Path,
    metrics: List[str],
    algos: List[str],
    n_checkpoints: int,
    deltas: Dict[str, Dict[str, np.ndarray]],
    colors: Dict[str, tuple],
    agg_mode: str,
    err_mode: str,
    scope_label: str,  # NEW: "all" or group name
    env_idxs: List[int] | None,  # NEW: which env columns to use (None => all)
):
    out_dir = input_path / "plots_overall_series"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = np.arange(n_checkpoints)
    for m in tqdm(metrics, desc=f"Plotting overall metrics ({scope_label})..."):
        plt.figure(dpi=500, figsize=(5.2, 4.4))
        for algo in algos:
            mat_full = deltas[algo][m]  # [T, E]
            mat = (
                mat_full if env_idxs is None else subset_by_indices(mat_full, env_idxs)
            )

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
        plt.title(f"{scope_label} — {m} (%Δ causal vs base)")
        ylabel = "% Δ (" + ("IQM" if agg_mode == "iqm" else "Mean") + " ± "
        ylabel += "IQR/2" if err_mode == "iqr" else "Std"
        ylabel += ")"
        plt.xlabel("Checkpoint")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=9)
        fname = f"{agg_mode}_{scope_label}_{m}_{err_mode}.pdf"
        savefig(out_dir / fname)


def plot_overall_errorbars(
    input_path: Path,
    metrics: List[str],
    algos: List[str],
    n_checkpoints: int,
    deltas: Dict[str, Dict[str, np.ndarray]],
    colors: Dict[str, tuple],
    agg_mode: str,
    err_mode: str,
    scope_label: str,  # NEW
    env_idxs: List[int] | None,  # NEW
):
    out_dir = input_path / "plots_overall_errorbars"
    out_dir.mkdir(parents=True, exist_ok=True)

    latex_rows = []
    title_center = "IQM" if agg_mode == "iqm" else "Mean"
    title_error = "IQR/2" if err_mode == "iqr" else "Std"

    def _fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return r"--"
        return f"{x:.2f}"

    def _escape_latex(s: str) -> str:
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

    for m in tqdm(metrics, desc=f"Error bars per metric ({scope_label})..."):
        plt.figure(dpi=500, figsize=(5.2, 4.4))
        xs = np.arange(len(algos))
        centers_mean, spreads_mean = [], []

        for algo in algos:
            mat_full = deltas[algo][m]  # [T, E]
            mat = (
                mat_full if env_idxs is None else subset_by_indices(mat_full, env_idxs)
            )
            centers = np.full(n_checkpoints, np.nan)
            spreads = np.full(n_checkpoints, np.nan)
            for t in range(n_checkpoints):
                c, s = aggregate(mat[t, :], agg_mode, err_mode)
                centers[t], spreads[t] = c, s

            centers_mean.append(np.nanmean(centers))
            spreads_mean.append(np.nanmean(spreads))

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
        plt.title(f"{scope_label} — {m} ({title_center} ± {title_error} across envs)")
        plt.ylabel("% Δ")
        plt.grid(True, alpha=0.3, axis="y")
        fname = f"{agg_mode}_{scope_label}_{m}_{err_mode}_summary.pdf"
        savefig(out_dir / fname)

        row_cells = [
            f"{_fmt(c)} $\\pm$ {_fmt(s)}" for c, s in zip(centers_mean, spreads_mean)
        ]
        latex_rows.append([_escape_latex(m), *row_cells])

    # LaTeX table per scope_label
    colspec = "l|" + "c" * len(algos)
    header = " & ".join(["Metric"] + [a.replace("_", r"\_") for a in algos]) + r" \\"
    caption = (
        f"{scope_label}: per-metric summary (%$\\Delta$). "
        f"Entries show {title_center} $\\pm$ {title_error} across environments, "
        f"averaged over checkpoints."
    )
    label = f"tab:{scope_label}_{agg_mode}_{err_mode}"

    table_lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        f"\\begin{{tabular}}{{{colspec}}}",
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
    latex_path = out_dir / f"{agg_mode}_{scope_label}_{err_mode}_summary_table.txt"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))


# ─────────── run ───────────
def plot_empirical_check(
    input_path: Path,
    algos: List[str],
    agg_mode: str,
    err_mode: str,
    how_to_group: str = "default",  # NEW
) -> None:

    # discover envs
    envs = list_envs_from_folder(input_path)
    if not envs:
        raise RuntimeError(f"No *_metrics.json found in {input_path}")

    # build delta tensor (across all envs)
    n_checkpoints, deltas = build_delta_tensor(
        folder=input_path, envs=envs, algos=algos, metrics=METRICS
    )

    colors = make_colors(algos)

    # 1) per-env curves (unchanged)
    plot_per_env(
        input_path=input_path,
        envs=envs,
        metrics=METRICS,
        algos=algos,
        n_checkpoints=n_checkpoints,
        folder=input_path,
        colors=colors,
        agg_mode=agg_mode,
    )

    # 2) overall "all envs"
    plot_overall_timeseries(
        input_path=input_path,
        metrics=METRICS,
        algos=algos,
        n_checkpoints=n_checkpoints,
        deltas=deltas,
        colors=colors,
        agg_mode=agg_mode,
        err_mode=err_mode,
        scope_label="all",
        env_idxs=None,
    )
    plot_overall_errorbars(
        input_path=input_path,
        metrics=METRICS,
        algos=algos,
        n_checkpoints=n_checkpoints,
        deltas=deltas,
        colors=colors,
        agg_mode=agg_mode,
        err_mode=err_mode,
        scope_label="all",
        env_idxs=None,
    )

    # 3) per-group overall plots + tables (NEW)
    groups = env_indices_by_group(envs, how_to_group)
    for g, idxs in groups.items():
        if len(idxs) == 0:
            continue
        # Timeseries per group
        plot_overall_timeseries(
            input_path=input_path,
            metrics=METRICS,
            algos=algos,
            n_checkpoints=n_checkpoints,
            deltas=deltas,
            colors=colors,
            agg_mode=agg_mode,
            err_mode=err_mode,
            scope_label=g,
            env_idxs=idxs,
        )
        # Errorbars + LaTeX table per group
        plot_overall_errorbars(
            input_path=input_path,
            metrics=METRICS,
            algos=algos,
            n_checkpoints=n_checkpoints,
            deltas=deltas,
            colors=colors,
            agg_mode=agg_mode,
            err_mode=err_mode,
            scope_label=g,
            env_idxs=idxs,
        )


if __name__ == "__main__":
    in_path = Path("../runs/gymnasium_ablation_ok")

    agg_mode = "iqm"  # "iqm" | "mean"
    err_mode = "iqr"  # "iqr" | "std"
    how_to_group = "hardness"  # "default" | "hardness"

    empirical_checks: List[str] = list(EMPIRICAL_CHECKS.keys())
    # (keep removing ppo/trpo as you had)
    empirical_checks.remove("ppo")
    empirical_checks.remove("trpo")

    plot_empirical_check(in_path, empirical_checks, agg_mode, err_mode, how_to_group)

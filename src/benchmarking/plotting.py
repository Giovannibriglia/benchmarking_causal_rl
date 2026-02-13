from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Global color map for consistent algorithm colors across plots
COLOR_MAP: Dict[str, tuple] = {}


def get_algo_color(algo: str, palette: str = "Set1"):
    """Return consistent color for each algo using a matplotlib palette (default Set1)."""
    import matplotlib

    if algo not in COLOR_MAP:
        idx = len(COLOR_MAP) % 10  # cycle palette if more than 10 algos
        COLOR_MAP[algo] = matplotlib.colormaps[palette](idx)
    return COLOR_MAP[algo]


NON_METRIC_COLS = {
    "algorithm",
    "environment",
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


def load_run(run_name: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    run_dir = Path("runs") / run_name
    cfg_path = run_dir / "config.yaml"
    train_path = run_dir / "train_metrics.csv"
    eval_path = run_dir / "eval_metrics.csv"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    if not train_path.exists() and not eval_path.exists():
        raise FileNotFoundError(
            "No metrics CSVs found (train_metrics.csv or eval_metrics.csv)"
        )

    with cfg_path.open("r") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    return config, train_df, eval_df


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


def compute_aggregates(
    df: pd.DataFrame,
    metric_name: str,
    mean_col: str,
    std_col: Optional[str],
    x_axis: str,
    aggregation: str,
    rollout_len: int,
) -> pd.DataFrame:
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
            ax.plot(x, y, label=algo, color=color)
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


def _escape_latex(text: str) -> str:
    return text.replace("_", "\\_")


def make_latex_table(
    df: pd.DataFrame,
    metric: str,
    split: str,
    aggregation: str,
    outdir: Path,
    precision: int = 3,
):
    if df.empty:
        print(f"[warn] no data for table {metric} ({split})")
        return
    envs = sorted(df["environment"].unique())
    algos = sorted(df["algorithm"].unique())
    rows = []
    fmt = f"{{:.{precision}f}}"

    for env in envs:
        row = [_escape_latex(env)]
        env_df = df[df["environment"] == env]
        for algo in algos:
            cell_df = env_df[env_df["algorithm"] == algo]
            if cell_df.empty:
                row.append("--")
                continue
            center = cell_df.iloc[0]["center"]
            spread = cell_df.iloc[0]["spread"]
            row.append(f"${fmt.format(center)} \\pm {fmt.format(spread)}$")
        rows.append(row)

    # overall row: mean of centers/spreads across envs where present
    overall_row = ["Overall"]
    for algo in algos:
        vals = []
        spreads = []
        for env in envs:
            cell = df[(df["environment"] == env) & (df["algorithm"] == algo)]
            if not cell.empty:
                vals.append(cell.iloc[0]["center"])
                spreads.append(cell.iloc[0]["spread"])
        if len(vals) == 0:
            overall_row.append("--")
        else:
            overall_row.append(
                f"${fmt.format(np.mean(vals))} \\pm {fmt.format(np.mean(spreads))}$"
            )
    rows.append(overall_row)

    # Bold headers; escape underscores
    header = ["\\textbf{Environment}"] + [
        f"\\textbf{{{_escape_latex(a)}}}" for a in algos
    ]
    col_spec = "l" + "|c" * len(algos)
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{metric.replace('_', '\\_')} ({split})}}",
        f"\\label{{tab:{split}_{metric}}}",
        "%\\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        " \\\\ ".join(header) + " \\\\",
        "\\hline\\hline",
    ]
    for r in rows:
        lines.append(" \\\\ ".join(r) + " \\\\")
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
        make_latex_table(agg_df, metric_name, split, aggregation, outdir)


def run_plotting(
    run_name: str,
    split: str,
    x_axis: str,
    aggregation: str,
    outdir: Path,
    formats: List[str],
):
    config, train_df, eval_df = load_run(run_name)
    rollout_len = _get_rollout_len(config)
    outdir = Path(outdir) / run_name
    x_label = "Frames" if x_axis == "frames" else "Episodes"

    if split in ("train", "both") and not train_df.empty:
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
            # overall figure
            overall = (
                agg.groupby(["algorithm", "x"])
                .agg({"center": "mean", "spread": "mean"})
                .reset_index()
            )
            overall["environment"] = "overall"
            plot_metric(
                overall, metric_name, "train", outdir, formats, x_label, overall=True
            )
        build_tables(train_df, train_metrics, "train", aggregation, outdir)

    if split in ("eval", "both") and not eval_df.empty:
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
            overall = (
                agg.groupby(["algorithm", "x"])
                .agg({"center": "mean", "spread": "mean"})
                .reset_index()
            )
            overall["environment"] = "overall"
            plot_metric(
                overall, metric_name, "eval", outdir, formats, x_label, overall=True
            )
        build_tables(eval_df, eval_metrics, "eval", aggregation, outdir)


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmarking runs and export tables"
    )
    parser.add_argument("--run", required=True, help="Run folder name under runs/")
    parser.add_argument(
        "--split",
        choices=["train", "eval", "both"],
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
    args = parser.parse_args()

    run_plotting(
        run_name=args.run,
        split=args.split,
        x_axis=args.x_axis,
        aggregation=args.aggregation,
        outdir=Path(args.outdir),
        formats=args.formats,
    )


if __name__ == "__main__":
    main()

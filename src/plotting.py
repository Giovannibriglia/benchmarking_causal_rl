from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

# ────────────────────────── helpers ───────────────────────────────────────────

EVAL_RE = re.compile(r"^evaluation_(return|length)_(\d+)$")

# Consistent colormap for algorithms
COLOR_MAP: dict[str, any] = {}


def get_algo_color(algo: str, palette: str = "Set1"):
    """Return consistent color for each algo using a matplotlib palette (default Set1)."""
    if algo not in COLOR_MAP:
        idx = len(COLOR_MAP) % 10  # cycle palette if more than 10 algos
        COLOR_MAP[algo] = matplotlib.colormaps[palette](idx)
    return COLOR_MAP[algo]


def _stack_runs(mdict: dict[str, list[float]], prefix: str) -> np.ndarray:
    """evaluation_<prefix>_<id>  ➜  (n_seeds, n_chkpts) array (nan-padded)."""
    keys = sorted(k for k in mdict if k.startswith(f"evaluation_{prefix}"))
    if not keys:
        return np.empty((0, 0))
    runs = [mdict[k] for k in keys]
    mlen = max(map(len, runs))
    padded = [np.pad(r, (0, mlen - len(r)), constant_values=np.nan) for r in runs]
    return np.vstack(padded)


def _mean_iqr_std(
    arr: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    q25, q75 = np.nanpercentile(arr, [25, 75], axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, (q25, q75), std


def _combine_losses(
    mdict: dict[str, list[float]], keys: tuple[str, ...]
) -> np.ndarray | None:
    """Element-wise sum of any existing loss curves in *keys* (nan-pad ragged)."""
    parts = [np.asarray(mdict[k], float) for k in keys if k in mdict]
    if not parts:
        return None
    mlen = max(map(len, parts))
    padded = [np.pad(p, (0, mlen - len(p)), constant_values=np.nan) for p in parts]
    return np.nansum(padded, axis=0)


def _iqm_iqr_of_series(series: pd.Series) -> tuple[float, float]:
    vals = np.asarray(series.dropna().values, dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    q25, q75 = np.percentile(vals, [25, 75])
    mid = vals[(vals >= q25) & (vals <= q75)]
    if mid.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(mid)), float(np.std(mid))


def _build_pairs_by_group_matrix(
    impr_per_env: pd.DataFrame,
) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Rows = algo pairs ("variant vs baseline"), Cols = group_env,
    Cell = IQM(improvement_pct) ± IQR-std across envs in that group.
    """
    if impr_per_env.empty:
        return [], [], []

    pairs = sorted(impr_per_env["pair"].unique().tolist())
    groups = sorted(impr_per_env["group_env"].unique().tolist())

    matrix = []
    for p in pairs:
        row = []
        any_val = False
        for g in groups:
            s = impr_per_env.loc[
                (impr_per_env["pair"] == p) & (impr_per_env["group_env"] == g),
                "improvement_pct",
            ]
            iqm, iqrstd = _iqm_iqr_of_series(s)
            if np.isnan(iqm):
                row.append("--")
            else:
                row.append(f"{iqm:.2f} $\\pm$ {iqrstd:.2f}")
                any_val = True
        if any_val:
            matrix.append(row)
        else:
            # if a pair never appears in any group, skip it entirely
            pairs.remove(p)
    return pairs, groups, matrix


def _build_pairs_overall_matrix(
    impr_per_env: pd.DataFrame,
) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Rows = algo pairs, single column 'ALL',
    Cell = IQM(improvement_pct) ± IQR-std across ALL envs.
    """
    if impr_per_env.empty:
        return [], [], []
    pairs = sorted(impr_per_env["pair"].unique().tolist())
    row_names, matrix = [], []
    for p in pairs:
        s = impr_per_env.loc[impr_per_env["pair"] == p, "improvement_pct"]
        iqm, iqrstd = _iqm_iqr_of_series(s)
        if not np.isnan(iqm):
            row_names.append(p)
            matrix.append([f"{iqm:.2f} $\\pm$ {iqrstd:.2f}"])
    return row_names, ["ALL"], matrix


def _summarise(y: np.ndarray) -> tuple[float, float, float, float]:
    """mean, std, iqr25, iqr75 ignoring nans."""
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    q25, q75 = map(float, np.nanpercentile(y, [25, 75]))
    return mean, std, q25, q75


def _iqm_and_iqr_std(y: np.ndarray) -> tuple[float, float]:
    """Interquartile mean and std over flattened values (ignore nans)."""
    yy = np.asarray(y, float).ravel()
    yy = yy[~np.isnan(yy)]
    if yy.size == 0:
        return float("nan"), float("nan")
    q25, q75 = np.percentile(yy, [25, 75])
    mid = yy[(yy >= q25) & (yy <= q75)]
    if mid.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(mid)), float(np.std(mid))


def _iqm_1d(arr_like) -> float:
    a = np.asarray(arr_like, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    q25, q75 = np.percentile(a, [25, 75])
    mid = a[(a >= q25) & (a <= q75)]
    if mid.size == 0:
        return float("nan")
    return float(np.mean(mid))


# Gymnasium environment grouping
def _env_group(env_name: str, how_to_group: str = "default") -> str:
    name = env_name.lower()
    if how_to_group == "default":
        # classic control
        classic = ("cartpole", "mountaincar", "acrobot", "pendulum")
        if any(k in name for k in classic):
            return "classic_control"
        # box2d
        box2d = ("lunarlander", "bipedalwalker", "carracing")
        if any(k in name for k in box2d):
            return "box2d"
        # toy text
        toytext = ("frozenlake", "cliffwalking", "taxi", "blackjack")
        if any(k in name for k in toytext):
            return "toy_text"
        # mujoco
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
        # classic control
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


# ─────────────────── causal variant improvement helpers ──────────────────────

# Auto-baseline detection (suffix stripping)
_SUFFIXES = (
    "_cc",
    "-cc",
)


def _latex_table_from_matrix(
    row_header: str,
    row_names: list[str],
    col_names: list[str],
    matrix_vals: list[list[str]],
    caption: str,
    label: str,
) -> str:
    """
    Build a LaTeX tabular with first column = row_header, remaining columns = algorithms.
    matrix_vals entries are already formatted strings (e.g., '12.3 ± 4.5').
    """
    colspec = "l" + "|c" * len(col_names)
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    header = [row_header] + col_names
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\\hline")
    for rname, row in zip(row_names, matrix_vals):
        lines.append(" & ".join([rname] + row) + r" \\")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _format_delta_pct(
    iqm_var: float, iqm_base: float, iqr_var: float, iqr_base: float, eps: float = 1e-8
) -> tuple[float, float]:
    """
    Return (delta_iqm_pct, delta_iqr_pct) where:
      delta_iqm_pct = 100 * (iqm_var - iqm_base) / max(eps, |iqm_base|)
      delta_iqr_pct ≈ 100 * sqrt(iqr_var^2 + iqr_base^2) / max(eps, |iqm_base|)
    """
    denom = max(eps, abs(iqm_base))
    delta = 100.0 * (iqm_var - iqm_base) / denom
    # simple independent error propagation
    delta_err = 100.0 * (float(np.sqrt(iqr_var**2 + iqr_base**2)) / denom)
    return float(delta), float(delta_err)


def _build_envwise_delta_matrix(
    df_wide: pd.DataFrame,
) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Build (row_names=envs, col_names=variant algos, matrix_vals=formatted 'x ± y').

    Uses iqm_evaluation_return and iqr_std_evaluation_return.
    Only compares variant with its proper baseline determined by _base_algo_name.
    """
    if not {
        "env",
        "algo",
        "iqm_evaluation_return",
        "iqr_std_evaluation_return",
    }.issubset(df_wide.columns):
        return [], [], []

    # work per env
    envs = sorted(df_wide["env"].unique().tolist())
    # discover all variant algos that have their baseline in at least one env
    variant_set = set()
    for env in envs:
        sub = df_wide[df_wide["env"] == env]
        algos = set(sub["algo"].astype(str))
        for a in algos:
            base = _base_algo_name(a)
            if base != a and base in algos:
                variant_set.add(a)
    col_names = sorted(variant_set)

    row_names = []
    matrix = []
    for env in envs:
        if env == "MEAN":
            continue
        sub = df_wide[df_wide["env"] == env]
        amap_iqm = dict(
            zip(sub["algo"].astype(str), sub["iqm_evaluation_return"].astype(float))
        )
        amap_iqr = dict(
            zip(sub["algo"].astype(str), sub["iqr_std_evaluation_return"].astype(float))
        )

        row_vals = []
        have_any = False
        for var in col_names:
            base = _base_algo_name(var)
            if var in amap_iqm and base in amap_iqm:
                d, e = _format_delta_pct(
                    amap_iqm[var], amap_iqm[base], amap_iqr[var], amap_iqr[base]
                )
                row_vals.append(f"{d:.2f} $\\pm$ {e:.2f}")
                have_any = True
            else:
                row_vals.append("--")
        if have_any:
            row_names.append(env)
            matrix.append(row_vals)
    return row_names, col_names, matrix


def _build_group_and_overall_delta_matrix(
    impr_per_env: pd.DataFrame,
    agg_type: str = "mean",  # "iqm" or "mean"
) -> tuple[
    tuple[list[str], list[str], list[list[str]]],
    tuple[list[str], list[str], list[list[str]]],
]:
    """
    From impr_per_env (columns: env, group_env, variant_algo, baseline_algo, pair, improvement_pct),
    compute:
      • group-wise: row = group_env, col = variant algo
      • overall: single row 'ALL', same columns
    Aggregation depends on agg_type:
      - "iqm": IQM(improvement_pct) ± IQR-std
      - "mean": mean(improvement_pct) ± std
    """
    if impr_per_env.empty:
        return ([], [], []), ([], [], [])

    col_names = sorted(impr_per_env["variant_algo"].unique().tolist())

    # ---- helpers ----
    def _iqm_iqr_of_series(s: pd.Series) -> tuple[float, float]:
        vals = np.asarray(s.dropna().values, dtype=float)
        if vals.size == 0:
            return float("nan"), float("nan")
        q25, q75 = np.percentile(vals, [25, 75])
        mid = vals[(vals >= q25) & (vals <= q75)]
        if mid.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(mid)), float(np.std(mid))

    def _mean_std_of_series(s: pd.Series) -> tuple[float, float]:
        vals = np.asarray(s.dropna().values, dtype=float)
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    agg_func = _iqm_iqr_of_series if agg_type == "iqm" else _mean_std_of_series

    # ---- group-wise ----
    row_names_g = []
    matrix_g = []
    for grp, sub in sorted(impr_per_env.groupby("group_env"), key=lambda x: x[0]):
        row = []
        have_any = False
        for var in col_names:
            series = sub.loc[sub["variant_algo"] == var, "improvement_pct"]
            mean_val, spread_val = agg_func(series)
            if np.isnan(mean_val):
                row.append("--")
            else:
                row.append(f"{mean_val:.2f} $\\pm$ {spread_val:.2f}")
                have_any = True
        if have_any:
            row_names_g.append(grp)
            matrix_g.append(row)

    # ---- overall ----
    row_names_o = []
    matrix_o = []
    sub_all = impr_per_env
    row = []
    have_any = False
    for var in col_names:
        series = sub_all.loc[sub_all["variant_algo"] == var, "improvement_pct"]
        mean_val, spread_val = agg_func(series)
        if np.isnan(mean_val):
            row.append("--")
        else:
            row.append(f"{mean_val:.2f} $\\pm$ {spread_val:.2f}")
            have_any = True
    if have_any:
        row_names_o.append("ALL")
        matrix_o.append(row)

    return (row_names_g, col_names, matrix_g), (row_names_o, col_names, matrix_o)


# ───────────────── NEW: Error-bar plots instead of bar plots ─────────────────


def _plot_env_improvement_errorbars(
    df_pairs_env: pd.DataFrame, out_path: Path, env: str
):
    """Per-env error-bar plot: x = variants, y = improvement_pct (points), no aggregation."""
    if df_pairs_env.empty:
        return
    # scatter points per pair (no yerr at per-env since single value per (env, pair))
    pairs = df_pairs_env["pair"].tolist()
    x = np.arange(len(pairs))
    y = df_pairs_env["improvement_pct"].astype(float).to_numpy()

    plt.figure(figsize=(max(6, 0.6 * len(pairs) + 2), 3.5), dpi=500)
    for i, p in enumerate(pairs):
        left = p.split(" vs ")[0]
        c = get_algo_color(left)
        plt.errorbar(
            x[i], y[i], yerr=None, fmt="o", capsize=5, lw=2, color=c, alpha=0.9
        )
    plt.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    plt.xticks(x, pairs, rotation=25, ha="right")
    plt.ylabel("% improvement (IQM return)")
    plt.title(f"{env}: improvement (points)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_group_or_overall_improvement_errorbars(
    df_pairs: pd.DataFrame, out_path: Path, title: str, agg: str = "iqm"
):
    """
    Error-bar plots with aggregation across envs:
      y = mean or IQM of improvement_pct per pair,
      err = std or IQR-std across envs for that pair.
    """
    if df_pairs.empty:
        return

    def _agg_vals(series: pd.Series) -> tuple[float, float]:
        vals = np.asarray(series.dropna().values, dtype=float)
        if vals.size == 0:
            return np.nan, np.nan
        if agg == "mean":
            return float(np.mean(vals)), float(np.std(vals))
        else:
            q25, q75 = np.percentile(vals, [25, 75])
            mid = vals[(vals >= q25) & (vals <= q75)]
            if mid.size == 0:
                return np.nan, np.nan
            return float(np.mean(mid)), float(np.std(mid))

    stats = (
        df_pairs.groupby("pair")["improvement_pct"]
        .apply(lambda s: pd.Series(_agg_vals(s)))
        .reset_index()
    )
    # Ensure stable column names across pandas versions
    stats = stats.rename(columns={0: "center", 1: "spread"})

    if "center" not in stats or "spread" not in stats:
        # nothing to plot safely
        return

    # Sort for stable plotting
    stats = stats.sort_values("pair")

    pairs = stats["pair"].tolist()
    centers = stats["center"].to_numpy(dtype=float)
    spreads = stats["spread"].to_numpy(dtype=float)

    x = np.arange(len(pairs))
    plt.figure(figsize=(max(7, 0.6 * len(pairs) + 3), 4), dpi=500)
    for i, p in enumerate(pairs):
        left = p.split(" vs ")[0]
        c = get_algo_color(left)
        plt.errorbar(
            x[i],
            centers[i],
            yerr=spreads[i],
            fmt="o",
            capsize=6,
            lw=2,
            color=c,
            alpha=0.9,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    plt.xticks(x, pairs, rotation=25, ha="right")
    ylabel = (
        ("mean" if agg == "mean" else "IQM")
        + " % improvement (IQM return) ± "
        + ("std" if agg == "mean" else "IQR-std")
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _base_algo_name(algo: str) -> str:
    """Strip known causal suffixes to get the baseline algo name."""
    s = algo
    for suf in _SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _parse_explicit_pairs(
    pairs_arg: Optional[str], pairs_file: Optional[str]
) -> List[Tuple[str, str]]:
    """
    Parse explicit pairs of the form "variant:baseline".
    Examples: "a2c_cc:a2c,ppo_cc:ppo,a2c_cc:trpo"
              file with one 'variant:baseline' per line (comments with '#').
    Returns list of (variant, baseline).
    """
    out: List[Tuple[str, str]] = []
    if pairs_arg:
        for item in pairs_arg.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                continue
            v, b = item.split(":", 1)
            v, b = v.strip(), b.strip()
            if v and b:
                out.append((v, b))
    if pairs_file:
        pf = Path(pairs_file)
        if pf.exists():
            for line in pf.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                v, b = line.split(":", 1)
                v, b = v.strip(), b.strip()
                if v and b:
                    out.append((v, b))
    # de-dup
    return sorted(set(out))


def _find_pairs_for_env(
    env_rows: pd.DataFrame, explicit_pairs: List[Tuple[str, str]] | None = None
) -> list[tuple[str, str]]:
    """
    For a single env (rows have 'algo'), return list of (variant, baseline).
    Includes:
      • auto-detected pairs via suffix stripping, IF baseline exists in env
      • any explicit (variant, baseline) where both algos exist in env
    """
    algos = sorted(set(env_rows["algo"].astype(str).tolist()))
    algo_set = set(algos)
    pairs = []

    # auto-detected
    for a in algos:
        base = _base_algo_name(a)
        if base != a and base in algo_set:
            pairs.append((a, base))

    # explicit
    if explicit_pairs:
        for v, b in explicit_pairs:
            if v in algo_set and b in algo_set:
                pairs.append((v, b))

    return sorted(set(pairs))


def _compute_improvement_table(
    df_env_only: pd.DataFrame, explicit_pairs: List[Tuple[str, str]] | None = None
) -> pd.DataFrame:
    """
    1) Compute % improvement per (env, pair) using IQM(evaluation_return).
       improvement_pct = 100 * (IQM_variant - IQM_baseline) / max(eps, |IQM_baseline|)
    """
    needed = {"env", "group_env", "algo", "iqm_evaluation_return"}
    if not needed.issubset(df_env_only.columns):
        return pd.DataFrame(
            columns=[
                "env",
                "group_env",
                "variant_algo",
                "baseline_algo",
                "pair",
                "improvement_pct",
            ]
        )

    out_rows = []
    eps = 1e-8
    for env, sub in df_env_only.groupby("env"):
        group_env = sub["group_env"].iloc[0] if not sub.empty else "other"
        pairs = _find_pairs_for_env(sub, explicit_pairs=explicit_pairs)
        if not pairs:
            continue

        iqm_map = dict(
            zip(sub["algo"].astype(str), sub["iqm_evaluation_return"].astype(float))
        )

        for variant, baseline in pairs:
            v = iqm_map.get(variant, np.nan)
            b = iqm_map.get(baseline, np.nan)
            if np.isnan(v) or np.isnan(b):
                continue
            denom = max(eps, abs(b))
            impr = 100.0 * (v - b) / denom
            out_rows.append(
                dict(
                    env=env,
                    group_env=group_env,
                    variant_algo=variant,
                    baseline_algo=baseline,
                    pair=f"{variant} vs {baseline}",
                    improvement_pct=float(impr),
                )
            )
    return pd.DataFrame(out_rows)


# ────────────────────────── plotting helpers ─────────────────────────────────


def _plot_mean_algo_errorbars_from_envrows(
    df_env_rows: pd.DataFrame, out_path: Path, title: str
):
    """
    Your original: env-mean of IQM(evaluation_return) with std error bars (raw).
    """
    needed = {"algo", "iqm_evaluation_return"}
    if not needed.issubset(df_env_rows.columns) or df_env_rows.empty:
        return

    stats = (
        df_env_rows.groupby("algo")["iqm_evaluation_return"]
        .agg(avg_over_env="mean", std_over_env="std")
        .reset_index()
    )

    if stats.empty:
        return

    algos = stats["algo"].tolist()
    vals = stats["avg_over_env"].to_numpy(dtype=float)
    errs = stats["std_over_env"].to_numpy(dtype=float)

    x = np.arange(len(algos))
    plt.figure(figsize=(5, 3), dpi=500)
    for i, algo in enumerate(algos):
        c = get_algo_color(algo)
        plt.errorbar(
            x[i],
            vals[i],
            yerr=errs[i],
            fmt="o",
            capsize=6,
            linewidth=2,
            color=c,
            alpha=0.8,
        )
    plt.xticks(x, algos, rotation=20)
    plt.ylabel("IQM evaluation_return (avg over envs)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_env_improvement_bars(df_pairs_env: pd.DataFrame, out_path: Path, env: str):
    """Per-env bar plot of improvement for all detected pairs."""
    if df_pairs_env.empty:
        return
    pairs = df_pairs_env["pair"].tolist()
    vals = df_pairs_env["improvement_pct"].to_numpy(dtype=float)

    x = np.arange(len(pairs))
    plt.figure(figsize=(max(6, 0.6 * len(pairs) + 2), 3.5), dpi=500)
    bars = plt.bar(x, vals)
    for i, p in enumerate(pairs):
        left = p.split(" vs ")[0]
        c = get_algo_color(left)
        bars[i].set_color(c)
        bars[i].set_alpha(0.9)
    plt.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    plt.xticks(x, pairs, rotation=25, ha="right")
    plt.ylabel("% improvement (IQM return)")
    plt.title(f"{env}: improvement")
    plt.grid(True, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_group_or_overall_improvement_bars(
    df_pairs: pd.DataFrame, out_path: Path, title: str, agg: str = "mean"
):
    """
    Bars showing AGGREGATED % improvement across envs (equal weight per env).
    agg in {"mean","iqm"}; default "iqm".
    """
    if df_pairs.empty:
        return
    if agg == "mean":
        stats = (
            df_pairs.groupby("pair")["improvement_pct"].mean().reset_index(name="value")
        )
        ylabel = "mean % improvement (IQM return)"
    else:
        stats = (
            df_pairs.groupby("pair")["improvement_pct"]
            .apply(_iqm_1d)
            .reset_index(name="value")
        )
        ylabel = "IQM % improvement (IQM return)"

    if stats.empty:
        return

    pairs = stats["pair"].tolist()
    vals = stats["value"].to_numpy(dtype=float)

    x = np.arange(len(pairs))
    plt.figure(figsize=(max(7, 0.6 * len(pairs) + 3), 4), dpi=500)
    bars = plt.bar(x, vals)
    for i, p in enumerate(pairs):
        left = p.split(" vs ")[0]
        c = get_algo_color(left)
        bars[i].set_color(c)
        bars[i].set_alpha(0.9)
    plt.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    plt.xticks(x, pairs, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ────────────────────────── main ──────────────────────────────────────────────


def plot_and_save_results(
    results_dir: str | Path,
    n_episodes: int,
    pairs_arg: Optional[str] = None,
    pairs_file: Optional[str] = None,
    agg_type: str = "iqm",
    how_to_group: str = "hardness",
):
    """
    Original pipeline + env-wise % improvement, group IQM, and overall IQM.
    NEW:
      • Explicit pairs via `pairs_arg` (comma-separated "variant:baseline")
        and/or `pairs_file` (one "variant:baseline" per line). Applied
        anywhere both algos exist in an environment.
    """
    fontsize = 25

    results_dir = Path(results_dir)
    plot_dir = results_dir / "plots"
    tables_dir = results_dir / "tables"
    (tables_dir / "plots").mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # parse explicit pairs once
    explicit_pairs = _parse_explicit_pairs(pairs_arg, pairs_file)

    # ────── load all *_metrics.json ───────────────────────────────────────────
    data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for jf in results_dir.glob("*_metrics.json"):
        env = jf.stem.replace("_metrics", "")
        data[env] = json.loads(jf.read_text()).get(env, {})

    summary_rows: list[dict[str, str | float]] = []
    robust_rows: list[dict[str, str | float]] = (
        []
    )  # env, group_env, algo, metric, iqm, iqr_std

    # ────── iterate envs ──────────────────────────────────────────────────────
    pbar = tqdm(data.items())
    for env, algo_dict in pbar:
        pbar.set_description(f"Plotting: {env}...")
        group_env = _env_group(env, how_to_group)

        # ─── evaluation_return & evaluation_length  (stats AND plots) ─────────
        for prefix in ("return", "length"):
            stats: dict[
                str, tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]
            ] = {}

            for algo, mdict in algo_dict.items():
                stacked = _stack_runs(mdict, prefix)  # (seeds, chkpts)
                if stacked.size == 0:
                    continue

                mean_curve, (lq, uq), sd_curve = _mean_iqr_std(stacked)
                stats[algo] = (mean_curve, (lq, uq), sd_curve)

                # Overall summary stats on all elems (raw)
                m, s, q25, q75 = _summarise(stacked)
                iqm, iqr_std = _iqm_and_iqr_std(stacked)
                summary_rows.append(
                    dict(
                        env=env,
                        algo=algo,
                        metric=f"evaluation_{prefix}",
                        mean=m,
                        std=s,
                        iqr25=q25,
                        iqr75=q75,
                    )
                )
                robust_rows.append(
                    dict(
                        env=env,
                        group_env=group_env,
                        algo=algo,
                        metric=f"evaluation_{prefix}",
                        iqm=iqm,
                        iqr_std=iqr_std,
                    )
                )

            # raw plots (mean ± std)
            if stats:
                x = np.linspace(0, n_episodes, next(iter(stats.values()))[0].size)
                plt.figure(figsize=(5, 3), dpi=500)
                for algo, (m, (lq, uq), sd) in stats.items():
                    c = get_algo_color(algo)
                    plt.plot(x, m, label=algo, linewidth=3, color=c)
                    plt.fill_between(x, m - sd, m + sd, color=c, alpha=0.2)
                plt.title(f"{env}", fontsize=fontsize + 3)
                plt.xlabel("episodes", fontsize=fontsize)
                plt.ylabel(f"eval {prefix}", fontsize=fontsize)
                plt.legend(loc="best")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_dir / f"{env}_evaluation_{prefix}.pdf")
                plt.close()

        # ─── collect non-eval metrics for overlaid plot -----------------------
        handled_loss_tags = {
            "actor_loss",
            "extra_actor_loss",
            "critic_loss",
            "extra_critic_loss",
            "extra_ctitic_loss",
            "total_loss",
            "",
        }
        for metric in sorted(
            {
                m
                for ad in algo_dict.values()
                for m in ad
                if not EVAL_RE.match(m) and m not in handled_loss_tags
            }
        ):
            plt.figure(figsize=(5, 3), dpi=500)
            drew = False
            for algo, mdict in algo_dict.items():
                y = mdict.get(metric)
                if not y:
                    continue
                y_arr = np.asarray(y, float)
                x = np.linspace(0, n_episodes, len(y_arr))
                c = get_algo_color(algo)
                plt.plot(x, y_arr, label=algo, linewidth=3, c=c)
                drew = True

                m, s, q25, q75 = _summarise(y_arr)
                iqm, iqr_std = _iqm_and_iqr_std(y_arr)

                summary_rows.append(
                    dict(
                        env=env,
                        algo=algo,
                        metric=metric,
                        mean=m,
                        std=s,
                        iqr25=q25,
                        iqr75=q75,
                    )
                )
                robust_rows.append(
                    dict(
                        env=env,
                        group_env=group_env,
                        algo=algo,
                        metric=metric,
                        iqm=iqm,
                        iqr_std=iqr_std,
                    )
                )

            if drew:
                plt.title(f"{env}", fontsize=fontsize + 3)
                plt.xlabel("episodes", fontsize=fontsize)
                plt.ylabel(f"{metric.replace("_", " ")}", fontsize=fontsize)
                plt.legend(loc="best")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_dir / f"{env}_{metric}.pdf")
                plt.close()

        # ─── three shared loss figures (algorithms overlaid) ------------------
        loss_specs = {
            "actor loss": ("actor_loss", "extra_actor_loss"),
            "critic loss": ("critic_loss", "extra_critic_loss", "extra_ctitic_loss"),
            "total loss": ("total_loss",),
        }

        for fig_tag, keys in loss_specs.items():
            plt.figure(figsize=(9, 6), dpi=500)
            drew = False
            for algo, mdict in algo_dict.items():
                if fig_tag == "total loss":
                    y = mdict.get("total_loss")
                    y_arr = None if y is None else np.asarray(y, float)
                else:
                    y_arr = _combine_losses(mdict, keys)

                if y_arr is None or len(y_arr) == 0:
                    continue

                x = np.linspace(0, n_episodes, len(y_arr))
                plt.plot(x, y_arr, label=algo, linewidth=2)
                drew = True

                m, s, q25, q75 = _summarise(y_arr)
                iqm, iqr_std = _iqm_and_iqr_std(y_arr)

                summary_rows.append(
                    dict(
                        env=env,
                        algo=algo,
                        metric=fig_tag,
                        mean=m,
                        std=s,
                        iqr25=q25,
                        iqr75=q75,
                    )
                )
                robust_rows.append(
                    dict(
                        env=env,
                        group_env=group_env,
                        algo=algo,
                        metric=fig_tag,
                        iqm=iqm,
                        iqr_std=iqr_std,
                    )
                )

            if drew:
                plt.title(f"{env}", fontsize=fontsize + 3)
                plt.xlabel("episodes", fontsize=fontsize)
                plt.ylabel(f"{fig_tag}", fontsize=fontsize)
                plt.legend(loc="best", fontsize=fontsize - 3)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_dir / f"{env}_{fig_tag.replace(' ', '_')}.pdf")
                plt.close()

    # ────── CSV with every metric’s mean/std/IQR ──────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(results_dir / "summary_table.csv", index=False)

    # ────── Robust (IQM / IQR std) tables, wide format with metric pairs ─────
    df_r = pd.DataFrame(robust_rows)
    if not df_r.empty:
        # pivot to wide: iqm_* and iqr_std_* columns
        iqm_wide = df_r.pivot_table(
            index=["env", "group_env", "algo"],
            columns="metric",
            values="iqm",
            aggfunc="mean",
        )
        iqm_wide.columns = [f"iqm_{c}" for c in iqm_wide.columns]
        iqr_wide = df_r.pivot_table(
            index=["env", "group_env", "algo"],
            columns="metric",
            values="iqr_std",
            aggfunc="mean",
        )
        iqr_wide.columns = [f"iqr_std_{c}" for c in iqr_wide.columns]
        df_wide = pd.concat([iqm_wide, iqr_wide], axis=1).reset_index()

        # Save overall wide table (raw stats)
        (results_dir / "robust_table.csv").write_text(df_wide.to_csv(index=False))

        # Also write an "overall_with_mean.csv" with a final MEAN row per algo (raw stats)
        num_cols = df_wide.select_dtypes(include=[np.number]).columns
        means_over_env = (
            df_wide.groupby("algo")[num_cols].mean(numeric_only=True).reset_index()
        )
        means_over_env.insert(0, "group_env", "ALL")
        means_over_env.insert(0, "env", "MEAN")
        overall_with_mean = pd.concat([df_wide, means_over_env], ignore_index=True)
        (tables_dir / "overall_with_mean.csv").write_text(
            overall_with_mean.to_csv(index=False)
        )

        # ───── Compute % improvements (env → group IQM → overall IQM) ─────
        df_env_only = df_wide[df_wide["env"] != "MEAN"].copy()

        # 1) Per-environment improvement table (FIRST RESULT)
        impr_per_env = _compute_improvement_table(
            df_env_only, explicit_pairs=explicit_pairs
        )
        (tables_dir / "improvement_per_env.csv").write_text(
            impr_per_env.to_csv(index=False)
        )

        # ───── LaTeX tables (TXT) with columns = variant algos, values = Δ% (IQM ± IQR-std) ─────
        latex_dir = tables_dir / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)

        # (A) ENV-wise table
        row_names, col_names, matrix_vals = _build_envwise_delta_matrix(
            df_wide[df_wide["env"] != "MEAN"]
        )
        if row_names:
            tex_env = _latex_table_from_matrix(
                row_header="env",
                row_names=row_names,
                col_names=col_names,
                matrix_vals=matrix_vals,
                caption="Per-environment IQM-based improvement (Δ%) of each variant over its baseline. Entries show IQM ± IQR-std.",
                label="tab:env_improvement_iqm",
            )
            (latex_dir / "env_delta_iqm_iqr_table.txt").write_text(tex_env)

        # (B) GROUP-wise and (C) OVERALL tables from per-env improvements
        (row_names_g, col_names_g, matrix_g), (row_names_o, col_names_o, matrix_o) = (
            _build_group_and_overall_delta_matrix(impr_per_env)
        )
        if row_names_g:
            tex_grp = _latex_table_from_matrix(
                row_header="group_env",
                row_names=row_names_g,
                col_names=col_names_g,
                matrix_vals=matrix_g,
                caption="Group-wise IQM % improvement of variants over their baselines. Entries show IQM ± IQR-std across environments in each group.",
                label="tab:group_improvement_iqm",
            )
            (latex_dir / "group_delta_iqm_iqr_table.txt").write_text(tex_grp)

        if row_names_o:
            tex_all = _latex_table_from_matrix(
                row_header="all",
                row_names=row_names_o,
                col_names=col_names_o,
                matrix_vals=matrix_o,
                caption="Overall IQM % improvement of variants over their baselines across all environments. Entries show IQM ± IQR-std.",
                label="tab:overall_improvement_iqm",
            )
            (latex_dir / "overall_delta_iqm_iqr_table.txt").write_text(tex_all)

        # ───── Error-bar plots (replace previous bar plots) ─────
        for env, sub in impr_per_env.groupby("env"):
            _plot_env_improvement_errorbars(
                sub.sort_values("pair"),
                tables_dir / "plots" / f"env_improvement_errorbars_{env}.pdf",
                env=env,
            )

        for grp, sub in impr_per_env.groupby("group_env"):
            _plot_group_or_overall_improvement_errorbars(
                sub,
                tables_dir
                / "plots"
                / f"{grp}_improvement_errorbars_{agg_type.upper()}.pdf",
                title=f"{grp}: {agg_type.upper()} % improvement (IQM of evaluation_return)",
                agg=("mean" if agg_type == "mean" else "iqm"),
            )

        _plot_group_or_overall_improvement_errorbars(
            impr_per_env,
            tables_dir
            / "plots"
            / f"overall_improvement_errorbars_{agg_type.upper()}.pdf",
            title=f"Overall: {agg_type.upper()} % improvement (IQM of evaluation_return)",
            agg=("mean" if agg_type == "mean" else "iqm"),
        )

        # (D) PAIRS × GROUP table
        row_pairs, col_groups, mat_pairs = _build_pairs_by_group_matrix(impr_per_env)
        if row_pairs:
            tex_pairs_grp = _latex_table_from_matrix(
                row_header="pair",
                row_names=row_pairs,
                col_names=col_groups,
                matrix_vals=mat_pairs,
                caption=(
                    "IQM % improvement (± IQR-std) of each requested algorithm pair "
                    "across environment groups. Each row is a 'variant vs baseline' pair."
                ),
                label="tab:pairs_by_group_improvement_iqm",
            )
            (latex_dir / "pairs_by_group_iqm_iqr_table.txt").write_text(tex_pairs_grp)

        # (E) PAIRS OVERALL table
        row_pairs_o, col_all, mat_pairs_o = _build_pairs_overall_matrix(impr_per_env)
        if row_pairs_o:
            tex_pairs_overall = _latex_table_from_matrix(
                row_header="pair",
                row_names=row_pairs_o,
                col_names=col_all,
                matrix_vals=mat_pairs_o,
                caption=(
                    "Overall IQM % improvement (± IQR-std) for each requested "
                    "'variant vs baseline' pair across all environments."
                ),
                label="tab:pairs_overall_improvement_iqm",
            )
            (latex_dir / "pairs_overall_iqm_iqr_table.txt").write_text(
                tex_pairs_overall
            )

    # Done


# ────────────────────────── optional CLI hook ────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir", type=str, help="Folder containing *_metrics.json files"
    )
    parser.add_argument(
        "--episodes", type=int, default=250, help="X-axis episodes used for plotting"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Comma-separated explicit pairs 'variant:baseline'",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        default="",
        help="Path to a file with one 'variant:baseline' per line",
    )
    parser.add_argument(
        "--agg-type",
        type=str,
        choices=["mean", "iqm"],
        default="mean",
        help="Aggregation type for improvements across environments (default: mean)",
    )
    args = parser.parse_args()

    plot_and_save_results(
        args.results_dir,
        n_episodes=args.episodes,
        pairs_arg=args.pairs or None,
        pairs_file=args.pairs_file or None,
        agg_type=args.agg_type,
    )

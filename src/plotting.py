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
def _env_group(env_name: str) -> str:
    name = env_name.lower()
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


# ─────────────────── causal variant improvement helpers ──────────────────────

# Auto-baseline detection (suffix stripping)
_SUFFIXES = (
    "_cc_cv",
    "-cc-cv",
    "_cc",
    "-cc",
    ".cc",
    "_cv",
    "-cv",
    ".cv",
)


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
        group_env = _env_group(env)

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
                plt.figure(figsize=(9, 6), dpi=500)
                for algo, (m, (lq, uq), sd) in stats.items():
                    c = get_algo_color(algo)
                    plt.plot(x, m, label=algo, linewidth=3, color=c)
                    plt.fill_between(x, m - sd, m + sd, color=c, alpha=0.2)
                plt.title(f"{env}", fontsize=fontsize + 3)
                plt.xlabel("episodes", fontsize=fontsize)
                plt.ylabel(f"evaluation {prefix}", fontsize=fontsize)
                plt.legend(loc="best", fontsize=fontsize - 3)
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
            plt.figure(figsize=(9, 6), dpi=500)
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
                plt.ylabel(f"{metric}", fontsize=fontsize)
                plt.legend(loc="best", fontsize=fontsize - 3)
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

        # Per-env plots
        for env, sub in impr_per_env.groupby("env"):
            _plot_env_improvement_bars(
                sub.sort_values("pair"),
                tables_dir / "plots" / f"env_improvement_{env}.pdf",
                env=env,
            )

        # 2) Aggregation per group
        if not impr_per_env.empty:
            if agg_type == "mean":
                agg_by_group = (
                    impr_per_env.groupby(["group_env", "pair"])["improvement_pct"]
                    .mean()
                    .reset_index(name="mean_improvement_pct")
                )
            else:  # iqm
                agg_by_group = (
                    impr_per_env.groupby(["group_env", "pair"])["improvement_pct"]
                    .apply(_iqm_1d)
                    .reset_index(name="iqm_improvement_pct")
                )
            (tables_dir / "improvement_mean_by_group.csv").write_text(
                agg_by_group.to_csv(index=False)
            )

            # Per-group bar plots (IQM only)
            for grp, sub in impr_per_env.groupby("group_env"):
                _plot_group_or_overall_improvement_bars(
                    sub,
                    tables_dir / "plots" / f"{grp}_improvement_bar_IQM.pdf",
                    title=f"{grp}: IQM % improvement (IQM of evaluation_return)",
                    agg="iqm",
                )

            if agg_type == "mean":
                agg_overall = (
                    impr_per_env.groupby("pair")["improvement_pct"]
                    .mean()
                    .reset_index(name="mean_improvement_pct")
                )
            else:
                agg_overall = (
                    impr_per_env.groupby("pair")["improvement_pct"]
                    .apply(_iqm_1d)
                    .reset_index(name="iqm_improvement_pct")
                )
            agg_overall.insert(0, "group_env", "ALL")
            (tables_dir / "improvement_mean_overall.csv").write_text(
                agg_overall.to_csv(index=False)
            )

            # Overall bar plot (IQM)
            _plot_group_or_overall_improvement_bars(
                impr_per_env,
                tables_dir / "plots" / "overall_improvement_bar_IQM.pdf",
                title="Overall: IQM % improvement (IQM of evaluation_return)",
                agg="iqm",
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

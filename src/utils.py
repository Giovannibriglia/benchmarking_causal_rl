import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

# ────────────────────────── helpers ───────────────────────────────────────────

EVAL_RE = re.compile(r"^evaluation_(return|length)_(\d+)$")


def _stack_runs(mdict: dict[str, list[float]], prefix: str) -> np.ndarray:
    """evaluation_<prefix>_<id>  ➜  (n_seeds, n_chkpts) array (nan-padded)."""
    keys = sorted(k for k in mdict if k.startswith(f"evaluation_{prefix}"))
    if not keys:
        return np.empty((0, 0))
    runs = [mdict[k] for k in keys]
    mlen = max(map(len, runs))
    padded = [np.pad(r, (0, mlen - len(r)), constant_values=np.nan) for r in runs]
    return np.vstack(padded)


def _mean_iqr_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # Keep values inside [q25, q75]
    mask = (yy >= q25) & (yy <= q75)
    mid = yy[mask]
    if mid.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(mid)), float(np.std(mid))


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


# ────────────────────────── main ──────────────────────────────────────────────


def plot_and_save_results(results_dir: str | Path, n_episodes: int):
    """
    • evaluation_return / evaluation_length: shaded mean±σ (also compute IQM/IQR std)
    • every *other* metric: algorithms compared in one figure
    • three shared loss plots per env:
        – actor   : (actor_loss + extra_actor_loss) per algorithm
        – critic  : (critic_loss + extra_critic_loss[|extra_ctitic_loss]) per algorithm
        – total   : total_loss per algorithm
    • CSVs written:
        1) summary_table.csv                – env, algo, metric, mean, std, iqr25, iqr75
        2) robust_table.csv                 – wide table: env, group_env, algo, iqm_*, iqr_std_*
        3) tables/group_<group>.csv         – per-group wide tables with final MEAN row
        4) tables/overall_with_mean.csv     – overall wide table with final MEAN row
    """
    fontsize = 25

    results_dir = Path(results_dir)
    plot_dir = results_dir / "plots"
    tables_dir = results_dir / "tables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ────── load all *_metrics.json ───────────────────────────────────────────
    data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for jf in results_dir.glob("*_metrics.json"):
        env = jf.stem.replace("_metrics", "")
        data[env] = json.loads(jf.read_text()).get(env, {})

    summary_rows: list[dict[str, str | float]] = []
    robust_rows: list[dict[str, str | float]] = []  # env, algo, metric, iqm, iqr_std

    # ────── iterate envs ──────────────────────────────────────────────────────
    pbar = tqdm(data.items())
    for env, algo_dict in pbar:
        pbar.set_description(f"Plotting: {env}...")
        group_env = _env_group(env)

        # ─── evaluation_return & evaluation_length  (stats AND plots) ─────────
        for prefix in ("return", "length"):
            stats = {}
            for algo, mdict in algo_dict.items():
                stacked = _stack_runs(mdict, prefix)  # (seeds, chkpts)
                if stacked.size == 0:
                    continue
                mean_curve, (lq, uq), sd_curve = _mean_iqr_std(stacked)

                # Overall summary stats on all elems
                m, s, q25, q75 = _summarise(stacked)
                # Robust stats (IQM & IQR std)
                iqm, iqr_std = _iqm_and_iqr_std(stacked)

                stats[algo] = mean_curve, (lq, uq), sd_curve

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

            if stats:
                x = np.linspace(0, n_episodes, next(iter(stats.values()))[0].size)
                plt.figure(figsize=(9, 6), dpi=500)
                for algo, (m, (lq, uq), sd) in stats.items():
                    plt.plot(x, m, label=algo, linewidth=3)
                    # You chose σ shading; keep it
                    plt.fill_between(x, m - sd, m + sd, alpha=0.1)
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
                plt.plot(x, y_arr, label=algo, linewidth=3)
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

        # Save overall wide table
        overall_path = results_dir / "robust_table.csv"
        df_wide.to_csv(overall_path, index=False)

        # Also write an "overall_with_mean.csv" with a final MEAN row per algo
        num_cols = df_wide.select_dtypes(include=[np.number]).columns
        means_over_env = (
            df_wide.groupby("algo")[num_cols].mean(numeric_only=True).reset_index()
        )
        means_over_env.insert(0, "group_env", "ALL")
        means_over_env.insert(0, "env", "MEAN")
        overall_with_mean = pd.concat([df_wide, means_over_env], ignore_index=True)
        overall_with_mean.to_csv(tables_dir / "overall_with_mean.csv", index=False)

        # Per-group tables with MEAN per algo (across envs in that group)
        for grp, sub in df_wide.groupby("group_env"):
            num_cols = sub.select_dtypes(include=[np.number]).columns
            grp_means = (
                sub.groupby("algo")[num_cols].mean(numeric_only=True).reset_index()
            )
            grp_means.insert(0, "group_env", grp)
            grp_means.insert(0, "env", "MEAN")
            sub_with_mean = pd.concat([sub, grp_means], ignore_index=True)
            sub_with_mean.to_csv(tables_dir / f"group_{grp}.csv", index=False)

    # Done
    # print(f"Plots in {plot_dir}\nSummary CSV: summary_table.csv\nRobust CSV: robust_table.csv")

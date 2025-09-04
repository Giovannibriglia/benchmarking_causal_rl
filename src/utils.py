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
    """evaluation_<prefix>_<id>  ➜  (n_seeds, n_chkpts) array (nan‑padded)."""
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
    """Element‑wise sum of any existing loss curves in *keys* (nan‑pad ragged)."""
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


# ────────────────────────── main ──────────────────────────────────────────────


def plot_and_save_results(results_dir: str | Path, n_episodes: int = 1_000):
    """
    • evaluation_return / evaluation_length: shaded mean±IQR±σ, algorithms overlaid
    • every *other* metric: algorithms compared in one figure
    • three shared loss plots per env:
        – actor   : (actor_loss + extra_actor_loss) per algorithm
        – critic  : (critic_loss + extra_critic_loss[|extra_ctitic_loss]) per algorithm
        – total   : total_loss per algorithm
    • CSV: env, algo, metric, mean, std, iqr25, iqr75  (for *all* metrics)
    """
    fontsize = 25

    results_dir = Path(results_dir)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ────── load all *_metrics.json ───────────────────────────────────────────
    data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for jf in results_dir.glob("*_metrics.json"):
        env = jf.stem.replace("_metrics", "")
        data[env] = json.loads(jf.read_text()).get(env, {})

    summary_rows: list[dict[str, str | float]] = []

    # ────── iterate envs ──────────────────────────────────────────────────────
    for env, algo_dict in tqdm(data.items(), desc="Plotting..."):

        # ─── evaluation_return & evaluation_length  (stats AND plots) ─────────
        for prefix in ("return", "length"):
            stats = {}
            for algo, mdict in algo_dict.items():
                stacked = _stack_runs(mdict, prefix)
                if stacked.size == 0:
                    continue
                mean, (lq, uq), sd = _mean_iqr_std(stacked)
                stats[algo] = mean, (lq, uq), sd

                m, s, q25, q75 = _summarise(stacked)
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

            if stats:
                x = np.linspace(0, n_episodes, next(iter(stats.values()))[0].size)
                plt.figure(figsize=(16, 9), dpi=500)
                for algo, (m, (lq, uq), sd) in stats.items():
                    plt.plot(x, m, label=algo, linewidth=3)
                    # plt.fill_between(x, lq, uq, alpha=0.25)
                    plt.fill_between(
                        x,
                        m - sd,
                        m + sd,
                        alpha=0.1,
                    )
                plt.title(f"{env}", fontsize=fontsize + 3)
                plt.xlabel("episodes", fontsize=fontsize)
                plt.ylabel(f"evaluation {prefix}", fontsize=fontsize)
                plt.legend(loc="best", fontsize=fontsize - 3)
                plt.tight_layout()
                plt.savefig(plot_dir / f"{env}_evaluation_{prefix}.png")
                plt.close()

        # ─── collect non‑eval metrics for overlaid plot -----------------------
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
            plt.figure(figsize=(16, 9), dpi=500)
            for algo, mdict in algo_dict.items():
                y = mdict.get(metric)
                if not y:
                    continue
                x = np.linspace(0, n_episodes, len(y))
                plt.plot(x, y, label=algo, linewidth=3)

                m, s, q25, q75 = _summarise(np.asarray(y, float))
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

            plt.title(f"{env}", fontsize=fontsize + 3)
            plt.xlabel("episodes", fontsize=fontsize)
            plt.ylabel(f"train {prefix}", fontsize=fontsize)
            plt.legend(loc="best", fontsize=fontsize - 3)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{env}_{metric}.png")
            plt.close()

        # ─── three shared loss figures (algorithms overlaid) ------------------
        loss_specs = {
            "actor loss": ("actor_loss", "extra_actor_loss"),
            "critic loss": ("critic_loss", "extra_critic_loss"),
            "total loss": ("total_loss",),
        }

        for fig_tag, keys in loss_specs.items():
            plt.figure(figsize=(16, 9), dpi=500)
            drew = False
            for algo, mdict in algo_dict.items():
                if fig_tag == "total_loss":
                    y = mdict.get("total_loss")
                else:
                    y = _combine_losses(mdict, keys)

                if y is None or len(y) == 0:
                    continue
                drew = True
                x = np.linspace(0, n_episodes, len(y))
                plt.plot(x, y, label=algo, linewidth=2)

                m, s, q25, q75 = _summarise(np.asarray(y, float))
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

            if drew:
                plt.title(f"{env}", fontsize=fontsize + 3)
                plt.xlabel("episodes", fontsize=fontsize)
                plt.ylabel(f"{fig_tag}", fontsize=fontsize)
                plt.legend(loc="best", fontsize=fontsize - 3)
                plt.tight_layout()
                plt.savefig(plot_dir / f"{env}_{fig_tag}.png")
                plt.close()

    # ────── CSV with every metric’s mean/std/IQR ──────────────────────────────
    pd.DataFrame(summary_rows).to_csv(results_dir / "summary_table.csv", index=False)
    # print(f"Plots saved in {plot_dir}\nSummary CSV written to summary_table.csv")

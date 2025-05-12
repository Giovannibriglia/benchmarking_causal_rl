import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt


def _iqr_mean(arr, percentile: int = 25):
    """
    Replace values outside the interquartile range with the 25th and 75th percentiles.

    Args:
        arr (np.ndarray): Input 1D or ND array.
        percentile (int): Lower percentile bound (default: 25). Upper is 100 - percentile.

    Returns:
        np.ndarray: Array with outliers replaced by percentile bounds.
    """
    lower = np.percentile(arr, percentile)
    upper = np.percentile(arr, 100 - percentile)

    # Clip values outside the percentile range
    arr_clipped = np.clip(arr, lower, upper)
    return arr_clipped


def plot_and_save_results(results_dir: str | Path, n_episodes: int):
    """
    Read every  *_metrics.json  in *results_dir*, aggregate metrics and
    generate one plot per metric plus a summary CSV (mean ± std of returns).

        results_dir/
            CartPole-v1_metrics.json
            FrozenLake-v1_metrics.json
            ...
    """

    fontsize = 25

    root = Path(results_dir)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- load all jsons -----------------------------
    data = {}
    for jf in root.glob("*_metrics.json"):
        env_id = jf.stem.replace("_metrics", "")
        data[env_id] = json.loads(jf.read_text()).get(env_id, {})

    table_rows = []

    # ---------------------- plotting & table ---------------------------
    for env_id, algo_dict in data.items():
        metrics = {m for ad in algo_dict.values() for m in ad}
        for metric in metrics:
            plt.figure(dpi=500, figsize=(16, 9))
            for algo_name, m_dict in algo_dict.items():
                y = m_dict.get(metric, [])
                # y = list(_iqr_mean(y))

                n_checkpoints = len(y)
                x = np.linspace(0, n_episodes, n_checkpoints)
                plt.plot(x, y, label=algo_name, linewidth=3)

                if metric.endswith("return"):
                    table_rows.append(
                        {
                            "env": env_id,
                            "algo": algo_name,
                            "metric": metric,
                            "value": f"{np.mean(y)} +- {np.std(y)}" if y else math.nan,
                        }
                    )

            plt.title(f"{env_id}", fontsize=fontsize + 5)
            # xt = np.linspace(0, n_episodes, n_checkpoints)
            plt.xticks(fontsize=fontsize - 2)
            plt.xlabel("episodes", fontsize=fontsize)
            plt.ylabel(f"{metric}", fontsize=fontsize)
            plt.legend(loc="best", fontsize=fontsize - 5)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{env_id}_{metric}.png", dpi=500)
            plt.close()

    # ---------------------- summary CSV --------------------------------
    df = pd.DataFrame(table_rows)
    df.to_csv(root / "summary_table.csv", index=False)
    print(f"Saved plots in {plot_dir} and CSV in {root/'summary_table.csv'}")


def compute_divergence(casual_dist, rl_dist, divergence: str = "kl"):
    if divergence == "kl":
        kl = (
            (
                F.softmax(rl_dist, -1)
                * (F.log_softmax(rl_dist, -1) - torch.log(casual_dist + 1e-8))
            )
            .sum(-1)
            .mean()
        )
        return kl
    else:
        raise NotImplementedError

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_and_save_results(results_dir: str | Path):
    """
    Read every  *_metrics.json  in *results_dir*, aggregate metrics and
    generate one plot per metric plus a summary CSV (mean ± std of returns).

        results_dir/
            CartPole-v1_metrics.json
            FrozenLake-v1_metrics.json
            ...
    """
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
                x = range(len(y))
                plt.plot(x, y, label=algo_name)

                if metric.endswith("return"):
                    table_rows.append(
                        {
                            "env": env_id,
                            "algo": algo_name,
                            "metric": metric,
                            "mean": np.mean(y) if y else math.nan,
                            "std": np.std(y) if y else math.nan,
                        }
                    )

            plt.title(f"{metric} – {env_id}")
            plt.xlabel("checkpoint #")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"{env_id}_{metric}.png", dpi=150)
            plt.close()

    # ---------------------- summary CSV --------------------------------
    df = pd.DataFrame(table_rows)
    df.to_csv(root / "summary_table.csv", index=False)
    print(f"Saved plots → {plot_dir} and CSV → {root/'summary_table.csv'}")

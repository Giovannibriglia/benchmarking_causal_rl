#!/usr/bin/env python3
"""β×δ grid heatmaps (Phase 6A deliverable):
(i) basic normalized regret, (ii) |naive OPE − true J| bias (basic rows),
(iii) gate z-scores (A–U and R–U).

Usage: python -m tools.grid_heatmaps <cell7_grid_run_dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.eval.regret import iqm

BETAS = ["0p5", "1", "2"]
DELTAS = ["0p25", "0p5", "1"]


def _tier_to_bd(tier: str):
    m = re.match(r"b(.+)-d(.+)$", tier)
    return m.group(1), m.group(2)


def _grid(df: pd.DataFrame, value_fn) -> np.ndarray:
    out = np.full((len(BETAS), len(DELTAS)), np.nan)
    for tier, grp in df.groupby("tier"):
        b, d = _tier_to_bd(str(tier))
        out[BETAS.index(b), DELTAS.index(d)] = value_fn(grp)
    return out


def _heat(ax, grid: np.ndarray, title: str, fmt: str = "{:.2f}"):
    im = ax.imshow(grid, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(DELTAS)))
    ax.set_xticklabels([d.replace("p", ".") for d in DELTAS])
    ax.set_yticks(range(len(BETAS)))
    ax.set_yticklabels([b.replace("p", ".") for b in BETAS])
    ax.set_xlabel("delta (U->reward)")
    ax.set_ylabel("beta (U->action)")
    ax.set_title(title, fontsize=10)
    for i in range(len(BETAS)):
        for j in range(len(DELTAS)):
            if not np.isnan(grid[i, j]):
                ax.text(
                    j,
                    i,
                    fmt.format(grid[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
    return im


def main() -> None:
    run_dir = Path(sys.argv[1])
    df = pd.read_csv(run_dir / "causal_cells_metrics.csv")
    gates = pd.read_csv(run_dir / "gate_reports.csv").drop_duplicates("tier")
    basic = df[df.role == "basic"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _heat(
        axes[0][0],
        _grid(basic, lambda g: iqm(g.normalized_regret.to_numpy(float))),
        "(i) basic (BC) normalized regret, IQM over 3 seeds",
    )
    _heat(
        axes[0][1],
        _grid(
            basic,
            lambda g: iqm(np.abs(g.ope_naive.to_numpy(float) - g.J.to_numpy(float))),
        ),
        "(ii) |naive OPE − true J| (basic), IQM",
        fmt="{:.0f}",
    )
    _heat(
        axes[1][0],
        _grid(gates, lambda g: float(g.action_u_zscore.iloc[0])),
        "(iii a) gate A–U conditional z-score",
        fmt="{:.1f}",
    )
    _heat(
        axes[1][1],
        _grid(gates, lambda g: float(g.reward_u_zscore.iloc[0])),
        "(iii b) gate R–U z-score (log10)",
        fmt="{:.1f}",
    )
    # R-U z explodes for constant-reward CartPole; show log10
    grid_rz = _grid(gates, lambda g: np.log10(max(float(g.reward_u_zscore.iloc[0]), 1)))
    axes[1][1].images[0].set_data(grid_rz)
    for txt in list(axes[1][1].texts):
        txt.remove()
    for i in range(len(BETAS)):
        for j in range(len(DELTAS)):
            if not np.isnan(grid_rz[i, j]):
                axes[1][1].text(
                    j,
                    i,
                    f"{grid_rz[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
    fig.suptitle(
        "Confounding-strength grid (cell 7, CartPole): regret, naive-OPE "
        "bias and gate statistics over (beta, delta)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path("outputs/causal_cells")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"grid_heatmaps.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {out}/grid_heatmaps.{{png,pdf}}")


if __name__ == "__main__":
    main()

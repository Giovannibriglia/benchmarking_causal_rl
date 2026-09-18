"""E1 figures aggregated across seeds: IQM +/- IQR-STD, line + fill_between."""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/e1/figures"
BASE, GRACE = "#2a78d6", "#eb6834"
ARM_C = {"base": BASE, "grace": GRACE}
CELL_ORDER = ["danull", "d100s0", "d100", "d025", "d010asym"]
CELL_TITLE = {
    "danull": "d_a_null",
    "d100s0": "d100  σ=0",
    "d100": "d100  σ=0.25",
    "d025": "d025  σ=0.25",
    "d010asym": "d010asym  σ=0.25",
}
ALGO_ORDER = ["cql", "iql"]


def iqm_iqrstd(a: np.ndarray):
    """Mean and std of the middle 50%, matching runner._aggregate_returns."""
    out_m, out_s = [], []
    for col in range(a.shape[1]):
        v = np.sort(a[:, col])
        n = v.size
        lo, hi = int(0.25 * (n - 1)), int(0.75 * (n - 1))
        mid = v[lo : hi + 1]
        out_m.append(mid.mean())
        out_s.append(mid.std())
    return np.asarray(out_m), np.asarray(out_s)


def load():
    runs = []
    for p in sorted(
        glob.glob("results/e1/*/beta_*/CartPole-v1/*/*/*/e1_provenance.json")
    ):
        d = json.load(open(p))
        leaf = os.path.dirname(p)
        d["eval"] = pd.read_csv(f"{leaf}/eval_metrics.csv")
        d["ca"] = pd.read_csv(f"{leaf}/critic_ablation_metrics.csv")
        dep = f"{leaf}/eval_deployment.csv"
        d["dep"] = pd.read_csv(dep) if os.path.exists(dep) else None
        runs.append(d)
    return runs


def series(runs, cell, algo, arm, frame, col):
    rs = [
        r
        for r in runs
        if r["cell"] == cell
        and r["algo"] == algo
        and r["arm"] == arm
        and r.get(frame) is not None
        # An ABSTAINED grace run is a byte-copy of its base (the transform never
        # fired), so pooling it into the grace aggregate double-counts base and
        # biases every grace-vs-base contrast toward zero. The serving layer's
        # own rule: abstentions are reported separately, never pooled.
        and not (r.get("grace") or {}).get("abstained", False)
    ]
    if not rs:
        return None, None, None, 0
    n = min(len(r[frame]) for r in rs)
    x = np.asarray(rs[0][frame]["episode"])[:n]
    a = np.vstack([np.asarray(r[frame][col])[:n] for r in rs])
    m, s = iqm_iqrstd(a)
    return x, m, s, len(rs)


def panel_fig(runs, frame, col, ylabel, title, fname, hline=None):
    cells = [
        c
        for c in CELL_ORDER
        if any(r["cell"] == c and r.get(frame) is not None for r in runs)
    ]
    if not cells:
        return
    fig, axes = plt.subplots(
        len(cells),
        len(ALGO_ORDER),
        figsize=(9.5, 2.8 * len(cells)),
        squeeze=False,
        sharex=True,
    )
    for i, cell in enumerate(cells):
        for j, algo in enumerate(ALGO_ORDER):
            ax = axes[i][j]
            for arm in ("base", "grace"):
                x, m, s, k = series(runs, cell, algo, arm, frame, col)
                if x is None:
                    continue
                n_abst = sum(
                    1
                    for r in runs
                    if r["cell"] == cell
                    and r["algo"] == algo
                    and r["arm"] == arm
                    and (r.get("grace") or {}).get("abstained", False)
                )
                lbl = f"{arm} (n={k}" + (f", {n_abst} abstained)" if n_abst else ")")
                ax.plot(x, m, color=ARM_C[arm], lw=2, label=lbl)
                ax.fill_between(x, m - s, m + s, color=ARM_C[arm], alpha=0.20, lw=0)
            if hline is not None:
                t = (
                    next((r["q1_truth"] for r in runs if r["cell"] == cell), None)
                    if hline == "truth"
                    else hline
                )
                if t is not None:
                    ax.axhline(t, color="black", lw=1.2, ls="--")
            ax.set_title(f"{CELL_TITLE[cell]} · {algo.upper()}", fontsize=10)
            ax.grid(True, lw=0.6, alpha=0.4)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if i == len(cells) - 1:
                ax.set_xlabel("gradient steps", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{OUT}/{fname}", dpi=170)
    plt.close(fig)
    print("wrote", os.path.abspath(f"{OUT}/{fname}"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    print(f"loaded {len(runs)} cells")
    panel_fig(
        runs,
        "eval",
        "eval_return_mean",
        "deployment return",
        "Deployment return (IQM ± IQR-STD across seeds)",
        "seeds_return.png",
    )
    panel_fig(
        runs,
        "ca",
        "q1_contrast_pred",
        "recovered do-contrast",
        "Recovered do-contrast (IQM ± IQR-STD across seeds)",
        "seeds_contrast.png",
        hline="truth",
    )
    panel_fig(
        runs,
        "dep",
        "eval_return_base_mean",
        "base return (steps)",
        "Base return component (IQM ± IQR-STD across seeds)",
        "seeds_base_return.png",
    )
    panel_fig(
        runs,
        "dep",
        "eval_bad_action_steps_mean",
        "a_bad steps",
        "a_bad step count (IQM ± IQR-STD across seeds)",
        "seeds_a_bad.png",
    )

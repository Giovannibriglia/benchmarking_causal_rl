"""Figures for the E1 cells completed so far. Reads leaves, never recomputes.

PARTIAL GRID, and the figures say so: d_a_null is complete (12 cells), d100 has
5 of 12, and d025 / d010_asym have not started. Anything read here is provisional.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# dataviz reference palette, slots 1 and 2 (validated all-pairs, light surface)
BASE, GRACE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
ARM_C = {"base": BASE, "grace": GRACE}
OUT = "results/e1/figures"
CELL_ORDER, ALGO_ORDER = ["danull", "d100"], ["cql", "iql"]
CELL_TITLE = {"danull": "d_a_null  (no confounding)", "d100": "d100  (σ=0.25)"}


def load():
    runs = []
    for p in sorted(
        glob.glob(
            "results/e1/*/beta_000_sigma_025/CartPole-v1/*/*/*/e1_provenance.json"
        )
    ):
        d = json.load(open(p))
        leaf = os.path.dirname(p)
        d["eval"] = pd.read_csv(f"{leaf}/eval_metrics.csv")
        d["ca"] = pd.read_csv(f"{leaf}/critic_ablation_metrics.csv")
        runs.append(d)
    return runs


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def _draw(ax, x, y, r, label_seed=False):
    if r["arm"] == "base":
        ax.plot(x, y, color=BASE, lw=3.2, alpha=0.9, solid_capstyle="round", zorder=2)
    else:
        ax.plot(
            x,
            y,
            color=GRACE,
            lw=1.8,
            ls=(0, (3, 2)),
            alpha=0.95,
            dash_capstyle="round",
            zorder=3,
        )
    if label_seed and r["arm"] == "base":
        ax.annotate(
            f" s{r['seed']}",
            (list(x)[-1], list(y)[-1]),
            fontsize=8,
            color=INK2,
            va="center",
            annotation_clip=False,
        )


def _grid_fig(runs, title, sub):
    present = [
        (c, a)
        for c in CELL_ORDER
        for a in ALGO_ORDER
        if any(r["cell"] == c and r["algo"] == a for r in runs)
    ]
    fig, axes = plt.subplots(
        len(CELL_ORDER), len(ALGO_ORDER), figsize=(10, 6.4), facecolor=SURFACE
    )
    fig.suptitle(title, x=0.012, ha="left", fontsize=14, color=INK, weight="bold")
    fig.text(0.012, 0.935, sub, ha="left", fontsize=9.5, color=INK2)
    return fig, axes, present


def fig_return(runs):
    fig, axes, _ = _grid_fig(
        runs,
        "Deployment return over training",
        "Analytic confounded-reward evaluation, 25 checkpoints per run. "
        "One line per seed. PARTIAL GRID: d025 and d010_asym not yet run.",
    )
    for i, cell in enumerate(CELL_ORDER):
        for j, algo in enumerate(ALGO_ORDER):
            ax = axes[i][j]
            _style(ax)
            sel = [r for r in runs if r["cell"] == cell and r["algo"] == algo]
            for r in sel:
                # base thick+solid UNDER, grace thin+dashed OVER: where the two
                # coincide the reader sees dashes on blue, not one arm erased.
                _draw(ax, r["eval"].episode, r["eval"].eval_return_mean, r)
            ax.set_title(
                f"{CELL_TITLE[cell]} · {algo.upper()}",
                fontsize=10,
                color=INK,
                loc="left",
                pad=6,
            )
            if not sel:
                ax.text(
                    0.5,
                    0.5,
                    "not run yet",
                    ha="center",
                    color=INK2,
                    fontsize=10,
                    transform=ax.transAxes,
                )
                continue
            lo = min(r["eval"].eval_return_mean.min() for r in sel)
            hi = max(r["eval"].eval_return_mean.max() for r in sel)
            ax.set_ylim(lo - 1.2, hi + 1.2)
            n_seeds = len({r["seed"] for r in sel})
            ax.text(
                0.985,
                0.06,
                f"range {lo:.0f}–{hi:.0f} over "
                f"{n_seeds} seed{'s' if n_seeds > 1 else ''} × 25 checkpoints",
                ha="right",
                fontsize=8,
                color=INK2,
                transform=ax.transAxes,
            )
            if i == 1:
                ax.set_xlabel("gradient steps", fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel("deployment return", fontsize=9, color=INK2)
    _legend(fig)
    fig.text(
        0.012,
        0.015,
        "READ THIS FIRST: every run sits at 2–3 for its whole training, with no trend "
        "(CartPole: an unlearned policy is ~9, a solved one 500). The only movement is\n"
        "d100·CQL flicking between 2 and 3. A return-based prediction has almost no "
        "room to fail here — the same S9 problem as d_a_null's zero-width interval.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    fig.subplots_adjust(top=0.855, bottom=0.155, hspace=0.42, wspace=0.2)
    fig.savefig(f"{OUT}/e1_return.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_contrast(runs):
    fig, axes, _ = _grid_fig(
        runs,
        "Quality of the learned reward model: recovered do-contrast vs truth",
        "The critic's estimate of E[R|do(a_bad)] − E[R|do(other)] at each checkpoint. "
        "Dashed line is the analytic truth. Closer to it is better.",
    )
    for i, cell in enumerate(CELL_ORDER):
        for j, algo in enumerate(ALGO_ORDER):
            ax = axes[i][j]
            _style(ax)
            sel = [r for r in runs if r["cell"] == cell and r["algo"] == algo]
            if not sel:
                ax.set_title(
                    f"{CELL_TITLE[cell]} · {algo.upper()}",
                    fontsize=10,
                    color=INK,
                    loc="left",
                    pad=6,
                )
                ax.text(
                    0.5,
                    0.5,
                    "not run yet",
                    ha="center",
                    color=INK2,
                    fontsize=10,
                    transform=ax.transAxes,
                )
                continue
            truth = sel[0]["q1_truth"]
            ax.axhline(truth, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=3)
            ax.text(
                sel[0]["ca"].episode.max(),
                truth,
                f"  truth = {truth:g}",
                va="center",
                fontsize=8.5,
                color=INK,
            )
            for r in sel:
                _draw(ax, r["ca"].episode, r["ca"].q1_contrast_pred, r, label_seed=True)
            ax.set_title(
                f"{CELL_TITLE[cell]} · {algo.upper()}",
                fontsize=10,
                color=INK,
                loc="left",
                pad=6,
            )
            if i == 1:
                ax.set_xlabel("gradient steps", fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel("recovered do-contrast", fontsize=9, color=INK2)
    _legend(fig)
    fig.text(
        0.012,
        0.015,
        "On d100 GRACE moves the recovered contrast toward truth in every paired run — "
        "the intended direction, though a large gap remains.\n"
        "On d_a_null the two arms sit on top of each other (max divergence 3.4e-4 across "
        "25 checkpoints): the transform fires but has almost nothing to change.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    fig.subplots_adjust(top=0.855, bottom=0.155, hspace=0.42, wspace=0.2)
    fig.savefig(f"{OUT}/e1_reward_model_quality.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_seam(runs):
    """The SEAM's own reward model, on the two axes P1a is registered against."""
    g = [r for r in runs if r["arm"] == "grace" and r.get("grace")]
    g.sort(key=lambda r: (-CELL_ORDER.index(r["cell"]), r["algo"], r["seed"]))
    lab = [f"{r['cell']} · {r['algo']} · s{r['seed']}" for r in g]
    yy = range(len(g))
    fig, (axL, axR) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.4),
        facecolor=SURFACE,
        gridspec_kw=dict(width_ratios=[1.25, 1]),
    )
    fig.suptitle(
        "The seam's reward model, per GRACE run",
        x=0.008,
        ha="left",
        fontsize=14,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.008,
        0.90,
        "LEFT: how well GRACE estimates the do-contrast.   "
        "RIGHT: how big the served pessimism is, against what P1a says it should track.",
        ha="left",
        fontsize=9.5,
        color=INK2,
    )

    # ---- LEFT: estimate − truth, with L4's interval around it. 0 = exact.
    _style(axL)
    axL.axvline(0, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=3)
    for y, r in enumerate(g):
        t, m = r["q1_truth"], r["grace"]["meta"]
        axL.plot(
            [r["grace"]["lo"] - t, r["grace"]["hi"] - t],
            [y, y],
            color=GRACE,
            lw=8,
            solid_capstyle="round",
            alpha=0.35,
            zorder=2,
        )
        axL.plot(
            [m["contrast_point"] - t],
            [y],
            "o",
            color=GRACE,
            ms=9,
            mec=SURFACE,
            mew=2,
            zorder=4,
        )
    axL.set_yticks(list(yy))
    axL.set_yticklabels(lab, fontsize=9, color=INK)
    axL.set_ylim(-0.6, len(g) - 0.4)
    axL.set_xlim(-0.030, 0.017)
    axL.set_xlabel("estimate − truth", fontsize=9, color=INK2)
    axL.set_title(
        "Estimation error   (0 = exact; bar is L4's interval)",
        fontsize=10,
        color=INK,
        loc="left",
        pad=6,
    )
    axL.text(
        0.0012,
        len(g) - 0.75,
        "truth",
        fontsize=8.5,
        color=INK,
        va="center",
    )

    # ---- RIGHT: applied shift vs the two candidate scales P1a distinguishes
    _style(axR)
    for y, r in enumerate(g):
        m = r["grace"]["meta"]
        half = (r["grace"]["hi"] - r["grace"]["lo"]) / 2
        axR.plot(
            [0, m["pessimism_applied"]],
            [y, y],
            color=GRACE,
            lw=8,
            solid_capstyle="round",
            alpha=0.35,
            zorder=2,
        )
        axR.plot([half], [y], "|", color=INK, ms=16, mew=2.2, zorder=5)
        axR.plot([r["q1_truth"]], [y], "x", color=INK2, ms=9, mew=2, zorder=5)
    axR.set_yticks(list(yy))
    axR.set_yticklabels([])
    axR.set_ylim(-0.6, len(g) - 0.4)
    axR.set_xscale("symlog", linthresh=1e-3)
    axR.set_xlim(-2e-4, 1.2)
    axR.set_xlabel("magnitude (symlog)", fontsize=9, color=INK2)
    axR.set_title(
        "Served shift vs the scale it should track",
        fontsize=10,
        color=INK,
        loc="left",
        pad=6,
    )
    import matplotlib.lines as ml

    axR.legend(
        handles=[
            ml.Line2D(
                [],
                [],
                color=GRACE,
                lw=6,
                alpha=0.35,
                label="shift applied  (point − lo)",
            ),
            ml.Line2D(
                [],
                [],
                color=INK,
                marker="|",
                ls="",
                ms=13,
                mew=2,
                label="interval half-width",
            ),
            ml.Line2D(
                [],
                [],
                color=INK2,
                marker="x",
                ls="",
                ms=8,
                mew=2,
                label="M · tilt  (= truth)",
            ),
        ],
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.99),
    )
    fig.text(
        0.008,
        0.022,
        "P1a reads directly off the right panel: on d100 the served shift is 0.0141 against a "
        "half-width of 0.0139 and an M·tilt of 0.50 — pessimism tracking the interval, "
        "not the bias.\nOn d_a_null all three collapse to zero: reward is constant, so the "
        "contrast is identically 0 for the data AND every resample. That is an untestable "
        "pass, not a clean one,\nand it is why the σ=0 arm is the no-harm measurement.",
        fontsize=8.5,
        color=INK,
        va="bottom",
    )
    fig.subplots_adjust(top=0.775, bottom=0.30, left=0.135, right=0.985, wspace=0.06)
    fig.savefig(f"{OUT}/e1_seam_interval.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def _legend(fig):
    import matplotlib.lines as ml

    fig.legend(
        handles=[
            ml.Line2D([], [], color=BASE, lw=3.2, label="base (unmodified critic)"),
            ml.Line2D(
                [],
                [],
                color=GRACE,
                lw=1.8,
                ls=(0, (3, 2)),
                label="grace (reward transform)",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.925),
        frameon=False,
        fontsize=9.5,
        labelcolor=INK2,
        ncol=2,
        handlelength=2.6,
        columnspacing=1.6,
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    print(f"loaded {len(runs)} completed cells")
    fig_return(runs)
    fig_contrast(runs)
    fig_seam(runs)
    for f in sorted(glob.glob(f"{OUT}/*.png")):
        print("wrote", os.path.abspath(f))

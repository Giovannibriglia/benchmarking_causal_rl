"""Figures for the (regime × L-sweep) ``results/`` tree — PRESENTATION ONLY.

This renderer CONSUMES ``regime_report.build_report(results_root, regime)`` and plots
from that single source of truth. It NEVER re-walks the tree, re-parses
``beta_*_sigma_*`` paths, or re-computes a mean/sd — that logic lives in
``regime_report``. If a figure needs a number ``build_report`` does not expose, the
figure is not produced here (see the biased-coverage note below), it is not smuggled
in by re-aggregating.

Figures written to ``<results_root>/_report/figures/``, PNG + PDF:

  1. Confounded σ-wedge (per env, algo) — ``value_mse_to_oracle_mean`` vs σ, one line
     per critic (the β=0 slice), error bars = ``_sd``. Fixes the ALGO, varies the
     CRITIC: does deconfounding help?
  3. pessimism_cost vs σ (per env, algo; sensitivity only) — pure cost at σ=0, buying
     robustness at σ>0. (A sweep cell runs one fixed Γ, so this is vs σ.)

  A. Fix-critic / vary-algo (per env) — ``value_mse_to_oracle_mean`` vs σ on the
     confounded slice at critic=observational, ONE LINE PER ALGO. The COMPLEMENT of the
     wedge: fixes the CRITIC, varies the ALGO — which base learner is most fragile to
     confounding? Error bars = ``_sd``.
  B. Return across the sweep (per env, algo) — the return column(s) vs σ on the β=0
     slice, one line per critic for the evaluation return (``eval_return_mean``), with
     the training/behavior return (``train_return_mean``) as a reference line WHEN it is
     logged. NB (honest caveat): the strategy critic-ablation shares ONE base actor
     across critics, so the return does not vary by critic in this cell — the per-critic
     eval lines coincide; the wedge (Fig 1) is where the critics separate. train_return
     is BLANK for offline algos (no per-episode rollout return is logged), so the
     reference line appears only for the online regimes.
  C. Biased-arm coverage (per env, algo) — ``action_coverage_mean`` vs β on the biased
     arm (σ=0, observational critic). Expect the (1-β)·coverage(0) line. Now unblocked:
     ``regime_report`` aggregates ``action_coverage`` from ``arm_diagnostics.csv``, so
     it is in ``build_report``'s output and no tree re-walk is needed here.

Guards (the 1-seed smoke hits all): an all-NaN ``_sd`` column -> plot means, skip error
bars; a missing critic/algo at some x -> just omit that point; an uncalibrated null-cal
row (n<2 or no reference) -> annotate, never fail.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402

from src.benchmarking.regime_report import build_report  # noqa: E402

_CRITIC_ORDER = ("observational", "proximal", "oracle_u", "sensitivity")


def _num(x) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(float(x))


def _safe(name: str) -> str:
    return str(name).replace("/", "-")


def _env_algos(agg: List[Dict]):
    return sorted({(r["env"], r["algo"]) for r in agg})


def _envs(agg: List[Dict]):
    return sorted({r["env"] for r in agg})


def _nc_row(nc: List[Dict], env: str, algo: str) -> Optional[Dict]:
    for r in nc:
        if r["env"] == env and r["algo"] == algo:
            return r
    return None


def _nc_annotation(row: Optional[Dict]) -> str:
    if row is None:
        return "null-calibration: no row"
    cal = row.get("null_calibrated")
    if cal is None:
        return "null-calibrated: uncalibrated (n<2 or no reference)"
    ratio, k = _num(row.get("ratio")), _num(row.get("k"))
    n = _num(row.get("n_seeds"))
    n_i = int(n) if _finite(n) else 0
    txt = f"null_calibrated={cal}  (ratio={ratio:.2g}, k={k:.2g}, n_seeds={n_i})"
    # The fixed-denominator gate computes a verdict even at n=1 (gap vs a STORED
    # noise_ref, not the cell's own noise), but a single-seed gap is statistically
    # meaningless — flag it so a smoke's verdict is never read as real.
    if n_i < 2:
        txt += "  — single-seed, NOT a real verdict"
    return txt


def _save(fig, out: Path, stem: str, formats) -> List[Path]:
    fig.tight_layout()
    written: List[Path] = []
    for fmt in formats:
        p = out / f"{stem}.{fmt}"
        fig.savefig(p, dpi=150)
        written.append(p)
    plt.close(fig)
    return written


def _plot_series(ax, xs, ys, es, **kw) -> None:
    """Error bars ONLY when every sd is finite and at least one is > 0 (a 1-seed
    smoke has all-NaN sd -> plot the means alone, don't crash)."""
    if all(_finite(e) for e in es) and any(e > 0 for e in es):
        ax.errorbar(xs, ys, yerr=es, capsize=3, marker="o", **kw)
    else:
        ax.plot(xs, ys, marker="o", **kw)


# --------------------------------------------------------------------------- #
# Figure 1 — the confounded σ-wedge                                            #
# --------------------------------------------------------------------------- #
def _fig_sigma_wedge(agg, nc, regime, out, formats) -> List[Path]:
    written: List[Path] = []
    for env, algo in _env_algos(agg):
        # β=0 slice = the σ-sweep (basic σ=0 + the confounded arm σ>0).
        rows = [
            r
            for r in agg
            if r["env"] == env and r["algo"] == algo and float(r["beta"]) == 0.0
        ]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        drew = False
        for critic in _CRITIC_ORDER:
            pts = sorted(
                (r for r in rows if r["critic"] == critic),
                key=lambda r: float(r["sigma"]),
            )
            triples = [
                (
                    float(r["sigma"]),
                    _num(r.get("value_mse_to_oracle_mean")),
                    _num(r.get("value_mse_to_oracle_sd")),
                )
                for r in pts
            ]
            triples = [
                (x, y, e) for x, y, e in triples if _finite(y)
            ]  # missing σ -> omit
            if not triples:
                continue
            xs, ys, es = zip(*triples)
            _plot_series(ax, list(xs), list(ys), list(es), label=critic)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("σ (confounding strength)")
        ax.set_ylabel("value_mse_to_oracle (mean over seeds)")
        ax.set_title(f"Confounded σ-wedge — {regime} / {env} / {algo}")
        ax.annotate(
            _nc_annotation(_nc_row(nc, env, algo)),
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=8,
        )
        ax.legend(fontsize=8, title="critic")
        written += _save(fig, out, f"sigma_wedge_{_safe(env)}_{_safe(algo)}", formats)
    return written


# --------------------------------------------------------------------------- #
# Figure 3 — pessimism_cost vs σ (sensitivity)                                 #
# --------------------------------------------------------------------------- #
def _fig_pessimism_cost(agg, regime, out, formats) -> List[Path]:
    written: List[Path] = []
    for env, algo in _env_algos(agg):
        rows = sorted(
            (
                r
                for r in agg
                if r["env"] == env
                and r["algo"] == algo
                and r["critic"] == "sensitivity"
                and float(r["beta"]) == 0.0
            ),
            key=lambda r: float(r["sigma"]),
        )
        triples = [
            (
                float(r["sigma"]),
                _num(r.get("pessimism_cost_mean")),
                _num(r.get("pessimism_cost_sd")),
            )
            for r in rows
        ]
        triples = [(x, y, e) for x, y, e in triples if _finite(y)]
        if not triples:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        xs, ys, es = zip(*triples)
        _plot_series(ax, list(xs), list(ys), list(es), color="C4")
        ax.axhline(0.0, color="0.7", lw=0.8, ls="--")
        ax.set_xlabel("σ (confounding strength)")
        ax.set_ylabel("pessimism_cost (sensitivity)")
        ax.set_title(f"Pessimism cost vs σ — {regime} / {env} / {algo}")
        ax.annotate(
            "Γ fixed per cell (library default); a Γ-sweep needs multiple cells",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=7,
        )
        written += _save(
            fig, out, f"pessimism_cost_{_safe(env)}_{_safe(algo)}", formats
        )
    return written


# --------------------------------------------------------------------------- #
# Figure A — fix-critic / vary-algo (base robustness to confounding)           #
# --------------------------------------------------------------------------- #
def _fig_fix_critic_vary_algo(agg, regime, out, formats, *, critic="observational"):
    """value_mse_to_oracle vs σ on the confounded slice (β=0) at a FIXED critic, one
    line per ALGO — the complement of the σ-wedge (wedge fixes algo, varies critic)."""
    written: List[Path] = []
    for env in _envs(agg):
        algos = sorted({r["algo"] for r in agg if r["env"] == env})
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        drew = False
        for algo in algos:
            pts = sorted(
                (
                    r
                    for r in agg
                    if r["env"] == env
                    and r["algo"] == algo
                    and r["critic"] == critic
                    and float(r["beta"]) == 0.0
                ),
                key=lambda r: float(r["sigma"]),
            )
            triples = [
                (
                    float(r["sigma"]),
                    _num(r.get("value_mse_to_oracle_mean")),
                    _num(r.get("value_mse_to_oracle_sd")),
                )
                for r in pts
            ]
            triples = [(x, y, e) for x, y, e in triples if _finite(y)]
            if not triples:
                continue
            xs, ys, es = zip(*triples)
            _plot_series(ax, list(xs), list(ys), list(es), label=algo)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("σ (confounding strength)")
        ax.set_ylabel("value_mse_to_oracle (mean over seeds)")
        ax.set_title(f"Base fragility (critic={critic}) — {regime} / {env}")
        ax.legend(fontsize=8, title="algo")
        written += _save(fig, out, f"fix_critic_vary_algo_{_safe(env)}", formats)
    return written


# --------------------------------------------------------------------------- #
# Figure B — train + evaluation reward across the sweep                        #
# --------------------------------------------------------------------------- #
def _fig_reward_sweep(agg, regime, out, formats):
    """eval_return_mean vs σ on the β=0 slice, one line per critic, with
    train_return_mean drawn as a reference line WHEN it is logged (blank for offline).
    """
    written: List[Path] = []
    for env, algo in _env_algos(agg):
        rows = [
            r
            for r in agg
            if r["env"] == env and r["algo"] == algo and float(r["beta"]) == 0.0
        ]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        drew = False
        for critic in _CRITIC_ORDER:
            pts = sorted(
                (r for r in rows if r["critic"] == critic),
                key=lambda r: float(r["sigma"]),
            )
            triples = [
                (
                    float(r["sigma"]),
                    _num(r.get("eval_return_mean_mean")),
                    _num(r.get("eval_return_mean_sd")),
                )
                for r in pts
            ]
            triples = [(x, y, e) for x, y, e in triples if _finite(y)]
            if not triples:
                continue
            xs, ys, es = zip(*triples)
            _plot_series(ax, list(xs), list(ys), list(es), label=f"eval / {critic}")
            drew = True
        # training/behavior return: ONE reference line (shared base actor), drawn only
        # where it is finite (blank -> NaN for offline algos, so it simply won't appear).
        train_pts = sorted(
            {
                float(r["sigma"]): _num(r.get("train_return_mean_mean")) for r in rows
            }.items()
        )
        train_pts = [(x, y) for x, y in train_pts if _finite(y)]
        if train_pts:
            xs, ys = zip(*train_pts)
            ax.plot(
                list(xs),
                list(ys),
                color="0.4",
                ls="--",
                marker="s",
                label="train / behavior",
            )
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("σ (confounding strength)")
        ax.set_ylabel("return (mean over seeds)")
        ax.set_title(f"Reward across the sweep — {regime} / {env} / {algo}")
        ax.annotate(
            "critic-ablation shares one base actor -> eval lines coincide; "
            "train_return blank for offline",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=7,
        )
        ax.legend(fontsize=8)
        written += _save(fig, out, f"reward_sweep_{_safe(env)}_{_safe(algo)}", formats)
    return written


# --------------------------------------------------------------------------- #
# Figure C — biased-arm coverage vs β (now unblocked by the STEP-1 aggregation) #
# --------------------------------------------------------------------------- #
def _fig_biased_coverage(agg, regime, out, formats, *, critic="observational"):
    """action_coverage vs β on the biased arm (σ=0, β>0). Expect (1-β)·coverage(0)."""
    written: List[Path] = []
    for env, algo in _env_algos(agg):
        rows = sorted(
            (
                r
                for r in agg
                if r["env"] == env
                and r["algo"] == algo
                and r["critic"] == critic
                and float(r["sigma"]) == 0.0  # biased arm holds σ at 0
            ),
            key=lambda r: float(r["beta"]),
        )
        triples = [
            (
                float(r["beta"]),
                _num(r.get("action_coverage_mean")),
                _num(r.get("action_coverage_sd")),
            )
            for r in rows
        ]
        triples = [(x, y, e) for x, y, e in triples if _finite(y)]
        if not triples:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        xs, ys, es = zip(*triples)
        _plot_series(
            ax, list(xs), list(ys), list(es), color="C2", label="action_coverage"
        )
        # the (1-β)·coverage(0) reference, anchored on the β=0 (basic) point.
        base = next((y for x, y, _ in zip(xs, ys, es) if x == 0.0), None)
        if base is not None:
            ref_xs = [x for x in xs]
            ax.plot(
                ref_xs,
                [(1.0 - x) * base for x in ref_xs],
                color="0.6",
                ls="--",
                label="(1-β)·coverage(0)",
            )
        ax.set_xlabel("β (fractional coverage loss)")
        ax.set_ylabel("action_coverage (mean over seeds)")
        ax.set_title(
            f"Biased-arm coverage (critic={critic}) — {regime} / {env} / {algo}"
        )
        ax.legend(fontsize=8)
        written += _save(
            fig, out, f"biased_coverage_{_safe(env)}_{_safe(algo)}", formats
        )
    return written


def render(
    results_root: str | Path, regime: str, *, formats=("png", "pdf")
) -> List[Path]:
    """Render the cell's figures from ``build_report`` output. Returns the file paths."""
    agg, nc = build_report(results_root, regime)  # THE single source of numbers
    out = Path(results_root) / "_report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    written += _fig_sigma_wedge(agg, nc, regime, out, formats)
    written += _fig_pessimism_cost(agg, regime, out, formats)
    written += _fig_fix_critic_vary_algo(agg, regime, out, formats)  # Fig A
    written += _fig_reward_sweep(agg, regime, out, formats)  # Fig B
    written += _fig_biased_coverage(agg, regime, out, formats)  # Fig C
    return written


def _main(argv: List[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Render figures from regime_report output (presentation only; "
        "consumes build_report, never re-aggregates)."
    )
    ap.add_argument("regime")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args(argv)
    written = render(args.results_root, args.regime, formats=tuple(args.formats))
    figdir = Path(args.results_root) / "_report" / "figures"
    print(f"[render_regime_report] wrote {len(written)} figure files under {figdir}/")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Figures for the CLASSICAL simulation's ``results/{regime}/classical/`` tree —
PRESENTATION ONLY.

The classical counterpart of ``render_regime_report``: it CONSUMES
``regime_report.build_classical_report(results_root, regime)`` and plots from that
single source of truth — it never re-walks the tree or re-aggregates. The classical
simulation has NO critic axis, so every figure varies the ALGO (the benchmark
question: which learner wins where, and how does that change across the L?).

Figures written to ``<results_root>/_report/figures/``, PNG + PDF:

  1. {regime}_return_vs_sigma_{env} — ``eval_return_mean`` vs σ on the β=0 slice (basic
     origin + confounded arm), ONE LINE PER ALGO; the training/behavior return is
     drawn as a same-color dashed reference where logged (blank -> NaN for
     offline algos, so it simply won't appear).
  2. {regime}_return_vs_beta_{env} — ``eval_return_mean`` vs β on the σ=0 slice (basic
     origin + biased arm), one line per algo: return degradation under coverage
     loss.
  3. {regime}_classical_coverage_{env} — ``action_coverage_mean`` vs β per algo with the
     (1-β)·coverage(0) reference (the biased arm's mechanism check).

Guards (the 1-seed smoke hits all): an all-NaN ``_sd`` -> plot means, skip error
bars; a missing algo at some x -> omit the point; an empty slice -> skip the figure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402

from src.benchmarking.regime_report import build_classical_report  # noqa: E402


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


def _envs(agg: List[Dict]):
    return sorted({r["env"] for r in agg})


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


def _slice_triples(agg, env, algo, *, axis: str, metric: str):
    """(x, mean, sd) triples for one (env, algo) along one L arm. axis='sigma'
    takes the β=0 slice ordered by σ; axis='beta' the σ=0 slice ordered by β."""
    hold, var = ("beta", "sigma") if axis == "sigma" else ("sigma", "beta")
    pts = sorted(
        (
            r
            for r in agg
            if r["env"] == env and r["algo"] == algo and float(r[hold]) == 0.0
        ),
        key=lambda r: float(r[var]),
    )
    triples = [
        (float(r[var]), _num(r.get(f"{metric}_mean")), _num(r.get(f"{metric}_sd")))
        for r in pts
    ]
    return [(x, y, e) for x, y, e in triples if _finite(y)]


# --------------------------------------------------------------------------- #
# Figure 1 — eval return vs σ, one line per algo                               #
# --------------------------------------------------------------------------- #
def _fig_return_vs_sigma(agg, regime, out, formats) -> List[Path]:
    written: List[Path] = []
    for env in _envs(agg):
        algos = sorted({r["algo"] for r in agg if r["env"] == env})
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        drew = False
        for i, algo in enumerate(algos):
            triples = _slice_triples(
                agg, env, algo, axis="sigma", metric="eval_return_mean"
            )
            if triples:
                xs, ys, es = zip(*triples)
                _plot_series(
                    ax, list(xs), list(ys), list(es), label=algo, color=f"C{i}"
                )
                drew = True
            # training/behavior return as a same-color dashed reference (finite
            # only for the online regimes — offline logs no rollout return).
            train = _slice_triples(
                agg, env, algo, axis="sigma", metric="train_return_mean"
            )
            if train:
                xs, ys, _ = zip(*train)
                ax.plot(list(xs), list(ys), color=f"C{i}", ls="--", alpha=0.6)
                drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("σ (confounding strength)")
        ax.set_ylabel("eval return (mean over seeds)")
        ax.set_title(f"Return vs σ (classical) — {regime} / {env}")
        ax.annotate(
            "dashed = training/behavior return (only where logged: on-policy online)",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=7,
        )
        ax.legend(fontsize=8, title="algo")
        written += _save(fig, out, f"{regime}_return_vs_sigma_{_safe(env)}", formats)
    return written


# --------------------------------------------------------------------------- #
# Figure 2 — eval return vs β, one line per algo                               #
# --------------------------------------------------------------------------- #
def _fig_return_vs_beta(agg, regime, out, formats) -> List[Path]:
    written: List[Path] = []
    for env in _envs(agg):
        algos = sorted({r["algo"] for r in agg if r["env"] == env})
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        drew = False
        for i, algo in enumerate(algos):
            triples = _slice_triples(
                agg, env, algo, axis="beta", metric="eval_return_mean"
            )
            if not triples:
                continue
            xs, ys, es = zip(*triples)
            _plot_series(ax, list(xs), list(ys), list(es), label=algo, color=f"C{i}")
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("β (behavior-policy bias / coverage loss)")
        ax.set_ylabel("eval return (mean over seeds)")
        ax.set_title(f"Return vs β (classical) — {regime} / {env}")
        ax.legend(fontsize=8, title="algo")
        written += _save(fig, out, f"{regime}_return_vs_beta_{_safe(env)}", formats)
    return written


# --------------------------------------------------------------------------- #
# Figure 3 — biased-arm coverage vs β per algo (mechanism check)               #
# --------------------------------------------------------------------------- #
def _fig_coverage(agg, regime, out, formats) -> List[Path]:
    written: List[Path] = []
    for env in _envs(agg):
        algos = sorted({r["algo"] for r in agg if r["env"] == env})
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        drew = False
        ref_drawn = False
        for i, algo in enumerate(algos):
            triples = _slice_triples(
                agg, env, algo, axis="beta", metric="action_coverage"
            )
            if not triples:
                continue
            xs, ys, es = zip(*triples)
            _plot_series(ax, list(xs), list(ys), list(es), label=algo, color=f"C{i}")
            drew = True
            # the (1-β)·coverage(0) reference, anchored on the first algo's β=0
            # point (the mechanism is policy-level, shared across learners).
            if not ref_drawn:
                base = next((y for x, y, _ in triples if x == 0.0), None)
                if base is not None:
                    ax.plot(
                        list(xs),
                        [(1.0 - x) * base for x in xs],
                        color="0.6",
                        ls="--",
                        label="(1-β)·coverage(0)",
                    )
                    ref_drawn = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("β (fractional coverage loss)")
        ax.set_ylabel("action_coverage (mean over seeds)")
        ax.set_title(f"Biased-arm coverage (classical) — {regime} / {env}")
        ax.legend(fontsize=8)
        written += _save(fig, out, f"{regime}_classical_coverage_{_safe(env)}", formats)
    return written


def render(
    results_root: str | Path, regime: str, *, formats=("png", "pdf")
) -> List[Path]:
    """Render the classical figures from ``build_classical_report`` output."""
    agg = build_classical_report(results_root, regime)  # THE single source
    out = Path(results_root) / "_report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    written += _fig_return_vs_sigma(agg, regime, out, formats)
    written += _fig_return_vs_beta(agg, regime, out, formats)
    written += _fig_coverage(agg, regime, out, formats)
    return written


def _main(argv: List[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Render figures for the classical simulation's "
        "results/{regime}/classical/ tree (presentation only; consumes "
        "build_classical_report, never re-aggregates)."
    )
    ap.add_argument("regime")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args(argv)
    written = render(args.results_root, args.regime, formats=tuple(args.formats))
    figdir = Path(args.results_root) / "_report" / "figures"
    print(
        f"[render_classical_report] wrote {len(written)} figure files under {figdir}/"
    )
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

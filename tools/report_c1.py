"""C1 — the observability-contract grid's report (Phases 4 and 6), WITHIN-COLUMN.

Reads every leaf under a campaign root (default results/c1) through its
``e1_provenance.json`` (cell, arm, algo, dataset seed, training seed, critic,
dataset id, the served GRACE record) plus the leaf's final ``eval_deployment``
and ``critic_ablation_metrics`` rows, and writes:

  * per column (truth): the return table per (cell, algo) — IQM over leaves
    with every leaf listed (never pooled with abstentions);
  * the critic-accuracy table (``q1_contrast_error`` / ``q1_contrast_pred``)
    per comparator — observational (floor), proximal, oracle_u (ceiling),
    sensitivity, and GRACE (declared MDP / declared POMDP) — with the
    comparators a cell EXCLUDED and why (L2);
  * the window table for the declared-POMDP cells: selected-k distribution,
    materiality margins (stage deltas vs L4 half-widths), the sufficient? /
    necessary? diagnostics where k was supplied;
  * the L5 record per cell (p, dR2, capacity shrink, base R^2, scale_invalid);
  * abstentions, in their OWN table, with reasons.

Cross-column comparisons are NOT produced: the two truths are never lined up
on a shared axis (contract plan §5). Predictions to read against: the
handoff's "PHASE 4 PRE-REGISTRATION".

    uv run python tools/report_c1.py [--root results/c1] [--out results/c1/report]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXCLUDED = {
    # (truth, critic) -> the L2 reason it is not run on that column
    ("tpomdp", "proximal"): "L2: on the masked construction (POMDP + U, the D-G "
    "shape) proximal's q1 verdict is bounds-only and q2 non-ID",
}
CRITIC_ORDER = (
    "oracle_u",
    "proximal",
    "grace_dmdp",
    "grace_dpomdp",
    "sensitivity",
    "observational",
)


def _last_row(path: Path, **match) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    for k, v in match.items():
        rows = [r for r in rows if r.get(k) == v]
    return rows[-1] if rows else None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _iqm(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if xs.size == 0:
        return float("nan")
    if xs.size < 4:
        return float(np.mean(xs))
    lo, hi = np.percentile(xs, [25, 75])
    mid = xs[(xs >= lo) & (xs <= hi)]
    return float(np.mean(mid)) if mid.size else float(np.mean(xs))


def _truth_of(cell: str) -> str:
    return "tpomdp" if "tpomdp" in cell else "tmdp"


def collect(root: Path) -> list:
    leaves = []
    for prov in sorted(root.rglob("e1_provenance.json")):
        leaf = prov.parent
        p = json.loads(prov.read_text())
        cell = str(p.get("cell", ""))
        arm = str(p.get("arm", ""))
        critic = str(p.get("critic", "observational"))
        grace = p.get("grace") or {}
        meta = (grace.get("meta") or {}) if isinstance(grace, dict) else {}
        dep = _last_row(leaf / "eval_deployment.csv") or {}
        crit = (
            _last_row(leaf / "critic_ablation_metrics.csv", critic=critic)
            or _last_row(leaf / "critic_ablation_metrics.csv")
            or {}
        )
        comparator = (
            critic
            if arm != "grace"
            else ("grace_dpomdp" if "dpomdp" in cell else "grace_dmdp")
        )
        leaves.append(
            dict(
                leaf=str(leaf),
                cell=cell,
                truth=_truth_of(cell),
                arm=arm,
                comparator=comparator,
                algo=str(p.get("algo", "")),
                ds=p.get("seed"),
                ts=p.get("train_seed", p.get("seed")),
                dataset_id=p.get("dataset_id"),
                abstained=(
                    bool(grace.get("abstained", False)) if arm == "grace" else False
                ),
                grace_label=grace.get("label", "") if isinstance(grace, dict) else "",
                grace_reason=(
                    (grace.get("reason") or meta.get("reason") or "")
                    if arm == "grace"
                    else ""
                ),
                ret=_f(dep.get("eval_return_mean")),
                ret_base=_f(dep.get("eval_return_base_mean")),
                bad_steps=_f(dep.get("eval_bad_action_steps_mean")),
                q1_err=_f(crit.get("q1_contrast_error")),
                q1_pred=_f(crit.get("q1_contrast_pred")),
                vmse=_f(crit.get("value_mse_to_oracle")),
                window_k=meta.get("window_k"),
                window_source=meta.get("window_source"),
                window_sufficient=meta.get("window_sufficient"),
                window_necessary=meta.get("window_necessary"),
                stage0_delta=_f(meta.get("window_stage0_delta")),
                stage0_w=_f(meta.get("window_stage0_w")),
                stage1_delta=_f(meta.get("window_stage1_delta")),
                stage1_w=_f(meta.get("window_stage1_w")),
                l5_p=_f(meta.get("l5_p")),
                l5_dr2=_f(meta.get("l5_dr2")),
                l5_shrink=_f(meta.get("l5_shrink")),
                l5_rejected=meta.get("l5_rejected"),
                l5_base_r2=meta.get("l5_base_r2", ""),
                l5_scale_invalid=meta.get("l5_scale_invalid", ""),
                q1_truth=p.get("q1_truth"),
            )
        )
    return leaves


def _md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render(leaves: list, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    served = [x for x in leaves if not x["abstained"]]
    abst = [x for x in leaves if x["abstained"]]
    md, rep = [], {"n_leaves": len(leaves), "columns": {}}
    md.append("# C1 — the observability-contract grid (WITHIN-COLUMN)\n")
    md.append(
        f"{len(leaves)} leaves; {len(abst)} abstained (own table below, never pooled).\n"
    )
    for truth in ("tmdp", "tpomdp"):
        col = [x for x in served if x["truth"] == truth]
        if not col:
            continue
        md.append(
            f"\n## Column: true {'MDP' if truth == 'tmdp' else 'POMDP'} ({truth})\n"
        )
        rep["columns"][truth] = {}
        # --- return per (cell, algo)
        rows = []
        by = defaultdict(list)
        for x in col:
            if x["comparator"] in ("observational", "grace_dmdp", "grace_dpomdp"):
                by[(x["cell"], x["algo"], x["comparator"])].append(x)
        for (cell, algo, comp), xs in sorted(by.items()):
            rets = [x["ret"] for x in xs]
            rows.append(
                (
                    cell,
                    algo,
                    comp,
                    len(xs),
                    f"{_iqm(rets):.1f}",
                    " ".join(f"{r:.0f}" for r in rets),
                    f"{_iqm([x['bad_steps'] for x in xs]):.1f}",
                )
            )
            rep["columns"][truth][f"return/{cell}/{algo}/{comp}"] = dict(
                n=len(xs), iqm=_iqm(rets), leaves=rets
            )
        md.append(
            "### Return (deployment, analytic E_U) — IQM over served leaves, every leaf listed\n"
        )
        md.append(
            _md_table(
                ("cell", "algo", "arm", "n", "IQM", "leaves", "bad-action steps IQM"),
                rows,
            )
        )
        # --- critic accuracy per comparator
        rows = []
        byc = defaultdict(list)
        for x in col:
            byc[(x["algo"], x["comparator"])].append(x)
        for algo in sorted({x["algo"] for x in col}):
            for comp in CRITIC_ORDER:
                xs = byc.get((algo, comp))
                if not xs:
                    reason = EXCLUDED.get((truth, comp))
                    if reason:
                        rows.append((algo, comp, "EXCLUDED", "", "", reason))
                    continue
                errs = [x["q1_err"] for x in xs]
                rows.append(
                    (
                        algo,
                        comp,
                        len(xs),
                        f"{_iqm(errs):.4f}",
                        f"{_iqm([x['q1_pred'] for x in xs]):.4f}",
                        " ".join(f"{e:.3f}" for e in errs),
                    )
                )
                rep["columns"][truth][f"q1_err/{algo}/{comp}"] = dict(
                    n=len(xs), iqm=_iqm(errs), leaves=errs
                )
        md.append(
            "\n### Critic accuracy — q1_contrast_error (oracle_u = ceiling, observational = floor)\n"
        )
        md.append(
            _md_table(
                (
                    "algo",
                    "comparator",
                    "n",
                    "|error| IQM",
                    "pred IQM",
                    "leaves / reason",
                ),
                rows,
            )
        )
        # --- windows (declared POMDP + the supplied-k diagnostics)
        rows = []
        for x in sorted(
            col, key=lambda z: (z["cell"], z["algo"], str(z["ds"]), str(z["ts"]))
        ):
            if x["arm"] != "grace":
                continue
            rows.append(
                (
                    x["cell"],
                    x["algo"],
                    f"ds{x['ds']}_ts{x['ts']}",
                    x["window_source"],
                    x["window_k"],
                    f"{x['stage0_delta']:.4f}/{x['stage0_w']:.4f}",
                    (
                        f"{x['stage1_delta']:.4f}/{x['stage1_w']:.4f}"
                        if np.isfinite(x["stage1_delta"])
                        else ""
                    ),
                    x["window_sufficient"],
                    x["window_necessary"],
                )
            )
        kd = defaultdict(lambda: defaultdict(int))
        for x in col:
            if x["arm"] == "grace" and x["window_source"] == "selected":
                kd[x["cell"]][str(x["window_k"])] += 1
        md.append(
            "\n### Windows — selected k, materiality margins (stage delta / L4 half-width), supplied-k diagnostics\n"
        )
        md.append(
            _md_table(
                (
                    "cell",
                    "algo",
                    "leaf",
                    "source",
                    "k",
                    "stage0 Δ/w",
                    "stage1 Δ/w",
                    "sufficient?",
                    "necessary?",
                ),
                rows,
            )
        )
        if kd:
            md.append(
                "\nSelected-k distribution (delegated selection): "
                + "; ".join(f"{c}: {dict(v)}" for c, v in kd.items())
            )
            rep["columns"][truth]["selected_k"] = {c: dict(v) for c, v in kd.items()}
        # --- L5 record
        rows = []
        for x in sorted(
            col, key=lambda z: (z["cell"], z["algo"], str(z["ds"]), str(z["ts"]))
        ):
            if x["arm"] != "grace":
                continue
            rows.append(
                (
                    x["cell"],
                    x["algo"],
                    f"ds{x['ds']}_ts{x['ts']}",
                    f"{x['l5_p']:.3f}",
                    f"{x['l5_dr2']:.2e}",
                    f"{x['l5_shrink']:.2f}",
                    x["l5_rejected"],
                    x["l5_scale_invalid"] or "-",
                )
            )
        md.append(
            "\n### L5 record at the served lag (report-only; on true-MDP data rejection at floor dR2 with shrink > 1 is S18, not a defect)\n"
        )
        md.append(
            _md_table(
                (
                    "cell",
                    "algo",
                    "leaf",
                    "p",
                    "dR2",
                    "shrink",
                    "rejected",
                    "scale_invalid",
                ),
                rows,
            )
        )
    # --- abstentions
    md.append(
        "\n## Abstentions (fit health, L4 family) — reported separately, never pooled\n"
    )
    if abst:
        md.append(
            _md_table(
                ("cell", "algo", "leaf", "reason"),
                [
                    (
                        x["cell"],
                        x["algo"],
                        f"ds{x['ds']}_ts{x['ts']}",
                        x["grace_reason"] or x["grace_label"],
                    )
                    for x in abst
                ],
            )
        )
    else:
        md.append("none")
    rep["abstentions"] = [
        dict(
            cell=x["cell"],
            algo=x["algo"],
            ds=x["ds"],
            ts=x["ts"],
            reason=x["grace_reason"] or x["grace_label"],
        )
        for x in abst
    ]
    rep["excluded"] = {f"{t}/{c}": r for (t, c), r in EXCLUDED.items()}
    (out / "c1_report.md").write_text("\n".join(md) + "\n")
    (out / "c1_report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "c1_leaves.json").write_text(json.dumps(leaves, indent=1, default=str))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/c1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    leaves = collect(root)
    out = Path(args.out) if args.out else root / "report"
    rep = render(leaves, out)
    print(f"{len(leaves)} leaves -> {out}/c1_report.md")
    print(
        json.dumps(
            {k: v for k, v in rep.items() if k != "columns"}, indent=1, default=str
        )[:2000]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

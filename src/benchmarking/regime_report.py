"""PR 6 — the reporting reader for the (regime × L-sweep) ``results/`` tree.

PR 5 writes ``results/{regime}/beta_{bbb}_sigma_{sss}/{env}/{algo}/{critic}/{seed}/``;
the legacy reporting layer (``plotting.py``) is anchored on ``runs/`` and parses run
identity from the FOLDER NAME. This module is the ADDITIVE reader for the new tree —
a path-MODEL change, not a name tweak:

  * identity comes from PATH SEGMENTS, not a folder-name regex (parse_results_leaf);
  * a σ-sweep's siblings live under DIFFERENT ``beta_*`` parents, so they are
    collected by walking the regime subtree and grouping on (env, algo, critic, seed)
    (collect_sigma_siblings) rather than a co-located ``confounded_sigma_*`` glob;
  * {basic, biased, confounded} labels are DERIVED from (β, σ) (regime_sweep.arm_label
    / reslice_results) — no label is ever stored in a path;
  * the new tree has a SEED axis the old one lacked, so everything aggregates per
    (regime, β, σ, env, algo, critic) across seeds (mean + across-seed sd);
  * null_calibration is a RELATIVE, seed-based, CELL-level property computed HERE
    (N1) — never the broken absolute per-run column.

The legacy ``runs/`` name-parsing stays untouched; ``plotting.py:_sigma_from_run``
gained a segment-first dispatch so it serves BOTH trees.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.benchmarking.regime_sweep import arm_label, parse_param_dir

_PARAM_RE = re.compile(r"beta_\d{3}_sigma_\d{3}")
ADAPTIVE_CRITICS = ("observational", "proximal", "oracle_u")

# Null-calibration factor (PR 6 follow-up 2, feat/null-cal-fixed-denominator).
# null_calibrated = gap < k * noise_ref, where noise_ref is the CORRECT pipeline's
# basic-point seed-sd — a STORED constant per (env, algo), see null_cal_reference.yaml —
# NOT the judged cell's OWN noise. The bare-DQN base-confound inflates the judged cell's
# seed variance almost as fast as the gap, so a gap/(cell-noise) gate greenlit the very
# confound it exists to catch; a fixed reference denominator separates the endpoints.
#
# k is a BUDGET-DEPENDENT calibration constant: it is re-derived (log-halfway between the
# correct and broken gap/noise_ref endpoints) whenever the endpoints move — exactly like
# noise_ref itself. RE-PINNED 2.4 -> 1.5 on 2026-07-21 (feat/repin-k-v3) for the NEW
# offline budget (offline_grad_steps=50_000, rollout_episodes=3000; endpoints measured
# CartPole-v1, 5 seeds):
#   cql: correct gap/noise_ref = 0.72   broken (obs=bare DQN vs CQL oracle) = 2.95
#   iql: correct = 0.58                 broken = 1784.6   (non-binding: huge margin)
# The BINDING interval is cql's (0.72, 2.95); log-halfway = sqrt(0.72*2.95) = 1.46 -> 1.5.
# WHY NOT keep 2.4: it was log-halfway of the OLD budget's endpoints (1.02, 5.75). At the
# new budget it sits only 23% below the broken endpoint (2.95), so a ~25% error in
# noise_ref (n=5) drops the broken ratio to ~2.36 and the confound PASSES. At k=1.5 both
# cql endpoints clear by ~2x (0.72/1.5 = 0.48; 2.95/1.5 = 1.97) and survive that
# uncertainty. NB the cql margin is intrinsically thin at this budget (see
# null_cal_reference.yaml) — k=1.5 is the best split of it, not a comfortable one.
NULL_CALIBRATION_K = 1.5

# The stored correct-pipeline reference denominators, keyed by (env, algo). A MISSING
# key -> the gate returns "uncalibrated" (None), never a silent pass.
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "reproducibility"
    / "rl_regimes"
    / "_base"
    / "null_cal_reference.yaml"
)


def load_null_cal_reference(
    path: str | Path | None = None,
) -> Dict[Tuple[str, str], float]:
    """Load the stored ``noise_ref`` (correct-pipeline basic-point pooled seed-sd of
    the adaptive critics), keyed by (env, algo). Returns {} if the file is absent — a
    missing reference must yield an UNCALIBRATED verdict, not a silent pass."""
    p = Path(path) if path is not None else _REFERENCE_PATH
    if not p.exists():
        return {}
    doc = yaml.safe_load(p.read_text()) or {}
    out: Dict[Tuple[str, str], float] = {}
    for env, by_algo in (doc.get("reference", {}) or {}).items():
        for algo, val in (by_algo or {}).items():
            if val is not None:
                out[(str(env), str(algo))] = float(val)
    return out


# --------------------------------------------------------------------------- #
# CHANGE 1 — parse SEGMENTS, not folder-name strings                          #
# --------------------------------------------------------------------------- #
def parse_results_leaf(path: str | Path) -> Dict:
    """A ``results/`` run-dir leaf path -> its identity, from PATH SEGMENTS:

        .../{regime}/beta_{bbb}_sigma_{sss}/{env}/{algo}/{critic}/{seed}[/...]

    The subcell label is DERIVED from (β, σ), never read from a segment (there is
    none to read). Raises ValueError if no ``beta_*_sigma_*`` segment is present."""
    parts = Path(path).parts
    idx = next((i for i, seg in enumerate(parts) if _PARAM_RE.fullmatch(seg)), None)
    if idx is None or idx == 0 or idx + 4 >= len(parts) + 1:
        raise ValueError(f"not a results/ leaf (no beta_*_sigma_* segment): {path!r}")
    beta, sigma = parse_param_dir(parts[idx])
    tail = parts[idx + 1 : idx + 5]  # env, algo, critic, seed
    if len(tail) < 4:
        raise ValueError(f"results/ leaf is missing env/algo/critic/seed: {path!r}")
    env, algo, critic, seed = tail
    return {
        "regime": parts[idx - 1],
        "beta": beta,
        "sigma": sigma,
        "arm": arm_label(beta, sigma),
        "env": env,
        "algo": algo,
        "critic": critic,
        "seed": int(seed),
        "path": str(Path(path)),
    }


def sigma_from_leaf(path: str | Path) -> Optional[float]:
    """σ from the ``beta_*_sigma_*`` PATH SEGMENT (the new-tree analogue of
    plotting._sigma_from_run's folder-name regex). None if absent."""
    for seg in Path(path).parts:
        if _PARAM_RE.fullmatch(seg):
            return parse_param_dir(seg)[1]
    return None


def iter_leaves(results_root: str | Path, regime: str) -> List[Dict]:
    """Every run-dir leaf under ``results/{regime}`` (a dir holding config.yaml),
    parsed to its segment identity."""
    root = Path(results_root) / regime
    out: List[Dict] = []
    if not root.is_dir():
        return out
    for cfg in sorted(root.rglob("config.yaml")):
        try:
            out.append(parse_results_leaf(cfg.parent))
        except ValueError:
            continue
    return out


def collect_sigma_siblings(
    results_root: str | Path, regime: str
) -> Dict[Tuple[str, str, str, int], Dict[float, str]]:
    """The σ-sweep, collected the NEW way (CHANGE 1). In ``runs/`` the σ variants
    were flat co-located siblings matched by a ``confounded_sigma_*`` name glob; in
    ``results/`` they live under DIFFERENT ``beta_*`` parents, so we walk the regime
    subtree, group by the (env, algo, critic, seed) tuple, and collect the siblings
    across those parents.

    The σ-sweep is the β=0 slice (the basic origin σ=0 + the confounded arm σ>0);
    the biased arm (β>0) is a separate axis and is not part of a σ-sweep. Returns
    ``{(env, algo, critic, seed): {sigma: leaf_path}}``."""
    groups: Dict[Tuple[str, str, str, int], Dict[float, str]] = {}
    for leaf in iter_leaves(results_root, regime):
        if leaf["beta"] != 0.0:  # σ-sweep is the β=0 slice
            continue
        key = (leaf["env"], leaf["algo"], leaf["critic"], leaf["seed"])
        groups.setdefault(key, {})[leaf["sigma"]] = leaf["path"]
    return groups


# --------------------------------------------------------------------------- #
# Metric reading + CHANGE 3 seed aggregation                                  #
# --------------------------------------------------------------------------- #
# Which per-leaf CSV each report metric is read from. The critic-ablation strategy
# schema (value_mse_to_oracle, apparent_q_mean, ...) is per-critic SLICED; the return
# and arm-diagnostics metrics live in SHARED per-leaf files copied unchanged onto every
# critic leaf (regime_sweep._run_point.shared_files), so a per-(env,algo,critic) cell
# still resolves them — the base actor is shared across critics, so those values are
# identical across the critic siblings of one (env, algo, σ, seed) point. A column not
# named here defaults to critic_ablation_metrics.csv (the existing behavior).
_METRIC_SOURCE_FILE: Dict[str, str] = {
    # evaluation return (the learned policy rolled out) — eval_metrics.csv
    "eval_return_mean": "eval_metrics.csv",
    "eval_return_std": "eval_metrics.csv",
    # training/behavior return — train_metrics.csv. NB: BLANK for offline algos (the
    # offline training loop logs no per-episode rollout return); populated only for the
    # online regimes. Aggregates to NaN where blank, which the renderer guards handle.
    "train_return_mean": "train_metrics.csv",
    "train_return_std": "train_metrics.csv",
    # per-checkpoint arm diagnostics — arm_diagnostics.csv (action_coverage is the
    # biased arm's metric; (1-β)·coverage(0) by construction).
    "action_coverage": "arm_diagnostics.csv",
    "separability": "arm_diagnostics.csv",
    "action_overlap": "arm_diagnostics.csv",
    "intervened_mean": "arm_diagnostics.csv",
}


def _read_last_checkpoint(csv_path: Path, column: str) -> Optional[float]:
    """The last-checkpoint (max ``episode``) value of ``column`` from a per-leaf CSV.
    None if the file/column is absent or the value is blank."""
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    rows = [r for r in rows if r.get(column, "") not in ("", None)]
    if not rows:
        return None
    last = max(rows, key=lambda r: int(r["episode"]))
    try:
        return float(last[column])
    except (ValueError, KeyError):
        return None


def read_critic_metric(leaf: str | Path, column: str) -> Optional[float]:
    """The last-checkpoint value of ``column`` from a leaf's sliced
    ``critic_ablation_metrics.csv`` (one critic per leaf). None if absent/blank."""
    return _read_last_checkpoint(Path(leaf) / "critic_ablation_metrics.csv", column)


def read_leaf_metric(leaf: str | Path, column: str) -> Optional[float]:
    """The last-checkpoint value of ``column`` from whichever per-leaf CSV owns it
    (``_METRIC_SOURCE_FILE`` dispatch; default critic_ablation_metrics.csv). Additive
    reader for the return / arm-diagnostics columns the new figures need — the file
    walker/aggregator is unchanged, only the source file per column is resolved."""
    filename = _METRIC_SOURCE_FILE.get(column, "critic_ablation_metrics.csv")
    return _read_last_checkpoint(Path(leaf) / filename, column)


def _mean_sd(values: List[float]) -> Tuple[float, float, int]:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan"), n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)  # sample sd (ddof=1)
    return mean, math.sqrt(var), n


def aggregate_over_seeds(
    results_root: str | Path,
    regime: str,
    metrics: Tuple[str, ...] = ("value_mse_to_oracle",),
) -> List[Dict]:
    """Per (regime, β, σ, arm, env, algo, critic): mean + across-seed sd of each
    metric (CHANGE 3). The seed axis is what the new tree adds and what the relative
    null-calibration gate needs. Returns one record per cell (seeds collapsed)."""
    cells: Dict[Tuple, Dict[str, List[float]]] = {}
    seeds_seen: Dict[Tuple, set] = {}
    for leaf in iter_leaves(results_root, regime):
        key = (
            leaf["regime"],
            leaf["beta"],
            leaf["sigma"],
            leaf["env"],
            leaf["algo"],
            leaf["critic"],
        )
        seeds_seen.setdefault(key, set()).add(leaf["seed"])
        bucket = cells.setdefault(key, {m: [] for m in metrics})
        for m in metrics:
            bucket[m].append(read_leaf_metric(leaf["path"], m))
    out: List[Dict] = []
    for key, bucket in sorted(cells.items(), key=lambda kv: str(kv[0])):
        regime_, beta, sigma, env, algo, critic = key
        rec = {
            "regime": regime_,
            "beta": beta,
            "sigma": sigma,
            "arm": arm_label(beta, sigma),  # DERIVED (CHANGE 2)
            "env": env,
            "algo": algo,
            "critic": critic,
            # n_seeds = the cell's seed count (leaves present), NOT a per-metric
            # non-null count — a metric that is legitimately blank for this critic
            # (e.g. pessimism_cost on an adaptive critic) must not zero it.
            "n_seeds": len(seeds_seen.get(key, ())),
        }
        for m in metrics:
            mean, sd, n = _mean_sd(bucket[m])
            rec[f"{m}_mean"] = mean
            rec[f"{m}_sd"] = sd
            rec[f"{m}_n"] = n  # per-metric non-null count (metric may be blank)
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# CHANGE 4 (N1) — RELATIVE, seed-based, CELL-level null_calibrated             #
# --------------------------------------------------------------------------- #
def _pooled_seed_sd(by_critic: Dict[str, Dict[int, float]]) -> float:
    """Pooled across-seed sd of MSE over the two NON-degenerate adaptive critics
    (observational, proximal). oracle_u is excluded — it scores against itself so its
    MSE is identically 0 (no seed noise)."""
    variances = []
    for c in ("observational", "proximal"):
        _, sd, n = _mean_sd(list(by_critic.get(c, {}).values()))
        if n >= 2 and not math.isnan(sd):
            variances.append(sd * sd)
    return math.sqrt(sum(variances) / len(variances)) if variances else float("nan")


def compute_null_calibration(
    results_root: str | Path,
    regime: str,
    *,
    k: float = NULL_CALIBRATION_K,
    metric: str = "value_mse_to_oracle",
    reference: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[Dict]:
    """Null-calibration on a FIXED REFERENCE DENOMINATOR (PR 6 follow-up 2).

    At the BASIC point (β=0, σ=0), per (env, algo), over seeds::

        gap             = | mean_seeds(MSE_obs) - mean_seeds(MSE_prox) |
        null_calibrated = gap < k * noise_ref        (ratio = gap / noise_ref)

    ``noise_ref`` is the CORRECT pipeline's basic-point seed-sd — a STORED constant per
    (env, algo) (``null_cal_reference.yaml``), NOT recomputed from the cell under
    judgment. That is the whole fix: the bare-DQN base-confound inflates the JUDGED
    cell's noise almost as fast as the gap, so a gap/(cell-noise) gate greenlit the
    confound (broken ratio 1.58 PASSED k=2.0). With a fixed reference the endpoints
    separate — correct gap/noise_ref ~= 1, broken ~= 5.75 — and k=2.4 splits them.

    A MISSING (env, algo) reference -> ``null_calibrated = None`` (UNCALIBRATED): a
    gate with no reference must not verdict. The judged cell's own pooled seed-sd is
    still logged as ``cell_noise`` (diagnostic — the quantity the finding shows moves
    with the defect), but it NEVER enters the verdict."""
    ref = load_null_cal_reference() if reference is None else reference
    per: Dict[Tuple[str, str], Dict[str, Dict[int, float]]] = {}
    for leaf in iter_leaves(results_root, regime):
        if leaf["arm"] != "basic":
            continue
        if leaf["critic"] not in ADAPTIVE_CRITICS:
            continue
        v = read_critic_metric(leaf["path"], metric)
        if v is None:
            continue
        per.setdefault((leaf["env"], leaf["algo"]), {}).setdefault(leaf["critic"], {})[
            leaf["seed"]
        ] = v

    out: List[Dict] = []
    for (env, algo), by_critic in sorted(per.items()):
        obs = list(by_critic.get("observational", {}).values())
        prox = list(by_critic.get("proximal", {}).values())
        obs_mean, _, n_obs = _mean_sd(obs)
        prox_mean, _, n_prox = _mean_sd(prox)
        gap = abs(obs_mean - prox_mean) if (n_obs and n_prox) else float("nan")
        cell_noise = _pooled_seed_sd(by_critic)  # diagnostic ONLY (moves with defect)
        noise_ref = ref.get((env, algo))
        if noise_ref is None or noise_ref <= 0 or math.isnan(gap):
            # UNCALIBRATED — no reference (or no gap): never a silent True.
            ratio: Optional[float] = None
            calibrated: Optional[bool] = None
        else:
            ratio = gap / noise_ref
            calibrated = bool(gap < k * noise_ref)
        out.append(
            {
                "regime": regime,
                "env": env,
                "algo": algo,
                "k": k,
                "gap": gap,
                "noise_ref": noise_ref,  # the FIXED gate denominator (stored)
                "cell_noise": cell_noise,  # diagnostic — NOT the denominator
                "ratio": ratio,  # gap / noise_ref
                "null_calibrated": calibrated,  # None = uncalibrated (no reference)
                "n_seeds": max(n_obs, n_prox),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# The aggregated report (CHANGE 2/3 + N1)                                      #
# --------------------------------------------------------------------------- #
_REPORT_METRICS = (
    "value_mse_to_oracle",
    "apparent_q_mean",
    "gap_closed_fraction",
    "pessimism_cost",
    # Return + coverage columns for Figs B/C (additive; read from the SHARED per-leaf
    # eval/train/arm_diagnostics files, not the critic slice). eval_return_mean is the
    # learned-policy rollout return; train_return_mean is the training/behavior return
    # (blank → NaN for offline algos); action_coverage is the biased arm's metric.
    "eval_return_mean",
    "train_return_mean",
    "action_coverage",
)


def build_report(
    results_root: str | Path, regime: str, *, k: float = NULL_CALIBRATION_K
) -> Tuple[List[Dict], List[Dict]]:
    """Return ``(aggregated_table, null_calibration_table)`` for a regime's
    ``results/`` tree: the per-cell mean+sd table with DERIVED arm labels, and the
    relative cell-level null-calibration verdict per (env, algo)."""
    return (
        aggregate_over_seeds(results_root, regime, _REPORT_METRICS),
        compute_null_calibration(results_root, regime, k=k),
    )


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _main(argv: List[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Aggregate a regime's results/ tree into a per-cell table + the "
        "relative cell-level null-calibration verdict (labels DERIVED from β/σ)."
    )
    ap.add_argument("regime")
    ap.add_argument("--results-root", default="results")
    ap.add_argument(
        "--out", default=None, help="output dir (default: <results-root>/_report)"
    )
    ap.add_argument(
        "--k",
        type=float,
        default=NULL_CALIBRATION_K,
        help=f"null-calibration factor (default {NULL_CALIBRATION_K}, see regime_report)",
    )
    args = ap.parse_args(argv)
    agg, nc = build_report(args.results_root, args.regime, k=args.k)
    out = Path(args.out) if args.out else Path(args.results_root) / "_report"
    _write_csv(agg, out / f"{args.regime}_aggregated.csv")
    _write_csv(nc, out / f"{args.regime}_null_calibration.csv")
    print(f"[regime_report] {len(agg)} cells, {len(nc)} null-cal rows -> {out}")
    for r in nc:
        nref = "MISSING" if r["noise_ref"] is None else f"{r['noise_ref']:.4g}"
        ratio = "n/a" if r["ratio"] is None else f"{r['ratio']:.4g}"
        print(
            f"  {r['env']}/{r['algo']}: gap={r['gap']:.4g} noise_ref={nref} "
            f"cell_noise={r['cell_noise']:.4g} ratio={ratio} "
            f"null_calibrated={r['null_calibrated']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

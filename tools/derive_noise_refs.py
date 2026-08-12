"""R-N1.1 — derive missing noise_ref values from E1's production basic-point
runs, under the review's guardrails.

Guardrails (all enforced here):
  1. A derived reference is computed EXCLUSIVELY from the observational-vs-
     proximal MSE distribution at the basic point (the same pooled-seed-sd
     convention as ``regime_report._pooled_seed_sd``) — never from any grace
     arm: the judged statistic (obs-vs-grace gap) and its reference never
     share an arm.
  2. Derived references share seeds/datasets/budget with the judged runs —
     every verdict resting on one is PROVISIONAL (coupled reference); the
     output yaml says so and report tables must carry the marking.
  3. Cross-check: for pairs with stored INDEPENDENT references, the ratio
     derived/stored must fall in [0.5, 2.0] for every such pair — otherwise
     ALL derived references are discarded and the missing pairs report
     UNCALIBRATED (never a silent pass).
  4. The cross-check doubles as the critics-only-vs-production open item in
     null_cal_reference.yaml; its outcome is recorded in the output header.

Output: ``reproducibility/rl_regimes/_base/null_cal_reference_derived.yaml``
(a SEPARATE file — the stored independent references are never edited).

Usage:
    uv run python tools/derive_noise_refs.py offline_mdp [--results-root results]
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import yaml

_BASE = Path(__file__).resolve().parents[1] / "reproducibility" / "rl_regimes" / "_base"
_OUT = _BASE / "null_cal_reference_derived.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("regime")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    from src.benchmarking.regime_report import (
        iter_leaves,
        load_null_cal_reference,
        read_critic_metric,
    )

    # Guardrail 1: obs + proximal ONLY.
    per: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for leaf in iter_leaves(args.results_root, args.regime):
        if leaf["arm"] != "basic" or leaf["critic"] not in (
            "observational",
            "proximal",
        ):
            continue
        v = read_critic_metric(leaf["path"], "value_mse_to_oracle")
        if v is not None:
            per[(leaf["env"], leaf["algo"])][leaf["critic"]][leaf["seed"]] = v

    def _sd(vals: list[float]) -> float:
        n = len(vals)
        if n < 2:
            return float("nan")
        mean = sum(vals) / n
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))

    derived: dict[tuple[str, str], float] = {}
    for (env, algo), by_critic in sorted(per.items()):
        variances = []
        for critic in ("observational", "proximal"):
            sd = _sd(list(by_critic.get(critic, {}).values()))
            if not math.isnan(sd):
                variances.append(sd * sd)
        if variances:
            derived[(env, algo)] = math.sqrt(sum(variances) / len(variances))

    stored = load_null_cal_reference()
    checks = []
    ok = True
    for pair, stored_ref in sorted(stored.items()):
        if pair in derived and stored_ref and stored_ref > 0:
            ratio = derived[pair] / stored_ref
            in_band = 0.5 <= ratio <= 2.0
            ok = ok and in_band
            checks.append(
                {
                    "env": pair[0],
                    "algo": pair[1],
                    "stored": float(stored_ref),
                    "derived": float(derived[pair]),
                    "ratio": float(ratio),
                    "in_band": bool(in_band),
                }
            )
    if not checks:
        ok = False  # no cross-check possible -> not trustworthy (guardrail 3)

    missing_pairs = {p for p in derived if p not in stored}
    print(
        f"[derive-noise-ref] {args.regime}: derived {len(derived)} pairs, "
        f"{len(missing_pairs)} missing from the stored reference"
    )
    for c in checks:
        print(
            f"[derive-noise-ref] cross-check {c['env']}/{c['algo']}: "
            f"derived {c['derived']:.4g} / stored {c['stored']:.4g} = "
            f"{c['ratio']:.3f} ({'in' if c['in_band'] else 'OUT OF'} [0.5, 2.0])"
        )
    verdict = (
        "ACCEPTED (provisional, coupled reference)"
        if ok
        else ("DISCARDED — cross-check failed; missing pairs stay UNCALIBRATED")
    )
    print(f"[derive-noise-ref] verdict: {verdict}")

    payload = {
        "derivation": {
            "regime": args.regime,
            "convention": "pooled seed-sd of {observational, proximal} "
            "value_mse_to_oracle at the basic point (grace arms excluded)",
            "status": "accepted_provisional" if ok else "discarded",
            "cross_check": checks,
        },
        "reference_derived": (
            {f"{env}/{algo}": float(v) for (env, algo), v in sorted(derived.items())}
            if ok
            else {}
        ),
    }
    header = (
        "# R-N1.1 DERIVED noise_ref values (feat/grace-critic) — PROVISIONAL.\n"
        "# Derived from the judged cell's own production basic-point runs\n"
        "# (obs-vs-prox pooled seed-sd ONLY; no grace arm enters). COUPLED\n"
        "# reference: every verdict resting on these is provisional and no\n"
        "# provisional pass may be a headline claim. Cross-check against the\n"
        "# stored independent references gates acceptance ([0.5, 2.0] on\n"
        "# derived/stored for every stored pair; any miss discards ALL).\n"
        "# This file never edits null_cal_reference.yaml (the independent\n"
        "# references stay authoritative). Doubles as the critics-only-vs-\n"
        "# production cross-check documented open in null_cal_reference.yaml.\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump(payload, sort_keys=True))
    print(f"[derive-noise-ref] wrote {args.out}")


if __name__ == "__main__":
    main()

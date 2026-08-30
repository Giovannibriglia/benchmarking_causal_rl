"""Merge the weak end's fit seeds 3-4 into its seeds 0-2 records.

WHY THIS EXISTS, recorded so the shortcut is auditable: the pre-S1c sweep ran
d010 and d005 with FIVE fit seeds while the post-S1c re-run used the tool's
default THREE. The tool reports the best-of-seeds fit BY LIKELIHOOD, so
best-of-5 against best-of-3 is not a like-for-like comparison -- and in this
regime it is decisive rather than cosmetic (the handoff's basin lottery: 2-3
seeds in 10 land in a bad basin whatever the configuration). Comparing them
would have read a seed-count artefact as a collapse to chance.

Re-running seeds 0-2 was unnecessary: fits are DETERMINISTIC in
(fit_seed, data, config) since 9625b85, so the completed seeds 0-2 records are
exactly what a five-seed run would have produced for those seeds. Only 3 and 4
were run, and this merges them under the tool's own rule -- ``max`` by
``final_ll``, means and maxima over all five -- so the merged row is identical
to a five-seed run's row rather than an approximation of it.

Verified on merge: the two sources must agree on the seeds they share (none
here, by construction) and must carry the same dataset row keys.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("results/dd_sweep_ablation_s1c")
TAGS = ("d010", "d005")
ARMS = ("with", "without", "with_proxyinit")


def _merge_arm(a: dict, b: dict) -> dict:
    per_seed = list(a["per_seed"]) + list(b["per_seed"])
    seeds = [d["fit_seed"] for d in per_seed]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"overlapping fit seeds in merge: {sorted(seeds)}")
    best = max(per_seed, key=lambda d: d["final_ll"])  # the tool's own rule
    return {
        **best,
        "per_seed": per_seed,
        "recovery_mean": float(sum(d["recovery"] for d in per_seed) / len(per_seed)),
        "recovery_max": float(max(d["recovery"] for d in per_seed)),
    }


def main() -> int:
    for tag in TAGS:
        p_main = BASE / f"d_d_sweep_{tag}.json"
        p_extra = BASE / f"seeds34_{tag}.json"
        rows = {(r["env"], r["seed"]): r for r in json.loads(p_main.read_text())}
        extra = {(r["env"], r["seed"]): r for r in json.loads(p_extra.read_text())}
        if set(rows) != set(extra):
            raise ValueError(f"{tag}: row keys differ between the two runs")
        out = []
        for key, r in rows.items():
            merged = dict(r)
            for arm in ARMS:
                if arm in r and arm in extra[key]:
                    merged[arm] = _merge_arm(r[arm], extra[key][arm])
            merged["delta_recovery"] = (
                merged["with"]["recovery"] - merged["without"]["recovery"]
            )
            merged["delta_separability"] = (
                merged["with"]["separability"] - merged["without"]["separability"]
            )
            merged["delta_contrast"] = (
                merged["with"]["do_contrast_a_bad_minus_other"]
                - merged["without"]["do_contrast_a_bad_minus_other"]
            )
            merged["n_fit_seeds"] = len(merged["with"]["per_seed"])
            out.append(merged)
            print(
                f"  {tag} {key[0]:<12} s{key[1]} "
                f"with={merged['with']['recovery']:.4f} "
                f"without={merged['without']['recovery']:.4f} "
                f"gap={merged['delta_recovery']:+.3f} "
                f"({merged['n_fit_seeds']} fit seeds)"
            )
        # Written to the canonical name so the renderer picks it up; the
        # 3-seed record stays at seeds012_*.json rather than being discarded.
        (BASE / f"seeds012_{tag}.json").write_text(
            json.dumps(list(rows.values()), indent=1)
        )
        p_main.write_text(json.dumps(out, indent=1))
    print("MERGE COMPLETE -- weak end now on the pre-fix run's 5-seed basis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

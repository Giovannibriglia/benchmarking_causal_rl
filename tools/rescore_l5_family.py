"""SHELVED (2026-09-03) — re-scoring is void: no calibration row gates anything.

The dr2-cut ruling ended the calibration campaign; the rows are kept solely
as S18 evidence (see tools/report_s18.py), for which the pre-S19 family is
valid as-is. This tool also imports `l5.select_window`, removed by the same
ruling — it will not run against current code. Kept in-tree as the record of
the re-score design (family tagging, code_version stamping, the measured
constant-reward invariance); nothing consumes it.

Original purpose:

Re-score pre-S19 calibration rows under the (O, A) history family.

The in-flight sweep imported the pre-S19 ``l5`` at process start, so every
row it wrote was scored with the (O, A, R) family — lagged R included. Under
S19 the family is (O, A) only (the served augmentation's columns), so rows
must be re-scored before the calibration report is read as the gate.

**Which rows are affected — derived, not listed.** A row whose verdict said R
was UNTESTABLE has a constant lagged-R column, so the (O, A, R) and (O, A)
designs differ by a constant feature the fit absorbs: the family statistic is
unchanged up to float noise, and the row is re-tagged, not re-run. Every row
with a testable R is re-run under the current code, with the SAME chunk,
sizes, masks, seeds and budgets (imported from ``calibrate_l5`` — one
construction site).

Output: ``rows_rescored.jsonl`` (all rows, each stamped ``family_history``:
re-tagged "OA=OAR" for the invariant ones, fresh "OA" rows for the re-runs)
plus a comparison summary of old-vs-new (p, dR2, k_selected) per re-run.

REFUSES to run while the disk ``l5`` is still the pre-S19 family (checked by
signature), so it cannot silently re-score with the very code being replaced.

Run (after the S19 push lands):  uv run python tools/rescore_l5_family.py
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

OUT = Path("results/l5_calibration")


def main() -> int:
    raise SystemExit(
        "rescore_l5_family is SHELVED (2026-09-03 ruling): no calibration row "
        "gates anything; the rows are S18 evidence only (tools/report_s18.py)."
    )
    import minari
    from src.rl.offline.grace import l5

    from tools.calibrate_l5 import (
        _chunks,
        _episodes_slice,
        _POWER_B,
        ALPHA,
        K_MAX,
        PLANS,
    )

    if "history_reward" not in inspect.signature(l5._build_design).parameters:
        raise SystemExit(
            "disk l5 is still the pre-S19 (O,A,R) family — re-scoring with it "
            "would reproduce the rows being replaced. Pull the S19 push first."
        )

    from src.rl.offline.grace.transform_cache import code_version

    cv = code_version()
    rows = [json.loads(x) for x in (OUT / "rows.jsonl").read_text().splitlines() if x]
    out_path = OUT / "rows_rescored.jsonl"
    done = set()
    if out_path.exists():
        for x in out_path.read_text().splitlines():
            r = json.loads(x)
            done.add((r["env"], r["dataset"], r["chunk"], r["kind"], r["n_ep"]))

    eps_cache: dict = {}

    def episodes(did):
        if did not in eps_cache:
            eps_cache.clear()  # one dataset in memory at a time
            eps_cache[did] = l5.episodes_from_minari(minari.load_dataset(did))
        return eps_cache[did]

    comparison = []
    with out_path.open("a") as f:
        for r in sorted(rows, key=lambda r: r["dataset"]):
            key = (r["env"], r["dataset"], r["chunk"], r["kind"], r["n_ep"])
            if key in done or "error" in r:
                continue
            if "R" in (r.get("untestable") or []):
                # Constant lagged-R column: family-invariant, re-tagged only.
                # Family-invariant BY MEASUREMENT (peer session, 2026-09-03):
                # on constant-reward data the post-S19 family reproduces the
                # pre-S19 statistic to 1e-15 at lags 0 and 1 — one shared
                # basis by construction (history RFF weights drawn at the
                # R-inclusive width and truncated, so the two expansions
                # coincide when R's standardised column is zero).
                f.write(
                    json.dumps({**r, "family_history": "OA=OAR", "code_version": cv})
                    + "\n"
                )
                f.flush()
                continue
            plan = PLANS[r["env"]]
            usize = plan["uniformity_size"]
            lo, hi = _chunks(10**9, usize, r["chunk"] + 1)[r["chunk"]]
            t0 = time.time()
            if r["kind"] == "null":
                sub = _episodes_slice(episodes(r["dataset"]), lo, hi, mask=())
                seed = 1000 * r["chunk"] + 17
                b = 199
            else:
                mname = r["kind"].split(":", 1)[1]
                sub = _episodes_slice(
                    episodes(r["dataset"]),
                    lo,
                    lo + r["n_ep"],
                    mask=plan["masks"][mname],
                )
                seed = 1000 * r["chunk"] + 17 + 7
                b = _POWER_B
            k_sel, kv = l5.select_window(sub, alpha=ALPHA, k_max=K_MAX, b=b, seed=seed)
            v = kv[0]
            new = {
                **r,
                "family_history": "OA",
                "code_version": cv,
                "p": v.p_value,
                "stat": v.statistic,
                "untestable": v.untestable,
                "base_r2_min": (
                    None
                    if v.base_r2 is None
                    else float(__import__("numpy").nanmin(v.base_r2[:-1]))
                ),
                "reward_channel": v.reward_channel,
                "capacity": v.capacity,
                "k_selected": k_sel,
                "k_tests": len(kv),
                "rescored_seconds": round(time.time() - t0, 1),
            }
            f.write(json.dumps(new) + "\n")
            f.flush()
            comparison.append(
                dict(
                    dataset=r["dataset"].split("/")[1][:40],
                    chunk=r["chunk"],
                    kind=r["kind"],
                    n_ep=r["n_ep"],
                    p_old=r["p"],
                    p_new=v.p_value,
                    dr2_old=r["stat"],
                    dr2_new=v.statistic,
                    k_old=r["k_selected"],
                    k_new=k_sel,
                )
            )
            print(
                f"rescored {comparison[-1]['dataset']} c{r['chunk']} {r['kind']}: "
                f"dR2 {r['stat']:.2e} -> {v.statistic:.2e}, p {r['p']:.3f} -> "
                f"{v.p_value:.3f} ({time.time()-t0:.0f}s)",
                flush=True,
            )
    (OUT / "rescore_comparison.json").write_text(json.dumps(comparison, indent=1))
    print(f"-> {out_path} + rescore_comparison.json ({len(comparison)} re-runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

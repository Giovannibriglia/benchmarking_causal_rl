"""The D-D proxy ablation, read against GROUND TRUTH rather than between arms.

An arm-vs-arm delta says the proxies changed something; it does not say which
arm is right. The interventional contrast has a closed form here, so the
comparison that matters is against it:

    E[R | do(a=a_bad), s] - E[R | do(a!=a_bad), s]  =  c_r * P(U = 1)

because ``interventional_sweep`` marginalises ``U`` over the fitted prior. With
``c_r = 1.0`` the reference is simply the logged ``P(U = 1)`` -- ground truth,
read from the generator's own logs, never from a second estimator (S4).

Reported alongside the latent-level numbers because the two can disagree, and
the disagreement is the finding: an arm can recover the labels and still get the
estimand badly wrong.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari

    from tools.recertify_diagram_arms import rebuild_samples

    rows = []
    for f in sorted(Path("results/dd_ablation").glob("*.json")):
        rows.extend(json.loads(f.read_text()))
    if not rows:
        print("no ablation results yet")
        return 1

    recert = {
        (r["env"], r["seed"], r["sigma"]): r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d"
    }

    print(f"=== D-D proxy ablation, {len(rows)} datasets, matched random init ===\n")
    hdr = (
        f"{'env':<12}{'sd':>3} | {'recovery':^19} | {'separability':^15} | "
        f"{'do-contrast vs truth':^30}"
    )
    print(hdr)
    print(
        f"{'':<12}{'':>3} | {'with':>8}{'w/o':>11} | {'with':>7}{'w/o':>8} | "
        f"{'truth':>7}{'with':>8}{'w/o':>8}{'|err| x':>7}"
    )
    print("-" * len(hdr))
    agg = []
    for r in sorted(rows, key=lambda x: (x["env"], x["seed"])):
        key = (r["env"], r["seed"], r["sigma"])
        ds = minari.load_dataset(recert[key]["dataset_id"])
        s, _ = rebuild_samples(ds, r["n_episodes"])
        ep = s["episode"]
        u_ep = np.array([s["u"][ep == e][0] for e in np.unique(ep)])
        truth = float(u_ep.mean())  # c_r = 1.0
        cw = r["with"]["do_contrast_a_bad_minus_other"]
        co = r["without"]["do_contrast_a_bad_minus_other"]
        ew, eo = abs(cw - truth), abs(co - truth)
        ratio = eo / ew if ew > 1e-9 else float("inf")
        agg.append(
            {
                **{k: r[k] for k in ("env", "seed")},
                "truth": truth,
                "err_with": ew,
                "err_without": eo,
                "ratio": ratio,
                "rec_with": r["with"]["recovery"],
                "rec_without": r["without"]["recovery"],
                "recmean_without": r["without"]["recovery_mean"],
            }
        )
        print(
            f"{r['env']:<12}{r['seed']:>3} | "
            f"{r['with']['recovery']:>8.4f}{r['without']['recovery']:>11.4f} | "
            f"{r['with']['separability']:>7.4f}{r['without']['separability']:>8.4f} | "
            f"{truth:>7.3f}{cw:>8.3f}{co:>8.3f}{ratio:>7.1f}"
        )

    print("\n--- latent level ---")
    print(
        f"  recovery, best-LL fit:   with {np.mean([a['rec_with'] for a in agg]):.4f}"
        f"   without {np.mean([a['rec_without'] for a in agg]):.4f}"
    )
    print(
        f"  recovery, MEAN over fit seeds: without {np.mean([a['recmean_without'] for a in agg]):.4f}"
        "   <- the gap here is EM stability, not identifying information"
    )
    print("\n--- value level, against ground truth ---")
    print(
        f"  |error| in the do-contrast:  with {np.mean([a['err_with'] for a in agg]):.4f}"
        f"   without {np.mean([a['err_without'] for a in agg]):.4f}"
        f"   (x{np.mean([a['ratio'] for a in agg]):.1f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

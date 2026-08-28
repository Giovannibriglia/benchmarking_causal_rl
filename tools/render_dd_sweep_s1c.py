"""The D-D gate-separation sweep, PRE-S1c against POST-S1c, side by side.

The proxy pseudo-replication fix changed the estimator materially (proxy
channels 78-90 nats -> 0.58-0.60), so every sweep number was re-measured
rather than re-labelled. This renders both, per the cell's binding reporting
constraint (PER SEED, never averaged across dataset seeds -- CartPole's s1
without-arm resists collapse where s0/s2 fall to chance, and a mean would
hide exactly that).

The registered expectation, recorded before the re-run: correctly weighted
proxies are LESS influential, so the decorative-proxies finding should
STRENGTHEN and the load-bearing transition should move to WEAKER R (smaller
d). A miss is the more interesting outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

PRE = Path("results/dd_sweep_ablation")
POST = Path("results/dd_sweep_ablation_s1c")
TAGS = (
    ("d100", 1.00, 1.0),
    ("d050", 0.50, 2.0),
    ("d025", 0.25, 4.0),
    ("d010", 0.10, 10.0),
    ("d005", 0.05, 20.0),
)
ENVS = ("CartPole-v1", "Acrobot-v1")


def _load(base: Path, tag: str):
    p = base / f"d_d_sweep_{tag}.json"
    if not p.exists():
        return {}
    return {(r["env"], r["seed"]): r for r in json.loads(p.read_text())}


def _fmt(rows, env, key):
    out = []
    for sd in (0, 1, 2):
        r = rows.get((env, sd))
        out.append("  .  " if r is None else f"{r[key]['recovery']:.3f}")
    return "/".join(out)


def _gaps(rows, env):
    out = []
    for sd in (0, 1, 2):
        r = rows.get((env, sd))
        out.append("  .   " if r is None else f"{r['delta_recovery']:+.3f}")
    return " ".join(out)


def main() -> int:
    print("D-D GATE-SEPARATION SWEEP -- PRE-S1c vs POST-S1c (corrected likelihood)")
    print("recovery per DATASET SEED (best-LL over fit seeds); M = 1.0; sigma = 0.25")
    print("per-seed by the cell's binding reporting constraint -- never averaged\n")
    header = (
        f"{'d':>5} {'c_r':>5} {'env':<12} {'fix':<5} "
        f"{'WITH {Z,W,V} (s0/s1/s2)':<26}{'WITHOUT (s0/s1/s2)':<26}"
        f"{'gap (per-seed paired)':<24}"
    )
    print(header)
    for tag, d, c_r in TAGS:
        pre, post = _load(PRE, tag), _load(POST, tag)
        for env in ENVS:
            for label, rows in (("pre", pre), ("post", post)):
                if not rows:
                    continue
                print(
                    f"{d:5.2f} {c_r:5.1f} {env:<12} {label:<5} "
                    f"{_fmt(rows, env, 'with'):<26}{_fmt(rows, env, 'without'):<26}"
                    f"{_gaps(rows, env):<24}"
                )
        print()

    # ---- the headline the re-run exists to test ---------------------------
    print("TRANSITION (per seed): smallest d whose gap exceeds the d=1.0 gap by 0.10")
    for env in ENVS:
        for label, base in (("pre", PRE), ("post", POST)):
            marks = []
            for sd in (0, 1, 2):
                found = "none"
                base_gap = None
                for tag, d, _ in TAGS:
                    rows = _load(base, tag)
                    r = rows.get((env, sd))
                    if r is None:
                        continue
                    if base_gap is None:
                        base_gap = r["delta_recovery"]
                    if r["delta_recovery"] - base_gap > 0.10:
                        found = f"d={d:g}"
                        break
                marks.append(f"s{sd}:{found}")
            print(f"  {env:<12} {label:<5} " + "  ".join(marks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""GATE for the D-D sigma = 0.25 decision: is the selection bias measurably nonzero?

sigma moves the U->A edge, i.e. the SELECTION BIAS the cell exists to exhibit
-- not the do-target, which is fixed by the reward mechanism (M = c_r * d).
Before the cell is re-declared at sigma = 0.25, confirm there is still
something to correct, in both environments, from logged data alone. No fits.

What is measured -- the selection-bias term on the DECLARED channel:

    bias = c_r * ( P(U=1 | a = a_bad) - P(U=1) )

which is exactly the wedge between the naive transition-pooled contrast and
the do-contrast that flows through the gate. Transition-pooled deliberately:
that is what a naive consumer computes, and the pooling is part of what makes
it naive (S1b noted, not overlooked). P(U=1) is taken at episode level.

``a_bad`` is INFERRED per environment from the U-tilt on the sigma > 0
datasets (argmax_k P(a=k|U=1) - P(a=k|U=0)) and asserted consistent across
them -- a value-based inference collided with Acrobot's terminal reward, so
identity comes from the tilt, not the reward. ``c_r`` = 1.0 from
reproducibility/rl_regimes/diagrams/d_d.yaml (confounder_c_r).

THE NULL IS EMPIRICAL: the sigma = 0.0 datasets have no U->A edge, so their
five per-seed biases are this statistic's own floor. The gate CLEARS when
every sigma = 0.25 bias sits outside the sigma = 0 range with visible margin,
in both environments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--c-r", type=float, default=1.0, help="confounder_c_r from d_d.yaml"
    )
    ap.add_argument("--out", default="results/cost/naive_bias_gate.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari

    from tools.recertify_diagram_arms import rebuild_samples

    rows = [
        r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d"
    ]
    samples = {}
    for r in rows:
        s, _ = rebuild_samples(minari.load_dataset(r["dataset_id"]), 10_000)
        samples[(r["env"], r["sigma"], r["seed"])] = s

    # a_bad per env from the sigma>0 tilt, asserted consistent.
    a_bad = {}
    for env in sorted({r["env"] for r in rows}):
        tilts = []
        for (e, sig, sd), s in samples.items():
            if e != env or sig == 0.0:
                continue
            a, u, ep = s["a"], s["u"], s["episode"]
            # EPISODE-level tilt (S1b): the transition-pooled version is
            # length-weighted, and length is a collider of U and the action
            # mix -- it flipped the inferred a_bad on low-sigma CartPole
            # seeds when this was first written row-level. One row per
            # episode: the episode's action-k fraction, then the mean.
            ep_ids = np.unique(ep)
            u_ep = np.array([u[ep == e2][0] for e2 in ep_ids])
            tilt = []
            for k in np.unique(a):
                frac = np.array([(a[ep == e2] == k).mean() for e2 in ep_ids])
                tilt.append(frac[u_ep == 1].mean() - frac[u_ep == 0].mean())
            tilts.append(int(np.argmax(tilt)))
        assert len(set(tilts)) == 1, f"{env}: inconsistent a_bad inference {set(tilts)}"
        a_bad[env] = tilts[0]
        print(
            f"  {env}: a_bad = {a_bad[env]} (consistent across {len(tilts)} sigma>0 datasets)"
        )

    out = []
    for (env, sig, sd), s in sorted(samples.items()):
        a, u, ep = s["a"], s["u"], s["episode"]
        ep_ids = np.unique(ep)
        u_ep = np.array([u[ep == e][0] for e in ep_ids])
        p_u_ep = float(u_ep.mean())
        m = a == a_bad[env]
        bias = args.c_r * (float(u[m].mean()) - p_u_ep)
        # EPISODE-granularity analogue: one row per episode, each episode
        # contributing its a_bad PROPENSITY once -- removing the
        # length x frequency weighting (the collider term) while keeping the
        # between-episode selection the naive estimator is naive about. (A
        # WITHIN-episode contrast would difference U out entirely -- that is
        # a fixed-effects estimator, not a naive one.)
        f_ep = np.array([(a[ep == e] == a_bad[env]).mean() for e in ep_ids])
        bias_ep = args.c_r * (float((u_ep * f_ep).sum() / f_ep.sum()) - p_u_ep)
        out.append(
            {
                "env": env,
                "sigma": sig,
                "seed": sd,
                "p_u1_episode": round(p_u_ep, 4),
                "p_u1_given_abad": round(float(u[m].mean()), 4),
                "bias": round(bias, 4),
                "bias_episode": round(bias_ep, 4),
            }
        )

    print(f"\n{'env':<12} {'sigma':>5}  transition-pooled per seed  |  episode-level")
    gate = {}
    for env in sorted({o["env"] for o in out}):
        for sig in sorted({o["sigma"] for o in out}):
            sel = [o for o in out if o["env"] == env and o["sigma"] == sig]
            b = [o["bias"] for o in sel]
            be = [o["bias_episode"] for o in sel]
            print(
                f"{env:<12} {sig:>5}  "
                + " ".join(f"{x:+.4f}" for x in b)
                + "  |  "
                + " ".join(f"{x:+.4f}" for x in be)
            )
            gate[(env, sig)] = b
    print()
    for env in sorted({o["env"] for o in out}):
        lo, hi = min(gate[(env, 0.0)]), max(gate[(env, 0.0)])
        b25 = gate[(env, 0.25)]
        clears = all(x > hi or x < lo for x in b25)
        margin = min((abs(x - hi) if x > hi else abs(x - lo)) for x in b25)
        ratio = min(abs(x) for x in b25) / max(abs(lo), abs(hi), 1e-12)
        print(f"  {env}: sigma=0 null [{lo:+.4f}, {hi:+.4f}]  sigma=0.25 {sorted(b25)}")
        print(
            f"    -> {'CLEARS' if clears else 'DOES NOT CLEAR'} (min margin {margin:.4f}, min|bias|/null-extreme = {ratio:.1f}x)"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

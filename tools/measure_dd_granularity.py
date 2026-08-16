"""Step 4 — D-D's Kruskal views measured at BOTH granularities, matched.

The recorded margins cannot settle this. Their denominator changed with the fix
(the `max` of 40 permutation draws became the `q99` of 200), so an old-vs-new
margin comparison conflates the granularity edit with the cutoff edit. The
comparison here is on the **denominator-free raw ratio** ``s2/s1`` of the
measurement matrix ``P(view | U)``, computed from the same stored samples by the
same code, differing only in granularity.

Two questions that are NOT the same, and that the numbers separate:

* **Informativeness** — how well a view separates the latent classes. That is
  the raw ``s2/s1``: higher means the two rows of ``P(view|U)`` are further from
  proportional. This is what the D-D coupling note is about and what drives the
  estimator's conditioning.
* **Binding** — which view would fail the rank test first. That is the margin
  ``obs / cutoff`` against the view's OWN permutation null, because k-rank 2 is
  declared exactly when the observed ratio clears that cutoff. Kruskal is
  exactly tight at |U| = 2, so the smallest margin is the constraint.

They can point at different views and on CartPole they do: a view can be the
least informative while sitting furthest from its own null, because a view with
few effective bins has a low null cutoff too. Reporting only one of them would
answer the wrong question.

Also measured: ``P(a = a_bad)``, the behaviour quantity the catalogue says R's
view is coupled to — at EPISODE granularity, since it is a proportion over an
episode-constant-U stratum and is exactly the marginal the S1b caveat warns
about.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recert", default="results/vb_recertification/report.json")
    ap.add_argument("--out", default="results/vb_recertification/dd_granularity.json")
    ap.add_argument("--max-episodes", type=int, default=600)
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    from src.envs.offline.arm_preflight import (
        _episode_constant,
        _episode_index,
        _episode_mean,
        _k_rank_permutation,
        _sv_ratio,
        _view_matrix,
    )

    from tools.recertify_diagram_arms import rebuild_samples

    rows = [r for r in json.loads(Path(args.recert).read_text()) if r["cell"] == "d_d"]
    out = []
    for n, r in enumerate(
        sorted(rows, key=lambda x: (x["env"], x["seed"], x["sigma"])), 1
    ):
        ds = minari.load_dataset(r["dataset_id"])
        s, _ = rebuild_samples(ds, args.max_episodes)
        uniq, inv = _episode_index(s["episode"])
        n_ep = int(uniq.size)

        # EPISODE granularity: the fixed views.
        ep_views = {
            "Z": _episode_constant(s["z"], inv, n_ep, "Z"),
            "W": _episode_constant(s["w"], inv, n_ep, "W"),
            "R": _episode_mean(s["r"], inv, n_ep),
        }
        u_ep = _episode_constant(s["u"], inv, n_ep, "U")

        # TRANSITION granularity: exactly what the old code measured -- the
        # episode-constant proxies replicated once per step, the reward raw.
        tr_views = {"Z": s["z"], "W": s["w"], "R": s["r"]}
        u_tr = s["u"]

        rec = {
            "env": r["env"],
            "seed": r["seed"],
            "sigma": r["sigma"],
            "n_episodes": n_ep,
            "n_transitions": int(s["episode"].size),
            # P(a = a_bad) at EPISODE granularity: the mean over episodes of the
            # within-episode proportion. The transition-level proportion is the
            # length-weighted one the S1b caveat warns about, so both are kept.
            "p_a_bad_episode": float(
                np.mean(_episode_mean((s["a"] == 1.0).astype(float), inv, n_ep))
            ),
            "p_a_bad_transition": float(np.mean(s["a"] == 1.0)),
            "ratio_episode": {},
            "ratio_transition": {},
            "margin_episode": {},
            "k_rank_episode": {},
        }
        for name in ("Z", "W", "R"):
            rec["ratio_transition"][name] = float(
                _sv_ratio(_view_matrix(tr_views[name], u_tr))
            )
            kr, ratio, cutoff, p = _k_rank_permutation(ep_views[name], u_ep)
            rec["ratio_episode"][name] = float(ratio)
            rec["margin_episode"][name] = (
                float(ratio / cutoff) if cutoff > 0 else float("inf")
            )
            rec["k_rank_episode"][name] = int(kr)
        rec["order_episode"] = sorted(
            ("Z", "W", "R"), key=lambda k: rec["ratio_episode"][k]
        )
        rec["order_transition"] = sorted(
            ("Z", "W", "R"), key=lambda k: rec["ratio_transition"][k]
        )
        rec["least_informative_episode"] = rec["order_episode"][0]
        rec["least_informative_transition"] = rec["order_transition"][0]
        rec["binding_episode"] = min(
            ("Z", "W", "R"), key=lambda k: rec["margin_episode"][k]
        )
        out.append(rec)
        print(
            f"[{n}/{len(rows)}] {r['env']:<12} s{r['seed']} sig={r['sigma']:<5} "
            f"ratio ep Z/W/R = "
            f"{rec['ratio_episode']['Z']:.3f}/{rec['ratio_episode']['W']:.3f}/{rec['ratio_episode']['R']:.3f}"
            f"  tr = "
            f"{rec['ratio_transition']['Z']:.3f}/{rec['ratio_transition']['W']:.3f}/{rec['ratio_transition']['R']:.3f}",
            flush=True,
        )
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

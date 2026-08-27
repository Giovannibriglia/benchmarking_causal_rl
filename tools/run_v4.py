"""V4 -- the L4 gate: coverage, width, collapse, dual bounds, procedural share.

Rows (as ruled 2026-08-24):
* COVERAGE/WIDTH: per-seed q1 intervals vs analytic truth on d_a_null and all
  five D-D sweep points (both envs, 3 seeds).
* COLLAPSE: d_a_null and D-D d >= 0.25 -> ~0 width; D-E and D-B-prime -> not.
* D-E DUAL BOUNDS: Balke-Pearl closed form (what the DECLARED DIAGRAM
  licenses -- the verdict-directed path) AND the I-blind LR-walk bounds (what
  the latent-class model alone licenses); their gap is the measured value of
  declaring the instrument. Walk bounds are INNER approximations, disclosed.
* D-B-PRIME: the LR walk in production -- the empirical exploration gate
  (under-coverage here reads as an OPTIMISER finding, kept separable).
* PROCEDURAL SHARE: reported per cell; the weak end (d <= 0.10) called out.

Exactness row: tools/test_lr_optimiser_exactness.py (anchor BP<->LP to four
decimals; walk-vs-validated-oracle hi reproduced, lo attributed to simplex-
face geometry production lacks). Limitations stated there travel here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

C_R = {"d100": 1.0, "d050": 2.0, "d025": 4.0, "d010": 10.0, "d005": 20.0}
ALPHA, B, FIT_SEED = 0.1, 19, 0
INIT_SEEDS = (1, 2)
FK = dict(max_iter=30, m_step_budget=400, batch_size=4096)


def main() -> int:
    import minari
    import numpy as np
    import torch
    from nbn.utils.batching import pack_parents
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator
    from src.rl.offline.grace.l4 import (
        balke_pearl_contrast_bounds,
        lr_region_bounds,
        point_id_interval,
    )

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path("results/v4/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # RESUME: rows already in the report are skipped on relaunch. Sound
    # because determinism makes a redone row bitwise identical -- skipping
    # loses nothing and repeats nothing.
    out: list = []
    if out_path.exists():
        out = json.loads(out_path.read_text())
        print(f"  resuming: {len(out)} rows already done", flush=True)
    done = {(r["row"], r["cell"], r["env"], r["seed"]) for r in out}

    def save():
        out_path.write_text(json.dumps(out, indent=1))

    def load(did):
        s, blocks = rebuild_samples(minari.load_dataset(did), 10_000)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        return s, state

    def mk(s, state, with_proxy):
        t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
        kw = {}
        if with_proxy and s["z"].size:
            kw["proxy"] = {"Z": t(s["z"]), "W": t(s["w"])}
            if s["v"].size:
                kw["proxy"]["V"] = t(s["v"])
        return EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(s["episode"], torch.long),
            **kw,
        )

    def sweep_target(state):
        rng = np.random.default_rng(0)
        idx = rng.choice(state.shape[0], size=min(256, state.shape[0]), replace=False)
        ev = torch.tensor(state[idx], dtype=torch.float32, device=device)

        def target(est, fit):
            bad = est.interventional_sweep(ev, [1] * idx.size, fit)
            oth = est.interventional_sweep(ev, [0] * idx.size, fit)
            return float((bad.value - oth.value).mean().detach())

        return target

    def diff_target(est, state):
        """Differentiable-in-parameters gate contrast for the LR walk."""
        rng = np.random.default_rng(0)
        idx = rng.choice(state.shape[0], size=min(128, state.shape[0]), replace=False)
        ev = torch.tensor(state[idx], dtype=torch.float32, device=device)
        n = ev.shape[0]
        levels = est._reward_levels

        def target(model, prior):
            total = 0.0
            for k in range(est.u_card):
                means = []
                for a_val in (1, 0):
                    d = {
                        "S": ev,
                        "A": torch.full((n,), float(a_val), device=device),
                        "U": torch.full((n,), float(k), device=device),
                    }
                    pa = pack_parents(d, model.dag.parents("R"))
                    dist = model.mechanisms["R"](pa)
                    if levels is not None:
                        mean = dist.probs @ levels.to(dist.probs.dtype).to(device)
                    else:
                        mean = dist.mean
                    means.append(mean.mean())
                total = total + prior[k] * (means[0] - means[1])
            return total

        return target

    # ================= interval cells: d_a_null + D-D sweep ==================
    recert = json.loads(Path("results/vb_recertification/report.json").read_text())
    gen = {
        (r["cell"], r["env"], r["seed"]): r["dataset_id"]
        for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text())
    }

    jobs = []
    for r in recert:
        if r["cell"] == "d_a_null" and r["seed"] in (0, 1, 2):
            jobs.append(("d_a_null", r["env"], r["seed"], r["dataset_id"], 0.0))
    for tag in ("d100", "d050", "d025", "d010", "d005"):
        for env in ("CartPole-v1", "Acrobot-v1"):
            for sd in (0, 1, 2):
                jobs.append(
                    (tag, env, sd, gen[(f"d_d_sweep_{tag}", env, sd)], C_R[tag] * 0.5)
                )

    for cell, env, sd, did, truth in jobs:
        if ("interval", cell, env, sd) in done:
            continue
        s, state = load(did)
        has_p = bool(s["z"].size)
        data = mk(s, state, has_p)
        pn = (("Z", "W", "V") if s["v"].size else ("Z", "W")) if has_p else ()
        res = point_id_interval(
            make_estimator=lambda seed, pn=pn, st=state, s_=s: LatentClassEstimator(
                state_dim=st.shape[1],
                n_actions=int(s_["a"].max()) + 1,
                proxy_names=pn,
                device=device,
                seed=seed,
            ),
            data=data,
            target=sweep_target(state),
            fit_kwargs=dict(FK, init="proxy" if has_p else "random"),
            alpha=ALPHA,
            b=B,
            fit_seed=FIT_SEED,
            init_seeds=INIT_SEEDS,
        )
        covered = (res.kind == "interval") and (res.lo - 1e-9 <= truth <= res.hi + 1e-9)
        out.append(
            {
                "row": "interval",
                "cell": cell,
                "env": env,
                "seed": sd,
                "truth": truth,
                "kind": res.kind,
                "lo": res.lo,
                "hi": res.hi,
                "width": res.width if res.kind == "interval" else None,
                "covered": covered,
                "reason": res.reason,
                "procedural_share": res.procedural_share,
                "failure_rate": res.failure_rate,
                "meta": {
                    k: v for k, v in res.meta.items() if k != "bootstrap_diagnostics"
                },
                "bootstrap_diagnostics": res.meta.get("bootstrap_diagnostics"),
            }
        )
        print(
            f"  {cell:<9} {env:<12} s{sd} {res.summary()[:110]} covered={covered}",
            flush=True,
        )
        save()

    # ================= bounds cells: D-E (dual) + D-B-prime ==================
    for cell in ("d_e", "d_b_prime"):
        rows = [
            r
            for r in recert
            if r["cell"] == cell and r["sigma"] == 1.0 and r["seed"] in (0, 1, 2)
        ]
        for r in rows:
            env, sd, did = r["env"], r["seed"], r["dataset_id"]
            if ("bounds", cell, env, sd) in done:
                continue
            s, state = load(did)
            rec = {"row": "bounds", "cell": cell, "env": env, "seed": sd}
            if cell == "d_e":
                ip = np.isin(s["a"], (0, 1))
                hi_val = s["r"].max()
                bonus = (s["r"] > (hi_val - 0.5)).astype(int)
                lo_bp, hi_bp = balke_pearl_contrast_bounds(
                    bonus=bonus[ip],
                    x=(s["a"][ip] == 1).astype(int),
                    z=s["i"][ip].astype(int),
                )
                rec["bp"] = [lo_bp, hi_bp]
                rec["bp_covered"] = lo_bp - 1e-9 <= 0.5 <= hi_bp + 1e-9
            data = mk(s, state, False)
            est = LatentClassEstimator(
                state_dim=state.shape[1],
                n_actions=int(s["a"].max()) + 1,
                proxy_names=(),
                device=device,
                seed=FIT_SEED,
            )
            fit = est.fit(data, init="random", **FK)
            res = lr_region_bounds(
                estimator=est,
                fit=fit,
                data=data,
                target_of_model=diff_target(est, state),
                make_estimator=lambda seed, st=state, s_=s: LatentClassEstimator(
                    state_dim=st.shape[1],
                    n_actions=int(s_["a"].max()) + 1,
                    proxy_names=(),
                    device=device,
                    seed=FIT_SEED,
                ),
                fit_kwargs=dict(FK, init="random"),
                alpha=ALPHA,
                b=B,
                fit_seed=FIT_SEED,
                # ``steps`` is now a SAFETY LIMIT: the walk stops when its
                # per-window gain falls below the bound's own Monte-Carlo
                # error (derived from c(alpha)'s B replicates), and reports
                # which of the two ended it. Generous, because the 600-step
                # probe on d_b_prime CartPole s0 was still descending
                # (0.760 -> 0.574) where the old fixed 150 stopped at 0.67.
                steps=4000,
                opt_lr=1e-3,
                n_starts=3,
            )
            rec.update(
                {
                    "walk_kind": res.kind,
                    "walk_lo": res.lo,
                    "walk_hi": res.hi,
                    "walk_covered": (res.kind == "bounds")
                    and (res.lo - 1e-9 <= 0.5 <= res.hi + 1e-9),
                    "inner_approximation": res.inner_approximation,
                    "budget_truncated": res.meta.get("budget_truncated"),
                    "walk_label": res.label,
                    "reason": res.reason,
                    "meta": {
                        k: v
                        for k, v in res.meta.items()
                        if k != "bootstrap_diagnostics"
                    },
                }
            )
            out.append(rec)
            print(
                f"  {cell:<9} {env:<12} s{sd} "
                + (
                    f"BP=[{rec['bp'][0]:+.3f},{rec['bp'][1]:+.3f}] "
                    if "bp" in rec
                    else ""
                )
                + f"walk={res.summary()[:80]}",
                flush=True,
            )
            save()
    print("V4 RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What does a CONVERGED GRACE fit cost at production scale?

The number decides an architecture, so it is reported against that fork rather
than as a wall-clock figure:

* **minutes** -> GRACE can refit on a cadence, and the online cells' refresh
  (N2: a refresh REFITS, since ``update_local`` refuses weights) is viable;
* **hours** -> GRACE is structurally a **fit-once-then-serve** critic — fit at
  the sequence-buffer handoff and serve thereafter, as v1 did — and the online
  refresh is off the table whatever N2 permits.

Measured with the FIXED-STEP-BUDGET M-step in place, which is the change that
decouples per-iteration cost from dataset size: ``epochs`` makes the M-step
O(n * epochs), while a fixed gradient-step budget makes it O(steps). Legitimate
under GEM, which asks only that the M-step increase the objective.

Also reported, because they are what make the number interpretable: iterations
to convergence (not a fixed cap), the per-iteration split, and every C3
condition on the resulting fit — a converged-looking timing from a saturated or
degenerate fit is not a cost measurement of anything.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000, help="production scale")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--m-step-budget", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument(
        "--epochs-baseline",
        type=int,
        default=30,
        help="epoch-based M-step to time against, at the SAME data scale, so "
        "the fixed-step lever has a MEASURED factor rather than an argued one",
    )
    ap.add_argument("--baseline-iters", type=int, default=3)
    ap.add_argument("--envs", nargs="+", default=["CartPole-v1", "Acrobot-v1"])
    ap.add_argument("--out", default="results/cost/grace_fit_cost.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recert = {
        (r["env"], r["seed"]): r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d" and r["sigma"] == 1.0
    }
    out = []
    for env in args.envs:
        ds = minari.load_dataset(recert[(env, 0)]["dataset_id"])
        t_load = time.time()
        s, blocks = rebuild_samples(ds, args.episodes)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        load_s = time.time() - t_load
        ep = s["episode"]
        u_ep = np.array([s["u"][ep == e][0] for e in np.unique(ep)], dtype=np.int64)

        def t(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=device)

        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(ep, torch.long),
            proxy={"Z": t(s["z"]), "W": t(s["w"])},
        )
        est = LatentClassEstimator(
            state_dim=state.shape[1],
            n_actions=int(s["a"].max()) + 1,
            proxy_names=("Z", "W"),
            device=device,
            seed=0,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        fit = est.fit(
            data,
            max_iter=args.max_iter,
            tol=args.tol,
            init="proxy",
            m_step_budget=args.m_step_budget,
            batch_size=args.batch_size,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        hard = fit.hard_assignment().cpu().numpy()

        # PER-MECHANISM SPLIT, measured rather than inferred. With R categorical
        # the only gradient-fitted continuous mechanisms left are the proxies
        # and S -- and **S is needed only for q2**, the sequential value. So if
        # S dominates the M-step, that is a direct measurement of what the
        # sequential query costs over the per-step one, rather than an argument
        # about it.
        mech_times = {}
        for name, mech in dict(est.model.mechanisms).items():
            frame = est._frame(
                data, torch.zeros(data.n, dtype=torch.long, device=device)
            )
            from nbn.core.network import pack_parents

            pa = pack_parents(frame, est.model.dag.parents(name))
            if device == "cuda":
                torch.cuda.synchronize()
            tm = time.time()
            for _ in range(3):
                mech.log_prob(frame[name], pa)
            if device == "cuda":
                torch.cuda.synchronize()
            mech_times[name] = round((time.time() - tm) / 3, 4)

        # THE LEVER, MEASURED. The fixed-step budget has been argued to decouple
        # per-iteration cost from dataset size; the V-D projection rests on it,
        # so it gets a measured factor. Timed over a few iterations of the
        # epoch-based M-step at the SAME scale and scaled to a per-iteration
        # figure -- running it to convergence would cost exactly what the lever
        # exists to avoid.
        est_b = LatentClassEstimator(
            state_dim=state.shape[1],
            n_actions=int(s["a"].max()) + 1,
            proxy_names=("Z", "W"),
            device=device,
            seed=0,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        tb = time.time()
        fit_b = est_b.fit(
            data,
            max_iter=args.baseline_iters,
            tol=0.0,  # no early stop: we are timing iterations, not converging
            init="proxy",
            epochs=args.epochs_baseline,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        wall_b = time.time() - tb
        per_iter_b = wall_b / max(fit_b.n_iter, 1)
        per_iter_f = wall / max(fit.n_iter, 1)

        # THE TRAJECTORY TAIL. "converged: False" at the cap is ambiguous in a
        # way that matters here: still improving steadily means convergence is
        # reachable and expensive, while plateaued just above tolerance means
        # the tolerance or the optimiser needs attention before any cost number
        # means anything.
        hist = list(fit.log_likelihood)
        tail = hist[-6:]
        deltas = [round(b - a, 4) for a, b in zip(tail, tail[1:])]
        rel = [round((b - a) / max(abs(b), 1e-12), 8) for a, b in zip(tail, tail[1:])]
        rec = {
            "env": env,
            "device": device,
            "n_episodes": int(np.unique(ep).size),
            "n_transitions": int(ep.size),
            "mean_T": round(float(ep.size / np.unique(ep).size), 1),
            "load_seconds": round(load_s, 1),
            "fit_seconds": round(wall, 1),
            "n_iter": fit.n_iter,
            "n_anneal": fit.n_anneal,
            "seconds_per_iter": round(wall / max(fit.n_iter, 1), 2),
            "converged": bool(fit.converged),
            "reached_tau_one": bool(fit.reached_tau_one),
            "monotone": bool(fit.monotone),
            "backtracks": fit.backtracks,
            "backtrack_exhausted": bool(fit.backtrack_exhausted),
            "initial_saturation": round(float(fit.initial_saturation), 3),
            "saturated_at_init": bool(fit.saturated_at_init),
            "separation_per_step": float(fit.separation_per_step),
            "degenerate_mechanism": bool(fit.degenerate_mechanism),
            "mechanism_degeneracy": dict(fit.mechanism_degeneracy),
            "recovery": float(max((hard == u_ep).mean(), (hard != u_ep).mean())),
            "m_step_budget": args.m_step_budget,
            "batch_size": args.batch_size,
            "seconds_per_iter_fixed_step": round(per_iter_f, 2),
            "seconds_per_iter_epoch_based": round(per_iter_b, 2),
            "fixed_step_speedup": round(per_iter_b / max(per_iter_f, 1e-9), 2),
            "epochs_baseline": args.epochs_baseline,
            "ll_tail": [round(x, 3) for x in tail],
            "ll_tail_deltas": deltas,
            "ll_tail_rel_deltas": rel,
            "tol": args.tol,
            "convergence_window": 2,
            "stationary": bool(fit.stationary),
            "finished": bool(fit.finished),
            "lr_reductions": int(fit.lr_reductions),
            "final_lr_scale": float(fit.final_lr_scale),
            "mechanism_log_prob_seconds": mech_times,
        }
        out.append(rec)
        print(json.dumps(rec, indent=1), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n=== THE FORK ===")
    unresolved = False
    for r in out:
        mins = r["fit_seconds"] / 60.0
        # A fit that did not finish, or that finished carrying a C3 failure, is
        # not a cost measurement. NOTE the corrected saturation rule: saturated
        # WITH the anneal active is a risk flag, not a failure, so it does not
        # appear here -- it fired on a converged, correct, recovery-0.991 fit.
        problems = []
        if not (r["converged"] or r.get("stationary")):
            problems.append(f"NOT FINISHED at {r['n_iter']} iters")
        elif r.get("stationary") and not r["converged"]:
            # A legitimate end state, but a different claim from the tolerance
            # window, so it is named rather than folded in.
            print(
                f"  {r['env']:<12} finished by STATIONARITY (no improving step at "
                f"any tried step size; last relative delta already < tol), not by "
                f"the tolerance window"
            )
        if r["backtrack_exhausted"] and not r.get("stationary"):
            problems.append("stopped on a decrease while still improving")
        if not r["reached_tau_one"]:
            problems.append("stopped while tempered")
        if r["degenerate_mechanism"]:
            problems.append("degenerate scale")
        side = (
            "MINUTES -> cadence refit viable"
            if mins < 60
            else "HOURS -> fit-once-then-serve"
        )
        if problems:
            unresolved = True
            print(
                f"  {r['env']:<12} {mins:8.1f} min  -- NOT A CLEAN MEASUREMENT: "
                + "; ".join(problems)
            )
            print(f"      would read: {side}")
        else:
            print(f"  {r['env']:<12} {mins:8.1f} min  clean  -> {side}")
        print(
            f"      relative ll deltas {r['ll_tail_rel_deltas']} against tol {r['tol']}"
        )
        print(
            f"      fixed-step M-step {r['seconds_per_iter_fixed_step']}s/iter vs "
            f"epoch-based {r['seconds_per_iter_epoch_based']}s/iter "
            f"(x{r['fixed_step_speedup']})"
        )
        print(
            f"      E-step log_prob split (NOT the M-step): "
            f"{r['mechanism_log_prob_seconds']}"
        )
    if unresolved:
        print(
            "\n  Read the 'would read' lines as INDICATIVE, not as the fork's "
            "answer: a wall-clock figure from a fit that stopped on a decrease "
            "is a lower bound on a converged fit, not a measurement of one."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

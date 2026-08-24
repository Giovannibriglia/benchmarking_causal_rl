"""V-C1 -- V1 (no harm) and V5 (point-ID accuracy) at estimator level.

Pre-registered in docs/grace_v2_vc1_prereg.md (committed before any fit
here ran); criteria, the analytic truth, the decomposed curve and its
falsifiers live there. This tool only measures.

Query: q1, the per-step gate contrast E[R|do(a_bad), s] - E[R|do(other), s].
Truth: analytic -- c_r * qbar (qbar = 0.5) on the sweep arms, 0 on d_a_null.
Floor: the naive transition-pooled contrast (headline) and its episode-level
companion. GRACE: production configuration, 3 fit seeds, best-LL selection,
full episodes, C3 binding flags recorded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

C_R = {"d100": 1.0, "d050": 2.0, "d025": 4.0, "d010": 10.0, "d005": 20.0}
QBAR = 0.5
M = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--m-step-budget", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--out", default="results/vc1/report.json")
    args = ap.parse_args()

    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"

    jobs = []  # (cell_tag, env, seed, dataset_id, c_r, truth)
    for r in json.loads(Path("results/vb_recertification/report.json").read_text()):
        if r["cell"] == "d_a_null":
            jobs.append(("d_a_null", r["env"], r["seed"], r["dataset_id"], 0.0, 0.0))
    for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text()):
        tag = r["cell"].split("_")[-1]
        c_r = C_R[tag]
        jobs.append((tag, r["env"], r["seed"], r["dataset_id"], c_r, c_r * QBAR))

    out = []
    for tag, env, sd, did, c_r, truth in jobs:
        s, blocks = rebuild_samples(minari.load_dataset(did), 10_000)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        ep = s["episode"]
        ep_ids = np.unique(ep)
        # a_bad = 1 (asserted per env earlier via episode-level tilt; constant).
        a_bad = 1
        m = s["a"] == a_bad
        naive_tr = float(s["r"][m].mean() - s["r"][~m].mean())
        # episode-level companion: within-episode contrast averaged over
        # episodes containing both action groups (one row per episode).
        contrasts = []
        for e in ep_ids:
            me = ep == e
            am = s["a"][me] == a_bad
            if am.any() and (~am).any():
                contrasts.append(float(s["r"][me][am].mean() - s["r"][me][~am].mean()))
        naive_ep = float(np.mean(contrasts)) if contrasts else float("nan")

        has_proxy = bool(s["z"].size)
        proxy_names = (
            (("Z", "W", "V") if s["v"].size else ("Z", "W")) if has_proxy else ()
        )

        def t(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=device)

        kw = {}
        if has_proxy:
            kw["proxy"] = {"Z": t(s["z"]), "W": t(s["w"])}
            if s["v"].size:
                kw["proxy"]["V"] = t(s["v"])
        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(ep, torch.long),
            **kw,
        )
        rng = np.random.default_rng(0)
        idx = rng.choice(state.shape[0], size=min(256, state.shape[0]), replace=False)
        eval_states = torch.tensor(state[idx], dtype=torch.float32, device=device)

        fits = []
        for fs in args.fit_seeds:
            est = LatentClassEstimator(
                state_dim=state.shape[1],
                n_actions=int(s["a"].max()) + 1,
                proxy_names=proxy_names,
                device=device,
                seed=fs,
            )
            fit = est.fit(
                data,
                max_iter=args.max_iter,
                init="proxy" if has_proxy else "random",
                m_step_budget=args.m_step_budget,
                batch_size=args.batch_size,
            )
            bad = est.interventional_sweep(eval_states, [a_bad] * idx.size, fit)
            other = est.interventional_sweep(
                eval_states, [0 if a_bad != 0 else 1] * idx.size, fit
            )
            fits.append(
                {
                    "fit_seed": fs,
                    "final_ll": float(fit.final_ll),
                    "do_contrast": float((bad.value - other.value).mean().detach()),
                    "converged": bool(fit.converged),
                    "finished": bool(fit.finished),
                    "stationary": bool(fit.stationary),
                    "backtrack_exhausted": bool(fit.backtrack_exhausted),
                    "tau1_budget_bound": bool(fit.tau1_budget_bound),
                    "backtracks": int(fit.backtracks),
                    "label": bad.label(),
                }
            )
        best = max(fits, key=lambda f: f["final_ll"])
        rec = {
            "cell": tag,
            "env": env,
            "seed": sd,
            "c_r": c_r,
            "truth": truth,
            "naive_transition": round(naive_tr, 4),
            "naive_episode": round(naive_ep, 4),
            "grace_contrast": round(best["do_contrast"], 4),
            "err_naive_tr": round(abs(naive_tr - truth), 4),
            "err_naive_ep": round(abs(naive_ep - truth), 4),
            "err_grace": round(abs(best["do_contrast"] - truth), 4),
            "fits": fits,
        }
        out.append(rec)
        print(
            f"  {tag:<9} {env:<12} s{sd}  truth={truth:5.2f}  "
            f"naive_tr={naive_tr:+7.3f}  grace={best['do_contrast']:+7.3f}  "
            f"|e|_naive={rec['err_naive_tr']:.3f}  |e|_grace={rec['err_grace']:.3f}  "
            f"conv={best['converged']}",
            flush=True,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

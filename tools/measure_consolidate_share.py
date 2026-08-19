"""How much of the M-step was EWC consolidation? A/B on the same data and budget.

NBN defaults ``consolidate=True`` on ``fit``/``fit_local``, so every M-step in
the EM loop has been paying a diagonal-Fisher pass -- up to ``sample_cap = 4096``
SEQUENTIAL per-sample backward passes, per node, per call.

The EM loop does not need it: consolidation exists for continual learning, where
a mechanism must retain earlier tasks, whereas an M-step is a fresh weighted fit
of the same nodes on the same rows -- and GRACE never calls ``update()``, so the
snapshot is never read.

Measured rather than inferred: identical dataset, identical budget, identical
seed, varying only the flag. Reported as a share so the fraction attributable to
GRACE's own algorithm is separated from library overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _state_hash(state_dict) -> str:
    """SHA-256 over all non-EWC entries, order-canonicalised."""
    import hashlib

    h = hashlib.sha256()
    for k in sorted(state_dict):
        if "_ewc" in k:
            continue
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/cost/consolidate_share.json")
    ap.add_argument(
        "--reseed-per-m-step",
        action="store_true",
        help="reseed the global torch RNG before every M-step, so the two arms "
        "see identical RNG state at each refit. Isolates the ONE channel by "
        "which an inert consolidation can still change the fit: the Fisher "
        "pass consumes global RNG (randperm at online_laplace.py:87 whenever "
        "n > sample_cap), and under restart-EM every subsequent refit is a "
        "draw from that stream. Identical fits under this flag establish "
        "pure overhead; divergence under it is a real effect.",
    )
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    row = next(
        r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d"
        and r["env"] == args.env
        and r["seed"] == args.dataset_seed
        and r["sigma"] == 1.0
    )
    s, blocks = rebuild_samples(minari.load_dataset(row["dataset_id"]), args.episodes)
    state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
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

    class ReseededEstimator(LatentClassEstimator):
        """Identical RNG state at the top of every M-step, in both arms."""

        def m_step(self, data, resp, **fit_kwargs):
            torch.manual_seed(0xC0FFEE)
            return super().m_step(data, resp, **fit_kwargs)

    est_cls = ReseededEstimator if args.reseed_per_m_step else LatentClassEstimator
    out = []
    for consolidate in (True, False):
        est = est_cls(
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
            epochs=args.epochs,
            init="proxy",
            consolidate=consolidate,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        h = fit.hard_assignment().cpu().numpy()
        rec = {
            "env": args.env,
            "episodes": int(np.unique(ep).size),
            "transitions": int(ep.size),
            "consolidate": consolidate,
            "seconds": round(dt, 1),
            "n_iter": fit.n_iter,
            "seconds_per_iter": round(dt / max(fit.n_iter, 1), 2),
            "recovery": float(max((h == u_ep).mean(), (h != u_ep).mean())),
            "final_ll": float(fit.final_ll),
            "converged": bool(fit.converged),
            "finished": bool(fit.finished),
            "reseed_per_m_step": bool(args.reseed_per_m_step),
            # Hash of the fitted parameters EXCLUDING the _ewc_* buffers,
            # which exist only in the consolidate=True arm by construction.
            # Equal hashes across arms = consolidation left the fit bitwise
            # untouched; check 1 (see the log) already shows it writes
            # nothing else.
            "state_sha256_ex_ewc": _state_hash(est.model.state_dict()),
        }
        out.append(rec)
        print(
            f"  consolidate={str(consolidate):<5} {dt:7.1f}s  "
            f"{rec['seconds_per_iter']:6.2f}s/iter  n_iter={fit.n_iter}  "
            f"rec={rec['recovery']:.3f}  ll={fit.final_ll:.1f}",
            flush=True,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))

    on, off = out[0], out[1]
    share = 1.0 - off["seconds_per_iter"] / max(on["seconds_per_iter"], 1e-9)
    print(f"\n  per-iteration: {on['seconds_per_iter']}s -> {off['seconds_per_iter']}s")
    print(
        f"  CONSOLIDATION SHARE of the M-step: {share * 100:.1f}%  "
        f"(speedup x{on['seconds_per_iter'] / max(off['seconds_per_iter'], 1e-9):.2f})"
    )
    print(
        "\n  PRE-WARM-START LABEL: measured under restart-EM (NBN R3 not yet "
        "merged). The share is a fraction of a baseline M-step that warm-start "
        "will make substantially cheaper, so the SHARE will shrink; the "
        "absolute per-call saving persists."
    )
    print(
        "\n  Two claims, different strength. The per-iteration RATE stands "
        "regardless of endpoints. The PURE-OVERHEAD claim needs identical "
        "fits across arms:"
    )
    identical = (
        on["n_iter"] == off["n_iter"]
        and on["final_ll"] == off["final_ll"]
        and on["state_sha256_ex_ewc"] == off["state_sha256_ex_ewc"]
    )
    print(
        f"    n_iter {on['n_iter']} vs {off['n_iter']}, "
        f"final_ll {on['final_ll']:.4f} vs {off['final_ll']:.4f}, "
        f"state(ex-EWC) {on['state_sha256_ex_ewc']} vs "
        f"{off['state_sha256_ex_ewc']}"
    )
    if identical:
        print(
            "    IDENTICAL -> pure overhead ESTABLISHED"
            + (
                " (divergence without reseeding was the Fisher pass consuming "
                "global RNG; see --reseed-per-m-step)"
                if args.reseed_per_m_step
                else ""
            )
        )
    elif args.reseed_per_m_step:
        print(
            "    DIVERGE UNDER PER-M-STEP RESEEDING -> consolidation affects "
            "the fit through something other than RNG consumption; NOT a "
            "free win, and that is the finding."
        )
    else:
        print(
            "    DIFFER -> unresolved between a real effect and RNG "
            "consumption by the Fisher pass; re-run with --reseed-per-m-step "
            "to separate them. Report the speedup, not the equivalence."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

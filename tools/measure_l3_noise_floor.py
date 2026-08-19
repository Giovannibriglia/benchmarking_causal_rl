"""Calibrate the L3 fit's run-to-run noise floor, and re-test determinism at scale.

Every ll gap read off the consolidate A/B runs (12, 143, 210 nats) was judged
against an uncalibrated floor. This measures it: ``k`` repeats of the IDENTICAL
fit (same data, seed, config) under the default kernels, reporting the spread
of final_ll, n_iter and recovery. Any future "these two fits differ" claim is
read against this spread.

Also re-runs the determinism probe's outcome-2 result at the SCALE THAT FAILED
(the 300-episode control diverged; the 100-episode probe did not): a pair of
fits under ``torch.use_deterministic_algorithms(True)``. Bitwise identity here
recovers the bitwise instrument at production scale, at the probe's measured
~5% cost.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--deterministic-pairs", type=int, default=2)
    ap.add_argument("--out", default="results/cost/l3_noise_floor.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import time

    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.measure_consolidate_share import _state_hash
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

    def one_fit():
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
            epochs=args.epochs,
            init="proxy",
            consolidate=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        h = fit.hard_assignment().cpu().numpy()
        return {
            "seconds": round(time.time() - t0, 1),
            "n_iter": fit.n_iter,
            "final_ll": float(fit.final_ll),
            "recovery": float(max((h == u_ep).mean(), (h != u_ep).mean())),
            "state_sha256_ex_ewc": _state_hash(est.model.state_dict()),
        }

    out = {"env": args.env, "episodes": args.episodes, "device": device}

    torch.use_deterministic_algorithms(False)
    runs = []
    for i in range(args.k):
        r = one_fit()
        runs.append(r)
        print(
            f"  default #{i + 1}: ll {r['final_ll']:.4f}  n_iter {r['n_iter']}  "
            f"rec {r['recovery']:.3f}  {r['state_sha256_ex_ewc']}  ({r['seconds']}s)",
            flush=True,
        )
    lls = [r["final_ll"] for r in runs]
    out["default"] = {
        "runs": runs,
        "ll_spread": max(lls) - min(lls),
        "n_iter_values": sorted({r["n_iter"] for r in runs}),
        "recovery_values": sorted({r["recovery"] for r in runs}),
        "distinct_states": len({r["state_sha256_ex_ewc"] for r in runs}),
    }
    print(
        f"\n  NOISE FLOOR over k={args.k} identical fits: ll spread "
        f"{out['default']['ll_spread']:.4f} nats, n_iter {out['default']['n_iter_values']}, "
        f"recovery {out['default']['recovery_values']}, "
        f"{out['default']['distinct_states']} distinct parameter states",
        flush=True,
    )

    det = []
    try:
        torch.use_deterministic_algorithms(True)
        for i in range(args.deterministic_pairs):
            r = one_fit()
            det.append(r)
            print(
                f"  deterministic #{i + 1}: ll {r['final_ll']:.4f}  "
                f"n_iter {r['n_iter']}  {r['state_sha256_ex_ewc']}  ({r['seconds']}s)",
                flush=True,
            )
        identical = len({r["state_sha256_ex_ewc"] for r in det}) == 1 and (
            len({r["final_ll"] for r in det}) == 1
        )
        out["deterministic"] = {"runs": det, "bitwise_identical": identical}
        print(
            f"\n  DETERMINISTIC MODE at the failing scale: "
            f"{'BITWISE IDENTICAL -> instrument recovered' if identical else 'STILL DIVERGES'}",
            flush=True,
        )
    except RuntimeError as e:
        out["deterministic"] = {"raised": str(e)}
        print(f"\n  deterministic mode RAISED -> {e}", flush=True)
    finally:
        torch.use_deterministic_algorithms(False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

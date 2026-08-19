"""Is the L3 fit's run-to-run nondeterminism removable? Three outcomes, all useful.

The consolidate A/B's determinism control found two bitwise-identical
``consolidate=False`` fits (same data, seed, config, RNG identity by
construction) returning different parameters and a ~12-nat final_ll gap --
which closed off bitwise equivalence testing project-wide. Before accepting
that as permanent, this probe asks torch to remove the cause:

1. ``torch.use_deterministic_algorithms(True)`` RAISES on an op with no
   deterministic implementation -> record the op; the distributional route
   (paired seeds, median) stands permanently, with a stated cause.
2. It runs and repeats are BITWISE IDENTICAL -> nondeterministic kernels were
   the whole cause; measure the cost, and bitwise equivalence testing is
   recoverable by enabling the flag for those tests specifically.
3. It runs and repeats STILL DIVERGE -> a stronger finding: the divergence has
   a source torch's flag does not govern, worth its own investigation.

Small on purpose (S11): subsampled REAL episodes, full fit, never a cut budget.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set before CUDA initialises; required by torch for deterministic
# cuBLAS (otherwise use_deterministic_algorithms raises at the first matmul).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/cost/l3_determinism_probe.json")
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

    out = {"env": args.env, "episodes": args.episodes, "device": device, "modes": {}}
    for mode, flag in (("default", False), ("deterministic", True)):
        try:
            torch.use_deterministic_algorithms(flag)
            a, b = one_fit(), one_fit()
            identical = a["state_sha256_ex_ewc"] == b["state_sha256_ex_ewc"]
            out["modes"][mode] = {"runs": [a, b], "bitwise_identical": identical}
            print(
                f"  {mode:>13}: {'BITWISE IDENTICAL' if identical else 'DIVERGE'}  "
                f"ll {a['final_ll']:.4f} vs {b['final_ll']:.4f}  "
                f"n_iter {a['n_iter']} vs {b['n_iter']}  "
                f"({a['seconds']}s, {b['seconds']}s)",
                flush=True,
            )
        except RuntimeError as e:
            # Outcome 1: an op with no deterministic implementation.
            out["modes"][mode] = {"raised": str(e)}
            print(f"  {mode:>13}: RAISED -> {e}", flush=True)
        finally:
            torch.use_deterministic_algorithms(False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

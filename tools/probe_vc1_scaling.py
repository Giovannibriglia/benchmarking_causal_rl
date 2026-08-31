"""The registered 1/(d*sqrt(n)) scaling check for the V5 d=0.05 failure.

Prediction registered in docs/grace_v2_vc1_prereg.md BEFORE this ran: GRACE's
M-normalised |error| ~ delta/d with delta ~ 1/sqrt(n), so at fixed d,
err(n)/err(3000) ~ sqrt(3000/n). CartPole, d in {0.10, 0.05}, n in {750, 1500}
(n = 3000 already measured by V-C1). Production configuration otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))


def main() -> int:
    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    C_R = {"d010": 10.0, "d005": 20.0}
    gen = {
        (r["cell"], r["env"], r["seed"]): r["dataset_id"]
        for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text())
    }
    out = []
    for tag in ("d010", "d005"):
        truth = C_R[tag] * 0.5
        for sd in (0, 1, 2):
            did = gen[(f"d_d_sweep_{tag}", "CartPole-v1", sd)]
            for n_ep in (750, 1500):
                s, blocks = rebuild_samples(minari.load_dataset(did), n_ep)
                state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
                ep = s["episode"]

                def t(x, dtype=torch.float32):
                    return torch.tensor(x, dtype=dtype, device=device)

                data = EpisodeData(
                    state=t(state),
                    action=t(s["a"], torch.long),
                    reward=t(s["r"]),
                    episode_ids=t(ep, torch.long),
                    proxy={"Z": t(s["z"]), "W": t(s["w"]), "V": t(s["v"])},
                )
                rng = np.random.default_rng(0)
                idx = rng.choice(
                    state.shape[0], size=min(256, state.shape[0]), replace=False
                )
                ev = torch.tensor(state[idx], dtype=torch.float32, device=device)
                fits = []
                for fs in (0, 1, 2):
                    est = LatentClassEstimator(
                        state_dim=state.shape[1],
                        n_actions=int(s["a"].max()) + 1,
                        proxy_names=("Z", "W", "V"),
                        device=device,
                        seed=fs,
                    )
                    fit = est.fit(
                        data,
                        max_iter=30,
                        init="proxy",
                        m_step_budget=400,
                        batch_size=4096,
                    )
                    bad = est.interventional_sweep(ev, [1] * idx.size, fit)
                    oth = est.interventional_sweep(ev, [0] * idx.size, fit)
                    fits.append(
                        {
                            "fit_seed": fs,
                            "final_ll": float(fit.final_ll),
                            "do": float((bad.value - oth.value).mean().detach()),
                        }
                    )
                best = max(fits, key=lambda f: f["final_ll"])
                err = abs(best["do"] - truth)
                out.append(
                    {"cell": tag, "seed": sd, "n_ep": n_ep, "err": round(err, 4)}
                )
                print(f"  {tag} s{sd} n={n_ep}: |e|/M = {err:.4f}", flush=True)
                Path("results/vc1/scaling_probe.json").write_text(
                    json.dumps(out, indent=1)
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

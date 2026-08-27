"""V4 follow-up -- attribute degenerate-mechanism replicate failures by CHANNEL.

The persisted bootstrap diagnostics count degenerate replicates but do not say
WHICH mechanism hit its floor. The Acrobot-s1 cluster is already attributed at
the data level (the one-terminating-episode rare reward level flipping
``_resolve_reward_type`` to MDN-R on ~6/19 resamples -- see the s1 note in the
handoff). This tool refits replicates for rows whose failures that path does
NOT explain and reads ``resolved_reward_mechanism`` + ``mechanism_degeneracy``
per replicate. Determinism: replicate i uses seed ``fit_seed + 1 + i`` exactly
as ``bootstrap_null`` assigns it, so these ARE the run's replicates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

FK = dict(max_iter=30, m_step_budget=400, batch_size=4096)
FIT_SEED = 0
B = 19
ROWS = (  # rows whose failures the support-growth path does NOT explain
    ("d005", "CartPole-v1", 1),  # 4/19 failed, all degenerate-mechanism
    ("d100", "CartPole-v1", 2),  # 3/19 failed, s2 -- the background rate
)


def main() -> int:
    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator
    from src.rl.offline.grace.l4 import _dirty, _resample_episode_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = {
        (r["cell"], r["env"], r["seed"]): r["dataset_id"]
        for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text())
    }
    out_path = Path("results/v4/replicate_attribution.json")
    report = []
    for tag, env, sd in ROWS:
        did = gen[(f"d_d_sweep_{tag}", env, sd)]
        from tools.recertify_diagram_arms import rebuild_samples

        s, blocks = rebuild_samples(minari.load_dataset(did), 10_000)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
        proxy = {"Z": t(s["z"]), "W": t(s["w"])}
        if s["v"].size:
            proxy["V"] = t(s["v"])
        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(s["episode"], torch.long),
            proxy=proxy,
        )
        pn = tuple(proxy.keys())
        print(f"\n=== {tag} {env} s{sd} ({did}) ===", flush=True)
        row = {"cell": tag, "env": env, "seed": sd, "replicates": []}
        for i in range(B):
            rep_seed = FIT_SEED + 1 + i
            rng = np.random.default_rng(rep_seed)
            rdata = _resample_episode_data(data, rng)
            est_r = LatentClassEstimator(
                state_dim=state.shape[1],
                n_actions=int(s["a"].max()) + 1,
                proxy_names=pn,
                device=device,
                seed=FIT_SEED,
            )
            fit_r = est_r.fit(rdata, init="proxy", **FK)
            dirty = _dirty(fit_r)
            deg = {
                k: round(v, 4) for k, v in fit_r.mechanism_degeneracy.items() if v > 0
            }
            rec = {
                "rep_seed": rep_seed,
                "resolved_R": est_r.resolved_reward_mechanism,
                "dirty": dirty,
                "degenerate_channels": deg,
                "finished": bool(fit_r.finished),
                "backtrack_exhausted": bool(fit_r.backtrack_exhausted),
            }
            row["replicates"].append(rec)
            print(
                f"  rep {rep_seed:2d}: R={est_r.resolved_reward_mechanism:<16} "
                f"degen={deg or '-'} dirty={dirty or 'clean'}",
                flush=True,
            )
        report.append(row)
        out_path.write_text(json.dumps(report, indent=1))
    print("\nATTRIBUTION COMPLETE ->", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

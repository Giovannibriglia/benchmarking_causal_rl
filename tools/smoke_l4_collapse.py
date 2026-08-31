"""L4 collapse smoke (cheap, loud): the property that catches a wrong engine.

Scoped per the V-C2 sequence: run BEFORE V4 proper, on subsampled real data
(S11 -- this is a diagnostic). Three probes:

1. D-D sweep d=1.0 CartPole (point-ID, proxies decorative, R strong): the
   interval must be NARROW around the analytic truth 0.5.
2. d_a_null CartPole: the refusal rules ABSTAIN here by design (the observed
   fit is the degenerate MDN-on-constant) -- surfaced as a DESIGNED CONFLICT
   with the V4 gate's "D-A-null collapses" wording, for review: abstention on
   a degenerate mechanism vs a collapsing interval are both defensible
   no-harm behaviours, but the gate must name which it expects.
3. D-E CartPole: Balke-Pearl bounds from the data -- must be NON-collapsed
   (finite width well above the D-D interval's) and contain the analytic
   within-pair truth q_bar.
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
    from src.rl.offline.grace.l4 import balke_pearl_contrast_bounds, point_id_interval

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(did, n_ep):
        s, blocks = rebuild_samples(minari.load_dataset(did), n_ep)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        return s, state

    def mk_data(s, state, with_proxy):
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

    def target_factory(state):
        rng = np.random.default_rng(0)
        idx = rng.choice(state.shape[0], size=min(256, state.shape[0]), replace=False)
        ev = torch.tensor(state[idx], dtype=torch.float32, device=device)

        def target(est, fit):
            bad = est.interventional_sweep(ev, [1] * idx.size, fit)
            oth = est.interventional_sweep(ev, [0] * idx.size, fit)
            return float((bad.value - oth.value).mean().detach())

        return target

    fk = dict(max_iter=30, init="proxy", m_step_budget=400, batch_size=4096)

    # ---- probe 1: D-D d=1.0, point-ID interval must be narrow around 0.5 ----
    gen = {
        (r["cell"], r["env"], r["seed"]): r["dataset_id"]
        for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text())
    }
    s, state = load(gen[("d_d_sweep_d100", "CartPole-v1", 0)], 400)
    data = mk_data(s, state, True)
    res = point_id_interval(
        make_estimator=lambda seed: LatentClassEstimator(
            state_dim=state.shape[1],
            n_actions=int(s["a"].max()) + 1,
            proxy_names=("Z", "W", "V"),
            device=device,
            seed=seed,
        ),
        data=data,
        target=target_factory(state),
        fit_kwargs=fk,
        alpha=0.1,
        b=19,
        fit_seed=0,
        init_seeds=(1, 2),
        n_jobs=1,
    )
    print("D-D d=1.0 :", res.summary(), " truth=0.5", flush=True)

    # ---- probe 2: d_a_null, expected ABSTAIN (designed conflict) ------------
    r0 = next(
        r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_a_null" and r["env"] == "CartPole-v1" and r["seed"] == 0
    )
    s2, state2 = load(r0["dataset_id"], 400)
    data2 = mk_data(s2, state2, False)
    fk2 = dict(fk, init="random")
    res2 = point_id_interval(
        make_estimator=lambda seed: LatentClassEstimator(
            state_dim=state2.shape[1],
            n_actions=int(s2["a"].max()) + 1,
            proxy_names=(),
            device=device,
            seed=seed,
        ),
        data=data2,
        target=target_factory(state2),
        fit_kwargs=fk2,
        alpha=0.1,
        b=19,
        fit_seed=0,
        init_seeds=(1, 2),
        n_jobs=1,
    )
    print("d_a_null  :", res2.summary(), flush=True)

    # ---- probe 3: D-E Balke-Pearl bounds, must NOT collapse -----------------
    de = next(
        r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_e"
        and r["env"] == "CartPole-v1"
        and r["seed"] == 0
        and r["sigma"] == 1.0
    )
    s3, _ = load(de["dataset_id"], 3000)
    in_pair = np.isin(s3["a"], (0, 1))
    bonus = (s3["r"] > 1.5).astype(int)  # CartPole: native 1, bonus 1+c_r=2
    lo, hi = balke_pearl_contrast_bounds(
        bonus=bonus[in_pair],
        x=(s3["a"][in_pair] == 1).astype(int),
        z=s3["i"][in_pair].astype(int),
    )
    md = dict(minari.load_dataset(de["dataset_id"]).storage.metadata)
    print(
        f"D-E BP    : bounds [{lo:+.4f}, {hi:+.4f}] width={hi-lo:.4f} "
        f"(gate_probs in metadata: {md.get('gate_probs', 'n/a')})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

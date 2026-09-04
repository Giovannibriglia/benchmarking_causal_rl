"""Phase 2 — speed, MEASURED: the bitwise reference + the memory gate + the levers.

    uv run python tools/phase2_speed.py reference   # n_jobs=1 fit into the GRID cache + peak GPU memory
    uv run python tools/phase2_speed.py parallel N  # n_jobs=N into a scratch cache; bitwise vs reference
    uv run python tools/phase2_speed.py sweep       # sweep chunk 4096 vs full: bitwise? and the timing

The rule: apply nothing on argument alone; a change that alters a served
number is not a speed change. The reference doubles as the grid's first
cache entry (d100 sigma=0.25 s0, the tmdp column's ds0 declared-MDP fit).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)

DID = "grace-v2/d_d_sweep_d100-cartpole/medium-bias_confounded_action-sigma025-seed0-d100-v0"
PROXIES = ("Z", "W", "V")
GRID_CACHE = "results/grace_cache"
OUT = "results/profile"


def _load(did, dev):
    import minari

    ds = minari.load_dataset(did)
    obs, act, rew, eps, prox = [], [], [], [], {p: [] for p in PROXIES}
    for i, ep in enumerate(ds.iterate_episodes()):
        t = len(ep.actions)
        obs.append(np.asarray(ep.observations[:-1], dtype=np.float32))
        act.append(np.asarray(ep.actions))
        rew.append(np.asarray(ep.rewards, dtype=np.float32))
        eps.append(np.full(t, i))
        for p in PROXIES:
            prox[p].append(
                np.asarray(ep.infos[f"proxy_{p.lower()}"], dtype=np.float32)[:t]
            )
    buf = dict(
        obs=torch.tensor(np.concatenate(obs), device=dev),
        actions=torch.tensor(np.concatenate(act), device=dev),
        rewards=torch.tensor(np.concatenate(rew), device=dev),
        episode_ids=torch.tensor(np.concatenate(eps), device=dev),
    )
    for p in PROXIES:
        buf[f"proxy_{p}"] = torch.tensor(np.concatenate(prox[p]), device=dev)
    return buf


def _digest(serving) -> dict:
    r = serving.rewards
    h = (
        hashlib.sha256(
            r.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest()
        if r is not None
        else None
    )
    return dict(
        rewards_sha256=h,
        lo=float(serving.lo),
        hi=float(serving.hi),
        contrast_point=float(serving.meta.get("contrast_point", float("nan"))),
        label=serving.label(),
        abstained=serving.abstained,
    )


def run_fit(cache_dir, n_jobs, sweep_chunk=4096, apply=False):
    from src.rl.offline.grace.serving import transform_offline_rewards

    dev = "cuda"
    buf = _load(DID, dev)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    s = transform_offline_rewards(
        buf,
        cache_dir=cache_dir,
        dataset_id=DID,
        apply=apply,
        proxy_names=PROXIES,
        device=dev,
        n_jobs=n_jobs,
        sweep_chunk=sweep_chunk,
    )
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    free, total = torch.cuda.mem_get_info()
    out = dict(
        mode=f"n_jobs={n_jobs} sweep_chunk={sweep_chunk}",
        wall_s=wall,
        peak_alloc_mib=peak,
        peak_reserved_mib=reserved,
        gpu_total_mib=total / 2**20,
        cache=s.meta.get("transform_cache_stored", "hit?"),
        **_digest(s),
    )
    print(json.dumps(out, indent=1), flush=True)
    return out


def main() -> int:
    mode = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if mode == "reference":
        out = run_fit(GRID_CACHE, 1)
        json.dump(out, open(f"{OUT}/phase2_reference.json", "w"), indent=1)
    elif mode == "parallel":
        n = int(sys.argv[2])
        out = run_fit(f"{OUT}/scratch_cache_njobs{n}", n)
        ref = json.load(open(f"{OUT}/phase2_reference.json"))
        same = (
            out["rewards_sha256"] == ref["rewards_sha256"]
            and out["lo"] == ref["lo"]
            and out["hi"] == ref["hi"]
        )
        out["bitwise_identical_to_reference"] = bool(same)
        out["speedup_vs_reference"] = ref["wall_s"] / out["wall_s"]
        print(
            f"BITWISE {'IDENTICAL' if same else 'DIFFERENT'}; speedup x{out['speedup_vs_reference']:.2f}",
            flush=True,
        )
        json.dump(out, open(f"{OUT}/phase2_parallel_njobs{n}.json", "w"), indent=1)
    elif mode == "sweep":
        from src.rl.offline.grace.estimator import LatentClassEstimator
        from src.rl.offline.grace.serving import (
            _episode_data_from_buffer,
            DEFAULT_FIT_KWARGS,
        )

        dev = "cuda"
        buf = _load(DID, dev)
        data, _, _ = _episode_data_from_buffer(buf, proxy_names=PROXIES, device=dev)
        est = LatentClassEstimator(
            state_dim=int(data.state.shape[1]),
            n_actions=int(data.action.max()) + 1,
            proxy_names=PROXIES,
            device=dev,
            seed=0,
        )
        fit = est.fit(data, **dict(DEFAULT_FIT_KWARGS, init="proxy"))
        res = {}
        chunks = [int(c) for c in sys.argv[2:]] or [4096, int(data.state.shape[0])]
        for chunk in chunks:
            t0 = time.time()
            vals = []
            for k in range(0, data.state.shape[0], chunk):
                c = data.state[k : k + chunk]
                vals.append(
                    est.interventional_sweep(c, [1] * c.shape[0], fit)
                    .value.reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            v = np.concatenate(vals)
            res[chunk] = dict(
                wall_s=time.time() - t0,
                sha256=hashlib.sha256(v.astype(np.float32).tobytes()).hexdigest(),
                mean=float(v.mean()),
            )
            print(chunk, res[chunk], flush=True)
        base = res[chunks[0]]
        for c in chunks[1:]:
            same = res[c]["sha256"] == base["sha256"]
            print(
                f"SWEEP chunk {c} vs {chunks[0]}: {'BITWISE IDENTICAL' if same else 'DIFFERENT'}; "
                f"speedup x{base['wall_s'] / res[c]['wall_s']:.2f}",
                flush=True,
            )
            res[c]["bitwise_identical_to_first"] = bool(same)
        json.dump(
            {str(k_): v for k_, v in res.items()},
            open(f"{OUT}/phase2_sweep_chunk.json", "w"),
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

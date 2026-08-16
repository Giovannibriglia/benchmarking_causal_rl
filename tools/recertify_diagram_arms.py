"""V-B re-certification — RECOMPUTE the preflight stamp from stored samples.

Certification is a STAMP, not generation. Every quantity the ground-truth
preflight consumes -- the logged ``U``, the proxies, the instrument, the
actions, the rewards, the observations and the episode blocks -- is already
persisted in each Minari dataset's ``infos``. So when the statistics themselves
are corrected (S1b granularity, the null of the max, a quantile tail), the 130
datasets do not need regenerating: they need re-reading. Minutes against the
original run's hours.

  uv run python tools/recertify_diagram_arms.py

FAITHFULNESS OF THE RECONSTRUCTION IS ITSELF CHECKED, and it has to be: the
whole point is to trust a number computed from a reassembled sample dict rather
than from the generator's own in-memory one. Two stamped quantities are
recomputed the OLD way and compared to what the dataset carries --

  * ``preflight_transitions`` / ``preflight_episodes``, pure counts, for every
    cell; and
  * a deterministic float that the code change did not touch --
    ``preflight_proxy_corr_z_u`` (a transition-level correlation) for D-D, and
    ``preflight_drift_realised_autocorr`` for D-B'.

-- so a reconstruction that has silently mis-ordered episodes, dropped a step or
mis-aligned an infos array cannot pass. A mismatch is a hard failure, never a
warning: a wrong reconstruction would produce a plausible certification table
with no other symptom.

The re-stamp writes back into the dataset metadata by default (``--no-write`` to
inspect first), because a dataset must carry its own validity proof rather than
depend on a verification someone remembers having run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CELLS = Path("reproducibility/rl_regimes/diagrams")


def rebuild_samples(ds, max_episodes: int):
    """Reassemble the generator's ``samples`` dict and per-episode observations.

    Mirrors ``_rollout_vectorized``'s sig_* construction exactly: ``sig_a`` is
    the action as a float (discrete envs here), ``sig_r`` the reward, ``sig_u``
    the logged latent, and the episode INDEX -- not a running counter -- is what
    lets the permutation nulls move whole episodes.
    """
    a, r, u, z, w, i, ep = [], [], [], [], [], [], []
    obs_blocks = []
    for k, episode in enumerate(ds.iterate_episodes()):
        if k >= max_episodes:
            break
        infos = getattr(episode, "infos", None) or {}
        acts = np.asarray(episode.actions, dtype=np.float64).reshape(-1)
        rews = np.asarray(episode.rewards, dtype=np.float64).reshape(-1)
        t = acts.size
        a.append(acts)
        r.append(rews)
        u.append(np.asarray(infos["confounder_u"], dtype=np.float64).reshape(-1))
        if "proxy_z" in infos:
            z.append(np.asarray(infos["proxy_z"], dtype=np.float64).reshape(-1))
            w.append(np.asarray(infos["proxy_w"], dtype=np.float64).reshape(-1))
        if "instrument_i" in infos:
            i.append(np.asarray(infos["instrument_i"], dtype=np.float64).reshape(-1))
        ep.append(np.full(t, k, dtype=np.int64))
        # Kept at FULL length T+1. ``_preflight_certification`` slices ``[:-1]``
        # itself to get each transition's source state, exactly as it does on the
        # generator's own buffers -- pre-trimming here silently shortens every
        # episode by one step, which numpy catches only because bincount is
        # strict about lengths.
        obs_blocks.append(np.asarray(episode.observations, dtype=np.float64))

    def cat(chunks):
        return np.concatenate(chunks) if chunks else np.zeros(0)

    samples = {
        "a": cat(a),
        "r": cat(r),
        "u": cat(u),
        "z": cat(z),
        "w": cat(w),
        "i": cat(i),
        "episode": cat(ep).astype(np.int64),
    }
    buffers = [SimpleNamespace(observations=o) for o in obs_blocks]
    return samples, buffers


def _pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def verify_reconstruction(meta: dict, samples: dict, cell: str) -> list[str]:
    """Hard checks that the reassembled samples ARE the certified ones.

    ``meta`` MUST come from the archived pre-re-certification baseline, never
    from the live dataset metadata -- this tool WRITES that metadata, so on a
    second run the check would be comparing the reconstruction against its own
    previous output. It bit exactly that way: run 1 re-stamped
    ``preflight_proxy_corr_z_u`` with the new EPISODE-level value, and run 2
    then flagged seven datasets because it was recomputing the OLD
    transition-level quantity and comparing it to an episode-level number. A
    verification whose reference is mutable, and mutated by the thing it
    verifies, degrades to a no-op or a false alarm on the second run.
    """
    problems = []
    n_ep = int(np.unique(samples["episode"]).size)
    n_tr = int(samples["episode"].size)
    if meta.get("preflight_episodes") not in (None, n_ep):
        problems.append(f"episodes {n_ep} != stamped {meta['preflight_episodes']}")
    if meta.get("preflight_transitions") not in (None, n_tr):
        problems.append(
            f"transitions {n_tr} != stamped {meta['preflight_transitions']}"
        )
    stamped = meta.get("preflight_proxy_corr_z_u")
    if stamped is not None and samples["z"].size:
        # Recomputed the OLD way -- transition level -- precisely because this
        # value must be reproduced bit-for-bit to prove the reassembly is right.
        got = _pearson(samples["z"], samples["u"])
        if abs(got - float(stamped)) > 1e-9:
            problems.append(f"corr(Z,U) {got:.12f} != stamped {float(stamped):.12f}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="results/vb_generation/report.json")
    ap.add_argument("--out", default="results/vb_recertification/report.json")
    ap.add_argument("--max-episodes", type=int, default=600)
    ap.add_argument(
        "--baseline",
        default="results/vb_recertification/pre_recert_metadata.json",
        help="archived PRE-re-certification metadata; the immutable reference "
        "the reconstruction check compares against",
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="recompute and report without re-stamping dataset metadata",
    )
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    from src.benchmarking.regime_sweep import load_sweep_spec
    from src.envs.offline.generate import _preflight_certification

    rows_in = json.loads(Path(args.report).read_text())
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(
            f"!! no baseline at {baseline_path}. The reconstruction check needs the "
            "PRE-re-certification stamps; re-stamped metadata cannot verify itself. "
            "Archive them first (see the module docstring)."
        )
        return 2
    baseline = json.loads(baseline_path.read_text())
    specs = {
        c: load_sweep_spec(CELLS / f"{c}.yaml") for c in {r["cell"] for r in rows_in}
    }
    local = set(minari.list_local_datasets())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []
    recon_failures: list[str] = []
    t0 = time.time()

    for n, row in enumerate(rows_in, 1):
        did, cell = row["dataset_id"], row["cell"]
        if did not in local:
            recon_failures.append(f"{did}: absent from the store")
            continue
        spec = specs[cell]
        ds = minari.load_dataset(did)
        meta = baseline.get(did, {})
        if not meta:
            recon_failures.append(f"{did}: no baseline stamp to verify against")
            continue
        samples, buffers = rebuild_samples(ds, args.max_episodes)

        bad = verify_reconstruction(meta, samples, cell)
        kw = spec.arm_generator_kwargs(row["sigma"])
        if spec.u_drift:
            # The one deterministic float for D-B'; check_drift's pooled
            # autocorrelation is untouched by the granularity fix, so it must
            # reproduce exactly.
            from src.envs.offline.arm_preflight import check_drift

            ep = samples["episode"]
            by_ep = [samples["u"][ep == e] for e in np.unique(ep)]
            got = check_drift(u_by_episode=by_ep, rho=float(spec.u_drift))
            stamped = meta.get("preflight_drift_realised_autocorr")
            if stamped is not None and abs(got.realised_autocorr - stamped) > 1e-9:
                bad.append(
                    f"drift autocorr {got.realised_autocorr:.12f} != stamped {stamped:.12f}"
                )
        if bad:
            recon_failures.append(f"{did}: " + "; ".join(bad))
            continue

        cert = _preflight_certification(
            samples,
            buffers,
            proxy_strength=kw.get("proxy_strength"),
            instrument_strength=kw.get("instrument_strength"),
            u_drift=kw.get("u_drift"),
            max_episodes=args.max_episodes,
            null_arm=not spec.confounder_c_r,
            a_bad=1,
        )
        if not args.no_write:
            ds.storage.update_metadata(cert)

        report.append(
            {
                **{k: row[k] for k in ("cell", "diagram", "env", "seed", "sigma")},
                "dataset_id": did,
                "gate_passed": row.get("gate_passed"),
                "was_preflight_passed": row.get("preflight_passed"),
                "was_reasons": row.get("preflight_reasons"),
                "was_k_ranks": row.get("proxy_k_ranks"),
                "was_margins": row.get("proxy_margins"),
                **{k: v for k, v in cert.items()},
            }
        )
        flip = ""
        if row.get("preflight_passed") is not cert.get("preflight_passed"):
            flip = (
                f"  <-- {row.get('preflight_passed')} -> {cert.get('preflight_passed')}"
            )
        print(
            f"[{n}/{len(rows_in)}] {cell} {row['env']} s{row['seed']} "
            f"sigma={row['sigma']:<5} preflight={cert.get('preflight_passed')}{flip}",
            flush=True,
        )
        out_path.write_text(json.dumps(report, indent=1))

    # S7: the completion invariant. A partial re-certification that reports a
    # clean table is worse than one that fails loudly.
    if recon_failures:
        print("\n=== RECONSTRUCTION FAILED — the table below is NOT valid ===")
        for f in recon_failures:
            print("  !", f)
        return 2
    if len(report) != len(rows_in):
        print(f"\n=== {len(report)} rows against {len(rows_in)} expected ===")
        return 2

    failed = [r for r in report if not r.get("preflight_passed")]
    print(
        f"\n=== re-certified {len(report)} datasets in {(time.time() - t0) / 60:.1f} min: "
        f"{len(failed)} still FAIL preflight ==="
    )
    for r in failed:
        print(
            " FAIL", r["cell"], r["env"], r["seed"], r["sigma"], r["preflight_reasons"]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

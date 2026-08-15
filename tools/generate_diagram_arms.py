"""V-B — generate the GRACE v2 diagram arms at production budget.

Generation only: no training, no critics. Each cell's channels are derived from
its declared diagram (never from the YAML), each dataset is certified by the
ground-truth preflight at write time, and the certification is stamped into the
Minari metadata so it travels with the data.

  uv run python tools/generate_diagram_arms.py --cells d_d d_e d_b_prime d_a_null

One shared pi_basic per (env, seed) across every point of a cell, exactly as the
regime sweep does — otherwise cross-point deltas are confounded by generator
variance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CELLS = Path("reproducibility/rl_regimes/diagrams")


def dataset_id_for(cell: str, k, env_id: str, seed: int, sigma: float) -> str:
    """THE id for one grid point — the single place identity is constructed.

    Every field that distinguishes a grid point must appear here. ``dataset_name``
    carries env, tier, policy and sigma but NOT the seed, and that omission cost
    an entire V-B run: all five seeds resolved to one id, and because the
    FINGERPRINT does include the seed, each seed saw the previous one's data as
    stale, deleted it and regenerated. 27 report rows collapsed to 8 datasets,
    silently.

    Factored out so ``test_diagram_arm_ids_are_injective`` can enumerate the
    whole grid and assert injectivity without generating anything — which makes
    this bug class unreachable rather than merely fixed.
    """
    from src.envs.offline.generate import dataset_name

    return (
        dataset_name(env_id, "medium", k.behavior_policy, sigma)
        .replace("generated/", f"grace-v2/{cell}-")
        .replace("-v0", f"-seed{seed}-v0")
    )


def grid_ids(cell: str, spec) -> list:
    """Every id this cell's driver run can produce, in order."""
    from src.envs.offline.diagram_arms import arm_knobs

    out = []
    for env_id in spec.envs:
        for seed in spec.seeds:
            for _, sigma in spec.points():
                k = arm_knobs(
                    spec.diagram,
                    sigma=sigma,
                    confounder_c_r=spec.confounder_c_r,
                    proxy_strength=spec.proxy_strength,
                    instrument_strength=spec.instrument_strength,
                    u_drift=spec.u_drift,
                    gate_probs=spec.gate_probs,
                )
                out.append(dataset_id_for(cell, k, env_id, seed, sigma))
    return out


def _row_from(ds, cell, spec, env_id, seed, sigma, did, ghash, seconds) -> dict:
    """One report row, built identically whether the dataset was just generated
    or reused — so a resumed run's report is not a different shape."""
    m = dict(ds.storage.metadata)
    return {
        "cell": cell,
        "diagram": spec.diagram,
        "env": env_id,
        "seed": seed,
        "sigma": sigma,
        "dataset_id": did,
        "generator_hash": ghash,
        "seconds": seconds,
        "gate_passed": m.get("gate_test_passed"),
        "gate_type": m.get("gate_type"),
        "preflight_passed": m.get("preflight_passed"),
        "preflight_reasons": m.get("preflight_reasons"),
        "proxy_k_ranks": m.get("preflight_proxy_k_ranks"),
        "proxy_margins": m.get("preflight_proxy_margins"),
        "instrument_null_sds": m.get("preflight_instrument_null_sds"),
        "instrument_exclusion_testable": m.get(
            "preflight_instrument_exclusion_testable"
        ),
        "drift_realised": m.get("preflight_drift_realised_autocorr"),
        "null_arm_u_inert": m.get("preflight_null_arm_u_inert"),
        "null_arm_null_sds": m.get("preflight_null_arm_null_sds"),
        "ok": bool(
            m.get("gate_test_passed") and m.get("preflight_passed") is not False
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells", nargs="+", default=["d_d", "d_e", "d_b_prime", "d_a_null"]
    )
    ap.add_argument("--out", default="results/vb_generation")
    ap.add_argument("--rollout-episodes", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep datasets whose generation_fingerprint already matches",
    )
    args = ap.parse_args()

    from src.benchmarking.regime_sweep import load_sweep_spec
    from src.benchmarking.registry import register_default_algorithms

    # NOTE: dataset_name is deliberately NOT imported here. Identity is
    # constructed in exactly one place (dataset_id_for) so that the injectivity
    # test covers the path the driver actually takes -- the previous spelling
    # left an inline construction in main() that the helper-level test could not
    # see, and it collided across seeds while the test passed.
    from src.envs.offline.generate import (
        build_generator_agent,
        generate_offline_dataset,
        generation_fingerprint,
    )
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []
    t0 = time.time()

    for cell in args.cells:
        spec = load_sweep_spec(CELLS / f"{cell}.yaml")
        envs = args.envs or spec.envs
        seeds = args.seeds or spec.seeds
        n_ep = args.rollout_episodes or spec.budget("rollout_episodes", 3000)
        print(
            f"\n=== {cell} ({spec.diagram}) — {len(envs)}x{len(seeds)}x"
            f"{len(spec.points())} points, {n_ep} episodes each ===",
            flush=True,
        )

        for env_id in envs:
            for seed in seeds:
                # ONE pi_basic per (env, seed), shared across the cell's points.
                agent, ghash = build_generator_agent(
                    env_id,
                    spec.generator_algo,
                    "medium",
                    seed=seed,
                    train_episodes=spec.budget("n_episodes", 250),
                    n_checkpoints=spec.budget("n_checkpoints", 25),
                    run_dir=str(out / "generator" / f"{env_id}_s{seed}"),
                )
                for beta, sigma in spec.points():
                    kw = spec.arm_generator_kwargs(sigma)
                    from src.envs.offline.diagram_arms import arm_knobs

                    k = arm_knobs(
                        spec.diagram,
                        sigma=sigma,
                        confounder_c_r=spec.confounder_c_r,
                        proxy_strength=spec.proxy_strength,
                        instrument_strength=spec.instrument_strength,
                        u_drift=spec.u_drift,
                        gate_probs=spec.gate_probs,
                    )
                    did = dataset_id_for(cell, k, env_id, seed, sigma)
                    # Idempotent: a partial or interrupted V-B run must be
                    # resumable, and Minari refuses to overwrite an existing id.
                    # --resume keeps a dataset whose generation_fingerprint
                    # matches (identical inputs => regenerating reproduces it),
                    # and deletes one whose fingerprint differs rather than
                    # silently serving data generated under other settings.
                    import minari

                    if did in minari.list_local_datasets():
                        existing = minari.load_dataset(did)
                        fp = dict(existing.storage.metadata).get(
                            "generation_fingerprint"
                        )
                        want = generation_fingerprint(
                            env_id=env_id,
                            generator_algo=spec.generator_algo,
                            tier="medium",
                            behavior_policy=k.behavior_policy,
                            behavior_strength=k.behavior_strength,
                            confounder_c_r=k.confounder_c_r,
                            pi_basic_epsilon=spec.pi_basic_epsilon,
                            a_bad=1,
                            rollout_episodes=n_ep,
                            seed=seed,
                            generator_hash=ghash,
                            rollout_device=spec.rollout_device,
                            rollout_n_envs=spec.rollout_n_envs,
                            legacy_rollout=spec.legacy_rollout,
                            **kw,
                        )
                        if args.resume and fp == want:
                            print(f"  {did}: reusing (fingerprint match)", flush=True)
                            report.append(
                                _row_from(
                                    existing,
                                    cell,
                                    spec,
                                    env_id,
                                    seed,
                                    sigma,
                                    did,
                                    ghash,
                                    0.0,
                                )
                            )
                            continue
                        minari.delete_dataset(did)
                    t = time.time()
                    ds = generate_offline_dataset(
                        env_id,
                        spec.generator_algo,
                        "medium",
                        behavior_policy=k.behavior_policy,
                        behavior_strength=k.behavior_strength,
                        confounder_c_r=k.confounder_c_r,
                        pi_basic_epsilon=spec.pi_basic_epsilon,
                        a_bad=1,
                        rollout_episodes=n_ep,
                        seed=seed,
                        dataset_id=did,
                        agent=agent,
                        rollout_device=spec.rollout_device,
                        rollout_n_envs=spec.rollout_n_envs,
                        **kw,
                    )
                    row = _row_from(
                        ds,
                        cell,
                        spec,
                        env_id,
                        seed,
                        sigma,
                        did,
                        ghash,
                        round(time.time() - t, 1),
                    )
                    report.append(row)
                    flag = "" if row["ok"] else "  <-- FAILED"
                    print(
                        f"  {env_id} s{seed} sigma={sigma:<5} gate={row['gate_passed']} "
                        f"preflight={row['preflight_passed']} "
                        f"({row['seconds']}s){flag}",
                        flush=True,
                    )
                    (out / "report.json").write_text(json.dumps(report, indent=1))

    # COMPLETION INVARIANT. surviving datasets == report rows == expected grid.
    # "27 rows against 8 datasets" was a contradiction the driver was in a
    # position to notice and did not, so it ran for 1h39m before the collision
    # surfaced. A mismatch is now a loud failure, never a success report.
    import minari

    expected = sum(
        len(grid_ids(c, load_sweep_spec(CELLS / f"{c}.yaml"))) for c in args.cells
    )
    if args.envs or args.seeds:
        expected = None  # a restricted sub-grid; the row/dataset check still holds
    produced = {r["dataset_id"] for r in report}
    local = set(minari.list_local_datasets())
    problems = []
    if len(produced) != len(report):
        problems.append(
            f"{len(report)} report rows collapsed to {len(produced)} distinct ids "
            "— dataset ids are COLLIDING and runs are overwriting each other"
        )
    missing = produced - local
    if missing:
        problems.append(f"{len(missing)} generated ids are absent from the store")
    if expected is not None and len(report) != expected:
        problems.append(f"{len(report)} rows against an expected grid of {expected}")
    if problems:
        print("\n=== COMPLETION INVARIANT VIOLATED ===")
        for x in problems:
            print("  !", x)
        return 2

    failed = [r for r in report if not r.get("ok")]
    print(
        f"\n=== V-B done in {(time.time() - t0) / 60:.1f} min: "
        f"{len(report)} datasets, {len(failed)} FAILED ==="
    )
    for r in failed:
        print(
            "  FAIL", r["cell"], r["env"], r["seed"], r["sigma"], r["preflight_reasons"]
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

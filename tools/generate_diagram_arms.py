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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells", nargs="+", default=["d_d", "d_e", "d_b_prime", "d_a_null"]
    )
    ap.add_argument("--out", default="results/vb_generation")
    ap.add_argument("--rollout-episodes", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--envs", nargs="+", default=None)
    args = ap.parse_args()

    from src.benchmarking.regime_sweep import load_sweep_spec
    from src.benchmarking.registry import register_default_algorithms
    from src.envs.offline.generate import (
        build_generator_agent,
        dataset_name,
        generate_offline_dataset,
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
                    did = dataset_name(
                        env_id, "medium", k.behavior_policy, sigma
                    ).replace("generated/", f"grace-v2/{cell}-")
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
                    m = dict(ds.storage.metadata)
                    row = {
                        "cell": cell,
                        "diagram": spec.diagram,
                        "env": env_id,
                        "seed": seed,
                        "sigma": sigma,
                        "dataset_id": did,
                        "generator_hash": ghash,
                        "seconds": round(time.time() - t, 1),
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
                    }
                    report.append(row)
                    # preflight_passed is None only when an arm has no channel
                    # to certify -- "not applicable", never a failure. Every arm
                    # here does have one (the null arm certifies U's INERTNESS),
                    # so None would now mean the certification did not run.
                    ok = row["gate_passed"] and row["preflight_passed"] is not False
                    flag = "" if ok else "  <-- FAILED"
                    row["ok"] = bool(ok)
                    print(
                        f"  {env_id} s{seed} sigma={sigma:<5} gate={row['gate_passed']} "
                        f"preflight={row['preflight_passed']} "
                        f"({row['seconds']}s){flag}",
                        flush=True,
                    )
                    (out / "report.json").write_text(json.dumps(report, indent=1))

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

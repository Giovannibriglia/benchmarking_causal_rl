"""E0 — null-calibrate the GRACE router thresholds on basic-arm data.

Generates INDEPENDENT basic-arm datasets (sigma=0 ``bias_confounded_action``
— the same U-recording, marginally-matched construction as the sweep's basic
origin, with its own seeds and dataset prefix so the reference stays
independent of the judged runs: the fixed-denominator philosophy), fits the
GRACE machinery on each (router components only — dataset quantities,
independent of the offline training budget; no base-learner training), and
writes per-env thresholds via ``RegimeRouter.calibrate`` (mean +/- k*sd,
k = NULL_CALIBRATION_K) to::

    reproducibility/rl_regimes/_base/grace_router_reference.yaml

A missing entry keeps the router UNCALIBRATED (serves Q_obs), so running this
BEFORE any defect cell is scored is a hard prerequisite (approved plan, E0).

Usage:
    uv run python tools/calibrate_grace_router.py \
        --cell reproducibility/rl_regimes/offline_mdp/critic_ablation.yaml \
        --seeds 100 101 102 103 104 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reproducibility"
    / "rl_regimes"
    / "_base"
    / "grace_router_reference.yaml"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, help="cell YAML (envs/budgets source)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-prefix", default="grace_nullcal")
    ap.add_argument("--out", default=str(_REFERENCE_PATH))
    args = ap.parse_args()

    import gymnasium as gym
    from src.benchmarking.regime_sweep import load_sweep_spec
    from src.config.seeding import set_seed
    from src.envs.offline.generate import (
        build_generator_agent,
        generate_offline_dataset,
    )
    from src.envs.offline.minari_loader import fill_sequence_buffer_from_minari
    from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer
    from src.rl.offline.grace import cell_graph, GraceMachinery, GraceOptions
    from src.rl.offline.grace.router import RegimeRouter

    spec = load_sweep_spec(args.cell)
    envs = args.envs or spec.envs
    device = torch.device(args.device)
    pomdp = spec.observability == "pomdp"
    graph = cell_graph(spec.observability, "template")
    re_val = spec.budget("rollout_episodes", 3000)
    gopts_kwargs = dict(spec.grace_options)
    gopts_kwargs.pop("_env_id", None)

    per_env_stats: dict[str, list[dict]] = {e: [] for e in envs}
    for env in envs:
        n_actions = int(gym.make(env).action_space.n)
        for seed in args.seeds:
            ds_id = (
                f"{args.dataset_prefix}/{spec.regime}/"
                f"{env.lower().replace('/', '-')}-seed{seed}-v0"
            )
            print(f"[grace-nullcal] {env} seed{seed}: dataset {ds_id}", flush=True)
            agent, _hash = build_generator_agent(
                env, spec.generator_algo, "random", seed=seed, device=args.device
            )
            try:
                import minari

                minari.delete_dataset(ds_id)
            except Exception:
                pass
            set_seed(seed, deterministic=True)
            generate_offline_dataset(
                env_id=env,
                generator_algo=spec.generator_algo,
                tier="random",
                behavior_policy="bias_confounded_action",
                behavior_strength=0.0,  # sigma=0: the basic (null) arm
                pi_basic_epsilon=spec.pi_basic_epsilon,
                confounder_c_r=spec.confounder_c_r,
                rollout_episodes=re_val,
                seed=seed,
                dataset_id=ds_id,
                agent=agent,
                device=args.device,
            )
            buf = SequenceReplayBuffer(capacity=2_000_000, device=device)
            fill_sequence_buffer_from_minari(
                ds_id,
                buf,
                device,
                mask_indices=(
                    tuple(spec.mask_indices.get(env, ())) or None if pomdp else None
                ),
                load_u=False,  # five-keys: the null calibration never reads U
            )
            machinery = GraceMachinery(
                graph,
                GraceOptions(**gopts_kwargs),
                n_actions=n_actions,
                device=device,
                gamma=0.99,
                env_id=None,  # calibration run: run-time verdicts irrelevant
            )
            machinery.fit_from_buffer(buf)
            stats = dict(machinery.verdict.stats)
            rounded = {k: round(float(v), 5) for k, v in stats.items()}
            print(f"[grace-nullcal]   components: {rounded}", flush=True)
            per_env_stats[env].append(stats)

    data = {}
    if Path(args.out).exists():
        data = yaml.safe_load(Path(args.out).read_text()) or {}
    ref = data.setdefault("reference", {})
    for env, stats in per_env_stats.items():
        if not stats:
            continue
        thr = RegimeRouter.calibrate(stats)
        ref[env] = {k: float(v) for k, v in thr.items()}
        print(f"[grace-nullcal] {env}: thresholds {ref[env]}")
    header = (
        "# GRACE router null-calibration reference (E0, feat/grace-critic).\n"
        "# Per-env thresholds = RegimeRouter.calibrate over basic-arm (sigma=0)\n"
        "# null runs generated INDEPENDENTLY of the judged cells (own seeds &\n"
        "# dataset prefix), k = NULL_CALIBRATION_K. Missing env -> the router\n"
        "# is UNCALIBRATED and serves Q_obs (never a silent causal route).\n"
        f"# Tool: tools/calibrate_grace_router.py (seeds {args.seeds}).\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump(data, sort_keys=True))
    print(f"[grace-nullcal] wrote {args.out}")


if __name__ == "__main__":
    main()

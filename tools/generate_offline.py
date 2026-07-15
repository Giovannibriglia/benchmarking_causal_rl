#!/usr/bin/env python3
"""Generate a tiered offline Minari dataset (B2): train -> snapshot -> rollout.

Trains an online generator (dqn/sac/ddpg), snapshots the requested tier BY
RETURN, rolls it out (optionally via an A-series collection policy for varied
provenance), and writes a Minari dataset to the local cache — consumed via B1's
``--offline-dataset``. ``--offline-tier`` lives HERE (not main.py), so this adds
no CLI surface to the training entrypoint.

    # clean expert CartPole, then consume via B1:
    python tools/generate_offline.py --env CartPole-v1 --algo dqn --offline-tier expert
    python main.py --algos offline_dqn --offline-dataset generated/cartpole/expert-v0

    # provenance: a confounded medium Pendulum dataset
    python tools/generate_offline.py --env Pendulum-v1 --algo sac \
        --offline-tier medium --behavior-policy bias_confounded
"""

from __future__ import annotations

import argparse

from src.envs.offline.generate import default_gate_for, generate_offline_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, help="generator/rollout env id")
    p.add_argument("--algo", default="dqn", help="online generator (dqn/sac/ddpg)")
    p.add_argument(
        "--offline-tier",
        default="expert",
        choices=["random", "medium", "expert"],
        help="snapshot tier by return: random=untrained, medium=1/3 of the "
        "random->expert range, expert=best checkpoint.",
    )
    p.add_argument(
        "--tier-fraction",
        type=float,
        default=1.0 / 3.0,
        help="medium target = R_random + fraction*(R_expert - R_random).",
    )
    p.add_argument(
        "--behavior-policy",
        default="agent",
        choices=[
            "agent",
            "anti_reward",
            "bias_skew",
            "bias_suboptimal",
            "curiosity",
            "bias_confounded",
            "bias_confounded_action",
        ],
        help="rollout behavior (provenance axis). Composes with --offline-tier.",
    )
    p.add_argument("--behavior-strength", type=float, default=None)
    # Declarative confounder config (action-dependent arm). c_r is the FIXED U->R
    # reward-shift magnitude (decoupled from sigma); the gate tolerances override the
    # action_dependent point-check defaults.
    p.add_argument("--confounder-c-r", type=float, default=None)
    p.add_argument("--pi-basic-epsilon", type=float, default=None)
    p.add_argument("--a-bad", type=int, default=1)
    p.add_argument("--gate-corr-tolerance", type=float, default=None)
    p.add_argument("--gate-ungated-reward-corr-max", type=float, default=None)
    p.add_argument("--gate-intervened-tolerance", type=float, default=None)
    p.add_argument("--train-episodes", type=int, default=50)
    p.add_argument("--n-checkpoints", type=int, default=10)
    p.add_argument("--rollout-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-id", default=None, help="override the generated id")
    p.add_argument(
        "--run-dir",
        default="outputs/generate",
        help="generator training dir (checkpoints + eval_metrics.csv)",
    )
    args = p.parse_args()

    # Build the declarative gate config from defaults + any CLI tolerance overrides.
    gate = default_gate_for(args.behavior_policy)
    for key, val in (
        ("corr_tolerance", args.gate_corr_tolerance),
        ("ungated_reward_corr_max", args.gate_ungated_reward_corr_max),
        ("intervened_tolerance", args.gate_intervened_tolerance),
    ):
        if val is not None:
            gate[key] = val

    ds = generate_offline_dataset(
        env_id=args.env,
        generator_algo=args.algo,
        tier=args.offline_tier,
        behavior_policy=args.behavior_policy,
        behavior_strength=args.behavior_strength,
        confounder_c_r=args.confounder_c_r,
        pi_basic_epsilon=args.pi_basic_epsilon,
        a_bad=args.a_bad,
        gate=gate,
        fraction=args.tier_fraction,
        train_episodes=args.train_episodes,
        n_checkpoints=args.n_checkpoints,
        rollout_episodes=args.rollout_episodes,
        seed=args.seed,
        dataset_id=args.dataset_id,
        run_dir=args.run_dir,
    )
    print(f"created {ds.id}: {ds.total_steps} steps, {ds.total_episodes} episodes")


if __name__ == "__main__":
    main()

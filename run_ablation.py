import argparse

from src.algos import EMPIRICAL_CHECKS

from src.empiricalchecks import EmpiricalChecks
from src.plot_empirical_check import plot_empirical_check


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark runner")

    parser.add_argument(
        "--env_suite",
        type=str,
        default="gymnasium",
        choices=["gymnasium"],
        help="Environment suite to benchmark (e.g. gymnasium, vmas, pettingzoo)",
    )
    parser.add_argument(
        "--n_episodes_train", type=int, default=250, help="Number of training episodes"
    )
    parser.add_argument(
        "--n_checkpoints",
        type=int,
        default=25,
        help="Number of checkpoints to save during training",
    )
    parser.add_argument(
        "--rollout_len",
        type=int,
        default=1024,
        help="Rollout length per environment per update",
    )
    parser.add_argument(
        "--n_train_envs", type=int, default=16, help="Number of training environments"
    )
    parser.add_argument(
        "--n_eval_envs", type=int, default=16, help="Number of evaluation environments"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (cpu, cuda, etc.)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    bench = EmpiricalChecks(
        env_suite=args.env_suite,
        n_episodes_train=args.n_episodes_train,
        n_checkpoints=args.n_checkpoints,
        rollout_len=args.rollout_len,
        n_train_envs=args.n_train_envs,
        n_eval_envs=args.n_eval_envs,
        seed=args.seed,
        device=args.device,
    )

    path_files = bench.run()

    plot_empirical_check(path_files, EMPIRICAL_CHECKS, "iqm", "iqr")

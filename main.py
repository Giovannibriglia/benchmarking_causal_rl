import argparse

from src.benchmark import Benchmark
from src.plotting import plot_and_save_results


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark runner")

    parser.add_argument(
        "--env_suite",
        type=str,
        default="gymnasium",
        choices=["gymnasium", "gymnasium-robotics"],
        help="Environment suite to benchmark (e.g. gymnasium, gymnasium-robotics, pettingzoo)",
    )
    parser.add_argument(
        "--n_episodes_train", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--n_checkpoints",
        type=int,
        default=50,
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

    bench = Benchmark(
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
    plot_and_save_results(path_files, bench.n_episodes_train)

    """
    (MyEnv) gbriglia@pascal:~/benchmarking_causal_rl$ export MUJOCO_GL=egl
    (MyEnv) gbriglia@pascal:~/benchmarking_causal_rl$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    """

    """
        (MyEnv) gbriglia@pascal:~/benchmarking_causal_rl$ export MUJOCO_GL=egl
        (MyEnv) gbriglia@pascal:~/benchmarking_causal_rl$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
        """

    """
    pip uninstall -y mujoco-py
    pip uninstall -y gymnasium-robotics  # we’ll reinstall right after

    pip install --upgrade "mujoco==3.3.6" gymnasium "gymnasium-robotics>=1.4"

    sudo apt-get update
    sudo apt-get install -y libgl1 libglfw3 libosmesa6 patchelf

    export MUJOCO_GL=egl
    """

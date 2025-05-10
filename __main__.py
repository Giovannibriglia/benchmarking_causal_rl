from src.benchmark import Benchmark

from src.utils import plot_and_save_results

if __name__ == "__main__":
    # TODO: fix video storing
    # TODO: fix plots: loss all in the same plot and fill_between
    # TODO: fix evaluation storing across envs

    bench = Benchmark(
        env_suite="gymnasium",
        n_episodes_train=100,
        n_checkpoints=5,
        rollout_len=256,
        n_train_envs=8,
        n_eval_envs=4,
        seed=42,
        device="cuda",
    )

    path_files = bench.run()

    plot_and_save_results(path_files)

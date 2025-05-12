from src.benchmark import Benchmark

from src.utils import plot_and_save_results

if __name__ == "__main__":
    # TODO: fix video storing
    bench = Benchmark(
        env_suite="gymnasium",
        n_episodes_train=2000,
        n_checkpoints=50,
        rollout_len=1024,
        n_train_envs=16,
        n_eval_envs=8,
        seed=42,
        device="cuda",
    )

    path_files = bench.run()
    plot_and_save_results(path_files, bench.n_episodes_train)

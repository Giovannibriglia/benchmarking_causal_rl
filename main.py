from src.benchmark import Benchmark

from src.utils import plot_and_save_results

if __name__ == "__main__":
    bench = Benchmark(env_suite="gymnasium", n_episodes_train=50)

    path_files = bench.run()
    plot_and_save_results(path_files, bench.n_episodes_train)

    """path_files = "runs/gymnasium_benchmark_20250905_103956"
    plot_and_save_results(path_files, 50)"""

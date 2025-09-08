from src.benchmark import Benchmark

from src.utils import plot_and_save_results

if __name__ == "__main__":
    bench = Benchmark(env_suite="gymnasium", n_episodes_train=500)

    path_files = bench.run()
    plot_and_save_results(path_files, bench.n_episodes_train)

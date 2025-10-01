import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
)
import torch
from tqdm import tqdm

from src.algos import AGENTS
from src.base import BasePolicy, DEFAULT_DEVICE
from src.envs import ENV_CLASSES, ENV_NAMES


class Benchmark:
    def __init__(
        self,
        env_suite="gymnasium",
        n_episodes_train=250,
        n_checkpoints=25,
        rollout_len=1024,
        n_train_envs=16,
        n_eval_envs=16,
        seed=42,
        device=DEFAULT_DEVICE,
    ):
        self.env_suite = env_suite
        self.n_episodes_train = n_episodes_train
        self.n_checkpoints = n_checkpoints
        self.rollout_len = rollout_len
        self.n_train_envs = n_train_envs
        self.n_eval_envs = n_eval_envs
        self.seed = seed
        self.device = device

        self._set_seed(seed)

        self.env_names = ENV_NAMES[env_suite]
        self.EnvClass = ENV_CLASSES[env_suite]
        self.dir_saving = Path("runs") / self._generate_simulation_name(
            f"{env_suite}_benchmark"
        )
        print("Experiment dir:", self.dir_saving)
        self.dir_saving.mkdir(parents=True, exist_ok=True)

        self.policy_path = self.dir_saving / "policies"
        self.policy_path.mkdir(parents=True, exist_ok=True)

        self.algorithms: Dict[str, type[BasePolicy]] = AGENTS
        self.results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

        cp = np.linspace(0, n_episodes_train - 1, n_checkpoints, dtype=np.int64)
        cp = np.unique(cp)  # drop duplicates from rounding
        self.checkpoints = [int(x) for x in cp.tolist()]  # cast to plain ints
        assert 0 in self.checkpoints, "first has to be in"
        assert n_episodes_train - 1 in self.checkpoints, "last has to be in"

        info_dict = {
            "env_suite": self.env_suite,  # str
            "n_episodes_train": self.n_episodes_train,  # int
            "n_checkpoints": self.n_checkpoints,  # int
            "rollout_len": self.rollout_len,  # int
            "n_train_envs": self.n_train_envs,  # int
            "n_eval_envs": self.n_eval_envs,  # int
            "seed": self.seed,  # int
            "device": str(self.device),  # str
            "env_names": self.env_names,  # list[str]
            "algorithms": list(self.algorithms.keys()),  # list[str]
            "checkpoints": self.checkpoints,  # list[int]
        }

        with open(self.dir_saving / "benchmark_info.json", "w") as f:
            json.dump(info_dict, f, indent=4)

    @staticmethod
    def _generate_simulation_name(base: str) -> str:
        return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def make_env(
        self,
        env_id: str,
        n_envs: int,
        seed: int,
        algo_name: str,
        record_video: bool = False,
    ):
        return self.EnvClass(
            env_id,
            n_envs=n_envs,
            seed=seed,
            device=self.device,
            record_video=record_video,
            video_dir=self.dir_saving / "videos" / env_id / algo_name,
        )

    def _update_json(self, env_id: str, algo_name: str, metrics: Dict[str, float]):
        path = self.dir_saving / f"{env_id}_metrics.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        algo_data = data.setdefault(env_id, {}).setdefault(algo_name, {})
        for k, v in metrics.items():
            algo_data.setdefault(k, []).append(v)
        path.write_text(json.dumps(data, indent=2))

    def run(self):
        """Main training/evaluation loop."""
        for env_id in self.env_names:
            self.results.setdefault(env_id, {})

            for algo_name, AlgoCls in self.algorithms.items():

                # Build training/evaluation envs once per environment
                train_env = self.make_env(
                    env_id, self.n_train_envs, self.seed, algo_name, record_video=False
                )
                eval_env = self.make_env(
                    env_id,
                    self.n_eval_envs,
                    self.seed + self.n_train_envs + 1,
                    algo_name,
                    record_video=True,
                )

                kwargs_agent = {}

                agent = AlgoCls(
                    train_env,
                    eval_env,
                    rollout_len=self.rollout_len,
                    device=self.device,
                    **kwargs_agent,
                )
                self.results[env_id].setdefault(algo_name, {})

                for ep in tqdm(
                    range(self.n_episodes_train), desc=f"{env_id} - {algo_name}"
                ):
                    agent.train()

                    # Checkpoint handling
                    if ep in self.checkpoints:

                        agent.evaluate()

                        # flush metrics
                        for buff_name, buff in [
                            ("train", agent.train_metrics),
                            ("eval", agent.eval_metrics),
                        ]:
                            metrics = buff.pop(divisor=1)
                            if not metrics:
                                continue
                            self._update_json(env_id, algo_name, metrics)
                            # Store in‑memory cache for plotting later
                            for k, v in metrics.items():
                                self.results[env_id][algo_name].setdefault(
                                    k, []
                                ).append(v)

                        # policy storing
                        policy_path = self.policy_path / env_id / algo_name
                        os.makedirs(policy_path, exist_ok=True)
                        agent.save_policy(policy_path / f"episode_{ep}")

                del agent

                # Close the envs (important for RecordVideo flush)
                train_env.close()
                eval_env.close()

        return self.dir_saving

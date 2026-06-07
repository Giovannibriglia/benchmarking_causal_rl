"""In-pipeline Cell-1 reference policies: train-or-load keyed by spec.

Closes Phase-3 flag 5: cell YAMLs no longer point at gitignored run
directories. A reference spec fully determines the training job
(env, algo, budget, seed, deterministic); the checkpoint is cached under
``references/<key>/`` and re-trained from scratch on a fresh clone — the
cache is an optimization, not a dependency.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

REFERENCES_DIR = Path("references")


def reference_key(spec: dict, seed: int) -> str:
    env = str(spec["env"]).replace("/", "-")
    algo = str(spec.get("algo", "ppo"))
    n_episodes = int(spec.get("n_episodes", 150))
    rollout = int(spec.get("rollout_len", 512))
    n_envs = int(spec.get("n_train_envs", 8))
    tag = str(spec.get("tag", "")).strip()
    suffix = f"_{tag}" if tag else ""
    return f"{env}_{algo}_seed{seed}_ep{n_episodes}_rl{rollout}_ne{n_envs}{suffix}"


def ensure_reference(spec: dict, seed: int, device: torch.device) -> str:
    """Return the path to the reference checkpoint, training it if absent."""
    key = reference_key(spec, seed)
    ckpt_path = REFERENCES_DIR / key / "reference.pt"
    if ckpt_path.is_file():
        return str(ckpt_path)

    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()

    env_id = str(spec["env"])
    algo = str(spec.get("algo", "ppo"))
    n_episodes = int(spec.get("n_episodes", 150))
    print(f"[reference] training {key} (cache miss) ...")
    train_dir = REFERENCES_DIR / key / "train_run"
    env_cfg = EnvConfig(
        env_id=env_id,
        n_train_envs=int(spec.get("n_train_envs", 8)),
        n_eval_envs=int(spec.get("n_eval_envs", 8)),
        rollout_len=int(spec.get("rollout_len", 512)),
        seed=int(seed),
    )
    train_cfg = TrainingConfig(
        n_episodes=n_episodes,
        # default 2 (first/last): the artifact is the final policy. Specs may
        # raise it to record a usable learning curve (continuous Cell-1 panel).
        n_checkpoints=int(spec.get("n_checkpoints", 2)),
        deterministic=bool(spec.get("deterministic", True)),
        device=str(device),
        algorithm=algo,
    )
    run_cfg = RunConfig(run_dir=str(train_dir))
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        run_cfg,
        registry.get(algo),
        progress_label=f"reference {key}",
    )
    runner.run()

    ckpt_dir = (
        train_dir / "checkpoints" / f"{env_id.replace('/', '-')}_{algo}_seed{seed}"
    )
    last = sorted(ckpt_dir.glob("ckpt_ep*.pt"))[-1]
    os.makedirs(ckpt_path.parent, exist_ok=True)
    shutil.copy(last, ckpt_path)
    print(f"[reference] cached {ckpt_path}")
    return str(ckpt_path)

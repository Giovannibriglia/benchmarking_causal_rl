from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from src.benchmarking.aux_models import AuxModelConfig
from src.benchmarking.critic_ablation import CriticAblationConfig, default_aux_critics
from src.benchmarking.registry import (
    expand_env_set,
    register_default_algorithms,
    registry,
)
from src.benchmarking.runner import BenchmarkRunner
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.device import detect_device
from src.envs.registry import register_default_env_wrappers


def _parse_mask_indices(raw) -> tuple[int, ...] | None:
    # Accepts a comma/space-separated string (CLI) or a list (reproduce YAML);
    # returns a tuple of ints, or None when unset/empty.
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        vals = [int(x) for x in raw]
    elif isinstance(raw, str):
        normalized = raw.replace(",", " ")
        vals = [int(x) for x in normalized.split() if x.strip()]
    else:
        vals = [int(raw)]
    return tuple(vals) if vals else None


def _validate_mask_list(val, env_id: str, source: str) -> tuple[int, ...] | None:
    # A single env's mask: must be a list of non-negative ints (no bools,
    # strings, negatives, or nested lists). Empty -> None (no masking).
    if not isinstance(val, (list, tuple)):
        raise ValueError(
            f"mask_indices for '{env_id}' in {source} must be a list of "
            f"non-negative ints, got {type(val).__name__}."
        )
    out: list[int] = []
    for x in val:
        if isinstance(x, bool) or not isinstance(x, int) or x < 0:
            raise ValueError(
                f"mask_indices for '{env_id}' in {source} must be a list of "
                f"non-negative ints; invalid entry {x!r}."
            )
        out.append(int(x))
    return tuple(out) if out else None


def _resolve_mask_indices_map(raw, env_ids, source: str) -> dict:
    # Build {env_id: tuple|None} for every env in env_ids. Three input shapes:
    #   None      -> no masking anywhere (default; YAMLs without the key are
    #                byte-identical to pre-mask runs).
    #   dict      -> strict per-env map (PR5 §5): EVERY env in env_ids must be a
    #                key; a missing env raises (never silently unmask).
    #   list/str  -> uniform: the same parsed indices for every env (CLI
    #                --mask-indices and the ad-hoc YAML scalar form).
    if raw is None:
        return {e: None for e in env_ids}
    if isinstance(raw, dict):
        result = {}
        for e in env_ids:
            if e not in raw:
                raise ValueError(
                    f"mask_indices map in {source} is missing env '{e}': it is "
                    "listed in 'envs' but absent from the mask_indices map. Add "
                    f"'{e}: [..]' to the map, or remove '{e}' from 'envs'."
                )
            result[e] = _validate_mask_list(raw[e], e, source)
        return result
    uniform = _parse_mask_indices(raw)
    return {e: uniform for e in env_ids}


def parse_args():
    p = argparse.ArgumentParser(description="Benchmarking Causal RL")
    p.add_argument(
        "--mode",
        choices=["benchmark", "critic_ablation"],
        default="benchmark",
        help="Run standard benchmarking or critic-ablation mode.",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Shortcut for --mode critic_ablation.",
    )
    p.add_argument(
        "--envs", nargs="+", help="List of env ids to benchmark", default=None
    )
    p.add_argument(
        "--algos", nargs="+", help="List of algorithms to benchmark", default=None
    )
    p.add_argument(
        "--env-set",
        type=str,
        default=None,
        help="Named environment set to expand into env ids (overrides --envs).",
    )
    p.add_argument(
        "--env-wrapper",
        type=str,
        default="auto",
        help="Env wrapper to use (auto selects by env name or entry point).",
    )
    p.add_argument(
        "--env-entry-point",
        type=str,
        default=None,
        help="Python entry point for custom envs, e.g. my_pkg.envs:make_env.",
    )
    p.add_argument(
        "--env-kwargs",
        type=str,
        default=None,
        help="JSON dict of kwargs for the env entry point.",
    )
    p.add_argument(
        "--offline-dataset",
        type=str,
        default=None,
        help=(
            "Minari dataset id for offline algorithms (data_regime='offline', "
            "e.g. --algos offline_dqn). The live env is still built for eval."
        ),
    )
    p.add_argument(
        "--behavior-policy",
        type=str,
        default="agent",
        choices=[
            "agent",
            "anti_reward",
            "bias_skew",
            "bias_suboptimal",
            "curiosity",
            "bias_confounded",
        ],
        help=(
            "Off-policy online collection behavior policy (opt-in; default "
            "'agent' is byte-identical to the standard agent.act collection). "
            "anti_reward=critic-pessimal; bias_skew=prob-p preferred action; "
            "bias_suboptimal=prob-beta agent else random; curiosity=novelty; "
            "bias_confounded=per-episode latent U biases action and reward."
        ),
    )
    p.add_argument(
        "--behavior-strength",
        type=float,
        default=None,
        help=(
            "Primary knob for --behavior-policy: anti_reward=epsilon, bias_skew=p, "
            "bias_suboptimal=beta, curiosity=strength. None keeps the policy default."
        ),
    )
    p.add_argument(
        "--mask-indices",
        type=str,
        default=None,
        help="Comma-separated integer indices to drop from the flat observation vector. "
        "When set on an online run, wraps the env. When set on an offline run, "
        "projects loaded transitions. None = no masking.",
    )
    p.add_argument("--n-train-envs", type=int, default=16)
    p.add_argument("--n-eval-envs", type=int, default=16)
    p.add_argument("--rollout-len", type=int, default=1024)
    p.add_argument("--n-episodes", type=int, default=250)
    p.add_argument("--n-checkpoints", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reproduce",
        type=str,
        help=(
            "Name of reproducibility YAML in reproducibility/. Accepts with or without "
            "extension (e.g., comoreai26 or comoreai26.yaml)."
        ),
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable torch deterministic algorithms - use it for final benchmarks to publish",
    )
    p.add_argument(
        "--aggregation",
        choices=["iqm", "mean"],
        default="iqm",
        help="Aggregation strategy for reported stats",
    )
    p.add_argument(
        "--ablation-critics",
        nargs="+",
        default=None,
        help=(
            "Auxiliary critics to train in critic_ablation mode. "
            f"Defaults to baseline critic: {' '.join(default_aux_critics())}"
        ),
    )
    p.add_argument(
        "--ablation-lr",
        type=float,
        default=3e-4,
        help="Learning rate for auxiliary critics in critic_ablation mode.",
    )
    p.add_argument(
        "--ablation-hidden-dims",
        "--ablation-hidded-dims",
        dest="ablation_hidden_dims",
        type=str,
        default="64,64",
        help="Comma-separated hidden layer sizes for auxiliary critics.",
    )
    p.add_argument(
        "--ablation-bins",
        type=int,
        default=32,
        help="Histogram bins for distribution metrics (MI, KL, JS).",
    )
    p.add_argument(
        "--aux-models",
        action="store_true",
        help=(
            "Train auxiliary reward r(s,a) + next-state s'(s,a) models alongside "
            "RL (logged to aux_metrics.csv; NOT folded into the RL loss). "
            "Off by default; works with any algorithm/regime."
        ),
    )
    p.add_argument(
        "--aux-lr",
        type=float,
        default=3e-4,
        help="Learning rate for the auxiliary models.",
    )
    p.add_argument(
        "--aux-hidden-dims",
        dest="aux_hidden_dims",
        type=str,
        default="64,64",
        help="Comma-separated hidden layer sizes for the auxiliary models.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    register_default_algorithms()
    register_default_env_wrappers()

    cfg_from_file: dict = {}
    repro_path = None
    if args.reproduce:
        repro_name = args.reproduce
        if not repro_name.endswith((".yaml", ".yml")):
            repro_name = f"{repro_name}.yaml"
        repro_path = Path("reproducibility") / repro_name
        cfg_from_file = yaml.safe_load(repro_path.read_text())

    env_cfg_src = (
        cfg_from_file.get("env", {}) if isinstance(cfg_from_file, dict) else {}
    )
    train_cfg_src = (
        cfg_from_file.get("training", {}) if isinstance(cfg_from_file, dict) else {}
    )

    def _maybe_list(val):
        if val is None:
            return None
        if isinstance(val, str):
            return val.split()
        return list(val)

    def _parse_hidden_dims(raw) -> tuple[int, ...]:
        if isinstance(raw, (list, tuple)):
            dims = [int(x) for x in raw]
        elif isinstance(raw, str):
            normalized = raw.replace(",", " ")
            dims = [int(x) for x in normalized.split() if x.strip()]
        elif raw is None:
            dims = [64, 64]
        else:
            dims = [int(raw)]
        if not dims:
            raise ValueError("ablation_hidden_dims cannot be empty.")
        return tuple(dims)

    env_set = env_cfg_src.get(
        "env_set", cfg_from_file.get("env_set", args.env_set if args.env_set else None)
    )
    env_wrapper = env_cfg_src.get(
        "env_wrapper",
        cfg_from_file.get(
            "env_wrapper", args.env_wrapper if args.env_wrapper else None
        ),
    )
    env_entry_point = env_cfg_src.get(
        "env_entry_point",
        cfg_from_file.get(
            "env_entry_point",
            args.env_entry_point if args.env_entry_point else None,
        ),
    )
    envs_from_cfg = _maybe_list(env_cfg_src.get("envs")) or _maybe_list(
        cfg_from_file.get("envs") if isinstance(cfg_from_file, dict) else None
    )
    envs = expand_env_set(env_set) if env_set else (envs_from_cfg or args.envs)
    algos = (
        _maybe_list(train_cfg_src.get("algos"))
        or _maybe_list(
            cfg_from_file.get("algos") if isinstance(cfg_from_file, dict) else None
        )
        or args.algos
    )

    # Reproduce mode takes precedence; fall back to CLI if missing
    seed = env_cfg_src.get("seed", cfg_from_file.get("seed", args.seed))
    n_train_envs = env_cfg_src.get(
        "n_train_envs", cfg_from_file.get("n_train_envs", args.n_train_envs)
    )
    n_eval_envs = env_cfg_src.get(
        "n_eval_envs", cfg_from_file.get("n_eval_envs", args.n_eval_envs)
    )
    rollout_len = env_cfg_src.get(
        "rollout_len", cfg_from_file.get("rollout_len", args.rollout_len)
    )
    offline_dataset = env_cfg_src.get(
        "offline_dataset",
        cfg_from_file.get("offline_dataset", args.offline_dataset),
    )
    behavior_policy = env_cfg_src.get(
        "behavior_policy",
        cfg_from_file.get("behavior_policy", args.behavior_policy),
    )
    behavior_strength = env_cfg_src.get(
        "behavior_strength",
        cfg_from_file.get("behavior_strength", args.behavior_strength),
    )
    # mask_indices may be a per-env map ({env_id: [..]}), a uniform list, or a
    # CLI string; resolve to a per-env {env_id: tuple|None} lookup. Strict for
    # the map form (every env must be present); see _resolve_mask_indices_map.
    mask_indices_raw = env_cfg_src.get(
        "mask_indices",
        cfg_from_file.get("mask_indices", args.mask_indices),
    )
    mask_source = str(repro_path) if repro_path else "the config"
    mask_by_env = _resolve_mask_indices_map(mask_indices_raw, envs or [], mask_source)
    env_kwargs = env_cfg_src.get("env_kwargs", cfg_from_file.get("env_kwargs", None))
    if env_kwargs is None and args.env_kwargs:
        env_kwargs = json.loads(args.env_kwargs)
    if isinstance(env_kwargs, str):
        env_kwargs = json.loads(env_kwargs)
    if env_kwargs is None:
        env_kwargs = {}
    if not isinstance(env_kwargs, dict):
        raise ValueError("env_kwargs must be a dict or JSON object.")

    n_episodes = train_cfg_src.get(
        "n_episodes", cfg_from_file.get("n_episodes", args.n_episodes)
    )
    n_checkpoints = train_cfg_src.get(
        "n_checkpoints", cfg_from_file.get("n_checkpoints", args.n_checkpoints)
    )
    # Checkpoint count validation: must be in [2, n_episodes]. Behavior: clamp to range.
    if n_checkpoints < 2:
        n_checkpoints = 2
    if n_checkpoints > n_episodes:
        n_checkpoints = n_episodes
    deterministic = train_cfg_src.get(
        "deterministic", cfg_from_file.get("deterministic", args.deterministic)
    )
    aggregation = train_cfg_src.get(
        "aggregation", cfg_from_file.get("aggregation", args.aggregation)
    )
    mode = train_cfg_src.get("mode", cfg_from_file.get("mode", args.mode))
    ablation_enabled = bool(
        train_cfg_src.get(
            "ablation_enabled",
            cfg_from_file.get("ablation_enabled", False),
        )
    ) or bool(args.ablation)
    if ablation_enabled:
        mode = "critic_ablation"
    if mode not in {"benchmark", "critic_ablation"}:
        raise ValueError(
            f"Unknown mode '{mode}'. Supported values: benchmark, critic_ablation."
        )
    ablation_cfg_src = train_cfg_src.get("ablation", {})
    if not isinstance(ablation_cfg_src, dict):
        ablation_cfg_src = {}
    ablation_critics = (
        _maybe_list(ablation_cfg_src.get("critics"))
        or _maybe_list(train_cfg_src.get("ablation_critics"))
        or _maybe_list(
            cfg_from_file.get("ablation_critics")
            if isinstance(cfg_from_file, dict)
            else None
        )
        or args.ablation_critics
        or default_aux_critics()
    )
    ablation_lr = ablation_cfg_src.get(
        "lr",
        train_cfg_src.get(
            "ablation_lr", cfg_from_file.get("ablation_lr", args.ablation_lr)
        ),
    )
    ablation_hidden_dims = _parse_hidden_dims(
        ablation_cfg_src.get(
            "hidden_dims",
            train_cfg_src.get(
                "ablation_hidden_dims",
                cfg_from_file.get("ablation_hidden_dims", args.ablation_hidden_dims),
            ),
        )
    )
    ablation_bins = int(
        ablation_cfg_src.get(
            "bins",
            train_cfg_src.get(
                "ablation_bins", cfg_from_file.get("ablation_bins", args.ablation_bins)
            ),
        )
    )
    if ablation_bins < 4:
        ablation_bins = 4

    # Auxiliary reward/next-state models (opt-in, off by default).
    aux_enabled = bool(
        train_cfg_src.get("aux_models", cfg_from_file.get("aux_models", False))
    ) or bool(args.aux_models)
    aux_lr = train_cfg_src.get("aux_lr", cfg_from_file.get("aux_lr", args.aux_lr))
    aux_hidden_dims = _parse_hidden_dims(
        train_cfg_src.get(
            "aux_hidden_dims",
            cfg_from_file.get("aux_hidden_dims", args.aux_hidden_dims),
        )
    )

    n_checkpoints = max(n_checkpoints, 2)
    n_checkpoints = min(n_checkpoints, n_episodes)

    if envs is None or algos is None:
        raise ValueError(
            "No algorithms or environments specified. Provide them via CLI or reproduce YAML."
        )

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.reproduce:
        repro_tag = args.reproduce.replace(".yaml", "")
        if mode != "benchmark":
            repro_tag = f"{repro_tag}_{mode}"
        run_dir = Path(f"runs/{repro_tag}_{base_timestamp}")
    else:
        prefix = "benchmark" if mode == "benchmark" else mode
        run_dir = Path(f"runs/{prefix}_{base_timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    device_str = str(
        train_cfg_src.get("device", cfg_from_file.get("device", str(detect_device())))
    )

    # Save config snapshot once
    config_snapshot = {
        "env": {
            "envs": envs,
            "env_set": env_set,
            "env_wrapper": env_wrapper,
            "env_entry_point": env_entry_point,
            "offline_dataset": offline_dataset,
            "env_kwargs": env_kwargs,
            "mask_indices": ({e: list(v) for e, v in mask_by_env.items() if v} or None),
            "n_train_envs": n_train_envs,
            "n_eval_envs": n_eval_envs,
            "rollout_len": rollout_len,
            "seed": seed,
        },
        "training": {
            "mode": mode,
            "algos": algos,
            "n_episodes": n_episodes,
            "n_checkpoints": n_checkpoints,
            "deterministic": deterministic,
            "aggregation": aggregation,
            "device": device_str,
            "ablation": {
                "critics": ablation_critics if mode == "critic_ablation" else [],
                "lr": ablation_lr,
                "hidden_dims": list(ablation_hidden_dims),
                "bins": ablation_bins,
            },
        },
        "timestamp": base_timestamp,
    }
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config_snapshot, f)
    with (run_dir / "metadata.json").open("w") as f:
        json.dump({"timestamp": base_timestamp}, f, indent=2)

    run_cfg = RunConfig(run_dir=str(run_dir), timestamp=base_timestamp)

    for env_id in envs:
        for algo in algos:
            env_cfg = EnvConfig(
                env_id=env_id,
                n_train_envs=n_train_envs,
                n_eval_envs=n_eval_envs,
                rollout_len=rollout_len,
                seed=seed,
                env_wrapper=env_wrapper or "auto",
                env_entry_point=env_entry_point,
                env_kwargs=env_kwargs,
                offline_dataset=offline_dataset,
                behavior_policy=behavior_policy,
                behavior_strength=behavior_strength,
                mask_indices=mask_by_env[env_id],
            )
            train_cfg = TrainingConfig(
                n_episodes=n_episodes,
                n_checkpoints=n_checkpoints,
                deterministic=deterministic,
                device=device_str,
                algorithm=algo,
                aggregation=aggregation,
            )
            critic_ablation_cfg = None
            if mode == "critic_ablation":
                critic_ablation_cfg = CriticAblationConfig(
                    critics=[str(x) for x in ablation_critics],
                    lr=float(ablation_lr),
                    hidden_dims=ablation_hidden_dims,
                    bins=ablation_bins,
                )
            aux_model_cfg = None
            if aux_enabled:
                aux_model_cfg = AuxModelConfig(
                    lr=float(aux_lr), hidden_dims=aux_hidden_dims
                )
            spec = registry.get(algo)
            runner = BenchmarkRunner(
                env_cfg,
                train_cfg,
                run_cfg,
                spec,
                critic_ablation_cfg=critic_ablation_cfg,
                aux_model_cfg=aux_model_cfg,
                progress_label=f"{algo} - {env_id}",
            )
            runner.run()


if __name__ == "__main__":
    main()

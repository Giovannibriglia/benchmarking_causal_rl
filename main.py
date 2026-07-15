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
from src.benchmarking.runner import (
    _validate_algos_against_behavior_policy,
    BenchmarkRunner,
)
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


def _validate_dataset_id(val, env_id: str, source: str) -> str:
    # A single env's offline dataset id: must be a non-empty string.
    if not isinstance(val, str) or not val.strip():
        raise ValueError(
            f"offline_dataset for '{env_id}' in {source} must be a non-empty "
            f"dataset-id string; got {val!r}."
        )
    return val


def _resolve_offline_dataset_map(raw, env_ids, source: str) -> dict:
    # Build {env_id: dataset_id|None} for every env in env_ids. Three shapes,
    # mirroring _resolve_mask_indices_map:
    #   None  -> no offline dataset anywhere (default; online runs are unchanged).
    #   dict  -> strict per-env map (Cell 3 tier sweep): EVERY env in env_ids must
    #            be a key; a missing env raises (never silently train on no data).
    #   str   -> uniform: the same dataset id for every env (CLI --offline-dataset
    #            and the ad-hoc single-string YAML form).
    if raw is None:
        return {e: None for e in env_ids}
    if isinstance(raw, dict):
        result = {}
        for e in env_ids:
            if e not in raw:
                raise ValueError(
                    f"offline_dataset map in {source} is missing env '{e}': it is "
                    "listed in 'envs' but absent from the offline_dataset map. Add "
                    f"'{e}: <dataset-id>' to the map, or remove '{e}' from 'envs'."
                )
            result[e] = _validate_dataset_id(raw[e], e, source)
        return result
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"offline_dataset in {source} must be a dataset-id string or a "
            f"per-env map; got {type(raw).__name__}."
        )
    return {e: raw for e in env_ids}


def _normalize_algo(entry) -> dict:
    """Normalize one ``algos`` entry to {name, actor, critic, network_kwargs}.

    A plain string ``"ppo"`` -> all-MLP default. A dict ``{name, networks:
    {actor, critic, hidden_dim?, num_layers?}}`` -> the parsed components
    (networks defaults to MLP per missing component). Raises on a dict without
    ``name`` or a non-map ``networks``."""
    if isinstance(entry, str):
        return {"name": entry, "actor": "mlp", "critic": "mlp", "network_kwargs": {}}
    if isinstance(entry, dict):
        if "name" not in entry:
            raise ValueError(f"algos dict entry missing required 'name': {entry!r}")
        nets = entry.get("networks", {}) or {}
        if not isinstance(nets, dict):
            raise ValueError(
                f"algos[{entry['name']!r}].networks must be a map "
                f"{{actor, critic, ...}}; got {type(nets).__name__}."
            )
        # network_kwargs is the per-algo builder-kwargs channel. hidden_dim/
        # num_layers feed recurrent trunks; gamma_sensitivity (Γ) feeds the
        # SensitivityBounds reweighter (offline_dqn/bcq/cql/iql _sensitivity). All are
        # spread into the builder as **kwargs; builders ignore keys they don't read.
        nkw = {
            k: nets[k]
            for k in ("hidden_dim", "num_layers", "gamma_sensitivity")
            if k in nets
        }
        return {
            "name": entry["name"],
            "actor": nets.get("actor", "mlp"),
            "critic": nets.get("critic", "mlp"),
            "network_kwargs": nkw,
        }
    raise ValueError(
        f"algos entries must be a string or a dict; got {type(entry).__name__}."
    )


def _canonical_algo_id(spec: dict) -> str:
    """Canonical run/CSV identifier for a normalized algo spec. On-policy algos
    disambiguate by per-component network (``name__actor__critic``);
    off-policy/offline (and unknown) keep the bare name, so their goldens and
    run-dir names are unchanged by this feature."""
    name = spec["name"]
    try:
        kind = registry.get(name).kind
    except KeyError:
        return name
    if kind == "on_policy":
        # On-policy always suffixes (per PR #49).
        return f"{name}__{spec['actor']}__{spec['critic']}"
    # Off-policy: suffix ONLY when the network config is non-default. This keeps
    # existing off-policy goldens (bare dqn/sac/ddpg) bitwise-green while enabling
    # disambiguation for recurrent off-policy (dqn__lstm__lstm) in PR-1C2.
    if spec["actor"] != "mlp" or spec["critic"] != "mlp":
        return f"{name}__{spec['actor']}__{spec['critic']}"
    return name


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
            "pi_basic",
            "biased",
            "anti_reward",
            "bias_skew",
            "bias_suboptimal",
            "curiosity",
            "bias_confounded",
            "bias_confounded_action",
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
            "biased=beta (skew-on-pi_basic), bias_suboptimal=beta, curiosity=strength, "
            "bias_confounded[_action]=sigma. None keeps the policy default."
        ),
    )
    p.add_argument(
        "--pi-basic-epsilon",
        type=float,
        default=None,
        help=(
            "FIXED exploration epsilon of the SHARED base policy pi_basic, read "
            "identically by the basic (pi_basic), biased and confounded arms so their "
            "(beta=0, sigma=0) origin is one policy. REQUIRED for arm runs."
        ),
    )
    p.add_argument(
        "--confounder-c-r",
        type=float,
        default=None,
        help="Action-dependent confounder fixed U->R reward-shift magnitude (c_r).",
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
    # offline_dataset may be a per-env map ({env_id: dataset_id}), a uniform
    # string, or a CLI string; resolve to a per-env {env_id: id|None} lookup.
    # Strict for the map form (every env must be present).
    offline_dataset_raw = env_cfg_src.get(
        "offline_dataset",
        cfg_from_file.get("offline_dataset", args.offline_dataset),
    )
    offline_source = str(repro_path) if repro_path else "the config"
    offline_by_env = _resolve_offline_dataset_map(
        offline_dataset_raw, envs or [], offline_source
    )
    behavior_policy = env_cfg_src.get(
        "behavior_policy",
        cfg_from_file.get("behavior_policy", args.behavior_policy),
    )
    behavior_strength = env_cfg_src.get(
        "behavior_strength",
        cfg_from_file.get("behavior_strength", args.behavior_strength),
    )
    pi_basic_epsilon = env_cfg_src.get(
        "pi_basic_epsilon",
        cfg_from_file.get("pi_basic_epsilon", args.pi_basic_epsilon),
    )
    confounder_c_r = env_cfg_src.get(
        "confounder_c_r",
        cfg_from_file.get("confounder_c_r", args.confounder_c_r),
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

    # Normalize the algos list to {name, actor, critic, network_kwargs} dicts.
    normalized_algos = [_normalize_algo(a) for a in algos]

    # Fail fast on structurally-incompatible (on-policy algo + action-bias-only
    # behavior_policy) combinations before building any run (registry is already
    # populated above via register_default_algorithms()).
    _validate_algos_against_behavior_policy(
        [s["name"] for s in normalized_algos], behavior_policy
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
            "offline_dataset": ({e: d for e, d in offline_by_env.items() if d} or None),
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

    # Run-level u0-schema flag: the sibling (env, algo) runners SHARE one
    # offline_value_trace.csv in run_dir, so if ANY selected algo is a U-variant
    # (requires_confounder_u) every runner must widen that file to the u0 SUPERSET
    # header (base runners blank-fill the u0 cells). Decided here because only the
    # full algo list is visible at this level, not inside a per-(env, algo) runner.
    def _algo_requires_u(name: str) -> bool:
        try:
            return bool(registry.get(name).requires_confounder_u)
        except KeyError:
            return False  # unknown algo surfaces in the run loop below

    run_requires_u = any(_algo_requires_u(a["name"]) for a in normalized_algos)
    run_cfg = RunConfig(
        run_dir=str(run_dir),
        timestamp=base_timestamp,
        value_trace_u0_schema=run_requires_u,
    )

    for env_id in envs:
        for algo_spec_norm in normalized_algos:
            algo = algo_spec_norm["name"]
            canonical_algo = _canonical_algo_id(algo_spec_norm)
            env_cfg = EnvConfig(
                env_id=env_id,
                n_train_envs=n_train_envs,
                n_eval_envs=n_eval_envs,
                rollout_len=rollout_len,
                seed=seed,
                env_wrapper=env_wrapper or "auto",
                env_entry_point=env_entry_point,
                env_kwargs=env_kwargs,
                offline_dataset=offline_by_env[env_id],
                behavior_policy=behavior_policy,
                behavior_strength=behavior_strength,
                pi_basic_epsilon=pi_basic_epsilon,
                confounder_c_r=confounder_c_r,
                mask_indices=mask_by_env[env_id],
            )
            train_cfg = TrainingConfig(
                n_episodes=n_episodes,
                n_checkpoints=n_checkpoints,
                deterministic=deterministic,
                device=device_str,
                algorithm=canonical_algo,
                aggregation=aggregation,
                actor_network=algo_spec_norm["actor"],
                critic_network=algo_spec_norm["critic"],
                network_kwargs=algo_spec_norm["network_kwargs"],
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
                progress_label=f"{canonical_algo} - {env_id}",
            )
            runner.run()


if __name__ == "__main__":
    main()

"""Hosted-dataset behavior-policy sweep — the hosted analog of the classical L-sweep.

One family config sweeps the RECORDED behavior-policy axis over algos x seeds:
the ``datasets:`` block names the arms (tiers like simple/medium/expert, or
dataset variants like partial vs fullobs), and each arm maps every env to its
hosted Minari dataset id. There is NO generation phase and NO confounding gate
— the datasets are fixed hosted inputs (downloaded on first use into the active
Minari store). Leaves mirror the classical results tree:

    results/{regime}/{simulation}/{arm}/{env}/{algo}/{seed}/

Dispatch: ``main.py --reproduce`` routes any config with a ``datasets:`` key
here (checked BEFORE the regime/sweep-driver dispatch, so family files may
carry a ``regime:`` key for the results path without being sent to the
generation sweep).

Arm schema — two forms::

    datasets:
      expert: {Hopper-v5: mujoco/hopper/expert-v0}       # flat env->dataset map
      fullobs:                                           # structured form
        offline_dataset: {BabyAI-X-v0: minigrid/BabyAI-X/optimal-fullobs-v0}
        env_kwargs: {full_obs: true}                     # per-arm env build

Per-arm ``env_kwargs`` merge over the family-level ``env_kwargs`` (the fullobs
BabyAI arm changes the eval env encoding to match its dataset). Seeds are
TRAINING seeds over the one fixed dataset per arm — the standard hosted-data
protocol; dataset variance is NOT in the cross-seed bands (unlike the generated
sweeps, where each seed regenerates its dataset).

Resumable: a leaf whose ``eval_metrics.csv`` already exists is skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class HostedArm:
    name: str
    offline_dataset: dict  # {env_id: dataset_id}
    env_kwargs: dict = field(default_factory=dict)


@dataclass
class HostedFamilySpec:
    regime: str
    simulation: str
    arms: list  # [HostedArm] in file order
    algos: list  # normalized [{name, actor, critic, network_kwargs}]
    seeds: list
    budgets: dict
    env_wrapper: str = "auto"
    env_kwargs: dict = field(default_factory=dict)
    aggregation: str = "iqm"
    eval_count_terminal_reward: bool = True
    config_path: str = ""

    @property
    def envs(self) -> list:
        return list(self.arms[0].offline_dataset.keys())


def _normalize_algos(raw: list) -> list:
    """Same semantics as the flat reproduce path: plain string -> all-MLP
    default; dict -> {name, networks: {actor, critic, ...}}. Reused from main
    (lazy import: main imports this module only inside its dispatch function,
    so there is no import cycle at module load)."""
    from main import _normalize_algo

    return [_normalize_algo(entry) for entry in raw]


def _canonical(spec: dict) -> str:
    from main import _canonical_algo_id

    return _canonical_algo_id(spec)


def parse_family_config(path: str) -> HostedFamilySpec:
    cfg = yaml.safe_load(Path(path).read_text())
    if "datasets" not in cfg or not isinstance(cfg["datasets"], dict):
        raise ValueError(
            f"{path}: hosted family config requires a 'datasets:' arm map."
        )
    arms = []
    for name, entry in cfg["datasets"].items():
        if not isinstance(entry, dict) or not entry:
            raise ValueError(f"{path}: arm '{name}' must be a non-empty map.")
        if "offline_dataset" in entry:
            arm = HostedArm(
                name=str(name),
                offline_dataset=dict(entry["offline_dataset"]),
                env_kwargs=dict(entry.get("env_kwargs") or {}),
            )
        else:
            arm = HostedArm(name=str(name), offline_dataset=dict(entry))
        arms.append(arm)
    envs = sorted(arms[0].offline_dataset)
    for arm in arms[1:]:
        if sorted(arm.offline_dataset) != envs:
            raise ValueError(
                f"{path}: arm '{arm.name}' covers envs {sorted(arm.offline_dataset)} "
                f"but arm '{arms[0].name}' covers {envs}; every arm must map the "
                "same env set (the cross-arm comparison is per-env)."
            )
    budget_keys = (
        "offline_grad_steps",
        "n_checkpoints",
        "n_train_envs",
        "n_eval_envs",
        "rollout_len",
    )
    budgets = {k: cfg[k] for k in budget_keys if k in cfg}
    if "offline_grad_steps" not in budgets:
        raise ValueError(
            f"{path}: hosted sweeps require an explicit offline_grad_steps."
        )
    return HostedFamilySpec(
        regime=str(cfg.get("regime", "offline_mdp")),
        simulation=str(cfg.get("simulation", Path(path).stem)),
        arms=arms,
        algos=_normalize_algos(list(cfg.get("algos") or [])),
        seeds=[int(s) for s in (cfg.get("seeds") or [0])],
        budgets=budgets,
        env_wrapper=str(cfg.get("env_wrapper", "auto")),
        env_kwargs=dict(cfg.get("env_kwargs") or {}),
        aggregation=str(cfg.get("aggregation", "iqm")),
        eval_count_terminal_reward=bool(cfg.get("eval_count_terminal_reward", True)),
        config_path=str(path),
    )


def _leaf_dir(
    root: Path, spec: HostedFamilySpec, arm, env: str, algo_id: str, seed: int
) -> Path:
    return (
        root
        / spec.regime
        / spec.simulation
        / arm.name
        / env.replace("/", "-")
        / algo_id
        / f"seed{seed}"
    )


def run_hosted_sweep(
    config_path: str, results_root: str = "results", device: str | None = None
) -> str:
    """Run every (arm x env x algo x seed) leaf; skip leaves whose
    eval_metrics.csv already exists. Returns a printable summary."""
    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.config.seeding import set_seed
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    spec = parse_family_config(config_path)
    root = Path(results_root)
    family_dir = root / spec.regime / spec.simulation
    family_dir.mkdir(parents=True, exist_ok=True)
    algo_ids = [_canonical(a) for a in spec.algos]
    (family_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": spec.config_path,
                "arms": [a.name for a in spec.arms],
                "envs": spec.envs,
                "algos": algo_ids,
                "seeds": spec.seeds,
            },
            indent=2,
        )
    )
    ran, skipped = 0, 0
    for arm in spec.arms:
        for env in spec.envs:
            for algo in spec.algos:
                algo_id = _canonical(algo)
                for seed in spec.seeds:
                    leaf = _leaf_dir(root, spec, arm, env, algo_id, seed)
                    if (leaf / "eval_metrics.csv").exists():
                        skipped += 1
                        continue
                    leaf.mkdir(parents=True, exist_ok=True)
                    # Per-leaf re-seed so a leaf's result is independent of
                    # which leaves ran before it (same discipline as the
                    # generated sweeps' per-point re-seeding).
                    set_seed(seed, deterministic=True)
                    env_cfg = EnvConfig(
                        env_id=env,
                        n_train_envs=int(spec.budgets.get("n_train_envs", 2)),
                        n_eval_envs=int(spec.budgets.get("n_eval_envs", 16)),
                        rollout_len=int(spec.budgets.get("rollout_len", 512)),
                        seed=seed,
                        env_wrapper=spec.env_wrapper,
                        env_kwargs={**spec.env_kwargs, **arm.env_kwargs},
                        offline_dataset=arm.offline_dataset[env],
                    )
                    n_ckpt = int(spec.budgets.get("n_checkpoints", 2))
                    train_cfg = TrainingConfig(
                        n_episodes=max(2, n_ckpt),
                        n_checkpoints=n_ckpt,
                        deterministic=True,
                        device=device or "cpu",
                        algorithm=algo_id,
                        aggregation=spec.aggregation,
                        actor_network=algo["actor"],
                        critic_network=algo["critic"],
                        network_kwargs=algo["network_kwargs"],
                        offline_grad_steps=int(spec.budgets["offline_grad_steps"]),
                        record_eval_video=False,
                        eval_count_terminal_reward=spec.eval_count_terminal_reward,
                    )
                    print(
                        f"[hosted_sweep] {arm.name}/{env}/{algo_id}/seed{seed} "
                        f"({spec.budgets['offline_grad_steps']} grad steps)"
                    )
                    BenchmarkRunner(
                        env_cfg,
                        train_cfg,
                        RunConfig(run_dir=str(leaf), timestamp="sweep"),
                        registry.get(algo["name"]),
                    ).run()
                    ran += 1
    return (
        f"[hosted_sweep] {spec.regime}/{spec.simulation}: {ran} leaves run, "
        f"{skipped} skipped (already complete) -> {family_dir}/\n"
        f"[hosted_sweep] report: uv run python -m src.benchmarking.render_hosted_report "
        f"{spec.regime} --simulation {spec.simulation}"
    )

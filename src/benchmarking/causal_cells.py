"""``--mode causal_cells``: offline-cell orchestration (Cells 3–8).

Per gate decision 6, online cells (1–2) run through plain benchmark mode on
``causal/`` env ids; this dispatcher handles the OFFLINE cells: load a logged
dataset (per-cell switches applied at load time), train the basic and variant
offline algorithms, OPE them on the dataset, evaluate TRUE per-episode J in
the genuine environment, and report regret against the Cell-1 reference —
all into ``causal_cells_metrics.csv`` (§6.6 schema).

Cell YAML schema (reproducibility/causal_cells/*.yaml):

    mode: causal_cells
    cell: 3
    anchor: discrete                # discrete | continuous
    task_env: CartPole-v1           # TRUE env for J
    eval_env: CartPole-v1           # learner's view at eval (masked id for 4/6/8)
    mask_spec: null                 # dataset-side mask indices for 4/6/8
    behavior_policy: known          # known | unknown (unknown discards propensities)
    tiers: {medium: causal/cartpole/medium-v0, ...}
    algos: {basic: bc, variant: cql}
    seeds: [0]
    training: {n_steps: 3000, batch_size: 256}
    ope: {ipw_behavior: auto, fqe_iters: 400, n_boot_episodes: null}
    evaluation: {n_episodes: 100}
    reference_checkpoint: <path to a Cell-1 PPO ckpt for this task>
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import yaml

from src.config.seeding import set_seed
from src.data.experience_source import OfflineDatasetSource
from src.data.minari_io import to_offline_source
from src.eval.ope import (
    DeterministicPolicyAdapter,
    DirectMethod,
    DoublyRobust,
    IPWEstimator,
    NaiveEstimator,
    StochasticPolicyAdapter,
    TargetPolicy,
)
from src.eval.regret import (
    CAUSAL_CELLS_COLUMNS,
    compute_regret,
    evaluate_policy,
    random_policy_returns,
)
from src.logging.logger import CSVLogger
from src.offline.bc import BehaviorCloning
from src.rl.on_policy.policy import ActorCriticMLP


def _env_dims(env_id: str):
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    if hasattr(env.action_space, "n"):
        action_type, action_dim = "discrete", int(env.action_space.n)
    else:
        action_type, action_dim = "continuous", int(env.action_space.shape[0])
    env.close()
    return obs_dim, action_dim, action_type


def _build_algo(
    name: str, obs_dim: int, action_dim: int, action_type: str, device, seed: int
):
    torch.manual_seed(seed)
    name = name.lower()
    if name == "bc":
        policy = ActorCriticMLP(obs_dim, action_dim, action_type, device)
        return BehaviorCloning(policy, device)
    if name == "cql":
        from src.offline.d3rlpy_wrappers import make_cql

        return make_cql(device, action_type)
    if name == "iql":
        from src.offline.d3rlpy_wrappers import make_iql

        return make_iql(device, action_type)
    raise KeyError(f"Unknown offline algorithm '{name}' (expected bc|cql|iql).")


def _act_fn(agent, device, action_type):
    def act(obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            a = agent.act(t, deterministic=True).action
        a = a.squeeze(0).detach().cpu().numpy()
        return int(a) if action_type == "discrete" else a

    return act


def _target_adapter(agent, device, action_type, action_dim) -> Optional[TargetPolicy]:
    if isinstance(agent, BehaviorCloning):
        return StochasticPolicyAdapter(agent.policy)
    if action_type == "discrete":

        def act(obs: torch.Tensor) -> torch.Tensor:
            return agent.act(obs.to(device)).action

        return DeterministicPolicyAdapter(act, action_dim)
    # deterministic continuous target: IPW/DR are ill-defined (zero-measure
    # actions); DM/naive only.
    return None


def _ope_block(
    source: OfflineDatasetSource,
    agent,
    device,
    action_type: str,
    action_dim: int,
    fqe_iters: int,
    fqe_sync_every: int,
    seed: int,
) -> Dict[str, float]:
    nan = float("nan")
    out = {"ope_naive": nan, "ope_dm": nan, "ope_ipw": nan, "ope_dr": nan}
    target = _target_adapter(agent, device, action_type, action_dim)
    out["ope_naive"] = NaiveEstimator().estimate(source, target).value
    if target is None:
        return out
    out["ope_dm"] = (
        DirectMethod(gamma=1.0, n_iters=fqe_iters, sync_every=fqe_sync_every, seed=seed)
        .estimate(source, target)
        .value
    )
    ipw_behavior = "known" if source.behavior_logprob is not None else "cloned"
    try:
        out["ope_ipw"] = (
            IPWEstimator(behavior=ipw_behavior, seed=seed)
            .estimate(source, target)
            .value
        )
        out["ope_dr"] = (
            DoublyRobust(
                behavior=ipw_behavior,
                gamma=1.0,
                n_fqe_iters=fqe_iters,
                fqe_sync_every=fqe_sync_every,
                seed=seed,
            )
            .estimate(source, target)
            .value
        )
    except NotImplementedError:
        pass  # continuous pi_b cloning arrives in Phase 6
    return out


def _load_reference_act_fn(ckpt_path: str, env_id: str, device):
    from src.benchmarking.checkpoints import load_checkpoint

    obs_dim, action_dim, action_type = _env_dims(env_id)
    policy = ActorCriticMLP(obs_dim, action_dim, action_type, device)
    policy.load_state_dict(load_checkpoint(ckpt_path)["policy_state"])

    def act(obs: np.ndarray):
        t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            a = policy.act_deterministic(t).squeeze(0).cpu().numpy()
        return int(a) if action_type == "discrete" else a

    return act


def run_causal_cells(cfg: dict, run_dir: str, device: torch.device) -> str:
    """Execute one offline-cell spec; returns the metrics CSV path."""
    cell = int(cfg["cell"])
    anchor = str(cfg.get("anchor", ""))
    task_env = str(cfg["task_env"])
    eval_env = str(cfg.get("eval_env", task_env))
    behavior_policy = str(cfg.get("behavior_policy", "known"))
    mask_spec = cfg.get("mask_spec")
    tiers: Dict[str, str] = dict(cfg["tiers"])
    algos = dict(cfg["algos"])
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    train_cfg = dict(cfg.get("training", {}))
    n_steps = int(train_cfg.get("n_steps", 3000))
    batch_size = int(train_cfg.get("batch_size", 256))
    ope_cfg = dict(cfg.get("ope", {}))
    fqe_iters = int(ope_cfg.get("fqe_iters", 400))
    fqe_sync_every = int(ope_cfg.get("fqe_sync_every", 50))
    eval_cfg = dict(cfg.get("evaluation", {}))
    n_eval_episodes = int(eval_cfg.get("n_episodes", 100))
    reference_ckpt = cfg.get("reference_checkpoint")

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump({"mode": "causal_cells", "cell": cell}, f, indent=2)

    obs_dim_task, action_dim, action_type = _env_dims(task_env)
    mask_indices = None
    if mask_spec:
        from src.causal.masking import resolve_mask_indices

        probe = gym.make(task_env)
        mask_indices = resolve_mask_indices(probe, mask_spec)
        probe.close()

    # reference + random floors (once per task)
    j_random = float(np.mean(random_policy_returns(task_env, n_eval_episodes)))
    if reference_ckpt:
        ref_act = _load_reference_act_fn(str(reference_ckpt), task_env, device)
        j_ref = float(np.mean(evaluate_policy(task_env, ref_act, n_eval_episodes)))
    else:
        raise ValueError(
            "reference_checkpoint is required: train Cell 1 first (benchmark "
            "mode on the causal/<anchor>-cell1 id) and point the YAML at the "
            "final PPO checkpoint."
        )

    csv_path = os.path.join(run_dir, "causal_cells_metrics.csv")
    with CSVLogger(csv_path, fieldnames=CAUSAL_CELLS_COLUMNS) as log:
        base_row = {"cell": cell, "task": task_env, "anchor": anchor}
        log.log(
            {
                **base_row,
                "tier": "",
                "algo": "ppo",
                "role": "reference",
                "seed": "",
                "J": j_ref,
                "regret": 0.0,
                "normalized_regret": 0.0,
            }
        )
        log.log(
            {
                **base_row,
                "tier": "",
                "algo": "random",
                "role": "random",
                "seed": "",
                "J": j_random,
                "regret": j_ref - j_random,
                "normalized_regret": 1.0,
            }
        )
        for tier, dataset_id in tiers.items():
            for seed in seeds:
                set_seed(seed, deterministic=False)
                source = to_offline_source(
                    dataset_id,
                    device,
                    behavior_policy=behavior_policy,
                    mask_indices=mask_indices,
                    rng_seed=seed,
                )
                obs_dim = source.obs.shape[-1]
                for role in ("basic", "variant"):
                    algo_name = algos.get(role)
                    if not algo_name:
                        continue
                    agent = _build_algo(
                        algo_name, obs_dim, action_dim, action_type, device, seed
                    )
                    agent.fit_source(source, n_steps, batch_size=batch_size)
                    returns = evaluate_policy(
                        eval_env,
                        _act_fn(agent, device, action_type),
                        n_eval_episodes,
                        seed_base=20_000 + 100 * seed,
                    )
                    j = float(returns.mean())
                    reg = compute_regret(j, j_ref, j_random)
                    ope = _ope_block(
                        source,
                        agent,
                        device,
                        action_type,
                        action_dim,
                        fqe_iters,
                        fqe_sync_every,
                        seed,
                    )
                    log.log(
                        {
                            **base_row,
                            "tier": tier,
                            "algo": algo_name,
                            "role": role,
                            "seed": seed,
                            "J": j,
                            "regret": reg.regret,
                            "normalized_regret": reg.normalized_regret,
                            **ope,
                        }
                    )
                    print(
                        f"[cell {cell}|{tier}|seed {seed}] {role}={algo_name}: "
                        f"J={j:.1f} regret={reg.regret:.1f} "
                        f"(norm {reg.normalized_regret:.2f}) "
                        f"naive={ope['ope_naive']:.1f} dm={ope['ope_dm']:.1f} "
                        f"ipw={ope['ope_ipw']:.1f} dr={ope['ope_dr']:.1f}"
                    )
    return csv_path

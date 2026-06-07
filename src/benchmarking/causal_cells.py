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
from src.eval.kallus_zhou import kz_interval
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


class _rng_guard:
    """Save/restore ALL global RNG state around intermediate curve evals so
    the final policy matches the unchunked training path exactly."""

    def __enter__(self):
        import random as _random

        self.torch_state = torch.get_rng_state()
        self.cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        self.py_state = _random.getstate()
        self.np_state = np.random.get_state()
        return self

    def __exit__(self, *exc):
        import random as _random

        torch.set_rng_state(self.torch_state)
        if self.cuda_state is not None:
            torch.cuda.set_rng_state_all(self.cuda_state)
        _random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        return False


CURVE_COLUMNS = [
    "cell",
    "task",
    "anchor",
    "tier",
    "algo",
    "role",
    "seed",
    "step",
    "J",
]


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
        # continuous control needs capacity comparable to the d3rlpy variants
        # (256x256) to be a fair basic baseline; discrete stays at the repo
        # default (64x64).
        hidden = (256, 256) if action_type == "continuous" else (64, 64)
        policy = ActorCriticMLP(
            obs_dim, action_dim, action_type, device, hidden_dims=hidden
        )
        return BehaviorCloning(policy, device)
    if name == "cql":
        from src.offline.d3rlpy_wrappers import make_cql

        return make_cql(device, action_type)
    if name == "iql":
        from src.offline.d3rlpy_wrappers import make_iql

        return make_iql(device, action_type)
    if name == "proximal":
        if action_type != "discrete":
            raise ValueError("proximal variant is discrete in Phase 5.")
        from src.offline.proximal import ProximalBC

        return ProximalBC(obs_dim, action_dim, device, seed=seed)
    if name == "latent_wm":
        if action_type != "discrete":
            raise ValueError("latent_wm variant is discrete in Phase 5.")
        from src.offline.latent_world_model import LatentWorldModelBC

        return LatentWorldModelBC(obs_dim, action_dim, device, seed=seed)
    if name in ("ens_pessimistic", "delphic"):  # "delphic" kept as alias
        if action_type != "discrete":
            raise ValueError("ens_pessimistic variant is discrete in Phase 4/5.")
        from src.offline.ensemble_pessimism import EnsemblePessimisticDQN

        return EnsemblePessimisticDQN(obs_dim, action_dim, device, seed=seed)
    raise KeyError(
        f"Unknown offline algorithm '{name}' (expected bc|cql|iql|ens_pessimistic)."
    )


def _act_fn(agent, device, action_type):
    # stateful (history/recurrent) agents supply their own eval act-fn with
    # a per-episode reset() (evaluate_policy calls it at episode starts)
    if hasattr(agent, "make_eval_act_fn"):
        return agent.make_eval_act_fn(device)

    def act(obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            a = agent.act(t, deterministic=True).action
        a = a.squeeze(0).detach().cpu().numpy()
        return int(a) if action_type == "discrete" else a

    return act


def _target_adapter(agent, device, action_type, action_dim) -> Optional[TargetPolicy]:
    # History-dependent targets (proximal, latent world model) cannot be
    # faithfully evaluated by memoryless OPE adapters - their action depends
    # on the trajectory prefix, which flat (obs, action) batches do not carry.
    # OPE for those rows is naive-only (honest), like deterministic continuous.
    if hasattr(agent, "make_eval_act_fn"):
        return None
    if isinstance(agent, BehaviorCloning):
        return StochasticPolicyAdapter(agent.policy)
    if action_type == "discrete":

        def act(obs: torch.Tensor) -> torch.Tensor:
            return agent.act(obs.to(device)).action

        return DeterministicPolicyAdapter(act, action_dim)
    # continuous deterministic target: smooth with fixed Gaussian eval noise
    # so IPW/DR densities are well-defined (Phase-6 prerequisite).
    from src.eval.ope import GaussianPolicyAdapter

    def act_c(obs: torch.Tensor) -> torch.Tensor:
        return agent.act(obs.to(device)).action

    return GaussianPolicyAdapter(act_c, sigma=0.1)


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
    kz_gamma = float(ope_cfg.get("kz_gamma", 2.0))
    eval_cfg = dict(cfg.get("evaluation", {}))
    n_eval_episodes = int(eval_cfg.get("n_episodes", 100))
    eval_every = int(train_cfg.get("eval_every", max(1, n_steps // 10)))
    dataset_expect = dict(cfg.get("dataset_expect", {}) or {})
    curve_episodes = int(eval_cfg.get("curve_episodes", 20))
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

    # random floor (once per task); references are PER SEED when a reference
    # spec is given (train-or-load, Phase-6), or a fixed legacy checkpoint.
    reference_spec = cfg.get("reference")
    if not reference_spec and not reference_ckpt:
        raise ValueError(
            "a reference is required: either reference: {env, algo, "
            "n_episodes, ...} (train-or-load, preferred) or a legacy "
            "reference_checkpoint path."
        )
    j_random = float(np.mean(random_policy_returns(task_env, n_eval_episodes)))
    j_ref_by_seed: Dict[int, float] = {}

    csv_path = os.path.join(run_dir, "causal_cells_metrics.csv")
    curve_path = os.path.join(run_dir, "learning_curve.csv")
    with CSVLogger(csv_path, fieldnames=CAUSAL_CELLS_COLUMNS) as log, CSVLogger(
        curve_path, fieldnames=CURVE_COLUMNS
    ) as curve_log:
        base_row = {"cell": cell, "task": task_env, "anchor": anchor}

        def _ensure_j_ref(seed: int) -> float:
            if seed in j_ref_by_seed:
                return j_ref_by_seed[seed]
            if reference_spec:
                from src.eval.references import ensure_reference

                path = ensure_reference(dict(reference_spec), seed, device)
            else:
                path = str(reference_ckpt)
            ref_act = _load_reference_act_fn(path, task_env, device)
            j_ref = float(np.mean(evaluate_policy(task_env, ref_act, n_eval_episodes)))
            j_ref_by_seed[seed] = j_ref
            log.log(
                {
                    **base_row,
                    "tier": "",
                    "algo": str((reference_spec or {}).get("algo", "ppo")),
                    "role": "reference",
                    "seed": seed if reference_spec else "",
                    "J": j_ref,
                    "regret": 0.0,
                    "normalized_regret": 0.0,
                }
            )
            return j_ref

        log.log(
            {
                **base_row,
                "tier": "",
                "algo": "random",
                "role": "random",
                "seed": "",
                "J": j_random,
                "regret": "",
                "normalized_regret": 1.0,
            }
        )
        for tier, dataset_id in tiers.items():
            # Confounded cells (7-8): the gate validates the data-generating
            # process on the FULL known-pi_b view BEFORE any cell switches
            # (mask / propensity discard) are applied. Hard error on failure.
            gate_passed = ""
            from src.causal.cells import get_cell

            if get_cell(cell).confounded:
                from src.causal.confounding import assert_confounded

                gate_view = to_offline_source(
                    dataset_id,
                    device,
                    behavior_policy="known",
                    expect=dataset_expect.get(tier),
                )
                report = assert_confounded(gate_view)  # raises if unconfounded
                gate_passed = True
                import csv as _csv

                gate_csv = os.path.join(run_dir, "gate_reports.csv")
                write_header = not os.path.exists(gate_csv)
                with open(gate_csv, "a", newline="") as gf:
                    w = _csv.DictWriter(
                        gf,
                        fieldnames=[
                            "tier",
                            "dataset_id",
                            "naive_value",
                            "ipw_value",
                            "naive_ipw_gap",
                            "action_u_tv",
                            "action_u_zscore",
                            "reward_u_zscore",
                        ],
                    )
                    if write_header:
                        w.writeheader()
                    w.writerow(
                        {
                            "tier": tier,
                            "dataset_id": dataset_id,
                            "naive_value": report.naive_value,
                            "ipw_value": report.ipw_value,
                            "naive_ipw_gap": report.naive_ipw_gap,
                            "action_u_tv": report.action_u_dependence,
                            "action_u_zscore": report.action_u_zscore,
                            "reward_u_zscore": report.reward_u_zscore,
                        }
                    )
                print(
                    f"[gate PASS] {dataset_id}: |naive-ipw|="
                    f"{report.naive_ipw_gap:.2f} A-U z={report.action_u_zscore:.1f} "
                    f"R-U z={report.reward_u_zscore:.1f}"
                )
            for seed in seeds:
                j_ref = _ensure_j_ref(seed)
                set_seed(seed, deterministic=False)
                source = to_offline_source(
                    dataset_id,
                    device,
                    behavior_policy=behavior_policy,
                    mask_indices=mask_indices,
                    rng_seed=seed,
                    expect=dataset_expect.get(tier),
                )
                obs_dim = source.obs.shape[-1]

                def _curve_cb(agent_, algo_label_, role_, tier_=tier, seed_=seed):
                    act = _act_fn(agent_, device, action_type)

                    def cb(step: int) -> None:
                        with _rng_guard():
                            rets = evaluate_policy(
                                eval_env,
                                act,
                                curve_episodes,
                                seed_base=60_000 + 100 * seed_,
                            )
                        curve_log.log(
                            {
                                **base_row,
                                "tier": tier_,
                                "algo": algo_label_,
                                "role": role_,
                                "seed": seed_,
                                "step": step,
                                "J": float(rets.mean()),
                            }
                        )

                    return cb

                trained = {}
                for role in ("basic", "variant"):
                    algo_name = algos.get(role)
                    if not algo_name:
                        continue
                    if algo_name == "kz_select":
                        # Cell-8 variant (gate condition 2): no new training
                        # loop — pick the Kallus-Zhou LOWER-BOUND maximizer
                        # among already-trained candidates.
                        candidates = dict(trained)
                        extra = str(algos.get("kz_candidates", "ens_pessimistic"))
                        for name in extra.split():
                            if name not in candidates:
                                cand = _build_algo(
                                    name, obs_dim, action_dim, action_type, device, seed
                                )
                                cand.fit_source(
                                    source,
                                    n_steps,
                                    batch_size=batch_size,
                                    on_step=_curve_cb(cand, name, "variant"),
                                    on_step_every=eval_every,
                                )
                                candidates[name] = cand
                        best_name, best_agent, best_iv = None, None, None
                        for name, cand in candidates.items():
                            tgt = _target_adapter(cand, device, action_type, action_dim)
                            if tgt is None:
                                continue
                            iv = kz_interval(
                                source, tgt, gamma=kz_gamma, clone_seed=seed
                            )
                            print(
                                f"[kz_select] {name}: LB={iv.lower:.1f} "
                                f"UB={iv.upper:.1f} (Gamma={kz_gamma})"
                            )
                            if best_iv is None or iv.lower > best_iv.lower:
                                best_name, best_agent, best_iv = name, cand, iv
                        agent, algo_label = best_agent, f"kz:{best_name}"
                        kz_cols = {
                            "ope_kz_lb": best_iv.lower,
                            "ope_kz_ub": best_iv.upper,
                            "kz_gamma": kz_gamma,
                        }
                    else:
                        agent = _build_algo(
                            algo_name, obs_dim, action_dim, action_type, device, seed
                        )
                        agent.fit_source(
                            source,
                            n_steps,
                            batch_size=batch_size,
                            on_step=_curve_cb(agent, algo_name, role),
                            on_step_every=eval_every,
                        )
                        trained[algo_name] = agent
                        algo_label = algo_name
                        kz_cols = {}
                        if gate_passed:  # confounded cells: report the interval too
                            tgt = _target_adapter(
                                agent, device, action_type, action_dim
                            )
                            if tgt is not None:
                                iv = kz_interval(
                                    source, tgt, gamma=kz_gamma, clone_seed=seed
                                )
                                kz_cols = {
                                    "ope_kz_lb": iv.lower,
                                    "ope_kz_ub": iv.upper,
                                    "kz_gamma": kz_gamma,
                                }
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
                            "algo": algo_label,
                            "role": role,
                            "seed": seed,
                            "J": j,
                            "regret": reg.regret,
                            "normalized_regret": reg.normalized_regret,
                            "gate_passed": gate_passed,
                            **ope,
                            **kz_cols,
                        }
                    )
                    print(
                        f"[cell {cell}|{tier}|seed {seed}] {role}={algo_label}: "
                        f"J={j:.1f} regret={reg.regret:.1f} "
                        f"(norm {reg.normalized_regret:.2f}) "
                        f"naive={ope['ope_naive']:.1f} dm={ope['ope_dm']:.1f} "
                        f"ipw={ope['ope_ipw']:.1f} dr={ope['ope_dr']:.1f}"
                        + (
                            f" kz=[{kz_cols['ope_kz_lb']:.1f},"
                            f"{kz_cols['ope_kz_ub']:.1f}]"
                            if kz_cols
                            else ""
                        )
                    )
    return csv_path

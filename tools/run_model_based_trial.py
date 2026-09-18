"""First trial of the model-based planners against DQN and PPO, online AND
offline, on plain Gymnasium tasks.

    uv run python tools/run_model_based_trial.py --plan-only
    uv run python tools/run_model_based_trial.py --shard 0/4     # one worker
    uv run python tools/run_model_based_trial.py --report        # tables + figure

Grid (declared here, not in a YAML — it is a trial, not a campaign):

* ONLINE, tabular envs (Discrete obs): FrozenLake-v1, Taxi-v3, CliffWalking-v1
  with ``tabular_vi`` vs ``dqn`` vs ``ppo``.
* ONLINE, vector envs: CartPole-v1, Acrobot-v1, MountainCar-v0, LunarLander-v3
  with ``mlp_vi`` vs ``dqn`` vs ``ppo``.
* OFFLINE, tabular envs on ``uniform`` (uniform-random actions: full
  coverage, the model-based sweet spot) and ``medium`` (the DQN generator's
  1/3-of-expert checkpoint, epsilon 0.1) datasets, generated here if missing,
  with ``offline_tabular_vi`` vs ``offline_dqn`` vs ``cql``.
* OFFLINE, vector envs on the existing ``generated/<env>/medium-v0`` datasets
  with ``offline_mlp_vi`` vs ``offline_dqn`` vs ``cql``.

PPO has no offline form (on-policy), so CQL stands in as the second offline
comparator. Same env-step budget for every online run; same gradient-step
budget for every offline run. Leaves land under
``results/model_based_trial/{online,offline}/<env>[/<tier>]/<algo>/seed<k>/``
and a finished leaf (``eval_metrics.csv`` present) is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("results/model_based_trial")
SEEDS = (0, 1)

# CliffWalking-v1 registers NO TimeLimit (a policy that never reaches the goal
# never terminates: the medium-tier generator ran 408k steps for 0 episodes),
# so the trial uses a 200-step-limited registration of the same env.
CLIFF_ID = "CliffWalking200-v1"
TABULAR_ENVS = ("FrozenLake-v1", "Taxi-v3", CLIFF_ID)


def register_trial_envs() -> None:
    import gymnasium as gym

    if CLIFF_ID not in gym.registry:
        gym.register(
            id=CLIFF_ID,
            entry_point="gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
            max_episode_steps=200,
        )


VECTOR_ENVS = ("CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v3")

# One env-step budget for every online run: 8 envs x 256 steps x 50 rounds.
ONLINE = dict(
    n_train_envs=8, n_eval_envs=16, rollout_len=256, n_episodes=50, n_checkpoints=10
)
# One optimiser-step budget for every offline run.
OFFLINE = dict(
    offline_grad_steps=10_000, n_checkpoints=10, n_eval_envs=16, eval_rollout_len=500
)
# Tabular offline datasets are generated on demand at this size.
TABULAR_DATASET_EPISODES = 400


def _slug(env_id: str) -> str:
    return env_id.split("-v")[0].lower()


def _dataset_id(env_id: str, tier: str) -> str:
    return f"generated/{_slug(env_id)}/{tier}-v0"


def enumerate_runs() -> list[dict]:
    runs = []
    for env in TABULAR_ENVS:
        for algo in ("tabular_vi", "dqn", "ppo"):
            for s in SEEDS:
                runs.append(
                    dict(regime="online", env=env, algo=algo, seed=s, tier=None)
                )
    for env in VECTOR_ENVS:
        for algo in ("mlp_vi", "dqn", "ppo"):
            for s in SEEDS:
                runs.append(
                    dict(regime="online", env=env, algo=algo, seed=s, tier=None)
                )
    for env in TABULAR_ENVS:
        for tier in ("uniform", "medium"):
            for algo in ("offline_tabular_vi", "offline_dqn", "cql"):
                for s in SEEDS:
                    runs.append(
                        dict(regime="offline", env=env, algo=algo, seed=s, tier=tier)
                    )
    for env in ("CartPole-v1", "Acrobot-v1", "LunarLander-v3"):
        for algo in ("offline_mlp_vi", "offline_dqn", "cql"):
            for s in SEEDS:
                runs.append(
                    dict(regime="offline", env=env, algo=algo, seed=s, tier="medium")
                )
    return runs


def leaf_dir(run: dict) -> Path:
    parts = [ROOT, run["regime"], run["env"]]
    if run["tier"]:
        parts.append(run["tier"])
    parts += [run["algo"], f"seed{run['seed']}"]
    return Path(*parts)


def _uniform_random_agent(env_id: str):
    """A fresh DQN with epsilon = 1: ``act`` draws uniformly at random, so the
    generator's rollout is a uniform-random behaviour policy (the 'random'
    tier is NOT that: a fresh net's epsilon-0.1 greedy concentrates on one or
    two actions -- measured 59% (s,a) coverage on FrozenLake)."""
    import gymnasium as gym
    import torch
    from gymnasium.spaces.utils import flatdim
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    register_trial_envs()
    probe = gym.make(env_id)
    obs_dim, n_actions = flatdim(probe.observation_space), probe.action_space.n
    probe.close()
    _, agent = registry.get("dqn").builder(
        obs_dim=obs_dim,
        action_dim=n_actions,
        action_type="discrete",
        device=torch.device("cpu"),
        action_space=None,
        obs_shape=(obs_dim,),
    )
    agent.epsilon = 1.0
    return agent


def ensure_dataset(env_id: str, tier: str, seed: int = 0) -> str:
    """Tabular datasets are generated here (uniform = epsilon-1 agent, medium =
    the DQN generator's 1/3-of-expert checkpoint); vector ones must exist."""
    import minari

    did = _dataset_id(env_id, tier)
    if did in set(minari.list_local_datasets()):
        return did
    if env_id not in TABULAR_ENVS:
        raise SystemExit(
            f"dataset {did!r} missing; generate it with tools/generate_offline.py"
        )
    from src.envs.offline.generate import generate_offline_dataset

    register_trial_envs()
    print(f"[gen] {did} ({TABULAR_DATASET_EPISODES} episodes)", flush=True)
    extra = {}
    if tier == "uniform":
        extra = dict(tier="random", agent=_uniform_random_agent(env_id))
    generate_offline_dataset(
        env_id,
        "dqn",
        extra.pop("tier", tier),
        rollout_episodes=TABULAR_DATASET_EPISODES,
        seed=seed,
        dataset_id=did,
        run_dir=str(ROOT / "_generation" / _slug(env_id) / tier),
        device="cpu",
        rollout_n_envs=8,
        **extra,
    )
    return did


def run_one(run: dict, device: str) -> None:
    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    register_trial_envs()
    out = leaf_dir(run)
    out.mkdir(parents=True, exist_ok=True)
    if run["regime"] == "online":
        env_cfg = EnvConfig(
            env_id=run["env"],
            n_train_envs=ONLINE["n_train_envs"],
            n_eval_envs=ONLINE["n_eval_envs"],
            rollout_len=ONLINE["rollout_len"],
            seed=run["seed"],
        )
        train_cfg = TrainingConfig(
            n_episodes=ONLINE["n_episodes"],
            n_checkpoints=ONLINE["n_checkpoints"],
            device=device,
            algorithm=run["algo"],
            aggregation="mean",
            record_eval_video=False,
            # Count the terminal step's reward (FrozenLake/Taxi pay ON it); the
            # legacy default drops it and reads 0.0 for a goal-reaching policy.
            eval_count_terminal_reward=True,
        )
    else:
        did = ensure_dataset(run["env"], run["tier"])
        env_cfg = EnvConfig(
            env_id=run["env"],
            n_train_envs=2,
            n_eval_envs=OFFLINE["n_eval_envs"],
            rollout_len=OFFLINE["eval_rollout_len"],
            eval_rollout_len=OFFLINE["eval_rollout_len"],
            seed=run["seed"],
            offline_dataset=did,
        )
        train_cfg = TrainingConfig(
            n_episodes=1,
            n_checkpoints=OFFLINE["n_checkpoints"],
            device=device,
            algorithm=run["algo"],
            aggregation="mean",
            offline_grad_steps=OFFLINE["offline_grad_steps"],
            record_eval_video=False,
            eval_count_terminal_reward=True,
        )
    t0 = time.time()
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(out), timestamp="trial"),
        registry.get(run["algo"]),
    ).run()
    (out / "trial_meta.json").write_text(
        json.dumps(
            dict(run, wall_seconds=round(time.time() - t0, 1), device=device), indent=1
        )
    )


def done(run: dict) -> bool:
    return (leaf_dir(run) / "eval_metrics.csv").exists()


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def _final_eval(run: dict):
    p = leaf_dir(run) / "eval_metrics.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return None
    r = rows[-1]
    return float(r["eval_return_mean"]), [float(x["eval_return_mean"]) for x in rows]


def report() -> None:
    import numpy as np

    runs = enumerate_runs()
    table: dict = {}
    curves: dict = {}
    for run in runs:
        got = _final_eval(run)
        key = (run["regime"], run["env"], run["tier"], run["algo"])
        if got is None:
            continue
        table.setdefault(key, []).append(got[0])
        curves.setdefault(key, []).append(got[1])
    lines = ["# Model-based value iteration vs DQN / PPO — first trial", ""]
    lines.append(
        f"Online budget: {ONLINE['n_train_envs']} envs x {ONLINE['rollout_len']} steps x "
        f"{ONLINE['n_episodes']} rounds = {ONLINE['n_train_envs']*ONLINE['rollout_len']*ONLINE['n_episodes']:,} "
        f"env steps per run. Offline budget: {OFFLINE['offline_grad_steps']:,} gradient steps per run. "
        f"Final-checkpoint eval return, mean ± sd over seeds {SEEDS} (n shown). "
        "PPO has no offline form; CQL is the second offline comparator. "
        "THE METRIC: the runner's eval sums rewards over a FIXED window of steps per env "
        f"({ONLINE['rollout_len']} online, {OFFLINE['eval_rollout_len']} offline) with autoreset, "
        "averaged over eval envs -- for episodic tasks that is reward per window, not per "
        "episode (CliffWalking: -1 per step, so the optimal 13-step loop reads about "
        "-(window - window/14); Taxi: +20 per delivery; FrozenLake: goals reached per window)."
    )
    for regime in ("online", "offline"):
        lines += ["", f"## {regime}", ""]
        algos = sorted(
            {k[3] for k in table if k[0] == regime}, key=lambda a: ("vi" not in a, a)
        )
        lines.append("| env | tier | " + " | ".join(algos) + " |")
        lines.append("|---|---|" + "---|" * len(algos))
        envs = [
            (e, t)
            for e in TABULAR_ENVS + VECTOR_ENVS
            for t in (None, "uniform", "medium")
        ]
        for env, tier in envs:
            cells = []
            any_ = False
            for a in algos:
                v = table.get((regime, env, tier, a))
                if v is None:
                    cells.append("—")
                    continue
                any_ = True
                arr = np.asarray(v)
                cells.append(f"{arr.mean():.1f} ± {arr.std():.1f} (n={len(arr)})")
            if any_:
                lines.append(f"| {env} | {tier or '—'} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "report.md").write_text(md)
    print(md)
    _figure(curves)


def _figure(curves: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = sorted({(k[0], k[1], k[2]) for k in curves})
    if not panels:
        return
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    colors = {"vi": "#eb6834", "dqn": "#2a78d6", "ppo": "#2ca02c", "cql": "#9467bd"}
    for ax, (regime, env, tier) in zip(axes.flat, panels):
        for k, seeds in curves.items():
            if (k[0], k[1], k[2]) != (regime, env, tier):
                continue
            n = min(len(c) for c in seeds)
            arr = np.asarray([c[:n] for c in seeds])
            x = np.arange(1, n + 1)
            fam = "vi" if "vi" in k[3] else k[3].replace("offline_", "")
            ax.plot(x, arr.mean(0), color=colors.get(fam, "k"), label=k[3])
            if arr.shape[0] > 1:
                ax.fill_between(
                    x, arr.min(0), arr.max(0), color=colors.get(fam, "k"), alpha=0.15
                )
        ax.set_title(f"{regime} · {env}" + (f" · {tier}" if tier else ""), fontsize=10)
        ax.set_xlabel("checkpoint")
        ax.set_ylabel("eval return")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "report.png", dpi=140)
    print(f"wrote {ROOT / 'report.png'}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shard", default="0/1", help="i/N: this worker runs every N-th pending run"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--only", default=None, help="substring filter on env/algo/regime")
    p.add_argument(
        "--datasets-only",
        action="store_true",
        help="generate the tabular datasets, then stop",
    )
    args = p.parse_args()
    if args.report:
        report()
        return 0
    if (
        args.datasets_only
    ):  # run BEFORE sharding: parallel shards must not race on generation
        for env in TABULAR_ENVS:
            for tier in ("uniform", "medium"):
                print("dataset", ensure_dataset(env, tier), flush=True)
        return 0
    i, n = (int(x) for x in args.shard.split("/"))
    runs = enumerate_runs()
    if args.only:
        runs = [
            r
            for r in runs
            if args.only in f"{r['regime']}/{r['env']}/{r['tier']}/{r['algo']}"
        ]
    mine = [r for j, r in enumerate(runs) if j % n == i]
    pending = [r for r in mine if not done(r)]
    print(f"shard {i}/{n}: {len(mine)} runs, {len(pending)} pending", flush=True)
    if args.plan_only:
        for r in pending:
            print("  ", leaf_dir(r))
        return 0
    from src.config.threads import configure_intraop_threads

    configure_intraop_threads()
    for k, r in enumerate(pending):
        print(
            f"=== [{k+1}/{len(pending)}] {leaf_dir(r)} ({time.strftime('%H:%M:%S')})",
            flush=True,
        )
        try:
            run_one(r, args.device)
        except Exception as exc:  # keep the shard going; the report shows the hole
            print(f"FAILED {leaf_dir(r)}: {type(exc).__name__}: {exc}", flush=True)
            (leaf_dir(r) / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    raise SystemExit(main())

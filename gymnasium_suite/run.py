from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import List, Sequence

import gymnasium
import gymnasium as gym
import numpy as np
import pandas as pd
import pygame
import torch
from gymnasium import envs
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import EXTRA_KEYS, generate_simulation_name, make_policy, safe_tensor

warnings.filterwarnings(
    "ignore",
    message=".*Overwriting existing videos.*",
    category=UserWarning,
)


def get_envs_names() -> List[str]:
    all_envs = envs.registry.keys()
    env_ids = _get_latest_envs(all_envs)

    env_ids.remove("GymV21Environment-v0")
    env_ids.remove("GymV26Environment-v0")

    return env_ids


def _get_latest_envs(env_keys):
    env_dict = {}
    for env in env_keys:
        env_name, version = env.split("-v")
        version = int(version)
        if env_name not in env_dict or env_dict[env_name] < version:
            env_dict[env_name] = version
    return [f"{name}-v{version}" for name, version in env_dict.items()]


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _iqr_stats(arr: np.ndarray, band: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Inter‑quartile‑range‑trimmed mean & std along axis‑0.

    Returns (mean, std) with the same shape as the input except for axis‑0.
    """
    q_lo, q_hi = np.percentile(arr, [band, 100 - band], axis=0)
    mask = (arr >= q_lo) & (arr <= q_hi)
    trimmed = np.where(mask, arr, np.nan)
    mean = np.nanmean(trimmed, axis=0)
    std = np.nanstd(trimmed, axis=0)
    return mean, std


def _update_benchmark_table(
    rows: list[dict], dest: Path, metric_cols: list[str]
) -> Path:
    """
    Append (or overwrite) results in a persistent CSV called *benchmark_table.csv*.

    * If a row for (env, algo) already exists it is replaced with the newer values.
    * Missing metrics are kept as ``None`` so all rows have the same shape.
    """
    table_path = dest / "benchmark_table.csv"

    df_new = pd.DataFrame(rows).reindex(columns=["env", "algo", *metric_cols])

    if table_path.exists():
        df_old = pd.read_csv(table_path)
        # drop duplicates that are about to be replaced
        duplicates = (
            df_old[["env", "algo"]]
            .apply(tuple, axis=1)
            .isin(df_new[["env", "algo"]].apply(tuple, axis=1))
        )
        df_combined = pd.concat([df_old[~duplicates], df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(table_path, index=False)
    return table_path


# ──────────────────────────────────────────────────────────────────────────────
# main entry point
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(
    json_path: str | Path,
    percentile_band: int = 25,
    base_reward_keys: Sequence[str] = ("min_reward", "max_reward", "mean_reward"),
    cmap_name: str = "tab10",
) -> list[Path]:
    """
    • Plots every scalar metric in *json_path* (upper subplot) together with
      ``n_steps`` (lower), one PNG per metric.

    • Updates/creates *benchmark_table.csv* (in the same folder as *json_path*)
      containing one row per “env + algo” pair and one column per metric, with
      values formatted « *IQR‑mean ± IQR‑std* ».  Missing metrics are ``None``.

    Returns the list of paths to the saved PNGs.
    """
    # ── load run data ───────────────────────────────────────────────────────
    jpath = Path(json_path).expanduser()
    with jpath.open() as f:
        data = json.load(f)

    env_name = data["info"]["env_name"]
    episodes = np.asarray(data["info"]["recorded_episodes"])
    algorithms = data["info"]["algorithms"]

    cmap = plt.get_cmap(cmap_name)
    colors = {algo: cmap(i) for i, algo in enumerate(algorithms)}

    out_dir = jpath.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── collect *all* metric keys (union across algos) ─────────────────────
    keys: list[str] = list(base_reward_keys)
    for algo in algorithms:
        for k in data[algo]:
            if k not in keys and k not in ("n_steps", *base_reward_keys):
                keys.append(k)
    keys.append("n_steps")  # treat steps like any other metric for the table

    saved_plots: list[Path] = []
    summary_rows: list[dict] = []  # ── benchmark table rows

    # ── one pass per algorithm to get numbers for the table ────────────────
    for algo in algorithms:
        row: dict[str, str | None] = {"env": env_name, "algo": algo}

        for key in keys:
            if key not in data[algo]:  # metric missing → None
                row[key] = None
                continue

            arr = np.asarray(data[algo][key])
            m_mean, m_std = _iqr_stats(arr, percentile_band)

            # 1‑D sequences → use the *final* value, 0‑D scalars stay as‑is
            mean_val = float(m_mean[-1] if np.ndim(m_mean) else m_mean)
            std_val = float(m_std[-1] if np.ndim(m_std) else m_std)
            row[key] = f"{mean_val:.3g} ± {std_val:.3g}"

        summary_rows.append(row)

    # ── benchmark table … ──────────────────────────────────────────────────
    bench_path = _update_benchmark_table(summary_rows, out_dir, keys)

    # ── plot one figure per metric (skip n_steps because it is always below)
    for key in keys:
        if key == "n_steps":
            continue

        fig, (ax_top, ax_steps) = plt.subplots(
            2, 1, figsize=(10, 8), dpi=350, sharex=True
        )
        drew_any = False

        for algo in algorithms:
            # steps (always present)
            steps_raw = np.asarray(data[algo]["n_steps"])
            s_mean, s_std = _iqr_stats(steps_raw, percentile_band)
            ax_steps.plot(episodes, s_mean, color=colors[algo], lw=2, label=algo)
            ax_steps.fill_between(
                episodes,
                np.clip(s_mean - s_std, 1e-8, None),
                s_mean + s_std,
                color=colors[algo],
                alpha=0.25,
            )

            # chosen metric
            if key in data[algo]:
                arr = np.asarray(data[algo][key])
                m_mean, m_std = _iqr_stats(arr, percentile_band)
                ax_top.plot(episodes, m_mean, color=colors[algo], lw=2, label=algo)
                ax_top.fill_between(
                    episodes,
                    m_mean - m_std,
                    m_mean + m_std,
                    color=colors[algo],
                    alpha=0.25,
                )
                drew_any = True

        if not drew_any:  # nobody logged this metric → skip figure
            plt.close(fig)
            continue

        # cosmetic touches
        ax_top.set(
            ylabel=key.replace("_", " ").title(),
            title=f"{env_name} – {key.replace('_', ' ').title()} per Episode",
        )
        ax_top.grid(True, which="both", lw=0.3)
        ax_top.legend(fontsize=9)

        ax_steps.set(
            xlabel="Episode",
            ylabel="Steps",
            title=f"{env_name} – Steps per Episode",
            yscale="log",
        )
        ax_steps.grid(True, which="both", lw=0.3)
        ax_steps.legend(fontsize=9)

        fig.tight_layout()
        outfile = out_dir / f"{env_name}_{key}.png"
        fig.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(outfile)

    print(f"✔ Benchmark table updated → {bench_path}")
    return saved_plots


# Define a factory function to apply TimeLimit
def _make_env(
    rank: int,
    env_name: str,
    path_simulation: str,
    safe_env_name: str,
    episodes_to_record: np.ndarray,
    algo_name: str,
    max_episode_steps: int = None,
):
    def _thunk():
        # 1. build the base env
        if "FrozenLake" in env_name:
            env = gym.make(
                env_name,
                is_slippery=False,
                render_mode="rgb_array",  # <<–– for video
                max_episode_steps=max_episode_steps,
            )
        else:
            env = gym.make(
                env_name,
                render_mode="rgb_array",  # <<–– for video
                max_episode_steps=max_episode_steps,
            )
        env._np_random_seed = 42 + rank

        # 2. give *each* seed its own video directory
        video_dir = os.path.join(
            path_simulation, "videos", safe_env_name, f"{algo_name}_seed{rank}"
        )
        os.makedirs(video_dir, exist_ok=True)

        # 3. wrap with RecordVideo – only when
        #    the current episode is in `episodes_to_record`
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: ep in episodes_to_record,
            name_prefix=f"{safe_env_name}_seed{rank}",
            disable_logger=True,
            fps=30,
        )
        return env

    return _thunk


def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    else:
        return obj


def run_gymnasium(
    n_seeds: int,
    n_episodes: int,
    algorithms_names: List,
    max_episode_steps: int = None,
    causal_knowledge_update_per_episode: int = 1,
    n_checkpoints: int = 10,
):

    if n_checkpoints > n_episodes:
        n_checkpoints = n_episodes

    episodes_to_record = np.linspace(
        start=0, stop=n_episodes - 1, num=n_checkpoints, dtype=int
    )

    path_simulation = "benchmarking/" + generate_simulation_name("gymnasium_suite")
    os.makedirs(path_simulation, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gymnasium_envs = get_envs_names()  # ["FrozenLake-v1"]  #

    for env_index, env_name in enumerate(gymnasium_envs):

        safe_env_name = env_name.replace("/", "_").replace("\\", "_")
        json_path = os.path.join(path_simulation, f"{safe_env_name}.json")
        metrics_dict = {
            "info": {
                "recorded_episodes": episodes_to_record,
                "env_name": safe_env_name,
                "algorithms": [],
            },
        }

        for algo_index, algo_name in enumerate(algorithms_names):

            # ─── build vectorised env with n_seeds instances ───────────────────
            envs = gym.vector.SyncVectorEnv(
                [
                    _make_env(
                        rank,
                        env_name,
                        path_simulation,
                        safe_env_name,
                        episodes_to_record,
                        algo_name,
                        max_episode_steps,
                    )
                    for rank in range(n_seeds)
                ]
            )
            pygame.init()

            act_space = envs.single_action_space
            obs_space = envs.single_observation_space

            if isinstance(act_space, gymnasium.spaces.Box) and algo_name == "dqn":
                envs.close()  # end‑algo
                continue

            metrics_dict["info"]["observation_space"] = (
                str(obs_space)
                if isinstance(obs_space, gymnasium.spaces.Discrete)
                else "continuous"
            )
            metrics_dict["info"]["action_space"] = (
                str(act_space)
                if isinstance(act_space, gymnasium.spaces.Discrete)
                else "continuous"
            )
            metrics_dict["info"]["algorithms"].append(algo_name)

            # ─── allocate metrics arrays ───────────────────────────────────────
            n_ckpt = len(episodes_to_record)
            base_keys = ["min_reward", "max_reward", "mean_reward", "n_steps"]
            algo_metrics = base_keys + EXTRA_KEYS[algo_name.lower()]
            metrics_dict[algo_name] = {
                k: np.zeros((n_seeds, n_ckpt)) for k in algo_metrics
            }

            # ─── create policy (once) ──────────────────────────────────────────
            policy = make_policy(
                algo_name, act_space, obs_space, n_seeds, n_episodes=n_episodes
            )

            pbar = tqdm(
                range(n_episodes),
                desc=(
                    f"Env {env_index + 1}/{len(gymnasium_envs)}  "
                    f"{env_name} | Algo {algo_index + 1}/{len(algorithms_names)}  "
                    f"{algo_name}"
                ),
            )

            for episode in pbar:
                # reset all envs with deterministic seeds
                obs, _ = envs.reset(seed=list(range(n_seeds)))

                if hasattr(policy, "update_episode"):
                    policy.update_episode(episode)

                done_mask = np.zeros(n_seeds, dtype=bool)
                dones_tensor = torch.zeros(n_seeds, dtype=torch.bool, device=device)
                step_count = np.zeros(n_seeds, dtype=int)
                ep_rewards = np.zeros(n_seeds)
                track_max = np.full(n_seeds, -np.inf)
                track_min = np.full(n_seeds, np.inf)

                # ─── ROLLOUT ‑‑ one episode across all envs ────────────────────
                while not done_mask.all():

                    # observation tensor (dtype depends on space)
                    obs_tensor = (
                        safe_tensor(obs, torch.long, device)
                        if isinstance(obs_space, gymnasium.spaces.Discrete)
                        else safe_tensor(obs, torch.float32, device)
                    )

                    # ─ actions & entropy etc.
                    actions_tensor = policy.get_actions(obs_tensor)
                    actions_np = policy.setup_actions(
                        actions_tensor, dones_tensor, out_type="numpy"
                    )

                    # special clip for CliffWalking
                    if "CliffWalking-v0" in env_name:
                        actions_np = np.clip(
                            np.round(actions_np).astype(np.int32), 0, 3
                        )

                    next_obs, rewards, terminated, truncated, _ = envs.step(actions_np)
                    dones = np.logical_or(terminated, truncated)
                    active = ~done_mask

                    # ─ stats
                    ep_rewards += rewards * active
                    track_max = np.where(
                        active, np.maximum(track_max, rewards), track_max
                    )
                    track_min = np.where(
                        active, np.minimum(track_min, rewards), track_min
                    )

                    # ─ policy update
                    rewards_tensor = safe_tensor(rewards, torch.float32, device)
                    next_obs_tensor = (
                        safe_tensor(next_obs, torch.long, device)
                        if isinstance(obs_space, gymnasium.spaces.Discrete)
                        else safe_tensor(next_obs, torch.float32, device)
                    )
                    dones_tensor = safe_tensor(dones, torch.bool, device)

                    policy.update(
                        obs_tensor,
                        actions_tensor,
                        rewards_tensor,
                        next_obs_tensor,
                        dones_tensor,
                    )

                    step_count += active
                    done_mask = np.logical_or(done_mask, dones)
                    obs = next_obs

                # ─── episode‑level aggregates ─────────────────────────────────
                ep_rewards /= np.maximum(step_count, 1)
                pbar.set_postfix(rew=f"{ep_rewards.mean():.3f}")

                # ─── checkpoint logging ───────────────────────────────────────
                if episode in episodes_to_record:
                    idx = np.where(episodes_to_record == episode)[0][0]
                    m = metrics_dict[algo_name]

                    m["min_reward"][:, idx] = track_min
                    m["max_reward"][:, idx] = track_max
                    m["mean_reward"][:, idx] = ep_rewards
                    m["n_steps"][:, idx] = step_count

                    # pull scalar diagnostics from policy
                    scalars = policy.pop_metrics()
                    for key, val in scalars.items():
                        m[key][:, idx] = val  # broadcast same value to all seeds

                    # save JSON
                    with open(json_path, "w") as fp:
                        json.dump(convert_ndarray(metrics_dict), fp, indent=2)

            envs.close()  # end‑algo

        plot_results(json_path)  # end‑env


if __name__ == "__main__":
    # pip install gymnasium[all]

    n_seeds = 5
    n_episodes = int(1e2)
    algorithms = ["dqn", "a2c", "ppo"]
    max_episode_steps = int(1e2)
    causal_knowledge_update_per_episode = 1
    n_checkpoints = int(1e1)

    run_gymnasium(
        n_seeds,
        n_episodes,
        algorithms,
        max_episode_steps,
        causal_knowledge_update_per_episode,
        n_checkpoints,
    )

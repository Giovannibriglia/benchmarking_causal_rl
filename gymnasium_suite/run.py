from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import List

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

warnings.filterwarnings(
    "ignore",
    message=".* Degrees of freedom <= 0",
    category=RuntimeWarning,
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


def _iqr_stats(arr: np.ndarray, q: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust mean/std along axis‑0 using the inter‑quartile range.
    Guaranteed to return finite numbers without numpy RuntimeWarnings.
    Falls back to plain mean/std for columns where the IQR mask removes
    every sample (or when there is only one sample row).
    """
    if arr.size == 0:  # completely empty array
        shape = arr.shape[1:] or (1,)
        return np.zeros(shape), np.zeros(shape)

    # single seed → raw stats
    if arr.shape[0] < 2:
        return arr.astype(float), np.zeros_like(arr, dtype=float)

    q1 = np.percentile(arr, q, axis=0)
    q3 = np.percentile(arr, 100 - q, axis=0)
    mask = (arr >= q1) & (arr <= q3)
    trimmed = np.where(mask, arr, np.nan)

    with np.errstate(all="ignore"):  # silence empty‑slice warnings
        mean_iqr = np.nanmean(trimmed, axis=0)
        std_iqr = np.nanstd(trimmed, axis=0)

    # columns where IQR wiped everything (all NaN) → fall back to full range
    fallback = np.isnan(mean_iqr)
    if fallback.any():
        mean_full = arr[:, fallback].mean(axis=0)
        std_full = arr[:, fallback].std(axis=0)
        mean_iqr[fallback] = mean_full
        std_iqr[fallback] = std_full

    return mean_iqr, std_iqr


# ------------------------------------------------------------------
#  plot_results
# ------------------------------------------------------------------
def plot_results(
    json_path: str | Path,
    percentile_band: int = 25,
    base_reward_keys=("min_reward", "max_reward", "mean_reward"),
    cmap_name: str = "tab10",
) -> list[Path]:
    jpath = Path(json_path).expanduser()
    with jpath.open() as f:
        data = json.load(f)

    env_name = data["info"]["env_name"]
    episodes = np.asarray(data["info"]["recorded_episodes"])
    algos = data["info"]["algorithms"]

    cmap = plt.get_cmap(cmap_name)
    colors = {a: cmap(i) for i, a in enumerate(algos)}
    outdir = jpath.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------ discover *all* metric keys ------------------------
    metric_keys: list[str] = list(base_reward_keys)
    for algo in algos:
        for k in data[algo]:
            if k not in metric_keys and k not in ("n_steps", *base_reward_keys):
                metric_keys.append(k)

    saved: list[Path] = []

    # ------------ helper to update benchmark table ------------------
    rows = []
    for algo in algos:
        row = {"env": env_name, "algo": algo}
        for k in metric_keys + ["n_steps"]:
            if k in data[algo]:
                arr = np.asarray(data[algo][k])
                mean, std = _iqr_stats(arr, percentile_band)
                mean = np.asarray(mean)
                std = np.asarray(std)
                v_mean = float(mean.mean())  # works whether mean.ndim == 0 or >0
                v_std = float(std.mean())
                row[k] = f"{v_mean:.3g} ± {v_std:.3g}"
            else:
                row[k] = None
        rows.append(row)
    _update_benchmark_table(rows, outdir, metric_keys + ["n_steps"])

    # ------------ plotting loop -------------------------------------
    for key in metric_keys:  # skip n_steps (always lower plot)
        fig, (ax_top, ax_low) = plt.subplots(
            2, 1, figsize=(10, 8), dpi=350, sharex=True
        )
        plotted = []

        for algo in algos:
            if key not in data[algo]:
                continue  # algo didn't log this metric

            # ---- upper: chosen metric
            metric_arr = np.asarray(data[algo][key])
            m_mean, m_std = _iqr_stats(metric_arr, percentile_band)
            ax_top.plot(episodes, m_mean, color=colors[algo], lw=2, label=algo)
            ax_top.fill_between(
                episodes, m_mean - m_std, m_mean + m_std, color=colors[algo], alpha=0.15
            )

            # ---- lower: n_steps   (only for same algos)
            steps_arr = np.asarray(data[algo]["n_steps"])
            s_mean, s_std = _iqr_stats(steps_arr, percentile_band)
            ax_low.plot(episodes, s_mean, color=colors[algo], lw=2, label=algo)
            ax_low.fill_between(
                episodes,
                np.clip(s_mean - s_std, 1e-8, None),
                s_mean + s_std,
                color=colors[algo],
                alpha=0.25,
            )
            plotted.append(algo)

        if not plotted:  # nobody logged -> skip file
            plt.close(fig)
            continue

        # ---- cosmetics
        nice = key.replace("_", " ").title()
        ax_top.set(title=f"{env_name} – {nice} per Episode", ylabel=nice)
        ax_top.grid(True, which="both", lw=0.3)
        ax_top.legend(fontsize=9)

        ax_low.set(
            title=f"{env_name} – Steps per Episode",
            xlabel="Episode",
            ylabel="Steps",
            yscale="log",
        )
        ax_low.grid(True, which="both", lw=0.3)
        ax_low.legend(fontsize=9)

        fig.tight_layout()
        out = outdir / f"{env_name}_{key}.png"
        fig.savefig(out, bbox_inches="tight")
        saved.append(out)
        # plt.show()
        plt.close(fig)

    return saved


class GoalTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env_id = str(env.spec.id)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Override terminated logic
        terminated = self.goal_check_fn(obs)

        return obs, reward, terminated, truncated, info

    def goal_check_fn(self, obs) -> bool:

        if self.env_id in [
            "FrozenLake-v1",
            "FrozenLake8x8-v1",
            "CliffWalking-v0",
            "Taxi-v3",
        ]:
            # Assume last state is the goal (can be adjusted per environment)
            return obs == self.env.observation_space.n - 1
        elif self.env_id in ["CartPole-v1", "phys2d/CartPole-v1"]:
            # Consider goal reached if pole angle is less than some small threshold
            return abs(obs[2]) < 0.01
        elif self.env_id in ["MountainCar-v0", "MountainCarContinuous-v0"]:
            return obs[0] > 0.5
        elif self.env_id in ["Pendulum-v1", "phys2d/Pendulum-v0"]:
            return abs(obs[2]) < 0.05
        elif self.env_id == "Acrobot-v1":
            return obs[1] > 0.95
        elif "MountainCar" in self.env_id:
            # Example: consider goal reached if the car’s position is above a threshold
            return obs[0] > 0.5
        elif self.env_id in ["LunarLander-v3", "LunarLanderContinuous-v3"]:
            return -0.1 < obs[0] < 0.1 and -0.1 < obs[1] < 0.1
        elif self.env_id in ["BipedalWalker-v3", "BipedalWalkerHardcore-v3"]:
            return obs[2] > 0.9
        elif self.env_id == "CarRacing-v3":
            # Goal condition is trickier and might depend on the reward structure
            return obs[2] > 0.95  # Placeholder condition
        if self.env_id == "Reacher-v5":
            return obs[-1] < 0.05  # Distance to target is small
        if self.env_id == "Pusher-v5":
            return obs[-1] < 0.05  # Placeholder condition
        if self.env_id in ["InvertedPendulum-v5", "InvertedDoublePendulum-v5"]:
            return abs(obs[2]) < 0.05
        if self.env_id in [
            "Hopper-v5",
            "HalfCheetah-v5",
            "Swimmer-v5",
            "Walker2d-v5",
            "Ant-v5",
            "Humanoid-v5",
        ]:
            return obs[-1] > 0.9  # Placeholder condition
        else:
            return False


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

        # Add the goal termination logic
        env = GoalTerminationWrapper(env)

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
    n_checkpoints: int = 10,
):

    if n_checkpoints > n_episodes:
        n_checkpoints = n_episodes

    if n_checkpoints > 0:
        episodes_to_record = np.linspace(
            start=0, stop=n_episodes - 1, num=n_checkpoints, dtype=int
        )
    else:
        episodes_to_record = []

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

            if isinstance(act_space, gymnasium.spaces.Box) and "dqn" in algo_name:
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

            """kwargs = {
                "causal_knowledge_update_per_episode": causal_knowledge_update_per_episode
            }"""
            # ─── create policy (once) ──────────────────────────────────────────
            policy = make_policy(
                algo_name,
                act_space,
                obs_space,
                n_seeds,
                n_episodes,
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

            path_saving_policy = (
                path_simulation + f"/{algo_name}_{safe_env_name}_policy"
            )
            policy.save_policy(path_saving_policy)

        if n_checkpoints > 0:
            plot_results(json_path)  # end‑env


if __name__ == "__main__":
    # pip install gymnasium[all]

    n_seeds = 10
    n_episodes = int(1e4)
    algorithms = [
        "ppo",
        "causal_ppo",
        "dqn",
        "causal_dqn",
        "a2c",
        "causal_a2c",
        "sac",
        "causal_sac",
    ]
    max_episode_steps = int(5e2)
    n_checkpoints = int(1e2)

    run_gymnasium(
        n_seeds,
        n_episodes,
        algorithms,
        max_episode_steps,
        n_checkpoints,
    )

import json
import os
import warnings
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import pygame
import torch
from gymnasium import envs
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import generate_simulation_name, make_policy

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


def plot_results(
    json_path: str,
    percentile_band: int = 25,
    metrics=("min_reward", "max_reward", "mean_reward"),
    cmap_name: str = "tab10",
):
    """
    Render reward/step curves from a results JSON and save separate PNGs.

    Parameters
    ----------
    json_path : str
        Path to the results‑file (like the one in your example).
    percentile_band : int, default 25
        Lower percentile that defines the central band; upper is 100‑percentile_band.
    metrics : Sequence[str], default ("min_reward", "max_reward", "mean_reward")
        Reward keys to plot.
    cmap_name : str, default "tab20"
        Matplotlib colormap used to pick algorithm colours.

    Returns
    -------
    list[pathlib.Path]
        Absolute paths of the images that were written.
    """

    # ───────────────────────── utility ──────────────────────────
    def iqr_mean_std(arr: np.ndarray, q: int = 25) -> tuple[np.ndarray, np.ndarray]:

        if arr.shape[0] >= 2:
            """IQR‑based mean/std along axis‑0, with graceful NaN fallback."""
            q1 = np.percentile(arr, q, axis=0)
            q3 = np.percentile(arr, 100 - q, axis=0)
            mask = (arr >= q1) & (arr <= q3)
            filtered = np.where(mask, arr, np.nan)

            mean = np.nanmean(filtered, axis=0)
            std = np.nanstd(filtered, axis=0)

            # if an entire column was filtered out → fall back to raw stats
            nan_cols = np.isnan(mean)
            if nan_cols.any():
                mean[nan_cols] = arr[:, nan_cols].mean(axis=0)
                std[nan_cols] = arr[:, nan_cols].std(axis=0)
        else:
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
        return mean, std

    # ──────────────────────── load & prep ───────────────────────
    json_path = Path(json_path).expanduser().resolve()
    with json_path.open() as f:
        data = json.load(f)

    env_name = data["info"]["env_name"]
    episodes = np.asarray(data["info"]["recorded_episodes"])
    algorithms = data["info"]["algorithms"]

    out_dir = json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap(cmap_name)
    algo_colors = {algo: cmap(i) for i, algo in enumerate(algorithms)}

    saved_files: list[Path] = []

    # ───────────────────────── plotting ─────────────────────────
    for metric in metrics:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=500)

        for algo in algorithms:
            # ── reward curve ──
            reward_raw = np.asarray(data[algo][metric])
            r_mean, r_std = iqr_mean_std(reward_raw, percentile_band)

            axs[0].plot(episodes, r_mean, lw=2, label=algo, color=algo_colors[algo])
            axs[0].fill_between(
                episodes,
                r_mean - r_std,
                r_mean + r_std,
                alpha=0.3,
                color=algo_colors[algo],
            )

            # ── steps curve ──
            steps_raw = np.asarray(data[algo]["n_steps"])
            s_mean, s_std = iqr_mean_std(steps_raw, percentile_band)

            lower = np.clip(s_mean - s_std, a_min=1e-8, a_max=None)  # avoid log ≤ 0
            axs[1].plot(episodes, s_mean, lw=2, label=algo, color=algo_colors[algo])
            axs[1].fill_between(
                episodes, lower, s_mean + s_std, alpha=0.3, color=algo_colors[algo]
            )

        # ── cosmetics ──
        title_metric = metric.replace("_", " ").title()
        axs[0].set(
            title=f"{env_name} – {title_metric} per Episode", ylabel=title_metric
        )
        axs[1].set(
            title=f"{env_name} – Steps per Episode",
            xlabel="Episode",
            ylabel="Steps",
            yscale="log",
        )

        for ax in axs:
            ax.grid(True, which="both", lw=0.3)
            ax.tick_params(labelsize=11)

        axs[0].legend(loc="lower right", fontsize=10)
        axs[1].legend(loc="upper right", fontsize=10)

        fig.tight_layout()

        out_path = out_dir / f"{env_name}_{metric}.png"
        fig.savefig(out_path, bbox_inches="tight")

        plt.show()

        plt.close(fig)

        saved_files.append(out_path)

    return saved_files


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

    ########################################################################
    # 3.  MAIN TRAIN LOOP (vectorised – works for any env / algo)
    ########################################################################
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

            # ---------- build SyncVectorEnv with n_seeds parallel instances ----------
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

            action_space = envs.single_action_space
            observation_space = envs.single_observation_space

            metrics_dict["info"]["observation_space"] = (
                str(observation_space)
                if isinstance(observation_space, gym.spaces.Discrete)
                else "continuous"
            )
            metrics_dict["info"]["action_space"] = (
                str(action_space)
                if isinstance(action_space, gym.spaces.Discrete)
                else "continuous"
            )

            metrics_dict["info"]["algorithms"].append(algo_name)
            metrics_dict[str(algo_name)] = {
                "min_reward": np.zeros((n_seeds, n_checkpoints)),
                "max_reward": np.zeros((n_seeds, n_checkpoints)),
                "mean_reward": np.zeros((n_seeds, n_checkpoints)),
                "n_steps": np.zeros((n_seeds, n_checkpoints)),
            }

            policy = None
            pbar = tqdm(
                range(n_episodes),
                desc=(
                    f"Env: {env_name}  {env_index + 1}/{len(gymnasium_envs)} | "
                    f"Algo: {algo_name}  {algo_index + 1}/{len(algorithms_names)}"
                ),
            )

            for episode in pbar:
                obs, _ = envs.reset(seed=[s for s in range(n_seeds)])

                # build policy once (needs n_episodes for DQN ε‑schedule)
                if policy is None:
                    policy = make_policy(
                        algo_name,
                        action_space,
                        observation_space,
                        n_seeds,
                        n_episodes=n_episodes,
                    )

                if hasattr(policy, "update_episode"):  # only DQN needs it
                    policy.update_episode(episode)

                done_mask = np.zeros(n_seeds, dtype=bool)
                dones_mask_tensor = torch.zeros(
                    n_seeds, dtype=torch.bool, device=device
                )
                step_count = np.zeros(n_seeds, dtype=int)
                episodic_rewards = np.zeros(n_seeds)

                tracking_max = np.full(n_seeds, -np.inf)
                tracking_min = np.full(n_seeds, np.inf)

                # --------------- EPISODE ROLL‑OUT ----------------
                while not done_mask.all():
                    # dtype: Discrete obs → long, Box obs → float32
                    obs_tensor = (
                        torch.tensor(obs, dtype=torch.long, device=device)
                        if isinstance(observation_space, gym.spaces.Discrete)
                        else torch.tensor(obs, dtype=torch.float32, device=device)
                    )

                    # ---- choose & format actions ----
                    actions_tensor = policy.get_actions(obs_tensor)
                    actions = policy.setup_actions(
                        actions_tensor, dones_mask_tensor, "numpy"
                    )

                    # CliffWalking fix (kept from your original):
                    if "CliffWalking-v0" in env_name:
                        actions = np.clip(np.round(actions).astype(np.int32), 0, 3)

                    next_obs, rewards, terminated, truncated, _ = envs.step(actions)
                    dones = np.logical_or(terminated, truncated)
                    active_env = ~done_mask

                    episodic_rewards += rewards * active_env
                    tracking_max = np.where(
                        active_env, np.maximum(tracking_max, rewards), tracking_max
                    )
                    tracking_min = np.where(
                        active_env, np.minimum(tracking_min, rewards), tracking_min
                    )

                    # tensors for policy.update(...)
                    rewards_tensor = torch.tensor(
                        rewards, dtype=torch.float32, device=device
                    )
                    next_obs_tensor = (
                        torch.tensor(next_obs, dtype=torch.long, device=device)
                        if isinstance(observation_space, gym.spaces.Discrete)
                        else torch.tensor(next_obs, dtype=torch.float32, device=device)
                    )
                    dones_mask_tensor = torch.tensor(
                        dones, dtype=torch.bool, device=device
                    )

                    policy.update(
                        obs_tensor,
                        actions_tensor,
                        rewards_tensor,
                        next_obs_tensor,
                        dones_mask_tensor,
                    )

                    step_count += active_env
                    done_mask = np.logical_or(done_mask, dones)
                    obs = next_obs

                # ----------- end‑of‑episode bookkeeping -----------
                episodic_rewards /= np.maximum(step_count, 1)
                pbar.set_postfix(ep_rew=f"{episodic_rewards.mean():.3f}")

                # save checkpoints
                if episode in episodes_to_record:
                    e_idx = np.where(episodes_to_record == episode)[0].item()
                    m = metrics_dict[algo_name]
                    m["min_reward"][:, e_idx] = tracking_min
                    m["max_reward"][:, e_idx] = tracking_max
                    m["mean_reward"][:, e_idx] = episodic_rewards
                    m["n_steps"][:, e_idx] = step_count

                    with open(json_path, "w") as fp:
                        json.dump(convert_ndarray(metrics_dict), fp, indent=2)

            envs.close()

        plot_results(json_path)


if __name__ == "__main__":
    # pip install gymnasium[all]

    n_seeds = 2
    n_episodes = int(1e2)
    algorithms = ["a2c", "ppo"]
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

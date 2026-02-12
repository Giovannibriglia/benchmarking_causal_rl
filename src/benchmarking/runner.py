from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from tqdm import tqdm

from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.seeding import set_seed
from src.envs.wrappers.gymnasium_env import GymnasiumEnv
from src.logging.logger import CSVLogger
from src.rl.on_policy.base_actor_critic import RolloutBatch

TRAIN_COLUMNS: List[str] = [
    "episode",
    "algorithm",
    "environment",
    "train_return_mean",
    "train_return_std",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "kl",
    "critic_loss",
    "actor_loss",
    "q_loss",
]

EVAL_COLUMNS: List[str] = [
    "episode",
    "algorithm",
    "environment",
    "eval_return_mean",
    "eval_return_std",
]


@dataclass
class AlgorithmSpec:
    builder: Callable
    kind: str  # "on_policy" or "off_policy"


class BenchmarkRunner:
    def __init__(
        self,
        env_cfg: EnvConfig,
        train_cfg: TrainingConfig,
        run_cfg: RunConfig,
        algo_spec: AlgorithmSpec,
        progress_label: str | None = None,
    ):
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.run_cfg = run_cfg
        self.algo_spec = algo_spec
        self.device = torch.device(train_cfg.device)
        self.progress_label = (
            progress_label or f"{self.train_cfg.algorithm} - {self.env_cfg.env_id}"
        )
        self._env_tag = self.env_cfg.env_id.replace("/", "-")
        self.aggregation = self.train_cfg.aggregation

        self.run_dir = run_cfg.resolve_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "videos"), exist_ok=True)

        self.train_env = GymnasiumEnv(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.n_train_envs,
            device=self.device,
            seed=env_cfg.seed,
            render=False,
            record_video=False,
        )
        self.eval_env = GymnasiumEnv(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.n_eval_envs,
            device=self.device,
            seed=env_cfg.seed + env_cfg.n_train_envs,
            render=True,
            record_video=False,
        )
        if len(self.train_env.obs_space.shape) == 0:
            self.obs_dim = 1
        else:
            self.obs_dim = int(
                torch.tensor(self.train_env.obs_space.shape).prod().item()
            )
        act_space = self.train_env.act_space
        if hasattr(act_space, "n"):
            self.action_type = "discrete"
            self.action_dim = act_space.n
        else:
            self.action_type = "continuous"
            self.action_dim = act_space.shape[0]

        self.policy, self.agent = self.algo_spec.builder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_type=self.action_type,
            device=self.device,
            action_space=act_space,
        )
        self.replay_buffer = None
        self.offpolicy_batch_size = 128
        self.offpolicy_warmup = 1000
        if self.algo_spec.kind == "off_policy":
            self.replay_buffer = self.agent.buffer  # type: ignore[attr-defined]

    def _collect_on_policy(self) -> RolloutBatch:
        T = self.env_cfg.rollout_len
        N = self.env_cfg.n_train_envs
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = torch.zeros(T, N, device=self.device)
        done_buf = torch.zeros(T, N, device=self.device)
        val_buf = torch.zeros(T, N, device=self.device)
        next_val_buf = torch.zeros(T, N, device=self.device)

        obs, _ = self.train_env.reset()
        ep_returns = torch.zeros(N, device=self.device)
        for t in range(T):
            val = self.policy.value(obs).detach()
            action, logp = self.policy.act(obs)
            next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
            done = torch.logical_or(terminated, truncated).float()
            ep_returns += reward

            obs_buf.append(obs)
            act_buf.append(action.detach())
            logp_buf.append(logp.detach())
            rew_buf[t] = reward
            done_buf[t] = done
            val_buf[t] = val
            # estimate next value
            next_val_buf[t] = self.policy.value(next_obs).detach()

            obs = next_obs

        advantages, returns = self.agent.compute_gae(
            rew_buf, done_buf, val_buf, next_val_buf
        )
        batch = RolloutBatch(
            obs=torch.stack(obs_buf).reshape(T * N, -1),
            actions=(
                torch.stack(act_buf).reshape(T * N, -1)
                if act_buf[0].ndim > 1
                else torch.stack(act_buf).reshape(T * N)
            ),
            log_probs=torch.stack(logp_buf).reshape(T * N),
            rewards=rew_buf.reshape(T * N),
            dones=done_buf.reshape(T * N),
            values=val_buf.reshape(T * N),
            next_values=next_val_buf.reshape(T * N),
            advantages=advantages.reshape(T * N),
            returns=returns.reshape(T * N),
        )
        train_return_mean, train_return_std = self._aggregate_returns(ep_returns)
        return batch, train_return_mean, train_return_std

    def _train_on_policy(self, train_logger: CSVLogger, eval_logger: CSVLogger) -> None:
        checkpoint_eps = self.train_cfg.checkpoint_episodes()
        for ep in tqdm(range(self.train_cfg.n_episodes), desc=self.progress_label):
            batch, train_ret_mean, train_ret_std = self._collect_on_policy()
            metrics = self.agent.update(batch)

            if ep in checkpoint_eps:
                self._save_checkpoint(ep)
                eval_metrics = self.evaluate(ep)
                eval_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                train_metrics = metrics.copy()
                train_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                        "train_return_mean": train_ret_mean,
                        "train_return_std": train_ret_std,
                    }
                )
                eval_row = self._make_row(EVAL_COLUMNS, eval_metrics, ep)
                train_row = self._make_row(TRAIN_COLUMNS, train_metrics, ep)
                eval_logger.log(eval_row)
                train_logger.log(train_row)

    def _save_checkpoint(self, episode: int):
        ckpt_dir = os.path.join(
            self.run_dir,
            "checkpoints",
            f"{self._env_tag}_{self.train_cfg.algorithm}_seed{self.env_cfg.seed}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"ckpt_ep{episode:04d}.pt")
        from .checkpoints import save_checkpoint

        save_checkpoint(
            path,
            {
                "episode": episode,
                "policy_state": self.policy.state_dict(),
                "agent_state": getattr(self.agent, "state_dict", lambda: {})(),
                "config": {
                    "env": self.env_cfg.__dict__,
                    "training": self.train_cfg.__dict__,
                },
            },
        )

    def evaluate(self, episode: int) -> Dict[str, float]:
        env = self.eval_env
        video_path = os.path.join(
            self.run_dir,
            "videos",
            f"{self._env_tag}_{self.train_cfg.algorithm}_seed{self.env_cfg.seed}_ckpt{episode:04d}.mp4",
        )
        env.start_video(video_path)
        obs, _ = env.reset()
        total_rewards = torch.zeros(env.n_envs, device=self.device)
        for _ in range(self.env_cfg.rollout_len):
            with torch.no_grad():
                if self.algo_spec.kind == "off_policy":
                    if self.action_type == "discrete":
                        action = self.agent.act(obs, epsilon=0.0)  # type: ignore[arg-type]
                    else:
                        action = self.agent.act(obs, noise=False)  # type: ignore[arg-type]
                else:
                    if hasattr(self.policy, "act_deterministic"):
                        action = self.policy.act_deterministic(obs)
                    else:
                        action, _ = self.policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = torch.logical_or(terminated, truncated)
            total_rewards += reward * (~done)
        env.stop_video()
        mean, std = self._aggregate_returns(total_rewards)
        return {"eval_return_mean": mean, "eval_return_std": std}

    def _aggregate_returns(self, returns: torch.Tensor) -> Tuple[float, float]:
        if returns.numel() == 0:
            return 0.0, 0.0
        if self.aggregation == "mean":
            mean = returns.mean().item()
            std = returns.std(unbiased=False).item()
            return mean, std
        # IQM: mean of middle 50%; IQR-STD: std of middle 50%
        sorted_vals = torch.sort(returns.flatten()).values
        n = sorted_vals.numel()
        lower = int(0.25 * (n - 1))
        upper = int(0.75 * (n - 1))
        mid = sorted_vals[lower : upper + 1]
        mean = mid.mean().item()
        std = mid.std(unbiased=False).item()
        return mean, std

    @staticmethod
    def _make_row(
        schema: list[str], metrics: Dict[str, float] | None, episode: int
    ) -> Dict[str, float]:
        metrics = metrics or {}
        base = {
            "episode": episode,
        }
        full = {**metrics, **base}
        return {col: full.get(col, "") for col in schema}

    def run(self) -> None:
        set_seed(self.env_cfg.seed, deterministic=self.train_cfg.deterministic)
        with CSVLogger(
            os.path.join(self.run_dir, "train_metrics.csv"), fieldnames=TRAIN_COLUMNS
        ) as train_log, CSVLogger(
            os.path.join(self.run_dir, "eval_metrics.csv"), fieldnames=EVAL_COLUMNS
        ) as eval_log:
            if self.algo_spec.kind == "on_policy":
                self._train_on_policy(train_log, eval_log)
            else:
                self._train_off_policy(train_log, eval_log)

    def _train_off_policy(
        self, train_logger: CSVLogger, eval_logger: CSVLogger
    ) -> None:
        assert self.replay_buffer is not None
        checkpoint_eps = self.train_cfg.checkpoint_episodes()
        obs, _ = self.train_env.reset()
        total_steps = self.env_cfg.rollout_len
        metrics_cache = None
        for ep in tqdm(
            range(self.train_cfg.n_episodes),
            desc=self.progress_label,
        ):
            for _ in range(total_steps):
                if self.action_type == "discrete":
                    actions = self.agent.act(obs)  # type: ignore[attr-defined]
                else:
                    actions = self.agent.act(obs)  # type: ignore[attr-defined]
                next_obs, reward, terminated, truncated, _ = self.train_env.step(
                    actions
                )
                done = torch.logical_or(terminated, truncated).float()
                # store each env transition separately
                for i in range(self.env_cfg.n_train_envs):
                    self.replay_buffer.add(
                        {
                            "obs": obs[i].detach(),
                            "actions": actions[i].detach(),
                            "rewards": reward[i].detach(),
                            "next_obs": next_obs[i].detach(),
                            "dones": done[i].detach(),
                        }
                    )
                obs = next_obs

                if len(self.replay_buffer) > max(
                    self.offpolicy_warmup, self.offpolicy_batch_size
                ):
                    batch = self.replay_buffer.sample(self.offpolicy_batch_size)
                    metrics = self.agent.update(batch)
                    metrics_cache = metrics

            if ep in checkpoint_eps:
                self._save_checkpoint(ep)
                eval_metrics = self.evaluate(ep)
                eval_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                eval_row = self._make_row(EVAL_COLUMNS, eval_metrics, ep)
                train_metrics = (metrics_cache or {}).copy()
                train_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                train_row = self._make_row(TRAIN_COLUMNS, train_metrics, ep)
                train_logger.log(train_row)
                eval_logger.log(eval_row)

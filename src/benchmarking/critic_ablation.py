from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.nets.mlp import MLP
from src.rl.on_policy.base_actor_critic import RolloutBatch

CRITIC_ABLATION_COLUMNS: list[str] = [
    "episode",
    "algorithm",
    "environment",
    "critic",
    "train_loss",
    "advantage_mean",
    "explained_variance",
    "pearson",
    "spearman",
    "mutual_information",
    "kl",
    "js_normalized",
    "mse",
    "td_error_mean",
    "real_reward_mean",
    "pred_reward_mean",
    "reward_explained_variance",
    "reward_pearson",
    "reward_spearman",
    "reward_mutual_information",
    "reward_kl",
    "reward_js_normalized",
    "reward_mse",
    "reward_error_mean",
]

_EPS = 1e-8


@dataclass(frozen=True)
class CriticSpec:
    target: str  # returns | td0
    loss: str  # mse | huber
    architecture: str = "mlp"  # mlp | residual_mlp


CRITIC_LIBRARY: Dict[str, CriticSpec] = {
    # Default baseline critic used when no critic list is provided.
    "standard_mlp": CriticSpec(target="returns", loss="mse", architecture="mlp"),
    # Custom example critic with residual connection; useful for ablation comparisons.
    "residual_reward_model": CriticSpec(
        target="returns", loss="mse", architecture="residual_mlp"
    ),
}


def default_aux_critics() -> list[str]:
    return ["standard_mlp"]


DEFAULT_AUX_CRITICS: list[str] = default_aux_critics()


class ResidualValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h) + self.skip(x))
        return self.out(h)


@dataclass
class CriticAblationConfig:
    critics: list[str]
    lr: float = 3e-4
    hidden_dims: tuple[int, ...] = (64, 64)
    bins: int = 32

    def to_dict(self) -> dict:
        return {
            "critics": list(self.critics),
            "lr": float(self.lr),
            "hidden_dims": list(self.hidden_dims),
            "bins": int(self.bins),
        }


class AuxiliaryCritic:
    def __init__(
        self,
        name: str,
        spec: CriticSpec,
        obs_dim: int,
        device: torch.device,
        lr: float,
        hidden_dims: tuple[int, ...],
        gamma: float,
    ) -> None:
        self.name = name
        self.spec = spec
        self.device = device
        self.gamma = gamma
        self.net = self._build_network(obs_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _build_network(
        self, obs_dim: int, hidden_dims: tuple[int, ...]
    ) -> torch.nn.Module:
        if self.spec.architecture == "residual_mlp":
            hidden_dim = int(hidden_dims[-1]) if hidden_dims else 64
            return ResidualValueNet(obs_dim, hidden_dim)
        return MLP(obs_dim, 1, hidden_dims=hidden_dims)

    def _training_target(self, batch: RolloutBatch) -> torch.Tensor:
        if self.spec.target == "returns":
            return batch.returns
        with torch.no_grad():
            next_values = self.net(batch.next_obs).squeeze(-1)
        return batch.rewards + self.gamma * next_values * (1.0 - batch.dones)

    def update(self, batch: RolloutBatch) -> float:
        pred = self.net(batch.obs).squeeze(-1)
        target = self._training_target(batch)
        if self.spec.loss == "huber":
            loss = F.smooth_l1_loss(pred, target)
        else:
            loss = F.mse_loss(pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(obs).squeeze(-1)

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "spec": {
                "target": self.spec.target,
                "loss": self.spec.loss,
                "architecture": self.spec.architecture,
            },
            "network": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }


class CriticAblationManager:
    def __init__(
        self,
        obs_dim: int,
        device: torch.device,
        config: CriticAblationConfig,
        gamma: float = 0.99,
    ) -> None:
        self.config = config
        self.gamma = gamma
        self.device = device
        selected_critics = (
            [str(name) for name in config.critics]
            if config.critics
            else default_aux_critics()
        )
        self.config.critics = list(selected_critics)

        critics: Dict[str, AuxiliaryCritic] = {}
        for name in selected_critics:
            key = name.lower().strip()
            if key not in CRITIC_LIBRARY:
                available = ", ".join(sorted(CRITIC_LIBRARY))
                raise ValueError(
                    f"Unknown ablation critic '{name}'. Available: {available}"
                )
            if key in critics:
                continue
            critics[key] = AuxiliaryCritic(
                name=key,
                spec=CRITIC_LIBRARY[key],
                obs_dim=obs_dim,
                device=device,
                lr=config.lr,
                hidden_dims=config.hidden_dims,
                gamma=gamma,
            )
        if not critics:
            raise ValueError("At least one ablation critic must be configured.")
        self.critics = critics

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        losses: Dict[str, float] = {}
        for name, critic in self.critics.items():
            losses[name] = critic.update(batch)
        return losses

    def checkpoint_rows(
        self,
        batch: RolloutBatch,
        episode: int,
        algorithm: str,
        environment: str,
        latest_losses: Dict[str, float] | None = None,
    ) -> list[dict]:
        losses = latest_losses or {}
        rows: list[dict] = []
        for name, critic in self.critics.items():
            metrics = _compute_metrics(
                batch=batch,
                critic=critic,
                gamma=self.gamma,
                bins=max(4, int(self.config.bins)),
            )
            rows.append(
                {
                    "episode": episode,
                    "algorithm": algorithm,
                    "environment": environment,
                    "critic": name,
                    "train_loss": losses.get(name, ""),
                    **metrics,
                }
            )
        return rows

    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "gamma": float(self.gamma),
            "critics": {
                name: critic.state_dict() for name, critic in self.critics.items()
            },
        }


def _to_vector(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).float().cpu()


def _explained_variance(target: torch.Tensor, pred: torch.Tensor) -> float:
    var_target = torch.var(target, unbiased=False)
    if var_target.item() < _EPS:
        return 0.0
    return float(
        (1.0 - torch.var(target - pred, unbiased=False) / (var_target + _EPS)).item()
    )


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum()))
    if denom.item() < _EPS:
        return 0.0
    return float((x_centered * y_centered).sum().div(denom + _EPS).item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearson(_rank(x), _rank(y))


def _distribution_metrics(
    target: torch.Tensor, pred: torch.Tensor, bins: int
) -> tuple[float, float, float]:
    combined = torch.cat([target, pred], dim=0)
    low = float(combined.min().item())
    high = float(combined.max().item())
    if abs(high - low) < _EPS:
        high = low + 1e-3
    edges = torch.linspace(low, high, bins + 1)

    target_idx = torch.bucketize(target, edges[1:-1])
    pred_idx = torch.bucketize(pred, edges[1:-1])

    target_counts = torch.bincount(target_idx, minlength=bins).float()
    pred_counts = torch.bincount(pred_idx, minlength=bins).float()
    p = (target_counts + _EPS) / (target_counts.sum() + _EPS * bins)
    q = (pred_counts + _EPS) / (pred_counts.sum() + _EPS * bins)

    kl = float((p * torch.log((p + _EPS) / (q + _EPS))).sum().item())
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * torch.log((p + _EPS) / (m + _EPS))).sum()
        + (q * torch.log((q + _EPS) / (m + _EPS))).sum()
    )
    js_normalized = float((js / torch.log(torch.tensor(2.0))).item())

    joint_idx = target_idx * bins + pred_idx
    joint_counts = torch.bincount(joint_idx, minlength=bins * bins).float()
    joint = (joint_counts.reshape(bins, bins) + _EPS) / (
        joint_counts.sum() + _EPS * bins * bins
    )
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    mutual_information = float(
        (joint * torch.log((joint + _EPS) / (px @ py + _EPS))).sum().item()
    )
    return mutual_information, kl, js_normalized


def _compute_metrics(
    batch: RolloutBatch,
    critic: AuxiliaryCritic,
    gamma: float,
    bins: int,
) -> dict:
    pred_obs = critic.predict(batch.obs)
    pred = _to_vector(pred_obs)
    target = _to_vector(batch.returns)

    advantage = target - pred
    mse = float(torch.mean((target - pred) ** 2).item())
    explained_variance = _explained_variance(target, pred)
    pearson = _pearson(target, pred)
    spearman = _spearman(target, pred)
    mutual_information, kl, js_normalized = _distribution_metrics(
        target, pred, bins=bins
    )

    value = _to_vector(pred_obs)
    next_value = _to_vector(critic.predict(batch.next_obs))
    rewards = _to_vector(batch.rewards)
    dones = _to_vector(batch.dones)
    pred_reward = value - gamma * next_value * (1.0 - dones)
    td_error = rewards - pred_reward
    td_error_mean = float(td_error.mean().item())
    reward_mse = float(torch.mean((rewards - pred_reward) ** 2).item())
    reward_explained_variance = _explained_variance(rewards, pred_reward)
    reward_pearson = _pearson(rewards, pred_reward)
    reward_spearman = _spearman(rewards, pred_reward)
    reward_mutual_information, reward_kl, reward_js_normalized = _distribution_metrics(
        rewards, pred_reward, bins=bins
    )

    return {
        "advantage_mean": float(advantage.mean().item()),
        "explained_variance": explained_variance,
        "pearson": pearson,
        "spearman": spearman,
        "mutual_information": mutual_information,
        "kl": kl,
        "js_normalized": js_normalized,
        "mse": mse,
        "td_error_mean": td_error_mean,
        "real_reward_mean": float(rewards.mean().item()),
        "pred_reward_mean": float(pred_reward.mean().item()),
        "reward_explained_variance": reward_explained_variance,
        "reward_pearson": reward_pearson,
        "reward_spearman": reward_spearman,
        "reward_mutual_information": reward_mutual_information,
        "reward_kl": reward_kl,
        "reward_js_normalized": reward_js_normalized,
        "reward_mse": reward_mse,
        "reward_error_mean": td_error_mean,
    }

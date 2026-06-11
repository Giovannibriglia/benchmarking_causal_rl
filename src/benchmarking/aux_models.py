from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F

from src.rl.models.reward_model import RewardModel
from src.rl.models.transition_model import TransitionModel

AUX_METRICS_COLUMNS: list[str] = [
    "episode",
    "algorithm",
    "environment",
    "model",
    "train_loss",
    "mse",
    "mae",
]


@dataclass
class AuxModelConfig:
    lr: float = 3e-4
    hidden_dims: tuple[int, ...] = (64, 64)

    def to_dict(self) -> dict:
        return {"lr": float(self.lr), "hidden_dims": list(self.hidden_dims)}


def _field(batch: Any, name: str) -> torch.Tensor:
    """Read a field from either a RolloutBatch (attrs) or a sampled dict (keys)."""
    if hasattr(batch, name):
        return getattr(batch, name)
    return batch[name]


class AuxModelManager:
    """Auxiliary reward r(s,a) and next-state s'(s,a) models trained alongside RL.

    Separate modules with their own optimizers — NOT folded into the RL loss.
    Trained on the RL update's already-sampled batch (no new env steps, no new
    sampling); losses logged to a separate ``aux_metrics.csv``.

    RNG isolation: the models' construction draws from the global torch RNG
    (weight init). To keep an aux-enabled run bitwise-identical to an aux-off
    run, the global CPU and CUDA RNG states are snapshotted before construction
    and restored after, so construction has zero net effect on the stream. The
    MLP forward/backward + Adam step draw nothing from the global RNG, so
    training is RNG-neutral too.
    """

    def __init__(
        self,
        obs_dim: int,
        obs_shape: Sequence[int],
        action_dim: int,
        action_type: str,
        device: torch.device,
        config: AuxModelConfig,
    ) -> None:
        self.config = config
        self.device = device

        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            self.reward_model = RewardModel(
                obs_dim, action_dim, action_type, obs_shape, config.hidden_dims
            ).to(device)
            self.transition_model = TransitionModel(
                obs_dim, action_dim, action_type, obs_shape, config.hidden_dims
            ).to(device)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

        self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=config.lr)
        self.transition_opt = torch.optim.Adam(
            self.transition_model.parameters(), lr=config.lr
        )
        # Last training losses, so checkpoint logging needs no extra batch
        # (re-sampling one would perturb the off-policy buffer RNG / golden).
        self.last_losses: Dict[str, float] = {}

    def update(self, batch: Any) -> Dict[str, float]:
        obs = _field(batch, "obs")
        actions = _field(batch, "actions")
        rewards = _field(batch, "rewards").float()
        next_obs = _field(batch, "next_obs").float()

        r_pred = self.reward_model(obs, actions)
        r_loss = F.mse_loss(r_pred, rewards)
        self.reward_opt.zero_grad(set_to_none=True)
        r_loss.backward()
        self.reward_opt.step()

        s_pred = self.transition_model(obs, actions)
        t_loss = F.mse_loss(s_pred, next_obs)
        self.transition_opt.zero_grad(set_to_none=True)
        t_loss.backward()
        self.transition_opt.step()

        self.last_losses = {
            "reward": float(r_loss.item()),
            "transition": float(t_loss.item()),
        }
        return dict(self.last_losses)

    def checkpoint_rows(
        self,
        batch: Any,
        episode: int,
        algorithm: str,
        environment: str,
        latest_losses: Dict[str, float] | None = None,
    ) -> list[dict]:
        losses = latest_losses if latest_losses is not None else self.last_losses
        obs = _field(batch, "obs")
        actions = _field(batch, "actions")
        rewards = _field(batch, "rewards").float()
        next_obs = _field(batch, "next_obs").float()

        rows: list[dict] = []
        with torch.no_grad():
            r_pred = self.reward_model(obs, actions)
            s_pred = self.transition_model(obs, actions)
            specs = [
                ("reward", r_pred, rewards),
                ("transition", s_pred, next_obs),
            ]
            for name, pred, target in specs:
                rows.append(
                    {
                        "episode": episode,
                        "algorithm": algorithm,
                        "environment": environment,
                        "model": name,
                        "train_loss": losses.get(name, ""),
                        "mse": float(F.mse_loss(pred, target).item()),
                        "mae": float(F.l1_loss(pred, target).item()),
                    }
                )
        return rows

    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "reward_model": self.reward_model.state_dict(),
            "transition_model": self.transition_model.state_dict(),
        }

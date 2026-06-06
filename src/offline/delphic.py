"""Delphic-style pessimistic offline RL (Cells 7–8 variant; Pace et al. 2024).

Simplified-but-faithful implementation: an ensemble of reward models trained
on bootstrap resamples of the logged data spans hypotheses compatible with
the observations; their DISAGREEMENT proxies delphic (confounding-induced)
uncertainty — under hidden confounding the data cannot pin down r(s, a), and
the ensemble spreads exactly where U distorted the logged rewards. Learning
is fitted Q-iteration on the pessimistic reward

    r_tilde(s, a) = mean_k r_k(s, a) − kappa · std_k r_k(s, a).

Everything is estimated from the dataset alone — no oracle quantities, no
latent U (§8; contrast with the recovered ``confounded_dqn``, which read the
env oracle inside the learner and was discarded for that reason).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.data.experience_source import OfflineDatasetSource
from src.rl.base import ActionOutput, Algorithm
from src.rl.nets.mlp import MLP


class DelphicOfflineDQN(Algorithm):
    paradigm = "offline"
    action_type = "discrete"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: torch.device,
        n_ensemble: int = 5,
        kappa: float = 1.0,
        penalty_cap: float = 1.0,
        gamma: float = 0.99,
        lr: float = 3e-4,
        reward_model_steps: int = 2000,
        target_sync: int = 200,
        hidden_dims=(64, 64),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.n_ensemble = int(n_ensemble)
        self.kappa = float(kappa)
        # Sensitivity budget (Kallus-Zhou flavored): the analyst's assumed
        # bound on per-step reward distortion by hidden confounding. Caps the
        # pessimism so total epistemic uncertainty (e.g. under masking, Cell 8)
        # cannot drown the reward signal: r_tilde = mean - min(kappa*std, cap).
        self.penalty_cap = float(penalty_cap) if penalty_cap is not None else None
        self.gamma = float(gamma)
        self.reward_model_steps = int(reward_model_steps)
        self.target_sync = int(target_sync)
        self.seed = int(seed)
        torch.manual_seed(seed)
        self.q_net = MLP(obs_dim, n_actions, hidden_dims=hidden_dims).to(device)
        self.q_target = MLP(obs_dim, n_actions, hidden_dims=hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.reward_models: List[MLP] = []

    # ------------------------------------------------------------------ act
    def act(
        self,
        obs: torch.Tensor,
        state: Optional[Any] = None,
        *,
        deterministic: bool = False,
    ) -> ActionOutput:
        _ = deterministic  # greedy by construction
        with torch.no_grad():
            q = self.q_net(obs.float().to(self.device))
        return ActionOutput(action=q.argmax(dim=-1), state=state)

    # ------------------------------------------------- delphic reward model
    def _fit_reward_ensemble(self, source: OfflineDatasetSource) -> None:
        n = len(source)
        self.reward_models = []
        for k in range(self.n_ensemble):
            g = torch.Generator().manual_seed(self.seed * 97 + k)
            torch.manual_seed(self.seed * 31 + k)
            net = MLP(self.obs_dim, self.n_actions).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            # bootstrap resample: each member sees a different draw of the data
            boot = torch.randint(0, n, (n,), generator=g).to(source.device)
            for _ in range(self.reward_model_steps):
                idx = boot[torch.randint(0, n, (256,), generator=g)]
                obs = source.obs[idx].float()
                pred = (
                    net(obs)
                    .gather(1, source.actions[idx].long().unsqueeze(-1))
                    .squeeze(-1)
                )
                loss = F.mse_loss(pred, source.rewards[idx].float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            net.eval()
            self.reward_models.append(net)

    def pessimistic_rewards(self, obs: torch.Tensor, actions: torch.Tensor):
        """r_tilde = mean − kappa·std over the ensemble; also returns the
        per-transition delphic uncertainty (std)."""
        with torch.no_grad():
            preds = torch.stack(
                [
                    m(obs.float()).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
                    for m in self.reward_models
                ]
            )
        mean, std = preds.mean(dim=0), preds.std(dim=0)
        penalty = self.kappa * std
        if self.penalty_cap is not None:
            penalty = penalty.clamp(max=self.penalty_cap)
        return mean - penalty, std

    # ---------------------------------------------------------------- learn
    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].float()
        q = self.q_net(obs).gather(1, batch["actions"].long().unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.q_target(batch["next_obs"].float()).max(dim=-1).values
            tgt = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * next_q
        loss = F.mse_loss(q, tgt)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def fit_source(
        self,
        source: OfflineDatasetSource,
        n_steps: int,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        self._fit_reward_ensemble(source)
        r_tilde, delphic_std = self.pessimistic_rewards(source.obs, source.actions)
        # FQI on the pessimism-adjusted dataset view
        pess_episodes = []
        start = 0
        for ep in source.episodes:
            T = int(ep["rewards"].shape[0])
            pess = dict(ep)
            pess["rewards"] = r_tilde[start : start + T]
            pess_episodes.append(pess)
            start += T
        pess_source = OfflineDatasetSource(
            pess_episodes,
            source.device,
            behavior_policy="unknown",  # propensities not needed for FQI
            rng_seed=self.seed,
        )
        metrics: Dict[str, float] = {}
        for it in range(int(n_steps)):
            metrics = self.update(pess_source.sample(batch_size), pess_source)
            if (it + 1) % self.target_sync == 0:
                self.q_target.load_state_dict(self.q_net.state_dict())
        metrics["delphic_std_mean"] = float(delphic_std.mean().item())
        return metrics

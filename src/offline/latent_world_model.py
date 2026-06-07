"""Latent-variable world model for Cell 6 ("Fog of History").

Cell 6 = offline, pi_b unknown, Z hidden: statistically unidentifiable in
general. The canonical literature response is a latent world model: a
recurrent encoder compresses the action-observation history into a belief
state trained to predict next observations and rewards; the policy is then
learned on the belief. EXPECTATION (§5, binding): MODEST recovery — full
recovery would be a red flag given non-identifiability, not a success.

Implementation: GRU(obs ⊕ onehot(prev_action)) → h_t with three heads
(next-obs, reward, action/BC), trained jointly on logged episodes; acting
is greedy on the BC head over the recurrent belief.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.experience_source import OfflineDatasetSource
from src.rl.base import ActionOutput, Algorithm


class LatentWorldModelBC(Algorithm):
    paradigm = "offline"
    action_type = "discrete"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: torch.device,
        latent_dim: int = 64,
        lr: float = 1e-3,
        wm_coef: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.latent_dim = int(latent_dim)
        self.wm_coef = float(wm_coef)
        torch.manual_seed(seed)
        in_dim = obs_dim + n_actions
        self.gru = nn.GRU(in_dim, latent_dim, batch_first=True).to(device)
        self.next_obs_head = nn.Linear(latent_dim, obs_dim).to(device)
        self.reward_head = nn.Linear(latent_dim, 1).to(device)
        self.action_head = nn.Linear(latent_dim, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ----------------------------------------------------------- sequences
    def _episode_inputs(self, ep) -> tuple:
        T = int(ep["rewards"].shape[0])
        obs = ep["obs"][: T + 1].float().to(self.device)
        prev_a = torch.zeros(T, self.n_actions, device=self.device)
        if T > 1:
            prev_a[1:] = F.one_hot(
                ep["actions"][:-1].long(), num_classes=self.n_actions
            ).float()
        x = torch.cat([obs[:T], prev_a], dim=-1)  # [T, in_dim]
        return (
            x,
            obs[1 : T + 1],
            ep["rewards"].float().to(self.device),
            ep["actions"].long().to(self.device),
        )

    def learn(self, batch: Dict[str, Any]) -> Dict[str, float]:
        x = batch["x"]  # [B, T, in_dim] padded
        mask = batch["mask"]  # [B, T]
        h, _ = self.gru(x)
        m = mask.unsqueeze(-1)
        next_obs_loss = (
            (self.next_obs_head(h) - batch["next_obs"]) ** 2 * m
        ).sum() / m.sum()
        rew_loss = (
            (self.reward_head(h).squeeze(-1) - batch["rewards"]) ** 2 * mask
        ).sum() / mask.sum()
        logits = self.action_head(h)
        bc_loss = (
            F.cross_entropy(
                logits.reshape(-1, self.n_actions),
                batch["actions"].reshape(-1),
                reduction="none",
            ).reshape_as(mask)
            * mask
        ).sum() / mask.sum()
        loss = self.wm_coef * (next_obs_loss + rew_loss) + bc_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "wm_next_obs": float(next_obs_loss.item()),
            "wm_reward": float(rew_loss.item()),
            "bc": float(bc_loss.item()),
        }

    def fit_source(
        self,
        source: OfflineDatasetSource,
        n_steps: int,
        batch_size: int = 256,
        on_step=None,
        on_step_every: int = 0,
    ) -> Dict[str, float]:
        # batch_size is transition-scale (dispatcher convention); sequence
        # batches use whole episodes, so convert with a safe cap.
        episodes_per_batch = max(4, min(16, int(batch_size) // 16))
        seqs = [self._episode_inputs(ep) for ep in source.episodes]
        g = torch.Generator().manual_seed(0)
        metrics: Dict[str, float] = {}
        for it in range(int(n_steps)):
            idx = torch.randint(0, len(seqs), (episodes_per_batch,), generator=g)
            chosen = [seqs[i] for i in idx.tolist()]
            T_max = max(c[0].shape[0] for c in chosen)
            B = len(chosen)
            x = torch.zeros(B, T_max, self.obs_dim + self.n_actions, device=self.device)
            next_obs = torch.zeros(B, T_max, self.obs_dim, device=self.device)
            rewards = torch.zeros(B, T_max, device=self.device)
            actions = torch.zeros(B, T_max, dtype=torch.long, device=self.device)
            mask = torch.zeros(B, T_max, device=self.device)
            for b, (xb, nb, rb, ab) in enumerate(chosen):
                T = xb.shape[0]
                x[b, :T] = xb
                next_obs[b, :T] = nb
                rewards[b, :T] = rb
                actions[b, :T] = ab
                mask[b, :T] = 1.0
            metrics = self.update(
                {
                    "x": x,
                    "next_obs": next_obs,
                    "rewards": rewards,
                    "actions": actions,
                    "mask": mask,
                },
                source,
            )
            if on_step and on_step_every and (it + 1) % on_step_every == 0:
                on_step(it + 1)
        return metrics

    # ------------------------------------------------------------------ act
    def act(
        self,
        obs: torch.Tensor,
        state: Optional[Any] = None,
        *,
        deterministic: bool = False,
    ) -> ActionOutput:
        """``state`` = (gru_hidden, prev_action_onehot); None = episode start."""
        _ = deterministic
        obs = obs.float().to(self.device)
        B = obs.shape[0]
        if state is None:
            h = torch.zeros(1, B, self.latent_dim, device=self.device)
            prev_a = torch.zeros(B, self.n_actions, device=self.device)
        else:
            h, prev_a = state
        x = torch.cat([obs, prev_a], dim=-1).unsqueeze(1)  # [B, 1, in_dim]
        with torch.no_grad():
            out, h_next = self.gru(x, h)
            action = self.action_head(out.squeeze(1)).argmax(dim=-1)
        next_state = (h_next, F.one_hot(action, num_classes=self.n_actions).float())
        return ActionOutput(action=action, state=next_state)

    def make_eval_act_fn(self, device: torch.device):
        agent = self

        class _ActFn:
            def __init__(self) -> None:
                self.state = None

            def reset(self) -> None:
                self.state = None

            def __call__(self, obs: np.ndarray) -> int:
                t = torch.as_tensor(
                    obs.reshape(1, -1), dtype=torch.float32, device=device
                )
                out = agent.act(t, state=self.state)
                self.state = out.state
                return int(out.action.item())

        return _ActFn()

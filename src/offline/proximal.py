"""Proxy-variable learner for Cell 4 (Tennenholtz et al. 2020-style).

Cell 4 = offline, pi_b known, Z hidden. The proximal-OPE insight is that
PAST and FUTURE observations serve as proxies (negative controls) for the
hidden state. On velocity-masked control tasks the natural proxies are
ADJACENT-STEP observations: the previous masked observation and previous
action carry exactly the finite-difference information that identifies the
hidden velocities.

HONEST FRAMING (Phase-5 gate): velocity masking is the history-RECOVERABLE
regime (docs/causal_cells.md), so this adaptation reduces the proximal idea
to proxy-augmented imitation — its success demonstrates proxy-based
restoration where proxies are informative, not the general proximal
identification machinery (spectral/bridge-function estimation on genuinely
latent state is future work alongside the epistemic Cell-2 variant).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.experience_source import OfflineDatasetSource
from src.rl.base import ActionOutput, Algorithm
from src.rl.nets.mlp import MLP


class ProximalBC(Algorithm):
    """Imitation on proxy-augmented features [o_t, o_{t-1}, onehot(a_{t-1})].

    At episode starts the proxies are initialized to (o_0, zero-action),
    matching the evaluation-time history buffer.
    """

    paradigm = "offline"
    action_type = "discrete"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: torch.device,
        lr: float = 1e-3,
        hidden_dims=(64, 64),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        torch.manual_seed(seed)
        in_dim = 2 * obs_dim + n_actions
        self.net = MLP(in_dim, n_actions, hidden_dims=hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ----------------------------------------------------------- features
    def _augment_episode(self, ep) -> torch.Tensor:
        T = int(ep["rewards"].shape[0])
        obs = ep["obs"][:T].float()
        prev_obs = torch.cat([obs[:1], obs[:-1]], dim=0)
        prev_a = torch.zeros(T, self.n_actions, device=obs.device)
        if T > 1:
            prev_a[1:] = F.one_hot(
                ep["actions"][:-1].long(), num_classes=self.n_actions
            ).float()
        return torch.cat([obs, prev_obs, prev_a], dim=-1)

    # --------------------------------------------------------------- learn
    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        logits = self.net(batch["features"])
        loss = F.cross_entropy(logits, batch["actions"].long())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def fit_source(
        self,
        source: OfflineDatasetSource,
        n_steps: int,
        batch_size: int = 256,
        on_step=None,
        on_step_every: int = 0,
    ) -> Dict[str, float]:
        feats = torch.cat([self._augment_episode(ep) for ep in source.episodes])
        actions = source.actions
        g = torch.Generator().manual_seed(0)
        metrics: Dict[str, float] = {}
        for it in range(int(n_steps)):
            idx = torch.randint(0, len(actions), (batch_size,), generator=g).to(
                feats.device
            )
            metrics = self.update(
                {"features": feats[idx], "actions": actions[idx]}, source
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
        """``state`` carries (prev_obs, prev_action); None = episode start."""
        _ = deterministic
        obs = obs.float().to(self.device)
        if state is None:
            prev_obs = obs
            prev_a = torch.zeros(obs.shape[0], self.n_actions, device=self.device)
        else:
            prev_obs, prev_a = state
        feats = torch.cat([obs, prev_obs, prev_a], dim=-1)
        with torch.no_grad():
            action = self.net(feats).argmax(dim=-1)
        next_state = (
            obs,
            F.one_hot(action, num_classes=self.n_actions).float(),
        )
        return ActionOutput(action=action, state=next_state)

    def make_eval_act_fn(self, device: torch.device):
        """Stateful numpy act-fn with per-episode reset (evaluate_policy hook)."""
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

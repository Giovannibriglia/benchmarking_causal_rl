"""Conservative Q-Learning for continuous action spaces (CQL-on-SAC).

CQL(H), Kumar et al. 2020 (arXiv:2006.04779), continuous form. A SAC base
(squashed-Gaussian actor, twin Q critics, auto-tuned temperature) plus the
conservative critic penalty. The discrete closed-form ``logsumexp_a Q(s,a)`` is
unavailable for continuous actions, so it is importance-sampled:

    logsumexp_a Q(s,a)  ~=  logmeanexp over { Q(s,a_i) - log mu(a_i) }

with ``a_i`` drawn half from the current policy (``mu = pi``, log-prob from the
actor) and half uniform on ``[-1, 1]^d`` (``log mu = -d*log2``). The penalty
pushes those sampled-action values down and the dataset-action value up:

    penalty = E_s[ logsumexp_a Q(s,a) ] - E_(s,a)~D[ Q(s,a_data) ]

added (x ``cql_alpha``) to BOTH critic losses. Unlike the discrete exact
logsumexp (which is provably >= 0), the importance-sampled estimator is NOT
sign-definite; the correctness bar is "present, finite, and gated by cql_alpha".

This is a STANDALONE ``BaseOffPolicy`` that reuses ``SquashedGaussianActor`` and
the twin-Q ``select_backbone`` pattern but never edits ``sac.py`` — so SAC's
bitwise determinism test is protected by construction.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.base import ActionOutput
from src.rl.off_policy.base_off_policy import BaseOffPolicy
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.off_policy.sac import SquashedGaussianActor


class ContinuousCQL(BaseOffPolicy):
    """CQL(H) on a SAC backbone, continuous actions. Q-centric + entropy actor."""

    action_type = "continuous"

    def __init__(
        self,
        actor: SquashedGaussianActor,
        q1: nn.Module,
        q2: nn.Module,
        q1_target: nn.Module,
        q2_target: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        action_scale: float = 1.0,
        cql_alpha: float = 5.0,
        num_sampled_actions: int = 10,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.actor = actor
        self.q1, self.q2 = q1, q2
        self.q1_target, self.q2_target = q1_target, q2_target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.buffer = buffer
        self.tau = float(tau)
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.cql_alpha = float(cql_alpha)
        self.num_sampled_actions = max(1, int(num_sampled_actions))
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        noise: bool = True,
    ) -> ActionOutput:
        if deterministic or not noise:
            with torch.no_grad():
                mean, _ = self.actor(obs)
                return ActionOutput(
                    action=torch.tanh(mean) * self.action_scale, state=state
                )
        with torch.no_grad():
            action, logp, _ = self.actor.sample(obs)
        return ActionOutput(
            action=action * self.action_scale, log_prob=logp, state=state
        )

    def _q_input(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, actions], dim=-1)

    def _logsumexp_q(self, obs: torch.Tensor, q_net: nn.Module) -> torch.Tensor:
        """Importance-sampled ``logsumexp_a Q(s,a)`` per state (shape ``[B]``).

        Half the samples come from the current policy (importance weight = its
        log-prob, detached), half uniform on ``[-1, 1]^d`` (log-density
        ``-d*log2``). ``logmeanexp`` over the ``2M`` samples estimates
        ``log E_a exp(Q)``.
        """
        b = obs.shape[0]
        m = self.num_sampled_actions
        obs_rep = obs.unsqueeze(1).expand(b, m, obs.shape[-1]).reshape(b * m, -1)

        pi_a, pi_logp, _ = self.actor.sample(obs_rep)
        q_pi = q_net(self._q_input(obs_rep, pi_a)).reshape(b, m)
        pi_logp = pi_logp.reshape(b, m).detach()

        unif_a = torch.empty(b * m, self.action_dim, device=obs.device).uniform_(
            -1.0, 1.0
        )
        q_unif = q_net(self._q_input(obs_rep, unif_a)).reshape(b, m)
        unif_logp = -self.action_dim * math.log(2.0)

        cat = torch.cat([q_pi - pi_logp, q_unif - unif_logp], dim=1)
        return torch.logsumexp(cat, dim=1) - math.log(2 * m)

    def conservative_penalty(
        self, obs: torch.Tensor, q_data: torch.Tensor, q_net: nn.Module
    ) -> torch.Tensor:
        """``E_s[logsumexp_a Q] - E[Q(s,a_data)]`` (the CQL gap, batch mean)."""
        return (self._logsumexp_q(obs, q_net) - q_data).mean()

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"] / self.action_scale
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Twin-Q TD target (SAC-style, with entropy bonus).
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_obs)
            tq = torch.min(
                self.q1_target(self._q_input(next_obs, next_a)),
                self.q2_target(self._q_input(next_obs, next_a)),
            ).squeeze(-1)
            target = rewards + self.gamma * (1.0 - dones) * (
                tq - self.alpha.detach() * next_logp
            )
        q1 = self.q1(self._q_input(obs, actions)).squeeze(-1)
        q2 = self.q2(self._q_input(obs, actions)).squeeze(-1)
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # Conservative penalty per critic (importance-sampled logsumexp).
        pen1 = self.conservative_penalty(obs, q1, self.q1)
        pen2 = self.conservative_penalty(obs, q2, self.q2)
        penalty = pen1 + pen2
        critic_loss = td_loss + self.cql_alpha * penalty
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # Actor (entropy-regularized) + temperature.
        new_a, logp, _ = self.actor.sample(obs)
        q_new = torch.min(
            self.q1(self._q_input(obs, new_a)),
            self.q2(self._q_input(obs, new_a)),
        ).squeeze(-1)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "loss": float((critic_loss + actor_loss).item()),
            "critic_loss": float(critic_loss.item()),
            "td_loss": float(td_loss.item()),
            "q_loss": float(td_loss.item()),
            "cql_penalty": float(penalty.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(-logp.mean().item()),
            "alpha": float(self.alpha.item()),
        }


def build_cql_continuous(**kwargs):
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    action_space = kwargs.get("action_space")
    obs_shape = kwargs.get("obs_shape", (obs_dim,))

    from src.rl.models.backbone import select_backbone

    actor = SquashedGaussianActor(obs_dim, action_dim, obs_shape=obs_shape).to(device)
    mk_q = lambda: select_backbone(  # noqa: E731
        (obs_dim + action_dim,),
        obs_dim + action_dim,
        1,
        hidden_dims=(256, 256),
        activation=nn.ReLU,
    ).to(device)
    q1, q2, q1t, q2t = (mk_q() for _ in range(4))
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    try:
        scale = float(abs(action_space.high[0]))
    except Exception:
        scale = 1.0
    agent = ContinuousCQL(
        actor,
        q1,
        q2,
        q1t,
        q2t,
        buffer,
        device=device,
        action_dim=action_dim,
        action_scale=scale,
    )
    return actor, agent

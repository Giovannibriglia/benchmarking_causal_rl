"""Implicit Q-Learning for continuous action spaces (IQL-Gaussian).

Kostrikov et al. 2021 (arXiv:2110.06169), continuous form. The value/Q core is
IDENTICAL to discrete IQL (the shared ``expectile_loss`` is reused verbatim):

    L_V = E[ expectile_loss(min_i Q'_i(s,a) - V(s), tau) ]
    L_Q = E[ (r + gamma (1-d) V(s') - Q_i(s,a))^2 ]   (twin Q, V-bootstrapped)

The only continuous-specific piece is the policy: a squashed-Gaussian actor
extracted by advantage-weighted regression,

    L_pi = -E[ min(exp(beta (Q'-V)), clip) * log pi(a_data | s) ]

which needs ``log pi`` of the DATASET action under the squashed Gaussian.
``SquashedGaussianActor.sample`` only gives the log-prob of a freshly sampled
action, so ``squashed_log_prob`` computes it for a given action via the inverse
tanh — with the action CLAMPED to ``[-1+eps, 1-eps]`` first, because saturated
dataset torques scale to exactly +-1 and ``atanh(+-1) = +-inf`` would poison the
AWR weight with NaN. (SAC's ``sample`` never hits this; it uses the pre-tanh
value. AWR does, because it inverts tanh on a logged action.)

Standalone ``BaseOffPolicy`` reusing ``SquashedGaussianActor`` + twin-Q + a V
net; never edits ``sac.py`` (protects its bitwise test).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.base import ActionOutput
from src.rl.off_policy.base_off_policy import BaseOffPolicy
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.off_policy.sac import _LOG_STD_MAX, _LOG_STD_MIN, SquashedGaussianActor
from src.rl.offline.iql import expectile_loss


def squashed_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``log pi(action | s)`` for a tanh-squashed Gaussian, given the action.

    ``action`` is the post-tanh action in ``[-1, 1]`` (already de-scaled). It is
    clamped to ``[-1+eps, 1-eps]`` before ``atanh`` to avoid +-inf at saturated
    actions. The ``-log(1 - tanh^2)`` Jacobian term mirrors
    ``SquashedGaussianActor.sample`` exactly (same ``+eps`` inside the log).
    """
    action = action.clamp(-1.0 + eps, 1.0 - eps)
    pre = torch.atanh(action)
    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    logp = dist.log_prob(pre) - torch.log1p(-action.pow(2) + eps)
    return logp.sum(-1)


class ContinuousIQL(BaseOffPolicy):
    """IQL with a squashed-Gaussian AWR policy (continuous actions)."""

    action_type = "continuous"

    def __init__(
        self,
        actor: SquashedGaussianActor,
        q1: nn.Module,
        q2: nn.Module,
        q1_target: nn.Module,
        q2_target: nn.Module,
        value_net: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        action_scale: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        beta: float = 3.0,
        adv_clip: float = 100.0,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.actor = actor
        self.q1, self.q2 = q1, q2
        self.q1_target, self.q2_target = q1_target, q2_target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.value_net = value_net
        self.buffer = buffer
        self.tau = float(tau)
        self.action_scale = float(action_scale)
        self.expectile = float(expectile)
        self.beta = float(beta)
        self.adv_clip = float(adv_clip)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.v_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        noise: bool = True,
    ) -> ActionOutput:
        with torch.no_grad():
            mean, log_std = self.actor(obs)
            if deterministic or not noise:
                action = torch.tanh(mean)
            else:
                std = log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp()
                action = torch.tanh(torch.distributions.Normal(mean, std).sample())
        return ActionOutput(action=action * self.action_scale, state=state)

    def _q_input(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, actions], dim=-1)

    def _min_twin_q(self, net1, net2, obs, actions) -> torch.Tensor:
        return torch.min(
            net1(self._q_input(obs, actions)), net2(self._q_input(obs, actions))
        ).squeeze(-1)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"] / self.action_scale
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Value: expectile regression toward the (target) min-twin-Q of a_data.
        with torch.no_grad():
            q_target_sa = self._min_twin_q(self.q1_target, self.q2_target, obs, actions)
        v = self.value_net(obs).squeeze(-1)
        v_loss = expectile_loss(q_target_sa - v, self.expectile).mean()
        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()

        # Q: TD target bootstrapped from V(next) (no OOD max over actions).
        with torch.no_grad():
            next_v = self.value_net(next_obs).squeeze(-1)
            q_target = rewards + self.gamma * next_v * (1.0 - dones)
        q1 = self.q1(self._q_input(obs, actions)).squeeze(-1)
        q2 = self.q2(self._q_input(obs, actions)).squeeze(-1)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        # Policy: advantage-weighted regression with clipped weights.
        with torch.no_grad():
            adv = self._min_twin_q(
                self.q1_target, self.q2_target, obs, actions
            ) - self.value_net(obs).squeeze(-1)
            weight = torch.clamp(torch.exp(self.beta * adv), max=self.adv_clip)
        mean, log_std = self.actor(obs)
        log_pi = squashed_log_prob(mean, log_std, actions)
        pi_loss = -(weight * log_pi).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()

        return {
            "loss": float(q_loss.item()),
            "q_loss": float(q_loss.item()),
            "critic_loss": float(q_loss.item()),
            "value_loss": float(v_loss.item()),
            "actor_loss": float(pi_loss.item()),
        }


def build_iql_continuous(**kwargs):
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
    value_net = select_backbone(obs_shape, obs_dim, 1).to(device)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    try:
        scale = float(abs(action_space.high[0]))
    except Exception:
        scale = 1.0
    agent = ContinuousIQL(
        actor,
        q1,
        q2,
        q1t,
        q2t,
        value_net,
        buffer,
        device=device,
        action_scale=scale,
    )
    return actor, agent

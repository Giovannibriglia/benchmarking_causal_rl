"""Batch-Constrained Q-Learning for continuous actions (CVAE-BCQ).

Fujimoto et al. 2019 (arXiv:1812.02900). The continuous variant constrains the
policy to the data manifold with a conditional VAE behavior model plus a small
bounded perturbation, instead of the discrete variant's behavior-probability
threshold. Three pieces, none shared with SAC:

  * CVAE behavior model: encoder ``q(z|s,a)`` (log-variance head), decoder
    ``p(a|s,z)`` (tanh). Trained by the ELBO = recon-MSE + beta * analytic KL.
  * Perturbation net ``xi(s,a)``: a Phi-bounded residual, applied as
    ``clip(a + Phi * xi(s,a), -1, 1)`` -- lets the policy nudge VAE samples
    toward higher Q without leaving the data support.
  * Twin Q critics (via ``select_backbone`` over ``obs (+) act``).

Action selection (eval AND the Q-target) samples N latents from the PRIOR
``N(0, I)`` (never the encoder -- that needs an action), decodes, perturbs, and
takes the best under the critics. The Q-target runs entirely through TARGET nets
(target q1/q2 + a Polyak-tracked target perturbation net) and blends the twin
critics with the soft-clipped double-Q rule ``lmbda*min + (1-lmbda)*max`` -- the
conservatism mechanism. This is NOT the discrete/Atari-BCQ double-DQN
online-select/target-value split; continuous BCQ is all-target-net.

Standalone ``BaseOffPolicy``; never imports ``SquashedGaussianActor`` or edits
``sac.py``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.base import ActionOutput
from src.rl.nets.mlp import MLP
from src.rl.off_policy.base_off_policy import BaseOffPolicy
from src.rl.off_policy.replay_buffer import ReplayBuffer

_LOGVAR_MIN, _LOGVAR_MAX = -8.0, 8.0


class ConditionalVAE(nn.Module):
    """CVAE behavior model: encoder ``q(z|s,a)`` + decoder ``p(a|s,z)``.

    The encoder emits a log-variance head (so ``std = exp(0.5*logvar)`` and the
    analytic KL is literal). The decoder is tanh-bounded to ``[-1, 1]`` (the
    de-scaled action range).
    """

    def __init__(
        self, obs_dim: int, action_dim: int, latent_dim: int, hidden=(256, 256)
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = MLP(
            obs_dim + action_dim, 2 * latent_dim, hidden_dims=hidden, activation=nn.ReLU
        )
        self.decoder = MLP(
            obs_dim + latent_dim,
            action_dim,
            hidden_dims=hidden,
            activation=nn.ReLU,
            output_activation=nn.Tanh,
        )

    def encode(self, obs: torch.Tensor, action: torch.Tensor):
        mu, logvar = self.encoder(torch.cat([obs, action], dim=-1)).chunk(2, dim=-1)
        return mu, logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX)

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([obs, z], dim=-1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        mu, logvar = self.encode(obs, action)
        # Reparameterization: z = mu + std*eps, gradient flows to the encoder.
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(obs, z), mu, logvar


class PerturbationNet(nn.Module):
    """``clip(a + Phi * tanh(net([s;a])), -1, 1)`` -- a Phi-bounded residual."""

    def __init__(
        self, obs_dim: int, action_dim: int, phi: float, hidden=(256, 256)
    ) -> None:
        super().__init__()
        self.phi = float(phi)
        self.net = MLP(
            obs_dim + action_dim,
            action_dim,
            hidden_dims=hidden,
            activation=nn.ReLU,
            output_activation=nn.Tanh,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.phi * self.net(torch.cat([obs, action], dim=-1))
        return (action + delta).clamp(-1.0, 1.0)


class ContinuousBCQ(BaseOffPolicy):
    """Continuous BCQ: CVAE + bounded perturbation + soft-clipped twin Q."""

    action_type = "continuous"

    def __init__(
        self,
        vae: ConditionalVAE,
        perturbation: PerturbationNet,
        perturbation_target: PerturbationNet,
        q1: nn.Module,
        q2: nn.Module,
        q1_target: nn.Module,
        q2_target: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        action_dim: int,
        action_scale: float = 1.0,
        phi: float = 0.05,
        n_sampled: int = 10,
        lmbda: float = 0.75,
        vae_lr: float = 1e-3,
        critic_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        beta_kl: float = 0.5,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.vae = vae
        self.perturbation = perturbation
        self.perturbation_target = perturbation_target
        self.perturbation_target.load_state_dict(self.perturbation.state_dict())
        self.q1, self.q2 = q1, q2
        self.q1_target, self.q2_target = q1_target, q2_target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.buffer = buffer
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.phi = float(phi)
        self.n_sampled = int(n_sampled)
        self.lmbda = float(lmbda)
        self.beta_kl = float(beta_kl)
        self.tau = float(tau)
        self.latent_dim = vae.latent_dim
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.actor_opt = torch.optim.Adam(self.perturbation.parameters(), lr=actor_lr)

    def _q_input(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, actions], dim=-1)

    def _soft_clipped(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """``lmbda*min + (1-lmbda)*max`` -- the BCQ conservatism blend."""
        return self.lmbda * torch.min(q1, q2) + (1.0 - self.lmbda) * torch.max(q1, q2)

    def _candidates(self, obs: torch.Tensor, n: int, perturbation: PerturbationNet):
        """Decode N prior-sampled latents per state, then perturb.

        Returns ``(obs_rep, actions)`` each with a leading ``B*n`` dim. ``z`` is
        drawn from the PRIOR ``N(0, I)`` -- the encoder is never used here (it
        would need an action, which is exactly what we are choosing).
        """
        b = obs.shape[0]
        obs_rep = obs.unsqueeze(1).expand(b, n, obs.shape[-1]).reshape(b * n, -1)
        z = torch.randn(b * n, self.latent_dim, device=obs.device)
        a = perturbation(obs_rep, self.vae.decode(obs_rep, z))
        return obs_rep, a

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        noise: bool = True,
    ) -> ActionOutput:
        # BCQ is inherently sample-based; the noise/deterministic flags the eval
        # loop passes are ignored (we always sample N from the prior + argmax).
        b = obs.shape[0]
        with torch.no_grad():
            obs_rep, a = self._candidates(obs, self.n_sampled, self.perturbation)
            q = torch.min(
                self.q1(self._q_input(obs_rep, a)),
                self.q2(self._q_input(obs_rep, a)),
            ).reshape(b, self.n_sampled)
            idx = q.argmax(dim=1)
            a = a.reshape(b, self.n_sampled, self.action_dim)[torch.arange(b), idx]
        return ActionOutput(action=a * self.action_scale, state=state)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"] / self.action_scale
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # --- CVAE ELBO: recon-MSE + beta * analytic Gaussian KL ---
        recon, mu, logvar = self.vae(obs, actions)
        recon_loss = F.mse_loss(recon, actions)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        vae_loss = recon_loss + self.beta_kl * kl
        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()

        # --- Critics: TD to the all-target-net soft-clipped target over N
        #     prior-decoded + target-perturbed next candidates ---
        with torch.no_grad():
            next_rep, next_a = self._candidates(
                next_obs, self.n_sampled, self.perturbation_target
            )
            soft = self._soft_clipped(
                self.q1_target(self._q_input(next_rep, next_a)),
                self.q2_target(self._q_input(next_rep, next_a)),
            ).reshape(next_obs.shape[0], self.n_sampled)
            best = soft.max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * best
        q1 = self.q1(self._q_input(obs, actions)).squeeze(-1)
        q2 = self.q2(self._q_input(obs, actions)).squeeze(-1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # --- Perturbation actor: raise Q1 of a perturbed fresh VAE sample ---
        z = torch.randn(obs.shape[0], self.latent_dim, device=obs.device)
        sampled = self.vae.decode(obs, z).detach()
        perturbed = self.perturbation(obs, sampled)
        actor_loss = -self.q1(self._q_input(obs, perturbed)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- Polyak: q1, q2, AND the perturbation net (VAE has no target) ---
        with torch.no_grad():
            for net, tgt in (
                (self.q1, self.q1_target),
                (self.q2, self.q2_target),
                (self.perturbation, self.perturbation_target),
            ):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "loss": float((critic_loss + actor_loss + vae_loss).item()),
            "critic_loss": float(critic_loss.item()),
            "q_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "kl_loss": float(kl.item()),
        }


def build_bcq_continuous(**kwargs):
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    action_space = kwargs.get("action_space")

    from src.rl.models.backbone import select_backbone

    phi = 0.05
    latent_dim = 2 * action_dim
    vae = ConditionalVAE(obs_dim, action_dim, latent_dim).to(device)
    perturbation = PerturbationNet(obs_dim, action_dim, phi).to(device)
    perturbation_target = PerturbationNet(obs_dim, action_dim, phi).to(device)
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
    agent = ContinuousBCQ(
        vae,
        perturbation,
        perturbation_target,
        q1,
        q2,
        q1t,
        q2t,
        buffer,
        device=device,
        action_dim=action_dim,
        action_scale=scale,
        phi=phi,
    )
    return perturbation, agent

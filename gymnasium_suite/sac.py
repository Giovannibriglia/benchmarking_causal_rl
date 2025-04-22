import torch
import torch.nn.functional as F

from gymnasium_suite.base import build_base_q_policy
from torch import nn
from torch.distributions import Categorical, Normal


def make_sac_policy(is_causal: bool = False):
    base_q_policy = build_base_q_policy(is_causal)

    class SACPolicy(base_q_policy):
        def __init__(self, *args, lr_q=3e-4, lr_pi=3e-4, tau=0.005, **kwargs):
            super().__init__(*args, lr=lr_q, pi_lr=lr_pi, **kwargs)
            self.tau = tau

            # — for discrete, install a small policy head —
            if self.is_discrete:
                self.actor = nn.Sequential(
                    nn.Linear(self.obs_dim + self.extra_input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.action_space.n),
                ).to(self.device)
                self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr_pi)

        def _get_actions(self, observations: torch.Tensor, mask: torch.Tensor = None):
            obs_enc = self._enc(observations.to(self.device))

            if self.is_discrete:
                logits = self.actor(obs_enc)
                dist = Categorical(logits=logits)
                return dist.sample()

            # continuous
            mu, log_std = self.actor(obs_enc)
            std = log_std.exp().clamp(min=1e-5)
            dist = Normal(mu, std)
            action = dist.rsample()
            return torch.tanh(action)

        def _update(self, obs, acts, rews, next_obs, dones):
            # 1) collect into buffer
            for i in range(self.n_envs):
                if dones[i]:
                    continue
                self.mem.put(
                    obs[i].cpu().numpy(),
                    acts[i].cpu().numpy() if not self.is_discrete else int(acts[i]),
                    rews[i].item(),
                    next_obs[i].cpu().numpy(),
                    1.0,
                )
            if len(self.mem) < self.batch:
                return

            # 2) sample a batch
            o, a, r, o2, m = self.mem.sample(self.batch, self.device)
            o_enc = self._enc(o)
            o2_enc = self._enc(o2)

            # 3) compute target Q
            with torch.no_grad():
                if self.is_discrete:
                    # discrete SAC target
                    logits_next = self.actor(o2_enc)
                    dist_next = Categorical(logits=logits_next)
                    probs = dist_next.probs
                    logp = dist_next.log_prob(dist_next.sample()).unsqueeze(-1)
                    q_next = self.target_q(o2_enc)  # [B, A]
                    # soft value: sum_a π(a|s') [Q(s',a) - α log π(a|s')]
                    soft_v = (probs * (q_next - self.entropy_coef * logp)).sum(
                        dim=1, keepdim=True
                    )
                else:
                    # continuous SAC target
                    mu2, log_std2 = self.actor(o2_enc)
                    std2 = log_std2.exp().clamp(min=1e-5)
                    dist2 = Normal(mu2, std2)
                    a2 = dist2.rsample()
                    logp2 = dist2.log_prob(a2).sum(-1, keepdim=True)
                    a2_t = torch.tanh(a2)
                    q1_t = self.target_q1(torch.cat([o2_enc, a2_t], dim=-1))
                    q2_t = self.target_q2(torch.cat([o2_enc, a2_t], dim=-1))
                    soft_v = torch.min(q1_t, q2_t) - self.entropy_coef * logp2

                target_q = r.unsqueeze(-1) + self.gamma * soft_v * m.unsqueeze(-1)

            # 4) critic update
            if self.is_discrete:
                q_pred = self.q_net(o_enc).gather(1, a.long().unsqueeze(-1))
                loss_q = F.mse_loss(q_pred, target_q)
            else:
                qa1 = self.q_net1(torch.cat([o_enc, a], dim=-1))
                qa2 = self.q_net2(torch.cat([o_enc, a], dim=-1))
                loss_q = F.mse_loss(qa1, target_q) + F.mse_loss(qa2, target_q)

            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()

            # 5) actor update
            o_enc_pi = o_enc.detach()
            if self.is_discrete:
                logits = self.actor(o_enc_pi)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(dist.sample()).unsqueeze(-1)
                q_val = self.q_net(o_enc_pi)
                # policy loss = E[α log π(a|s) - Q(s,a)]
                loss_pi = (
                    (dist.probs * (self.entropy_coef * logp - q_val)).sum(dim=1).mean()
                )
            else:
                mu, log_std = self.actor(o_enc_pi)
                std = log_std.exp().clamp(min=1e-5)
                dist = Normal(mu, std)
                a_pi = dist.rsample()
                logp = dist.log_prob(a_pi).sum(-1, keepdim=True)
                a_t = torch.tanh(a_pi)
                q1_pi = self.q_net1(torch.cat([o_enc_pi, a_t], dim=-1))
                q2_pi = self.q_net2(torch.cat([o_enc_pi, a_t], dim=-1))
                q_pi = torch.min(q1_pi, q2_pi)
                loss_pi = (self.entropy_coef * logp - q_pi).mean()

            self.opt_pi.zero_grad()
            loss_pi.backward()
            self.opt_pi.step()

            # 6) soft‐update targets
            with torch.no_grad():
                if self.is_discrete:
                    # single Q‑net target update
                    for p, tp in zip(
                        self.q_net.parameters(), self.target_q.parameters()
                    ):
                        tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                else:
                    # twin critics soft‐update
                    for p, tp in zip(
                        self.q_net1.parameters(), self.target_q1.parameters()
                    ):
                        tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                    for p, tp in zip(
                        self.q_net2.parameters(), self.target_q2.parameters()
                    ):
                        tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    return SACPolicy

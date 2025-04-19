import torch
import torch.nn.functional as F

from gymnasium_suite.base import BaseACPolicy, safe_clone


class PPOPolicy(BaseACPolicy):
    def __init__(
        self,
        *args,
        gae_lambda=0.95,
        eps_clip=0.2,
        K_epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
        **kw
    ):
        super().__init__(*args, **kw)
        self.lam = gae_lambda
        self.eps_clip = eps_clip
        self.K = K_epochs
        self.entropy_c = entropy_coef
        self.value_c = value_coef

    # helper: compute GAE & flatten rollout
    def _process_rollout(self, next_obs):
        s = torch.cat(self.buf["s"]).to(self.device)
        a = torch.cat(self.buf["a"]).to(self.device)
        lp = torch.cat(self.buf["logp"]).to(self.device)
        r = torch.stack(self.buf["r"]).to(self.device)  # [T, n_envs]
        d = torch.stack(self.buf["d"]).to(self.device)
        v = torch.cat(self.buf["v"]).to(self.device).view(self.rollout_len, self.n_envs)

        with torch.no_grad():
            next_v = self.net.value(self._enc(next_obs)).view(self.n_envs)

        adv = torch.zeros_like(r)
        gae = 0
        for t in reversed(range(self.rollout_len)):
            delta = r[t] + self.gamma * next_v * (1 - d[t]) - v[t]
            gae = delta + self.gamma * self.lam * (1 - d[t]) * gae
            adv[t] = gae
            next_v = v[t]
        ret = (adv + v).view(-1)
        return s, a, lp, adv.view(-1), ret

    def update(self, obs, acts, rews, next_obs, dones):
        # overwrite reward/done
        self.buf["r"][-1] = safe_clone(rews, torch.float32, self.device)
        self.buf["d"][-1] = safe_clone(dones, torch.float32, self.device)

        if len(self.buf["r"]) < self.rollout_len:
            return

        s, a, lp_old, adv, ret = self._process_rollout(next_obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.K):
            dist = self.net.dist(s)
            logp = dist.log_prob(a)
            entropy = dist.entropy()
            if logp.ndim > 1:
                logp, entropy = logp.sum(-1), entropy.sum(-1)
            ratio = torch.exp(logp - lp_old)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            value = self.net.value(s)
            value_loss = F.mse_loss(value, ret)

            loss = (
                actor_loss + self.value_c * value_loss - self.entropy_c * entropy.mean()
            )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self.metrics.add(
                actor_loss=actor_loss.item(),
                value_loss=value_loss.item(),
                kl=(lp_old - logp).mean().item(),
            )

        self._reset_buf()

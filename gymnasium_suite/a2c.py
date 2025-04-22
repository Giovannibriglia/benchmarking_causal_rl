import torch
import torch.nn.functional as F

from gymnasium_suite.base import build_base_acnet, safe_clone


def make_a2c_policy(causal: bool = False):
    BaseActorCriticPolicy = build_base_acnet(causal)

    class A2CPolicy(BaseActorCriticPolicy):
        def __init__(self, *args, entropy_coef=0.01, value_coef=0.5, **kw):
            super().__init__(*args, **kw)
            self.entropy_c = entropy_coef
            self.value_c = value_coef

        def _update(self, obs, acts, rews, next_obs, dones):
            # overwrite last step’s reward & done
            self.buf["r"][-1] = safe_clone(rews, torch.float32, self.device)
            self.buf["d"][-1] = safe_clone(dones, torch.float32, self.device)

            if len(self.buf["r"]) < self.rollout_len:  # not yet time to learn
                return

            # stack rollout
            s = torch.cat(self.buf["s"]).to(self.device)
            a = torch.cat(self.buf["a"]).to(self.device)
            logp_old = torch.cat(self.buf["logp"]).to(self.device)
            r = torch.stack(self.buf["r"]).to(self.device)  # [T, n_envs]
            d = torch.stack(self.buf["d"]).to(self.device)

            # bootstrap with V(s_T)
            with torch.no_grad():
                next_v = self.net.value(self._enc(next_obs)).view(self.n_envs)

            T = self.rollout_len
            returns = torch.zeros_like(r)
            running = next_v
            for t in reversed(range(T)):
                running = r[t] + self.gamma * running * (1 - d[t])
                returns[t] = running
            returns = returns.view(-1)

            # critic
            values = self.net.value(s)
            adv = returns - values.detach()

            # actor
            dist = self.net.dist(s)
            logp = dist.log_prob(a)
            if logp.ndim > 1:
                logp = logp.sum(-1)
            entropy = dist.entropy()
            if entropy.ndim > 1:
                entropy = entropy.sum(-1)

            actor_loss = -(logp * adv).mean()
            value_loss = F.mse_loss(values, returns)
            loss = (
                actor_loss + self.value_c * value_loss - self.entropy_c * entropy.mean()
            )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.metrics.add(
                actor_loss=actor_loss.item(),
                value_loss=value_loss.item(),
                kl=(logp_old - logp).mean().item(),
            )
            self._reset_buf()

    return A2CPolicy

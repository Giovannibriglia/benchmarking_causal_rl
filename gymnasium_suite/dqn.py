import random

import gymnasium as gym
import torch
import torch.nn.functional as F

from gymnasium_suite.base import build_base_q_policy


def make_dqn_policy(is_causal: bool = False):
    base_q_policy = build_base_q_policy(is_causal)

    class DQNPolicy(base_q_policy):
        def __init__(
            self,
            algo_name: str,
            act_space: gym.spaces.Discrete,
            obs_space: gym.spaces.Space,
            n_envs: int,
            n_episodes: int,
            **kwargs,
        ):
            if not isinstance(act_space, gym.spaces.Discrete):
                raise ValueError("DQN → discrete actions only.")
            super().__init__(
                algo_name,
                act_space,
                obs_space,
                n_envs,
                n_episodes,
            )

            self.eps_hi, self.eps_lo = 0.9, 0.02
            self.decay = (self.eps_hi - self.eps_lo) / n_episodes

        def _epsilon(self):
            return max(self.eps_lo, self.eps_hi - self.decay * self.cur_ep)

        def _get_actions(self, obs, mask=None):
            q = self.q_net(self._enc(obs))
            if random.random() < self._epsilon():
                return torch.randint(
                    0, self.action_space.n, (self.n_envs,), device=self.device
                )
            return q.argmax(1)

        def _update(self, obs, acts, rews, next_obs, dones):
            # 1) store transitions, with 1.0 for non‑terminal, 0.0 for terminal
            for i in range(self.n_envs):
                if dones[i]:
                    continue
                self.mem.put(
                    obs[i].cpu().numpy(),
                    int(acts[i]),
                    float(rews[i]),
                    next_obs[i].cpu().numpy(),
                    1.0,
                )

            # 2) wait until buffer has enough
            if len(self.mem) < self.batch:
                return

            # 3) sample batch
            s, a, r, s2, m = self.mem.sample(self.batch, self.device)

            # 4) current Q(s,a)
            q_sa = self.q_net(self._enc(s)).gather(1, a.unsqueeze(1)).squeeze(1)

            # 5) build target: r + γ·max_a' Q(s',a')·m
            with torch.no_grad():
                max_q_s2 = self.target_q(self._enc(s2)).max(1)[0]
                target = r + self.gamma * max_q_s2 * m

            # 6) Huber loss and optimize
            loss = F.smooth_l1_loss(q_sa, target)
            self.opt_q.zero_grad()
            loss.backward()
            self.opt_q.step()

            # 7) periodically sync target network
            self.update_ct += 1
            if self.update_ct % self.tgt_sync == 0:
                self.target_q.load_state_dict(self.q_net.state_dict())

    return DQNPolicy

"""feat/on-policy-recurrent-integration — POMDP demonstration (headline test).

A minimal cue-recall POMDP that REQUIRES memory: at t=0 the agent observes a
one-hot cue (action ignored); at t=1 the cue is hidden (obs is all-zeros) and the
agent is rewarded iff its action matches the t=0 cue. A memoryless MLP policy
sees identical t=1 observations regardless of the cue, so it cannot exceed chance
(~0.5 mean return). A recurrent policy can carry the cue in its hidden state and
solve the task (mean return -> 1.0).

This is the structural validator for the whole PR: if recurrent PPO does not beat
MLP PPO here, the hidden-state threading / BPTT is broken.
"""

from __future__ import annotations

import pytest
import torch
from src.data.experience_source import OnlineSource
from src.rl.on_policy.actor_critic import ActorCritic
from src.rl.on_policy.ppo import PPO


class CueRecallVecEnv:
    """2-step recall POMDP (vectorized). t=0: obs = one-hot cue, action ignored,
    reward 0. t=1: obs = zeros (cue hidden), reward = 1 iff action == cue, done."""

    def __init__(self, n_envs: int, device, seed: int = 0):
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.obs_dim = 2
        self.n_actions = 2
        self._gen = torch.Generator(device=self.device).manual_seed(seed)
        self._t = 0
        self._cue = self._sample_cue()

    def _sample_cue(self):
        return torch.randint(
            0, 2, (self.n_envs,), generator=self._gen, device=self.device
        )

    def _cue_obs(self):
        return torch.stack([self._cue == 0, self._cue == 1], dim=-1).float()

    def reset(self, seed=None):
        self._t = 0
        self._cue = self._sample_cue()
        return self._cue_obs(), {}

    def step(self, action):
        if self._t == 0:
            self._t = 1
            obs = torch.zeros(self.n_envs, self.obs_dim, device=self.device)
            reward = torch.zeros(self.n_envs, device=self.device)
            terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        else:
            self._t = 0
            reward = (action.long() == self._cue).float()
            terminated = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
            self._cue = self._sample_cue()  # autoreset: next episode's cue
            obs = self._cue_obs()
        truncated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        return obs, reward, terminated, truncated, {}


def _train_and_eval(actor_net: str, critic_net: str, *, n_updates: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_envs = 64
    env = CueRecallVecEnv(n_envs, device, seed=seed)
    src = OnlineSource(env, device)
    policy = ActorCritic(
        (2,),
        2,
        2,
        "discrete",
        device,
        actor_network=actor_net,
        critic_network=critic_net,
    )
    # lr=3e-3 is gentle enough for the vanilla RNN to train stably (lr=1e-2
    # destabilizes Elman RNNs on this task); LSTM/GRU solve comfortably too, and
    # the MLP is structurally incapable regardless of lr.
    agent = PPO(policy, device=device, lr=3e-3, entropy_coef=0.01, train_iters=4)
    recurrent = policy.is_recurrent
    roll = src.rollout_recurrent if recurrent else src.rollout
    for _ in range(n_updates):
        batch, _ = roll(policy, agent, n_steps=2, n_envs=n_envs)
        agent.update(batch)
    # Eval: mean episode return over several fresh rollouts (no grad).
    returns = []
    with torch.no_grad():
        for _ in range(20):
            _, ep_returns = roll(policy, agent, n_steps=2, n_envs=n_envs)
            returns.append(ep_returns.mean().item())
    return sum(returns) / len(returns)


def test_mlp_cannot_solve_pomdp():
    """Memoryless MLP PPO is stuck at chance (~0.5) — it cannot recall the cue."""
    ret = _train_and_eval("mlp", "mlp", n_updates=400, seed=0)
    assert ret < 0.7, f"MLP unexpectedly solved the POMDP (return={ret:.3f})"


@pytest.mark.parametrize("net", ["lstm", "gru", "rnn"])
def test_recurrent_solves_pomdp(net):
    """Recurrent PPO carries the cue in its hidden state and solves the task."""
    ret = _train_and_eval(net, net, n_updates=400, seed=0)
    assert ret > 0.8, f"{net} PPO failed to solve the POMDP (return={ret:.3f})"

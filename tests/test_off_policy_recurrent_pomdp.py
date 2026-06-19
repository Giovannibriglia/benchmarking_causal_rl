"""feat/off-policy-recurrent-integration — POMDP demonstration (headline test).

A 2-step cue-recall POMDP that REQUIRES memory: at t=0 the agent observes a cue
(action ignored); at t=1 the cue is hidden (obs all-zeros) and reward depends on
matching the cue. Memoryless MLP off-policy agents cannot exceed chance (~0.5);
recurrent (LSTM/GRU/RNN) agents carry the cue in their hidden state and solve it.

Discrete variant -> DQN; continuous variant -> SAC. This is the structural
validator for off-policy recurrence: if recurrent DQN/SAC do not beat their MLP
counterparts here, the sequence sampling / BPTT / target alignment is broken.
"""

from __future__ import annotations

import pytest
import torch
from src.data.experience_source import OnlineSource
from src.rl.models.backbone import build_trunk
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.off_policy.sac import SAC, SquashedGaussianActor
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer
from src.rl.policies.behavior_policy import AgentBehaviorPolicy

DEV = torch.device("cpu")
N_ENVS = 64


class _CueRecall:
    """t=0: obs = one-hot cue, reward 0. t=1: obs = zeros, episode ends. Discrete
    reward = 1 iff action == cue; continuous reward = 1 iff (action>0)==(cue==1)."""

    def __init__(self, n_envs, continuous, seed=0):
        self.n_envs = n_envs
        self.continuous = continuous
        self._g = torch.Generator(device=DEV).manual_seed(seed)
        self._t = 0
        self._cue = self._draw()

    def _draw(self):
        return torch.randint(0, 2, (self.n_envs,), generator=self._g, device=DEV)

    def _cue_obs(self):
        return torch.stack([self._cue == 0, self._cue == 1], dim=-1).float()

    def reset(self, seed=None):
        self._t = 0
        self._cue = self._draw()
        return self._cue_obs(), {}

    def step(self, action):
        if self._t == 0:
            self._t = 1
            obs = torch.zeros(self.n_envs, 2, device=DEV)
            reward = torch.zeros(self.n_envs, device=DEV)
            terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=DEV)
        else:
            self._t = 0
            if self.continuous:
                pred = (action.squeeze(-1) > 0).long()
            else:
                pred = action.long()
            reward = (pred == self._cue).float()
            terminated = torch.ones(self.n_envs, dtype=torch.bool, device=DEV)
            self._cue = self._draw()
            obs = self._cue_obs()
        truncated = torch.zeros(self.n_envs, dtype=torch.bool, device=DEV)
        return obs, reward, terminated, truncated, {}


def _buf(net):
    # Recurrent -> episode-aware sequence buffer; MLP -> flat buffer (mirrors the
    # runner's routing). MLP flat learn calls buffer.sample (absent on the
    # sequence buffer), so the buffer type must match the path.
    if net == "mlp":
        return ReplayBuffer(capacity=50_000, device=DEV)
    return SequenceReplayBuffer(capacity=50_000, device=DEV)


def _make_dqn(net):
    q = build_trunk(net, (2,), 2, 2).to(DEV)
    tgt = build_trunk(net, (2,), 2, 2).to(DEV)
    return DQN(q, tgt, _buf(net), device=DEV, lr=3e-3, epsilon=0.2)


def _make_sac(net):
    buf = _buf(net)
    actor = SquashedGaussianActor(2, 1, network=net).to(DEV)

    def mk():
        return build_trunk(net, (3,), 3, 1).to(DEV)

    return SAC(
        actor,
        mk(),
        mk(),
        mk(),
        mk(),
        buf,
        device=DEV,
        action_dim=1,
        actor_lr=3e-3,
        critic_lr=3e-3,
        alpha_lr=3e-3,
        action_scale=1.0,
    )


def _train_and_eval(agent, continuous, n_iter, seed):
    torch.manual_seed(seed)
    env = _CueRecall(N_ENVS, continuous, seed=seed)
    src = OnlineSource(env, DEV)
    obs, _ = env.reset()
    cpol = AgentBehaviorPolicy(agent)  # flat path needs a collection policy
    for _ in range(n_iter):
        if agent.is_recurrent:
            obs, _, _ = src.collect_off_policy_recurrent(
                agent,
                agent.buffer,
                obs,
                collection_policy=None,
                n_steps=2,
                n_envs=N_ENVS,
                warmup=256,
                batch_size=64,
                seq_len=2,
            )
        else:
            obs, _, _ = src.collect_off_policy(
                agent,
                agent.buffer,
                obs,
                collection_policy=cpol,
                n_steps=2,
                n_envs=N_ENVS,
                warmup=256,
                batch_size=64,
            )
    # Eval: greedy/deterministic, average reward over fresh episodes.
    returns = []
    with torch.no_grad():
        for _ in range(20):
            o, _ = env.reset()
            state = agent.initial_state(N_ENVS, device=DEV)
            ep = torch.zeros(N_ENVS, device=DEV)
            for _t in range(2):
                out = agent.act(o, state, deterministic=True)
                state = out.state
                o, r, _term, _trunc, _ = env.step(out.action)
                ep += r
            returns.append(ep.mean().item())
    return sum(returns) / len(returns)


def test_dqn_mlp_cannot_solve_pomdp():
    ret = _train_and_eval(_make_dqn("mlp"), continuous=False, n_iter=400, seed=0)
    assert ret < 0.7, f"DQN-MLP unexpectedly solved the POMDP (return={ret:.3f})"


@pytest.mark.parametrize("net", ["lstm", "gru", "rnn"])
def test_dqn_recurrent_solves_pomdp(net):
    ret = _train_and_eval(_make_dqn(net), continuous=False, n_iter=400, seed=0)
    assert ret > 0.8, f"DQN-{net} failed to solve the POMDP (return={ret:.3f})"


def test_sac_mlp_cannot_solve_pomdp():
    ret = _train_and_eval(_make_sac("mlp"), continuous=True, n_iter=400, seed=0)
    assert ret < 0.7, f"SAC-MLP unexpectedly solved the POMDP (return={ret:.3f})"


@pytest.mark.parametrize("net", ["lstm", "gru", "rnn"])
def test_sac_recurrent_solves_pomdp(net):
    ret = _train_and_eval(_make_sac(net), continuous=True, n_iter=400, seed=0)
    assert ret > 0.8, f"SAC-{net} failed to solve the POMDP (return={ret:.3f})"

"""feat/off-policy-sequence-buffer — recurrent off-policy collection path.

Tests OnlineSource.collect_off_policy_recurrent (preserves per-env episode
structure into a SequenceReplayBuffer; threads hidden state through act) and the
runner's flat-vs-recurrent routing toggle. The existing flat collect_off_policy
is untouched and covered by the off-policy goldens.
"""

from __future__ import annotations

import types

import torch
from src.benchmarking.runner import BenchmarkRunner
from src.data.experience_source import OnlineSource
from src.rl.base import ActionOutput
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

D = 3


class _StubVecEnv:
    """Vec env where env i ends its episode every ``periods[i]`` steps. obs[i]
    encodes (env_id, within-episode step) so sequence coherence is checkable."""

    def __init__(self, periods, device="cpu"):
        self.periods = periods
        self.n_envs = len(periods)
        self.device = torch.device(device)
        self._k = [0] * self.n_envs

    def _obs(self):
        return torch.tensor(
            [[float(i), float(self._k[i]), 0.0] for i in range(self.n_envs)],
            device=self.device,
        )

    def reset(self, seed=None):
        self._k = [0] * self.n_envs
        return self._obs(), {}

    def step(self, action):
        term = []
        for i in range(self.n_envs):
            self._k[i] += 1
            done_i = (self._k[i] % self.periods[i]) == 0
            if done_i:
                self._k[i] = 0
            term.append(done_i)
        obs = self._obs()
        reward = torch.ones(self.n_envs, device=self.device)
        terminated = torch.tensor(term, device=self.device)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        return obs, reward, terminated, truncated, {}


class _StubRecurrentAgent:
    """Off-policy-shaped stub: act(obs, state) threads state through (records the
    states it was called with) and exposes initial_state / reset_state_where."""

    def __init__(self, n_actions=2):
        self.n_actions = n_actions
        self.seen_states = []

    def initial_state(self, n, device=None):
        return {"h": torch.zeros(n, 1, device=device)}

    def reset_state_where(self, state, mask):
        state["h"][mask] = 0.0
        return state

    def act(self, obs, state=None):
        self.seen_states.append(state)
        b = obs.shape[0]
        nxt = None if state is None else {"h": state["h"] + 1.0}
        return ActionOutput(action=torch.zeros(b, dtype=torch.long), state=nxt)


def test_collect_off_policy_recurrent_preserves_episodes():
    env = _StubVecEnv(periods=[10, 15])
    src = OnlineSource(env, torch.device("cpu"))
    agent = _StubRecurrentAgent()
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    obs, _ = env.reset()
    src.collect_off_policy_recurrent(
        agent,
        buf,
        obs,
        collection_policy=None,
        n_steps=50,
        n_envs=2,
        warmup=0,
        batch_size=4,
    )
    assert len(buf) == 100  # 50 steps * 2 envs
    # Sampled sequences are coherent single-env contiguous runs.
    seq = buf.sample_sequences(batch_size=16, seq_len=8)["obs"]  # (16, 8, D)
    for row in seq:
        env_ids = row[:, 0].long()
        steps = row[:, 1].long()
        assert torch.all(env_ids == env_ids[0]), "sequence mixed envs"
        assert torch.all(steps[1:] - steps[:-1] == 1), "sequence not contiguous"


def test_collect_off_policy_recurrent_passes_state_through():
    env = _StubVecEnv(periods=[7, 7])
    src = OnlineSource(env, torch.device("cpu"))
    agent = _StubRecurrentAgent()
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    obs, _ = env.reset()
    src.collect_off_policy_recurrent(
        agent,
        buf,
        obs,
        collection_policy=None,
        n_steps=5,
        n_envs=2,
        warmup=0,
        batch_size=2,
    )
    # First call gets the initial (zeros) state; subsequent calls get a threaded
    # (advanced) non-None state -> threading did not crash and is wired.
    assert agent.seen_states[0] is not None  # initial_state provided
    assert agent.seen_states[1] is not None
    assert torch.all(agent.seen_states[1]["h"] >= agent.seen_states[0]["h"])


def _toggle_probe(is_recurrent: bool):
    calls = []
    fake_src = types.SimpleNamespace(
        collect_off_policy=lambda *a, **k: (calls.append("flat"), ("o", None, None))[1],
        collect_off_policy_recurrent=lambda *a, **k: (
            calls.append("recurrent"),
            ("o", None, None),
        )[1],
    )
    fake = types.SimpleNamespace(
        policy=types.SimpleNamespace(is_recurrent=is_recurrent),
        experience_source=fake_src,
        agent=None,
        replay_buffer=None,
        collection_policy=None,
        aux_models=None,
        offpolicy_warmup=0,
        offpolicy_batch_size=4,
        env_cfg=types.SimpleNamespace(n_train_envs=2),
    )
    BenchmarkRunner._collect_off_policy(
        fake, obs="o", total_steps=3, metrics_cache=None
    )
    return calls


def test_runner_toggle_routes_to_correct_collection_method():
    assert _toggle_probe(is_recurrent=True) == ["recurrent"]
    assert _toggle_probe(is_recurrent=False) == ["flat"]

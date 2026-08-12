"""Shared synthetic fixtures for the GRACE test suite (feat/grace-critic).

A tiny known MDP with an episode-static binary confounder U:

  * 2-D observations drawn uniformly; the true state is ``s = 1[obs_0 > 0.5]``
  * behavior: P(a=1 | U=1) = 0.8, P(a=1 | U=0) = 0.2 when ``confounded``,
    else P(a=1) = 0.5 (A independent of U)
  * reward: ``r = 1[s=1] + c_r * U * 1[a=1]`` (the action-gated confounder)
  * transitions: uniform re-draw (state-independent), fixed horizon

Tests run on cuda when available (session preference).
"""

from __future__ import annotations

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FakeSeqBuffer:
    """Duck-typed SequenceReplayBuffer: just ``iter_episodes``."""

    def __init__(self, episodes):
        self._eps = episodes

    def iter_episodes(self):
        return iter(self._eps)


def make_confounded_episodes(
    n_ep: int = 300,
    t_len: int = 15,
    confounded: bool = True,
    c_r: float = 1.0,
    seed: int = 0,
    poison_u: bool = True,
    state_reward: bool = True,
):
    """Returns (episodes, u_true). ``poison_u`` stores NaN in the
    ``confounder_u`` key so any estimator-side read explodes loudly (R5)."""
    g = torch.Generator().manual_seed(seed)
    episodes, u_true = [], []
    for _ in range(n_ep):
        u = int(torch.randint(0, 2, (1,), generator=g))
        u_true.append(u)
        ep = []
        obs = torch.rand(2, generator=g)
        for t in range(t_len):
            p1 = (0.8 if u == 1 else 0.2) if confounded else 0.5
            a = int(torch.rand(1, generator=g) < p1)
            s_bit = float(obs[0] > 0.5) if state_reward else 0.0
            r = s_bit + c_r * u * (a == 1)
            nxt = torch.rand(2, generator=g)
            ep.append(
                {
                    "obs": obs.clone(),
                    "actions": torch.tensor(a),
                    "rewards": torch.tensor(float(r)),
                    "next_obs": nxt.clone(),
                    "dones": torch.tensor(float(t == t_len - 1)),
                    "confounder_u": torch.tensor(
                        float("nan") if poison_u else float(u)
                    ),
                }
            )
            obs = nxt
        episodes.append(ep)
    return episodes, torch.tensor(u_true, dtype=torch.float32)

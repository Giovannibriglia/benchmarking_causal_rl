#!/usr/bin/env python3
"""Belief+CQL probe (Phase-5 gate addendum 2).

Hypothesis: in the history-recoverable Cell-6 instantiation, the latent world
model's ceiling is its BC head, not unidentifiability — so reusing the SAME
trained GRU encoder with an offline-RL (CQL) head on the belief states should
recover the Cell-5 result (~299), demonstrating the 6→5 reduction.
"""

from __future__ import annotations

import numpy as np
import torch
from src.config.seeding import set_seed
from src.data.experience_source import OfflineDatasetSource
from src.data.minari_io import to_offline_source
from src.envs.registry import register_default_env_wrappers
from src.eval.regret import evaluate_policy
from src.offline.d3rlpy_wrappers import make_cql
from src.offline.latent_world_model import LatentWorldModelBC


def main() -> None:
    register_default_env_wrappers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0, deterministic=False)

    # cell-6 view: medium tier, velocities masked, propensities discarded
    src = to_offline_source(
        "causal/cartpole/medium-v0",
        device,
        behavior_policy="unknown",
        mask_indices=[1, 3],
        rng_seed=0,
    )

    # 1) train the latent world model exactly as the cell-6 run does
    wm = LatentWorldModelBC(2, 2, device, seed=0)
    wm.fit_source(src, 4000, batch_size=256)

    # 2) encode every episode into belief sequences (gru hidden states)
    belief_eps = []
    with torch.no_grad():
        for ep in src.episodes:
            x, _, _, _ = wm._episode_inputs(ep)
            h, _ = wm.gru(x.unsqueeze(0))
            beliefs = h.squeeze(0)  # [T, latent]
            belief_eps.append(
                {
                    # belief obs need T+1 rows; repeat the last belief as the
                    # terminal "next" (episodes end by truncation/termination)
                    "obs": torch.cat([beliefs, beliefs[-1:]], dim=0).cpu(),
                    "actions": ep["actions"].cpu(),
                    "rewards": ep["rewards"].cpu(),
                    "terminations": ep["terminations"].cpu(),
                    "truncations": ep["truncations"].cpu(),
                }
            )
    belief_src = OfflineDatasetSource(belief_eps, device, behavior_policy="unknown")

    # 3) CQL head on the beliefs
    cql = make_cql(device, "discrete")
    cql.fit_source(belief_src, 4000)

    # 4) stateful composite policy: GRU encode -> CQL argmax
    class BeliefCQL:
        def __init__(self) -> None:
            self.state = None

        def reset(self) -> None:
            self.state = None

        def __call__(self, obs: np.ndarray) -> int:
            t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
            B = 1
            if self.state is None:
                h = torch.zeros(1, B, wm.latent_dim, device=device)
                prev_a = torch.zeros(B, wm.n_actions, device=device)
            else:
                h, prev_a = self.state
            x = torch.cat([t, prev_a], dim=-1).unsqueeze(1)
            with torch.no_grad():
                out, h_next = wm.gru(x, h)
                belief = out.squeeze(1)
                a = int(cql.act(belief).action.item())
            pa = torch.zeros(B, wm.n_actions, device=device)
            pa[0, a] = 1.0
            self.state = (h_next, pa)
            return a

    returns = evaluate_policy(
        "causal/cartpole-cell2", BeliefCQL(), n_episodes=100, seed_base=40_000
    )
    print(
        f"belief+CQL probe: J = {returns.mean():.1f} +- {returns.std():.1f} "
        "(cell-6 latent_wm BC head was 212.1; cell-5 CQL was 299.1)"
    )


if __name__ == "__main__":
    main()

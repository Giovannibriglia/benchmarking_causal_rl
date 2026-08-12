"""§D.3 — canonicalization stability: two fits from different seeds on the
same confounded dataset yield the SAME U-labeling (the proximal label-swap
convention: U=1 = the higher mean-residual stratum)."""

from __future__ import annotations

import torch
from src.rl.offline.grace import cell_graph, GraceMachinery, GraceOptions
from tests._grace_test_utils import DEV, FakeSeqBuffer, make_confounded_episodes


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def test_mdp_labels_are_seed_stable_and_canonical():
    eps, u_true = make_confounded_episodes(n_ep=300, t_len=15, seed=5)
    q1 = {}
    for grace_seed in (0, 99):
        m = GraceMachinery(
            cell_graph("mdp", "template"),
            GraceOptions(
                n_bins=3,
                em_iters=10,
                ensemble_k=0,
                interval=False,
                router=False,
                seed=grace_seed,
            ),
            n_actions=2,
            device=DEV,
            gamma=0.9,
        )
        m.fit_from_buffer(FakeSeqBuffer(eps))
        q1[grace_seed] = m.cbn.params.q_u[:, 1].cpu()
    # Both fits recover U in the SAME canonical orientation (index 1 = the
    # high-residual stratum = the true U=1 under the action-gated bonus).
    assert _corr(q1[0], u_true) > 0.9, _corr(q1[0], u_true)
    assert _corr(q1[99], u_true) > 0.9
    assert _corr(q1[0], q1[99]) > 0.99


def test_pomdp_labels_are_seed_stable_and_canonical():
    # 1-D observations (a masked view); latent chain fit by Baum-Welch whose
    # INIT depends on the grace seed — the labeling must not.
    g = torch.Generator().manual_seed(2)
    eps, u_all = [], []
    for _ in range(150):
        u = int(torch.randint(0, 2, (1,), generator=g))
        u_all.append(u)
        ep, z = [], 0
        for t in range(12):
            a = int(torch.rand(1, generator=g) < (0.75 if u else 0.25))
            r = float(z) + float(u * (a == 1))
            obs = torch.tensor([z + 0.3 * float(torch.randn(1, generator=g))])
            z_next = int(torch.rand(1, generator=g) < (0.8 if a else 0.3))
            ep.append(
                {
                    "obs": obs,
                    "actions": torch.tensor(a),
                    "rewards": torch.tensor(r),
                    "next_obs": torch.tensor([float(z_next)]),
                    "dones": torch.tensor(float(t == 11)),
                }
            )
            z = z_next
        eps.append(ep)
    u_true = torch.tensor(u_all, dtype=torch.float32)
    q1 = {}
    for grace_seed in (0, 31):
        m = GraceMachinery(
            cell_graph("pomdp", "template"),
            GraceOptions(
                n_bins=4,
                n_latent=4,
                pomdp_em_iters=8,
                em_iters=4,
                ensemble_k=0,
                interval=False,
                router=False,
                seed=grace_seed,
            ),
            n_actions=2,
            device=DEV,
            gamma=0.9,
        )
        m.fit_from_buffer(FakeSeqBuffer(eps))
        q1[grace_seed] = m.cbn.params.q_u[:, 1].cpu()
    assert _corr(q1[0], u_true) > 0.8, _corr(q1[0], u_true)
    assert _corr(q1[31], u_true) > 0.8, _corr(q1[31], u_true)

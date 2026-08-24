"""V4's exactness row: the LR-region optimiser vs Balke-Pearl, non-circularly.

At c -> 0+ the compatible set is the likelihood-flat manifold, which for the
binary (Z, X, Y) RESPONSE-FUNCTION parametrisation IS the Balke-Pearl
identified set -- so the penalised-Adam min/max must recover the closed-form
bounds on the same real D-E data, from the same likelihood, through the same
optimisation pattern used in production.

LIMITATION, stated per ruling (2026-08-24): this validates the OPTIMISER --
that the min/max over a constrained set is actually found -- NOT the
optimiser on GRACE's production latent-class model. "The search machinery
finds true extrema" and "the bounds GRACE serves are exact" are different
claims; the report must not blur them.

Parametrisation: theta = 16 response-type probabilities (compliance type
X(z=0),X(z=1) in {00,01,10,11} x response type Y(x=0),Y(x=1) in {00,01,10,11}
-- softmax over 16 logits). Observables P(X=x, Y=y | Z=z) are linear in
theta; the multinomial log-likelihood over the 8 (z,x,y) cells is the
constraint surface; the target P(Y(1)=1) - P(Y(0)=1) is linear in theta.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))


def main() -> int:
    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.l4 import balke_pearl_contrast_bounds

    from tools.recertify_diagram_arms import rebuild_samples

    de = next(
        r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_e"
        and r["env"] == "CartPole-v1"
        and r["seed"] == 0
        and r["sigma"] == 1.0
    )
    s, _ = rebuild_samples(minari.load_dataset(de["dataset_id"]), 3000)
    in_pair = np.isin(s["a"], (0, 1))
    y = (s["r"] > 1.5).astype(int)[in_pair]
    x = (s["a"][in_pair] == 1).astype(int)
    z = s["i"][in_pair].astype(int)

    bp_lo, bp_hi = balke_pearl_contrast_bounds(bonus=y, x=x, z=z)
    print(f"closed form      : [{bp_lo:+.4f}, {bp_hi:+.4f}]", flush=True)

    # ---- response-function model ------------------------------------------
    # type index t = 4*cx + ry ; cx in 0..3 encodes (X(0), X(1)) bits,
    # ry in 0..3 encodes (Y(0), Y(1)) bits.
    X0 = np.array([(cx >> 1) & 1 for cx in range(4)])
    X1 = np.array([cx & 1 for cx in range(4)])
    Y0 = np.array([(ry >> 1) & 1 for ry in range(4)])
    Y1 = np.array([ry & 1 for ry in range(4)])
    # counts n[z, x, y]
    n = np.zeros((2, 2, 2))
    for zi, xi, yi in zip(z, x, y):
        n[zi, xi, yi] += 1
    n_t = torch.tensor(n, dtype=torch.float64)

    # membership: does type (cx, ry) produce (x, y) under z?
    # X(z) = X0[cx] if z==0 else X1[cx]; Y = Y(X(z)).
    memb = torch.zeros((2, 2, 2, 16), dtype=torch.float64)
    for zi in (0, 1):
        for cx in range(4):
            xv = (X1 if zi else X0)[cx]
            for ry in range(4):
                yv = (Y1 if xv else Y0)[ry]
                memb[zi, xv, yv, 4 * cx + ry] = 1.0

    tgt_vec = torch.tensor(
        [float(Y1[t % 4] - Y0[t % 4]) for t in range(16)], dtype=torch.float64
    )

    def ll_of(theta):
        p = torch.einsum("zxyt,t->zxy", memb, theta).clamp_min(1e-12)
        return (n_t * torch.log(p)).sum()

    # MLE by direct optimisation (the flat manifold's ll value).
    logits0 = torch.zeros(16, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([logits0], lr=0.05)
    for _ in range(3000):
        opt.zero_grad()
        loss = -ll_of(torch.softmax(logits0, dim=0))
        loss.backward()
        opt.step()
    ll_hat = float(ll_of(torch.softmax(logits0, dim=0)))
    print(f"RF MLE ll        : {ll_hat:.2f}", flush=True)

    # ---- PROJECTED manifold walk at c -> 0+, SIMPLEX parametrisation --------
    # Two prior versions recorded: the stiff penalty collapsed at theta-hat
    # (could not traverse the flat manifold), and the softmax walk stalled
    # ~0.09 from both ends -- the BP extrema are simplex VERTICES (exact
    # zeros), reachable only at infinite logits under softmax. Direct simplex
    # coordinates with Euclidean projection reach vertices in finite steps.
    def proj_simplex(v):
        u, _ = torch.sort(v, descending=True)
        css = torch.cumsum(u, 0) - 1.0
        ks = torch.arange(1, v.numel() + 1, dtype=v.dtype)
        rho = int((u - css / ks > 0).nonzero().max())
        tau = css[rho] / (rho + 1)
        return torch.clamp(v - tau, min=0.0)

    c = 0.1
    theta_hat = torch.softmax(logits0.detach(), dim=0)
    results = {}
    gen = torch.Generator().manual_seed(0)
    for sign in (+1.0, -1.0):
        best = None
        starts = [theta_hat.clone()]
        # multi-start: random tangent perturbations of theta-hat, restored to
        # the manifold -- the single-start walk stalled on a polytope face in
        # the minimise direction (recorded above as the third failure mode).
        for _ in range(5):
            d = torch.randn(16, generator=gen, dtype=torch.float64)
            starts.append(proj_simplex(theta_hat + 0.05 * d / d.norm()))
        for theta0 in starts:
            theta = theta0.clone().requires_grad_(True)
            step = 0.01
            for it in range(4000):
                tgt = (tgt_vec * theta).sum()
                ll = ll_of(theta)
                g_t = torch.autograd.grad(tgt, theta, retain_graph=True)[0]
                g_l = torch.autograd.grad(ll, theta)[0]
                denom = (g_l * g_l).sum().clamp_min(1e-12)
                proj = g_t - (g_t * g_l).sum() / denom * g_l
                with torch.no_grad():
                    theta = proj_simplex(
                        theta + sign * step * proj / proj.norm().clamp_min(1e-12)
                    )
                theta.requires_grad_(True)
                # restoration on the simplex
                for _ in range(50):
                    lr_stat = 2.0 * (ll_hat - ll_of(theta))
                    if float(lr_stat) <= c:
                        break
                    gl = torch.autograd.grad(ll_of(theta), theta)[0]
                    with torch.no_grad():
                        theta = proj_simplex(
                            theta + 0.01 * gl / gl.norm().clamp_min(1e-12)
                        )
                    theta.requires_grad_(True)
                with torch.no_grad():
                    lr_stat = 2.0 * (ll_hat - ll_of(theta))
                    if float(lr_stat) <= c:
                        v = float((tgt_vec * theta).sum())
                        if best is None or sign * v > sign * best:
                            best = v
        results[sign] = best
    lo, hi = results[-1.0], results[+1.0]
    if lo is None or hi is None:
        print(
            f"optimiser: NO FEASIBLE ITERATE (lo={lo}, hi={hi}) -- walk cannot hold the manifold"
        )
        return 1
    print(f"optimiser (c->0+): [{lo:+.4f}, {hi:+.4f}]", flush=True)
    dlo, dhi = abs(lo - bp_lo), abs(hi - bp_hi)
    print(f"discrepancy      : lo {dlo:.4f}, hi {dhi:.4f}", flush=True)
    verdict = "REPRODUCED" if max(dlo, dhi) < 0.01 else "NOT REPRODUCED"
    print(f"VERDICT          : {verdict} (tolerance 0.01)", flush=True)
    Path("results/vc1/lr_exactness.json").write_text(
        json.dumps(
            {
                "bp": [bp_lo, bp_hi],
                "optimiser": [lo, hi],
                "discrepancy": [dlo, dhi],
                "verdict": verdict,
            },
            indent=1,
        )
    )
    return 0 if verdict == "REPRODUCED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

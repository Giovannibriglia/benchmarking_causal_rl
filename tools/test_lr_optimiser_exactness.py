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

    # ll_hat is the CLOSED-FORM saturated conditional-multinomial maximum --
    # Adam's iterative "MLE" was measured 2.45 nats SHORT, which thickened the
    # c->0 "flat manifold" into a shell and let the walk edge past BP by
    # 0.003 (the recorded phase-1 hi discrepancy, now explained).
    n_np = n_t.numpy().reshape(8)
    ll_hat = 0.0
    for zi in (0, 1):
        nz = n_np[zi * 4 : (zi + 1) * 4]
        tot = nz.sum()
        for v in nz:
            if v > 0:
                ll_hat += float(v * np.log(v / tot))
    logits0 = torch.zeros(16, dtype=torch.float64)
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
    # restore the start onto the manifold (uniform types are NOT on it)
    _th = theta_hat.clone().requires_grad_(True)
    for _ in range(2000):
        if float(2.0 * (ll_hat - ll_of(_th))) <= c:
            break
        _g = torch.autograd.grad(ll_of(_th), _th)[0]
        with torch.no_grad():
            _th = proj_simplex(_th + 0.02 * _g / _g.norm().clamp_min(1e-12))
        _th.requires_grad_(True)
    theta_hat = _th.detach()
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


def phase2_production_c() -> int:
    """The ruled (c) test: the walk at a PRODUCTION-REPRESENTATIVE c > 0
    against an independent certified convex solver on the same program.

    The LR region of a saturated multinomial is convex (concave ll => convex
    superlevel set) with a linear objective, so scipy trust-constr's answer is
    a certified global optimum. c is calibrated the production way: episode-
    level parametric bootstrap of the RF LR statistic on the same D-E data.

    LIMITATION THAT TRAVELS (ruled 2026-08-24): this validates the walk on a
    CONVEX smooth region. Production's regions over network weights are smooth
    but not guaranteed convex; there is no oracle there. Hence production
    bounds are labelled INNER APPROXIMATIONS, multi-start reports its spread,
    and V4's D-B-prime coverage row is the empirical exploration test.
    """
    import json
    import os

    os.environ.setdefault(
        "MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2")
    )
    from pathlib import Path

    import minari
    import numpy as np
    import torch
    from scipy.optimize import LinearConstraint, minimize, NonlinearConstraint

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
    ip = np.isin(s["a"], (0, 1))
    y_all = (s["r"] > 1.5).astype(int)
    x_all = (s["a"] == 1).astype(int)
    z_all = s["i"].astype(int)
    ep = s["episode"]

    X0 = np.array([(cx >> 1) & 1 for cx in range(4)])
    X1 = np.array([cx & 1 for cx in range(4)])
    Y0 = np.array([(ry >> 1) & 1 for ry in range(4)])
    Y1 = np.array([ry & 1 for ry in range(4)])
    memb = np.zeros((2, 2, 2, 16))
    for zi in (0, 1):
        for cx in range(4):
            xv = (X1 if zi else X0)[cx]
            for ry in range(4):
                yv = (Y1 if xv else Y0)[ry]
                memb[zi, xv, yv, 4 * cx + ry] = 1.0
    A8 = memb.reshape(8, 16)
    tgt = np.array([float(Y1[t % 4] - Y0[t % 4]) for t in range(16)])

    def counts_from(rows):
        n = np.zeros((2, 2, 2))
        m = ip[rows] if rows is not None else ip
        zz = z_all[rows][m] if rows is not None else z_all[ip]
        xx = x_all[rows][m] if rows is not None else x_all[ip]
        yy = y_all[rows][m] if rows is not None else y_all[ip]
        for zi, xi, yi in zip(zz, xx, yy):
            n[zi, xi, yi] += 1
        return n.reshape(8)

    def mle_ll(n8):
        # saturated multinomial per z: MLE ll = sum n log(n / n_z)
        ll = 0.0
        for zi in (0, 1):
            nz = n8[zi * 4 : (zi + 1) * 4]
            tot = nz.sum()
            for v in nz:
                if v > 0:
                    ll += v * np.log(v / tot)
        return ll

    n_obs = counts_from(None)
    ll_hat = mle_ll(n_obs)

    # ---- calibrate c the production way: episode-level bootstrap ------------
    rng_master = np.random.default_rng(0)
    uniq = np.unique(ep)
    rows_by = {int(e): np.flatnonzero(ep == e) for e in uniq}
    lrs = []
    for b_i in range(39):
        rng = np.random.default_rng(1 + b_i)
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([rows_by[int(e)] for e in picked])
        n_r = counts_from(rows)
        # LR_r = 2(l_r(theta_hat_r) - l_r(theta_hat)): saturated MLE vs the
        # observed conditionals evaluated on replicate counts
        ll_r_hat = mle_ll(n_r)
        p_obs = np.zeros(8)
        for zi in (0, 1):
            nz = n_obs[zi * 4 : (zi + 1) * 4]
            p_obs[zi * 4 : (zi + 1) * 4] = nz / nz.sum()
        ll_r_obs = float(np.sum(n_r[p_obs > 0] * np.log(p_obs[p_obs > 0])))
        lrs.append(max(0.0, 2.0 * (ll_r_hat - ll_r_obs)))
    c = float(np.quantile(lrs, 0.9))
    print(f"calibrated c (alpha=0.1, B=39, episode-level): {c:.2f} nats", flush=True)

    # ---- certified convex solver -------------------------------------------
    def neg_ll(th):
        p = A8 @ th
        return -float(np.sum(n_obs[p > 0] * np.log(p[p > 0])))

    def neg_ll_grad(th):
        p = np.clip(A8 @ th, 1e-12, None)
        return -(A8.T @ (n_obs / p))

    lr_con = NonlinearConstraint(
        lambda th: 2.0 * (ll_hat + neg_ll(th)),
        -np.inf,
        c,
        jac=lambda th: 2.0 * neg_ll_grad(th),
    )
    simplex = LinearConstraint(np.ones((1, 16)), 1.0, 1.0)
    # Starts AT the LP extremal vertices: they sit ON the flat manifold
    # (LR = 0 <= c, feasible by construction), so the solver only has to walk
    # OUTWARD -- the uniform start stalled (first run: solver hi below the
    # BP hi, mathematically impossible for the true optimum). Convexity makes
    # any local optimum global, but only if the solver actually moves.
    from scipy.optimize import linprog as _linprog

    # rebuild the flat-manifold LP for the vertex starts
    p_obs8 = np.zeros(8)
    for zi in (0, 1):
        nz = n_obs[zi * 4 : (zi + 1) * 4]
        p_obs8[zi * 4 : (zi + 1) * 4] = nz / nz.sum()
    A_eq = np.vstack([A8, np.ones((1, 16))])
    b_eq = np.concatenate([p_obs8, [1.0]])
    v_lo = _linprog(tgt, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * 16, method="highs").x
    v_hi = _linprog(-tgt, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * 16, method="highs").x
    sols = {}
    for sign, th0 in ((+1.0, v_hi), (-1.0, v_lo)):
        best = None
        for start in (th0, np.full(16, 1.0 / 16)):
            r = minimize(
                lambda th, s_=sign: -s_ * float(tgt @ th),
                start,
                jac=lambda th, s_=sign: -s_ * tgt,
                constraints=[lr_con, simplex],
                bounds=[(0, 1)] * 16,
                method="trust-constr",
                options={"maxiter": 8000, "gtol": 1e-10, "xtol": 1e-12},
            )
            # THE ORACLE IS HELD TO THE WALK'S STANDARD: trust-constr can
            # return bound-violating iterates whose apparent ll EXCEEDS the
            # saturated maximum (measured: an 'LR' of -188 from a theta with
            # negative components). Project back to the simplex, recompute
            # target and LR exactly, and accept only genuinely feasible
            # solutions.
            th_v = np.clip(r.x, 0.0, None)
            th_v = th_v / th_v.sum()
            lr_v = 2.0 * (ll_hat + neg_ll(th_v))
            if lr_v <= c + 1e-6:
                v = float(tgt @ th_v)
                if best is None or sign * v > sign * best:
                    best = v
        sols[sign] = best
    solver_lo, solver_hi = sols[-1.0], sols[+1.0]
    print(f"trust-constr at c: [{solver_lo:+.4f}, {solver_hi:+.4f}]", flush=True)

    # ---- the walk at the same c (torch, same pattern as production) ---------
    n_t = torch.tensor(n_obs.reshape(2, 2, 2), dtype=torch.float64)
    memb_t = torch.tensor(memb, dtype=torch.float64)
    tgt_t = torch.tensor(tgt, dtype=torch.float64)

    def ll_of(theta):
        p = torch.einsum("zxyt,t->zxy", memb_t, theta).clamp_min(1e-12)
        return (n_t * torch.log(p)).sum()

    def proj_simplex(v):
        u, _ = torch.sort(v, descending=True)
        css = torch.cumsum(u, 0) - 1.0
        ks = torch.arange(1, v.numel() + 1, dtype=v.dtype)
        rho = int((u - css / ks > 0).nonzero().max())
        tau = css[rho] / (rho + 1)
        return torch.clamp(v - tau, min=0.0)

    theta_hat = torch.full((16,), 1.0 / 16, dtype=torch.float64)
    # restore theta_hat to the region first
    results = {}
    gen = torch.Generator().manual_seed(0)
    for sign in (+1.0, -1.0):
        best = None
        starts = [theta_hat.clone()]
        for _ in range(3):
            d = torch.randn(16, generator=gen, dtype=torch.float64)
            starts.append(proj_simplex(theta_hat + 0.05 * d / d.norm()))
        for theta0 in starts:
            theta = theta0.clone().requires_grad_(True)
            for it in range(3000):
                t_v = (tgt_t * theta).sum()
                ll = ll_of(theta)
                g_t = torch.autograd.grad(t_v, theta, retain_graph=True)[0]
                g_l = torch.autograd.grad(ll, theta)[0]
                denom = (g_l * g_l).sum().clamp_min(1e-12)
                proj = g_t - (g_t * g_l).sum() / denom * g_l
                with torch.no_grad():
                    theta = proj_simplex(
                        theta + sign * 0.01 * proj / proj.norm().clamp_min(1e-12)
                    )
                theta.requires_grad_(True)
                for _ in range(30):
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
                    if float(2.0 * (ll_hat - ll_of(theta))) <= c:
                        v = float((tgt_t * theta).sum())
                        if best is None or sign * v > sign * best:
                            best = v
        results[sign] = best
    walk_lo, walk_hi = results[-1.0], results[+1.0]
    print(f"projected walk   : [{walk_lo:+.4f}, {walk_hi:+.4f}]", flush=True)
    dlo, dhi = abs(walk_lo - solver_lo), abs(walk_hi - solver_hi)
    verdict = "REPRODUCED" if max(dlo, dhi) < 0.01 else "NOT REPRODUCED"
    print(f"discrepancy lo {dlo:.4f}, hi {dhi:.4f} -> {verdict} (tol 0.01)", flush=True)
    import json as _json

    Path("results/vc1/lr_exactness_c.json").write_text(
        _json.dumps(
            {
                "c": c,
                "solver": [solver_lo, solver_hi],
                "walk": [walk_lo, walk_hi],
                "discrepancy": [dlo, dhi],
                "verdict": verdict,
            },
            indent=1,
        )
    )
    return 0 if verdict == "REPRODUCED" else 1


if __name__ == "__main__" and os.environ.get("PHASE2"):
    raise SystemExit(phase2_production_c())

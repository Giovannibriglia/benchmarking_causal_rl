"""V4 follow-up -- diagnose the projected walk's width-0 degeneration.

8 of 12 bounds rows returned width 0 with all three starts on an IDENTICAL
endpoint. That is not under-exploration; the walk is not moving at all -- and
the fallback path in ``lr_region_bounds`` (never-feasible -> theta-hat's own
target) returns the SAME value for every start, so "identical endpoints" is
its signature. The same walk moved on the RF parametrisation in the exactness
test, so the cause is specific to the production parametrisation.

Four checks, on one FROZEN row and one MOVING row for contrast:
  A. are the multi-start perturbations actually distinct (pairwise parameter
     distances)?
  B. is the target's gradient non-zero at theta-hat (the query_batch
     non-differentiability trap class)?
  C. does the feasibility gate reject every iterate (accept/reject + LR per
     step, restoration iterations used)?
  D. what is c, and what LR change does a single walk step produce (step size
     vs geometry)?
Plus the hypothesis check H: does the recomputed differentiable log-lik at
theta-hat equal ``fit.final_ll``? If not, LR(clone(theta-hat)) > c before any
step, everything is infeasible from step 0, and the fallback fires on every
start. Diagnosis only -- no fix is proposed here.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

FK = dict(max_iter=30, m_step_budget=400, batch_size=4096)
FIT_SEED = 0
OPT_LR = 1e-3
N_STEPS = 30  # enough to see the pattern; production ran 150

ROWS = (  # (cell, env, seed) -- one frozen, one moving
    ("d_e", "CartPole-v1", 0),  # FROZEN: c=821.36, endpoints identical
    ("d_b_prime", "CartPole-v1", 0),  # MOVED: c=790.20, spread across starts
)


def main() -> int:
    import minari
    import numpy as np
    import torch
    from nbn.utils.batching import pack_parents
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator
    from src.rl.offline.grace.l4 import _observed_ll_differentiable

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recert = json.loads(Path("results/vb_recertification/report.json").read_text())
    v4 = json.loads(Path("results/v4/report.json").read_text())
    c_by_row = {
        (r["cell"], r["env"], r["seed"]): r["meta"]["c_alpha"]
        for r in v4
        if r["row"] == "bounds"
    }
    out_path = Path("results/v4/walk_diagnosis.json")
    report = []

    for cell, env, sd in ROWS:
        did = next(
            r["dataset_id"]
            for r in recert
            if r["cell"] == cell
            and r["env"] == env
            and r["seed"] == sd
            and r["sigma"] == 1.0
        )
        c = float(c_by_row[(cell, env, sd)])
        print(f"\n=== {cell} {env} s{sd}  c={c:.4f}  ({did}) ===", flush=True)
        s, blocks = rebuild_samples(minari.load_dataset(did), 10_000)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(s["episode"], torch.long),
        )
        est = LatentClassEstimator(
            state_dim=state.shape[1],
            n_actions=int(s["a"].max()) + 1,
            proxy_names=(),
            device=device,
            seed=FIT_SEED,
        )
        fit = est.fit(data, init="random", **FK)
        ll_hat = float(fit.final_ll)
        print(f"fit: finished={fit.finished} final_ll={ll_hat:.4f}", flush=True)

        # target identical to run_v4's diff_target
        rng = np.random.default_rng(0)
        idx = rng.choice(state.shape[0], size=min(128, state.shape[0]), replace=False)
        ev = torch.tensor(state[idx], dtype=torch.float32, device=device)
        n = ev.shape[0]
        levels = est._reward_levels

        def target(model, prior):
            total = 0.0
            for k in range(est.u_card):
                means = []
                for a_val in (1, 0):
                    d = {
                        "S": ev,
                        "A": torch.full((n,), float(a_val), device=device),
                        "U": torch.full((n,), float(k), device=device),
                    }
                    pa = pack_parents(d, model.dag.parents("R"))
                    dist = model.mechanisms["R"](pa)
                    if levels is not None:
                        mean = dist.probs @ levels.to(dist.probs.dtype).to(device)
                    else:
                        mean = dist.mean
                    means.append(mean.mean())
                total = total + prior[k] * (means[0] - means[1])
            return total

        # ---- check H first: recomputed ll at theta-hat vs final_ll ---------
        model0 = copy.deepcopy(est.model)
        for p_ in model0.parameters():
            p_.requires_grad_(True)
        prior_logits0 = torch.nn.Parameter(
            torch.log(fit.prior.detach().clamp_min(1e-8)).clone()
        )
        ll0 = _observed_ll_differentiable(model0, prior_logits0, est, data)
        lr0 = 2.0 * (ll_hat - float(ll0.detach()))
        print(
            f"[H] final_ll={ll_hat:.4f}  recomputed ll(theta-hat)={float(ll0.detach()):.4f}  "
            f"LR(theta-hat)={lr0:.4f}  c={c:.4f}  feasible_at_start={lr0 <= c}",
            flush=True,
        )
        row = {
            "cell": cell,
            "env": env,
            "seed": sd,
            "c": c,
            "final_ll": ll_hat,
            "recomputed_ll_at_theta_hat": float(ll0.detach()),
            "lr_at_theta_hat": lr0,
            "starts": [],
        }
        del ll0

        # ---- instrumented walk over the 3 production starts (hi sign) ------
        flat0 = None
        for k in range(3):
            model_c = copy.deepcopy(est.model)
            if k:  # seed ONCE per start, exactly as extremum_multi does
                torch.manual_seed(FIT_SEED * 1000 + k)
            for p_ in model_c.parameters():
                p_.requires_grad_(True)
                if k:
                    with torch.no_grad():
                        p_ += (
                            (0.01 * k) * torch.randn_like(p_) * p_.abs().clamp_min(1e-3)
                        )
            prior_logits = torch.nn.Parameter(
                torch.log(fit.prior.detach().clamp_min(1e-8)).clone()
            )
            params = [p_ for p_ in model_c.parameters() if p_.requires_grad]
            params.append(prior_logits)
            with torch.no_grad():
                flat = torch.cat([p_.reshape(-1) for p_ in params])
            if flat0 is None:
                flat0 = flat
            dist0 = float((flat - flat0).norm())
            # [A] start distinctness

            def flat_grad(scalar, retain=False):
                gs = torch.autograd.grad(
                    scalar, params, retain_graph=retain, allow_unused=True
                )
                return [
                    torch.zeros_like(q) if g is None else g for g, q in zip(gs, params)
                ]

            def dot(a, b):
                return sum((x * y).sum() for x, y in zip(a, b))

            def lr_now():
                ll = _observed_ll_differentiable(model_c, prior_logits, est, data)
                v = 2.0 * (ll_hat - float(ll.detach()))
                del ll
                return v

            lr_start = lr_now()
            prior = torch.softmax(prior_logits, dim=0)
            tgt = target(model_c, prior)
            tgt_start = float(tgt.detach())
            g_t = flat_grad(tgt, retain=False)
            gt_norm = float(torch.sqrt(dot(g_t, g_t)).detach())
            srec = {
                "start": k,
                "param_dist_from_start0": dist0,
                "lr_at_start": lr_start,
                "target_at_start": tgt_start,
                "g_target_norm": gt_norm,
                "steps": [],
            }
            print(
                f"[A/B] start {k}: |theta-theta0|={dist0:.4f} LR={lr_start:.4f} "
                f"tgt={tgt_start:.4f} |g_t|={gt_norm:.3e}",
                flush=True,
            )

            best = None
            for step in range(N_STEPS):
                prior = torch.softmax(prior_logits, dim=0)
                tgt = target(model_c, prior)
                ll = _observed_ll_differentiable(model_c, prior_logits, est, data)
                lr_pre = 2.0 * (ll_hat - float(ll.detach()))
                g_t = flat_grad(tgt, retain=True)
                g_l = flat_grad(ll)
                denom = dot(g_l, g_l).clamp_min(1e-12)
                coef = dot(g_t, g_l) / denom
                gl_norm = float(torch.sqrt(denom).detach())
                with torch.no_grad():
                    norm = torch.sqrt(
                        sum(((a - coef * b) ** 2).sum() for a, b in zip(g_t, g_l))
                    ).clamp_min(1e-12)
                    for q, a, b_ in zip(params, g_t, g_l):
                        q += 1.0 * OPT_LR * (a - coef * b_) / norm
                proj_norm = float(norm)
                lr_after_step = lr_now()  # [D] LR change from ONE tangent step
                n_rest = 0
                for _ in range(20):
                    ll = _observed_ll_differentiable(model_c, prior_logits, est, data)
                    if float(2.0 * (ll_hat - float(ll.detach()))) <= c:
                        del ll
                        break
                    g_l = flat_grad(ll)
                    with torch.no_grad():
                        n2 = torch.sqrt(dot(g_l, g_l)).clamp_min(1e-12)
                        for q, b_ in zip(params, g_l):
                            q += OPT_LR * b_ / n2
                    n_rest += 1
                lr_final = lr_now()
                feasible = lr_final <= c
                with torch.no_grad():
                    prior = torch.softmax(prior_logits, dim=0)
                    tgt_v = float(target(model_c, prior))
                if feasible and (best is None or tgt_v > best):
                    best = tgt_v
                srec["steps"].append(
                    {
                        "step": step,
                        "lr_pre": lr_pre,
                        "lr_after_tangent_step": lr_after_step,
                        "restoration_iters": n_rest,
                        "lr_final": lr_final,
                        "feasible": feasible,
                        "target": tgt_v,
                        "g_l_norm": gl_norm,
                        "proj_grad_norm": proj_norm,
                    }
                )
                if step < 5 or step % 10 == 0:
                    print(
                        f"  [C/D] step {step}: LR {lr_pre:.2f} -> {lr_after_step:.2f} "
                        f"(tangent) -> {lr_final:.2f} after {n_rest} restorations "
                        f"feasible={feasible} tgt={tgt_v:.4f} |g_l|={gl_norm:.3e} "
                        f"|proj|={float(proj_norm):.3e}",
                        flush=True,
                    )
            srec["best_feasible_target"] = best
            n_feas = sum(1 for x in srec["steps"] if x["feasible"])
            print(
                f"  start {k}: {n_feas}/{N_STEPS} feasible iterates, " f"best={best}",
                flush=True,
            )
            row["starts"].append(srec)
        report.append(row)
        out_path.write_text(json.dumps(report, indent=1))
    print("\nWALK DIAGNOSIS COMPLETE ->", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

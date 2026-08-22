"""Are D-D's proxies LOAD-BEARING, or decorative?

The concern this answers. `R`'s episode mean separates `U` at AUC 1.0000, so
`U` is very nearly *observed* in D-D. If an estimator can recover it from
`(S, A, R)` alone, then conditioning on the recovered `U` closes the back door
without the proxies doing any work — D-D would be a **back-door cell wearing
proximal clothing**, and V-C would report success on proximal machinery it never
exercised. That threatens the cell's purpose, not merely a parameter choice.

**THE INITIALISATION HAS TO BE MATCHED, or the ablation measures the wrong
thing.** ``fit(init="proxy")`` -- the production default -- seeds EM by bucketing
the first proxy's episode mean at the median. Since ``AUC(Z | U) = 0.98``, that
is a ~98%-correct warm start. An ablated arm falls back to a RANDOM init, so a
naive with/without comparison conflates *information available to the
likelihood* with *quality of the starting point*, and would report decorative
proxies as load-bearing. Measured on CartPole s0 exactly that way: recovery
0.995 against 0.598, which is not an information result.

So the arms are, all at the SAME budget:

  * **with**    — `proxy_names = ("Z", "W")`, RANDOM init;
  * **without** — `proxy_names = ()`,          RANDOM init;
  * **with_proxyinit** — the production setting, reported for context only, and
    never used for the verdict.

Several fit seeds per arm, because a single random init can land in a confident
local optimum and EM will not leave it: the ablated arm's separability was
**0.98 with recovery 0.60** -- maximally confident and wrong -- which is the
signature of a local optimum, not of an underpowered fit.

Reported per fit, because latent recovery alone would not settle it:

  * **recovery** — label-swap-invariant accuracy of the hard assignment against
    the logged `U` (ground truth; S4 — the estimator is validated against the
    generator, never the reverse);
  * **separability** — mean max-responsibility, 0.5 is chance at K = 2. A fit can
    recover the labels while barely separating, and the two failure modes need
    different remedies;
  * **the value-level consequence** — the interventional contrast
    `E[R|do(a=a_bad), s] − E[R|do(a≠a_bad), s]`, which is what any downstream
    number actually rests on. Recovery that does not move the estimand is not a
    difference that matters, and an estimand that moves without a recovery
    change would say the proxies act through a channel other than the latent.

C3 is respected throughout: every number carries `monotone` / `converged` /
`separability`, so a fit that never ascended cannot be read as a clean result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recert", default="results/vb_recertification/report.json")
    ap.add_argument("--out", default="results/vb_recertification/dd_ablation.json")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument(
        "--m-step-budget",
        type=int,
        default=None,
        help="fixed gradient-step budget per M-step (estimator's O(steps) "
        "path). None = epoch-based O(n*epochs) -- which is what made the "
        "2026-08-19 run 17 h; the fixed-step path is ~50x cheaper at Acrobot "
        "scale (measured lever x148 at n=475k) and is REQUIRED for any sweep "
        "or converged value-level re-run.",
    )
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument(
        "--skip-proxyinit",
        action="store_true",
        help="drop the context-only production-init arm; it never enters the "
        "verdict, so it is the first thing to cut when the budget is tight",
    )
    ap.add_argument("--envs", nargs="+", default=["CartPole-v1", "Acrobot-v1"])
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument(
        "--cell",
        default="d_d",
        help="report cell name -- 'd_d' for the frozen arms, 'd_d_sweep_dNNN' "
        "for the 2026-08-21 revision's gate-separation sweep points.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to cuda when available. The MDN reward and "
        "proxy mechanisms are the whole cost here, so this is worth having.",
    )
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    def accuracy(hard, truth):
        """Label-swap invariant: a mixture is identified only up to permutation."""
        h = np.asarray(hard, dtype=np.int64)
        t = np.asarray(truth, dtype=np.int64)
        return float(max((h == t).mean(), (h != t).mean()))

    rows = [
        r
        for r in json.loads(Path(args.recert).read_text())
        if r["cell"] == args.cell
        and r["env"] in args.envs
        and r["seed"] in args.seeds
        and r["sigma"] == args.sigma
    ]
    out = []
    for r in sorted(rows, key=lambda x: (x["env"], x["seed"])):
        ds = minari.load_dataset(r["dataset_id"])
        s, obs_blocks = rebuild_samples(ds, args.episodes)
        # obs_blocks are T+1 rows per episode; the transition's SOURCE state is
        # the first T, matching the action/reward alignment everywhere else.
        state = np.concatenate([b.observations[:-1] for b in obs_blocks], axis=0)
        ep = s["episode"]
        u_ep = np.array([s["u"][ep == e][0] for e in np.unique(ep)], dtype=np.int64)

        def make_data(with_proxies: bool) -> EpisodeData:
            def t(x, dtype=torch.float32):
                return torch.tensor(x, dtype=dtype, device=device)

            kw = {}
            if with_proxies:
                # The with-arm uses EVERY proxy the arm declares: {Z, W} on the
                # frozen two-proxy datasets, {Z, W, V} on the revision's -- a
                # with-arm whose proxy count varied along the sweep would
                # confound the axis with the thing being swept.
                kw["proxy"] = {"Z": t(s["z"]), "W": t(s["w"])}
                if s["v"].size:
                    kw["proxy"]["V"] = t(s["v"])
            return EpisodeData(
                state=t(state),
                action=t(s["a"], torch.long),
                reward=t(s["r"]),
                episode_ids=t(ep, torch.long),
                **kw,
            )

        rec = {
            "env": r["env"],
            "seed": r["seed"],
            "sigma": r["sigma"],
            "n_episodes": int(u_ep.size),
            "n_transitions": int(ep.size),
        }
        # A fixed evaluation grid, shared by both arms, so the value-level
        # contrast is compared on identical inputs.
        rng = np.random.default_rng(0)
        idx = rng.choice(state.shape[0], size=min(256, state.shape[0]), replace=False)
        eval_states = torch.tensor(state[idx], dtype=torch.float32, device=device)

        arms = [
            ("with", True, "random"),
            ("without", False, "random"),
            # Production setting, for context only. Excluded from the verdict:
            # its warm start is ~98% correct, so it cannot separate information
            # from initialisation.
            ("with_proxyinit", True, "proxy"),
        ]
        if args.skip_proxyinit:
            arms = [a for a in arms if a[0] != "with_proxyinit"]
        for label, with_p, init in arms:
            per_seed = []
            for fs in args.fit_seeds:
                data = make_data(with_p)
                est = LatentClassEstimator(
                    state_dim=state.shape[1],
                    n_actions=int(s["a"].max()) + 1,
                    proxy_names=(
                        (("Z", "W", "V") if s["v"].size else ("Z", "W"))
                        if with_p
                        else ()
                    ),
                    device=device,
                    seed=fs,
                )
                budget_kw = (
                    {"m_step_budget": args.m_step_budget, "batch_size": args.batch_size}
                    if args.m_step_budget
                    else {}
                )
                fit = est.fit(
                    data,
                    max_iter=args.max_iter,
                    epochs=args.epochs,
                    init=init,
                    **budget_kw,
                )
                hard_ep = fit.hard_assignment().cpu().numpy()
                bad = est.interventional_sweep(
                    eval_states, [1] * eval_states.shape[0], fit
                )
                good = est.interventional_sweep(
                    eval_states, [0] * eval_states.shape[0], fit
                )
                print(
                    f"    {r['env']} s{r['seed']} {label}/{init} fit_seed={fs} "
                    f"ll={float(fit.final_ll):.1f} conv={fit.converged}",
                    flush=True,
                )
                per_seed.append(
                    {
                        "fit_seed": fs,
                        "recovery": accuracy(hard_ep, u_ep),
                        "separability": float(fit.separability()),
                        "monotone": bool(fit.monotone),
                        "converged": bool(fit.converged),
                        "final_ll": float(fit.final_ll),
                        "do_contrast_a_bad_minus_other": float(
                            (bad.value - good.value).mean().cpu()
                        ),
                    }
                )
            # The best-of-seeds fit by LIKELIHOOD, which is the only criterion
            # available without the ground truth -- picking by recovery would
            # use the answer to choose the method.
            best = max(per_seed, key=lambda d: d["final_ll"])
            rec[label] = {
                **best,
                "per_seed": per_seed,
                "recovery_mean": float(np.mean([d["recovery"] for d in per_seed])),
                "recovery_max": float(max(d["recovery"] for d in per_seed)),
            }
            print(
                f"  {r['env']:<12} s{r['seed']} {label:<15} init={init:<7} "
                f"recovery(best-LL)={best['recovery']:.4f} "
                f"mean={rec[label]['recovery_mean']:.4f} "
                f"max={rec[label]['recovery_max']:.4f} "
                f"sep={best['separability']:.4f} "
                f"do={best['do_contrast_a_bad_minus_other']:+.4f} "
                f"conv={best['converged']}",
                flush=True,
            )
        rec["delta_recovery"] = rec["with"]["recovery"] - rec["without"]["recovery"]
        rec["delta_separability"] = (
            rec["with"]["separability"] - rec["without"]["separability"]
        )
        rec["delta_contrast"] = (
            rec["with"]["do_contrast_a_bad_minus_other"]
            - rec["without"]["do_contrast_a_bad_minus_other"]
        )
        out.append(rec)
        Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n=== summary ===")
    for k in ("delta_recovery", "delta_separability", "delta_contrast"):
        v = [o[k] for o in out]
        print(
            f"  {k:<22} mean {np.mean(v):+.4f}   range [{min(v):+.4f}, {max(v):+.4f}]"
        )
    print(
        "\nA delta near zero on ALL THREE means the proxies are decorative: the "
        "estimator gets the same latent, the same confidence and the same "
        "estimand from (S, A, R) alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

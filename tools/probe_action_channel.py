"""Pre-check for the D-D revision: does the ACTION channel alone identify U?

At sigma > 0 the behaviour policy mixes by U, so the (S, A) sequence is itself
a U-proxy whose informativeness scales with episode length. If it alone
identifies U at production sigma, then NO amount of R-weakening makes the
proxies load-bearing -- the compensated-gate-separation sweep could not reach
a regime where they matter, and the revision needs a different lever. Measured
before anything is built, because the alternative is discovering it after
regeneration and a 54-fit campaign.

The R channel is killed DATA-SIDE, no regeneration: the reward column is
permuted among rows sharing the same action value. That preserves P(r | a)
exactly (support, rates, action-dependence) and destroys the dependence on U
(and on S), so the fitted per-class reward mechanisms carry no discrimination.
R keeps its two-valued support, so type resolution stays categorical.

Arms, all random-init (the ablation's matched-init discipline):
  * action_only      -- proxies withheld, R dead: the question itself.
  * weak_end_preview -- proxies present,  R dead: the sweep's d -> 0 endpoint,
    approximated a week early. If THIS fails too, the sweep has no working
    end and the entry must be rethought regardless of the first arm.
Comparator (not re-run): the 2026-08-19 ablation's without arm on unmodified
data -- recovery 0.995-1.000 -- is the "R alive" reference point.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["CartPole-v1", "Acrobot-v1"])
    ap.add_argument("--dataset-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--m-step-budget", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results/cost/action_channel_probe.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recert = {
        (r["env"], r["seed"]): r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d" and r["sigma"] == args.sigma
    }

    out = []
    for env in args.envs:
        for sd in args.dataset_seeds:
            s, blocks = rebuild_samples(
                minari.load_dataset(recert[(env, sd)]["dataset_id"]), args.episodes
            )
            state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
            ep = s["episode"]
            u_ep = np.array([s["u"][ep == e][0] for e in np.unique(ep)], dtype=np.int64)

            # Kill R's U-dependence: permute rewards WITHIN action strata.
            r_dead = s["r"].copy()
            rng = np.random.default_rng(12345)
            for a_val in np.unique(s["a"]):
                m = s["a"] == a_val
                r_dead[m] = rng.permutation(r_dead[m])
            # The operation must not have changed P(r | a):
            for a_val in np.unique(s["a"]):
                m = s["a"] == a_val
                assert np.isclose(r_dead[m].mean(), s["r"][m].mean())
                assert set(np.unique(r_dead[m])) == set(np.unique(s["r"][m]))
            # THE KILL METRIC IS CONDITIONAL (S2): P(r | a) flat across U.
            # The episode-mean AUC below is kept as the MARGINAL SHADOW -- it
            # stays high after the kill because mean-R tracks the episode's
            # ACTION MIX, i.e. the very channel under test wearing R's
            # clothes. Reading it as "the kill failed" is the marginal
            # mistake S2 names.
            hi = s["r"].max()
            cond_gap = 0.0
            for a_val in np.unique(s["a"]):
                m = s["a"] == a_val
                if not (m & (s["u"] == 1)).any() or not (m & (s["u"] == 0)).any():
                    continue
                g = abs(
                    float((r_dead[m & (s["u"] == 1)] == hi).mean())
                    - float((r_dead[m & (s["u"] == 0)] == hi).mean())
                )
                cond_gap = max(cond_gap, g)
            ep_ids = np.unique(ep)

            def _auc(rvec):
                means = np.array([rvec[ep == e].mean() for e in ep_ids])
                pos, neg = means[u_ep == 1], means[u_ep == 0]
                if not len(pos) or not len(neg):
                    return float("nan")
                gt = (pos[:, None] > neg[None, :]).mean()
                eq = (pos[:, None] == neg[None, :]).mean()
                return float(gt + 0.5 * eq)

            auc_alive, auc_dead = _auc(s["r"]), _auc(r_dead)

            def t(x, dtype=torch.float32):
                return torch.tensor(x, dtype=dtype, device=device)

            for label, with_p in (("action_only", False), ("weak_end_preview", True)):
                data = EpisodeData(
                    state=t(state),
                    action=t(s["a"], torch.long),
                    reward=t(r_dead),
                    episode_ids=t(ep, torch.long),
                    proxy=({"Z": t(s["z"]), "W": t(s["w"])} if with_p else {}),
                )
                per_seed = []
                for fs in args.fit_seeds:
                    est = LatentClassEstimator(
                        state_dim=state.shape[1],
                        n_actions=int(s["a"].max()) + 1,
                        proxy_names=("Z", "W") if with_p else (),
                        device=device,
                        seed=fs,
                    )
                    fit = est.fit(
                        data,
                        max_iter=args.max_iter,
                        init="random",
                        m_step_budget=args.m_step_budget,
                        batch_size=args.batch_size,
                        consolidate=False,
                    )
                    h = fit.hard_assignment().cpu().numpy()
                    per_seed.append(
                        {
                            "fit_seed": fs,
                            "recovery": float(
                                max((h == u_ep).mean(), (h != u_ep).mean())
                            ),
                            "final_ll": float(fit.final_ll),
                            "converged": bool(fit.converged),
                            "n_iter": fit.n_iter,
                        }
                    )
                    print(
                        f"    {env} s{sd} {label} fs={fs} "
                        f"rec={per_seed[-1]['recovery']:.4f} "
                        f"ll={per_seed[-1]['final_ll']:.1f} "
                        f"conv={per_seed[-1]['converged']}",
                        flush=True,
                    )
                best = max(per_seed, key=lambda r: r["final_ll"])
                rec = {
                    "env": env,
                    "dataset_seed": sd,
                    "arm": label,
                    "auc_R_alive": auc_alive,
                    "auc_R_dead_marginal_shadow": auc_dead,
                    "max_conditional_bonus_gap_dead": cond_gap,
                    "recovery_best_ll": best["recovery"],
                    "recovery_mean": float(np.mean([r["recovery"] for r in per_seed])),
                    "fits": per_seed,
                }
                out.append(rec)
                print(
                    f"  {env:<12} s{sd} {label:<17} "
                    f"cond-kill max|gap|={cond_gap:.4f}  "
                    f"AUC(R) {auc_alive:.4f}->{auc_dead:.4f} (marginal shadow)  "
                    f"recovery(best-LL)={rec['recovery_best_ll']:.4f} "
                    f"mean={rec['recovery_mean']:.4f}",
                    flush=True,
                )
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n  READING: action_only HIGH -> the action channel alone identifies U;")
    print("  the sweep cannot reach a proxies-load-bearing regime at this sigma")
    print("  and the revision needs a different lever. action_only LOW and")
    print("  weak_end_preview HIGH -> the sweep is viable and its weak end works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

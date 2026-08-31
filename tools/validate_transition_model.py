"""Q2-A step 1 -- transition-model validation, its own reported result.

q1 never needed P(S' | S, A); q2's fitted iteration does, and that mechanism
has never been validated (the profile only ever measured scoring cost). This
tool measures held-out predictive accuracy -- one-step and multi-step rollout
error against the true environment -- BEFORE any q2 number exists, so a poor
transition model is found here and not mis-attributed to the causal layer.

Load-bearing decisions, each argued rather than assumed:

* **Parents are (S, A) -- no U.** The declared diagram is the only assumption
  (A1), and catalogue fact 3 says no wired cell has a U -> S_next edge; q2's
  point-ID argument RESTS on unconfounded dynamics. The pre-registration's
  "P(S'|S,A,U)" phrasing resolves to the declared parent set, which is (S, A).
* **The truth is the logged trajectory.** CartPole and Acrobot dynamics are
  deterministic given (s, a), so a held-out episode's logged states ARE the
  true environment's response to its logged actions. One-step error against
  logged s' and open-loop multi-step rollouts along logged action sequences
  therefore measure against ground truth exactly, with no env re-stepping and
  no policy reconstruction. (A closed-loop rollout under the target policy is
  the d_a_null machinery check's business -- there the fitted iteration itself
  is under test, not the mechanism.)
* **Both candidate mechanisms are fitted** -- LinearGaussian and the
  estimator's continuous default MDN(3, (64,64)) -- because q2's mechanism
  choice is exactly what this result should inform.
* **Degeneracy is measured, not assumed.** Deterministic dynamics have zero
  conditional noise, the same finite-support property that made MDN-R drive
  its scale onto ``min_scale`` (the discrete-R lesson). If the same happens
  here, predictive means can be fine while log-densities pin at the ceiling --
  which matters the moment any likelihood-bearing layer reads this mechanism.
  The floor share is reported per mechanism; no magnitude cutoff (A2).
* **The split is at EPISODE granularity** (C1's rule): states within an
  episode are serially dependent, and a transition-level split would leak.
* Deterministic kernels on (the repo default since 9625b85); the MDN fit gets
  a FIXED STEP budget (S11's lesson: cut data, never the optimisation budget
  -- here the full budget on full data, since this is a reported result).

Scope: the Q2-A cells' datasets -- d_a_null (machinery), d_d d100 (the d=1.0
cell) and d005 (the weak end), both environments, seeds 0-2. Dynamics are
per-environment but the STATE DISTRIBUTION is per-dataset (policy and gate
differ), and q2 fits per dataset, so validation is per dataset.

Resume: rows already in the report are skipped on relaunch (determinism makes
a redone row identical).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

SPLIT_SEED = 0
TEST_FRACTION = 0.2
MDN_STEP_BUDGET = 4000  # gradient steps, converted to epochs per dataset size
BATCH = 4096
HORIZONS = (1, 2, 5, 10, 20, 50, 100, 200, 499)
MAX_ROLLOUT_EPISODES = 200  # test episodes used for the rollout curves


def main() -> int:
    import minari
    import numpy as np
    import torch
    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
    from nbn.mechanisms.parametric.mdn import MDNMechanism

    from tools.recertify_diagram_arms import rebuild_samples

    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path("results/q2a_transition/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: list = []
    if out_path.exists():
        out = json.loads(out_path.read_text())
        print(f"  resuming: {len(out)} rows already done", flush=True)
    done = {(r["cell"], r["env"], r["seed"]) for r in out}

    def save():
        out_path.write_text(json.dumps(out, indent=1))

    # ---- jobs: d_a_null + d_d sweep endpoints ------------------------------
    recert = json.loads(Path("results/vb_recertification/report.json").read_text())
    gen = {
        (r["cell"], r["env"], r["seed"]): r["dataset_id"]
        for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text())
    }
    jobs = []
    for r in recert:
        if r["cell"] == "d_a_null" and r["seed"] in (0, 1, 2):
            jobs.append(("d_a_null", r["env"], r["seed"], r["dataset_id"]))
    for tag in ("d100", "d005"):
        for env in ("CartPole-v1", "Acrobot-v1"):
            for sd in (0, 1, 2):
                jobs.append((tag, env, sd, gen[(f"d_d_sweep_{tag}", env, sd)]))

    for cell, env, sd, did in jobs:
        if (cell, env, sd) in done:
            continue
        t0 = time.time()
        s, blocks = rebuild_samples(minari.load_dataset(did), 10_000)
        obs = [np.asarray(b.observations, dtype=np.float64) for b in blocks]
        ep = s["episode"]
        acts = [s["a"][ep == k] for k in range(len(obs))]
        for o, a in zip(obs, acts):
            assert o.shape[0] == a.shape[0] + 1, "obs must be one longer than actions"

        # ---- episode-granularity split (C1) --------------------------------
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(obs))
        n_test = max(1, int(round(TEST_FRACTION * len(obs))))
        test_eps, train_eps = perm[:n_test], perm[n_test:]

        def flat(idx):
            S = np.concatenate([obs[k][:-1] for k in idx], axis=0)
            SN = np.concatenate([obs[k][1:] for k in idx], axis=0)
            A = np.concatenate([acts[k] for k in idx], axis=0)
            return (
                torch.tensor(S, dtype=torch.float32, device=device),
                torch.tensor(A, dtype=torch.float32, device=device).reshape(-1, 1),
                torch.tensor(SN, dtype=torch.float32, device=device),
            )

        S_tr, A_tr, SN_tr = flat(train_eps)
        S_te, A_te, SN_te = flat(test_eps)
        pa_tr = torch.cat([S_tr, A_tr], dim=1)
        pa_te = torch.cat([S_te, A_te], dim=1)
        # Normalisation reference: per-dim std of s' on TRAIN, fixed. A raw
        # RMSE would let large-scale dims (Acrobot velocities) dominate.
        ref_std = SN_tr.std(dim=0).clamp_min(1e-8)

        rec = {
            "cell": cell,
            "env": env,
            "seed": sd,
            "dataset_id": did,
            "n_episodes": len(obs),
            "n_train_eps": int(train_eps.size),
            "n_test_eps": int(test_eps.size),
            "n_train_transitions": int(S_tr.shape[0]),
            "n_test_transitions": int(S_te.shape[0]),
            "state_dim": int(S_tr.shape[1]),
            "mechanisms": {},
        }

        for name in ("linear_gaussian", "mdn", "mdn1"):
            torch.manual_seed(SPLIT_SEED)
            t_fit = time.time()
            if name == "linear_gaussian":
                mech = LinearGaussianMechanism()
                mech.fit_local(SN_tr, pa_tr)
            else:
                # ``mdn1`` -- ONE component: the missing middle for
                # deterministic dynamics. LG is linear (hence Acrobot's
                # 0.2-0.4 one-step error); a 3-component mixture is MORE than
                # deterministic dynamics need (hence the mixture pathology and
                # the fitted-iteration divergence on Acrobot s2). A single
                # component is a flexible neural mean with a learned scale and
                # nothing to destabilise.
                #
                # THE FLOOR-PINNING CAVEAT DOES NOT APPLY HERE, deliberately:
                # the transition mechanism is NOT likelihood-bearing --
                # ``_episode_log_liks`` scores A, R and the proxies, never S --
                # so if the scale collapses onto ``min_scale`` on deterministic
                # dynamics that is the CORRECT answer, and the near-
                # deterministic sampling it produces is exactly what the q2
                # backups want. Do not "fix" it.
                mech = MDNMechanism(
                    num_components=1 if name == "mdn1" else 3, hidden=(64, 64)
                )
                steps_per_epoch = max(1, int(np.ceil(S_tr.shape[0] / BATCH)))
                epochs = max(1, round(MDN_STEP_BUDGET / steps_per_epoch))
                mech.fit_local(
                    SN_tr,
                    pa_tr,
                    epochs=epochs,
                    lr=1e-3,
                    batch_size=BATCH,
                    consolidate=False,
                )
            fit_seconds = time.time() - t_fit

            with torch.no_grad():
                # ---- one-step, held out ------------------------------------
                dist = mech(pa_te)
                pred = dist.mean
                err = (pred - SN_te) / ref_std
                per_dim = torch.sqrt((err**2).mean(dim=0))
                logp = mech.log_prob(SN_te, pa_te)
                # ---- scale-floor degeneracy (the discrete-R lesson) --------
                if name.startswith("mdn"):
                    _, _, scale = mech._params_from_parents(pa_te, (pa_te.shape[0],))
                else:
                    scale = mech._scale().reshape(1, -1)
                floor = float(mech.min_scale) * 1.001
                m = {
                    "fit_seconds": round(fit_seconds, 2),
                    "one_step_rmse_norm_per_dim": [round(float(v), 6) for v in per_dim],
                    "one_step_rmse_norm_mean": round(float(per_dim.mean()), 6),
                    "heldout_logp_mean": round(float(logp.mean()), 4),
                    "heldout_logp_q10": round(float(logp.quantile(0.1)), 4),
                    "scale_floor_share": round(
                        float((scale <= floor).float().mean()), 4
                    ),
                    "scale_min": float(scale.min()),
                    "scale_median": float(scale.median()),
                }

                # ---- open-loop multi-step rollouts on test episodes --------
                # Roll the model along each held-out episode's LOGGED action
                # sequence from its logged start state; the logged states are
                # the deterministic truth. Two variants: MEAN (isolates bias)
                # and SAMPLED (what q2's sample(do=)-based backup does).
                roll_idx = test_eps[:MAX_ROLLOUT_EPISODES]
                lens = np.array([acts[k].shape[0] for k in roll_idx])
                tmax = int(lens.max())
                E = len(roll_idx)
                D = S_tr.shape[1]
                obs_pad = torch.zeros(E, tmax + 1, D, device=device)
                act_pad = torch.zeros(E, tmax, device=device)
                for j, k in enumerate(roll_idx):
                    T = lens[j]
                    obs_pad[j, : T + 1] = torch.tensor(
                        obs[k], dtype=torch.float32, device=device
                    )
                    act_pad[j, :T] = torch.tensor(
                        acts[k], dtype=torch.float32, device=device
                    )
                lens_t = torch.tensor(lens, device=device)
                curves = {}
                for variant in ("mean", "sampled"):
                    torch.manual_seed(SPLIT_SEED)
                    s_hat = obs_pad[:, 0].clone()
                    err_sum = {h: 0.0 for h in HORIZONS}
                    err_cnt = {h: 0 for h in HORIZONS}
                    for t in range(tmax):
                        active = lens_t > t
                        pa = torch.cat(
                            [s_hat[active], act_pad[active, t : t + 1]], dim=1
                        )
                        if variant == "mean":
                            nxt = mech(pa).mean
                        else:
                            nxt = mech.sample(pa, 1).reshape(-1, D)
                        s_hat[active] = nxt
                        h = t + 1
                        if h in err_sum:
                            e = ((s_hat[active] - obs_pad[active, h]) / ref_std).norm(
                                dim=1
                            ) / np.sqrt(D)
                            err_sum[h] += float(e.sum())
                            err_cnt[h] += int(e.numel())
                    curves[variant] = {
                        str(h): round(err_sum[h] / err_cnt[h], 6)
                        for h in HORIZONS
                        if err_cnt[h] > 0
                    }
                m["rollout_norm_err"] = curves
            rec["mechanisms"][name] = m

        rec["seconds"] = round(time.time() - t0, 1)
        out.append(rec)
        save()
        lg = rec["mechanisms"]["linear_gaussian"]
        md = rec["mechanisms"]["mdn"]
        print(
            f"  {cell:<9} {env:<12} s{sd} "
            f"LG 1step={lg['one_step_rmse_norm_mean']:.4f} "
            f"MDN 1step={md['one_step_rmse_norm_mean']:.4f} "
            f"MDN floor={md['scale_floor_share']:.0%} "
            f"({rec['seconds']:.0f}s)",
            flush=True,
        )

    print("TRANSITION VALIDATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

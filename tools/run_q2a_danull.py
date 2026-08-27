"""Q2-A step 2 -- the d_a_null MACHINERY check: fitted iteration on the do-channel.

Scope (prereg): does the sampling-based fitted value iteration reproduce the
MC return-to-go on the null cell, and WHAT DOES IT COST -- measured before any
substantive cell, because "many sample(do=) calls per backup; if prohibitive,
that changes the design and is better known at cell one". This is the
machinery check, not the full no-harm gate (that belongs to the assembled
Q2-A block).

Decisions, each argued:

* **Target policy = the generator's dqn-medium GREEDY** (``act(...,
  deterministic=True)``), rebuilt reproducibly with the cell spec's own
  budgets (``build_generator_agent`` seeds globally per (env, seed), so the
  rebuild IS the collection agent; ``ghash`` recorded on the row). The
  behaviour policy wrapped it in epsilon = 0.5; the target strips epsilon,
  per the pre-registration ("the generator's greedy policy, fixed and
  declared"). A fresh run_dir is used so the V-B generation artifacts are
  never touched.
* **Anchor: deterministic env + deterministic policy => the discounted RTG
  along an on-policy rollout IS V^pi at every visited state** -- no MC
  averaging needed. N_EVAL rollouts from distinct reset seeds; gamma = 0.99,
  the benchmark's own (iql/cql default), so the anchor is consistent with the
  offline algorithms the critic will sit beside.
* **Termination inside backups is the environment's PUBLISHED predicate**
  evaluated on sampled next states (CartPole: |x|>2.4 or |theta|>12 deg;
  Acrobot: -cos t1 - cos(t1+t2) > 1). Env-spec constants, not calibration
  constants (same class as reading max_steps). The backup horizon matches
  the anchor's truncation (500).
* **r-hat routes through interventional_sweep** (``query_batch(do=)``):
  backup targets are read-only constants for the V regression, N1a's
  read-only case. The sampling cost lands on the TRANSITION draws (m per
  backup row) -- counted and timed, which is the cost question step 2 exists
  to answer.
* **Both transition mechanisms run** (LinearGaussian and MDN(3,(64,64))) --
  step 1 left the Acrobot mechanism choice to this check.
* On d_a_null the per-step reward error is ~0 (Dirac / categorical R fitted
  on its exact support), so the return-level error isolates the transition
  model + termination handling -- the cleanest reading the null cell offers.
  Registered prediction 1 (error compounds ~ 1/(1-gamma)) is evaluated
  against the ONE-STEP transition error from step 1's report.

Resume: rows already in the report are skipped on relaunch.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

GAMMA = 0.99  # the benchmark's own (iql.py / cql defaults)
N_EVAL = 50  # anchor rollouts; deterministic policy+env => RTG is exact V^pi
MAX_STEPS = 500
BUFFER_CAP = 20_000  # backup buffer: dataset states, subsampled
M_SAMPLES = 8  # transition draws per backup row
K_ITERS = 60  # fitted-iteration sweeps (final sup-change REPORTED, not gated)
V_EPOCHS = 3  # regression passes per sweep
BATCH = 4096
FK = dict(max_iter=30, m_step_budget=400, batch_size=4096)
MDN_STEP_BUDGET = 4000


def _terminal(env_id, s):
    """The environment's published termination predicate on a state batch."""
    if env_id.startswith("CartPole"):
        return (s[:, 0].abs() > 2.4) | (s[:, 2].abs() > 0.2094395)
    if env_id.startswith("Acrobot"):
        c1, s1, c2, s2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        return (-c1 - (c1 * c2 - s1 * s2)) > 1.0
    raise ValueError(env_id)


def main() -> int:
    import gymnasium as gym
    import minari
    import numpy as np
    import torch
    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
    from nbn.mechanisms.parametric.mdn import MDNMechanism
    from src.benchmarking.regime_sweep import load_sweep_spec
    from src.envs.offline.generate import build_generator_agent
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path("results/q2a_danull/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: list = []
    if out_path.exists():
        out = json.loads(out_path.read_text())
        print(f"  resuming: {len(out)} rows already done", flush=True)
    done = {(r["env"], r["seed"]) for r in out}

    recert = json.loads(Path("results/vb_recertification/report.json").read_text())
    spec = load_sweep_spec(Path("reproducibility/rl_regimes/diagrams/d_a_null.yaml"))
    jobs = [
        (r["env"], r["seed"], r["dataset_id"])
        for r in recert
        if r["cell"] == "d_a_null" and r["seed"] in (0, 1, 2)
    ]

    for env_id, sd, did in jobs:
        if (env_id, sd) in done:
            continue
        t0 = time.time()
        print(f"\n=== {env_id} s{sd} ===", flush=True)

        # ---- target policy: the generator's greedy, rebuilt reproducibly ----
        agent, ghash = build_generator_agent(
            env_id,
            spec.generator_algo,
            "medium",
            seed=sd,
            train_episodes=spec.budget("n_episodes", 250),
            n_checkpoints=spec.budget("n_checkpoints", 25),
            run_dir=f"results/q2a_danull/generator/{env_id}_s{sd}",
        )
        t_agent = time.time() - t0

        def pi(s_batch):  # greedy target policy on a [n, D] tensor
            with torch.no_grad():
                return agent.act(s_batch, deterministic=True).action.reshape(-1)

        # ---- anchor: on-policy RTG from the true env (deterministic) --------
        env = gym.make(env_id)
        anchor_s, anchor_rtg, ep_lens = [], [], []
        for k in range(N_EVAL):
            obs, _ = env.reset(seed=10_000 + k)
            states, rewards = [], []
            for _t in range(MAX_STEPS):
                states.append(np.asarray(obs, dtype=np.float32))
                a = int(
                    pi(
                        torch.tensor(obs, dtype=torch.float32, device=device).reshape(
                            1, -1
                        )
                    )[0]
                )
                obs, r, term, trunc, _ = env.step(a)
                rewards.append(float(r))
                if term or trunc:
                    break
            rtg = np.zeros(len(rewards), dtype=np.float64)
            acc = 0.0
            for i in range(len(rewards) - 1, -1, -1):
                acc = rewards[i] + GAMMA * acc
                rtg[i] = acc
            anchor_s.append(np.stack(states))
            anchor_rtg.append(rtg)
            ep_lens.append(len(rewards))
        env.close()
        A_s = torch.tensor(np.concatenate(anchor_s), device=device)
        A_rtg = torch.tensor(
            np.concatenate(anchor_rtg), dtype=torch.float32, device=device
        )

        # ---- dataset, estimator fit, r-hat under the target policy ----------
        s, blocks = rebuild_samples(minari.load_dataset(did), 10_000)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        s_next = np.concatenate([b.observations[1:] for b in blocks], axis=0)
        t_ = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
        data = EpisodeData(
            state=t_(state),
            action=t_(s["a"], torch.long),
            reward=t_(s["r"]),
            episode_ids=t_(s["episode"], torch.long),
        )
        est = LatentClassEstimator(
            state_dim=state.shape[1],
            n_actions=int(s["a"].max()) + 1,
            proxy_names=(),
            device=device,
            seed=0,
        )
        t1 = time.time()
        fit = est.fit(data, init="random", **FK)
        t_fit = time.time() - t1

        rng = np.random.default_rng(0)
        idx = rng.choice(
            state.shape[0], size=min(BUFFER_CAP, state.shape[0]), replace=False
        )
        S_buf = t_(state[idx])
        a_buf = pi(S_buf)
        r_hat = est.interventional_sweep(S_buf, a_buf.tolist(), fit).value.reshape(-1)
        # query_batch runs under inference mode; its tensors raise if saved for
        # backward. The regression target must leave that lineage.
        r_hat = torch.as_tensor(
            r_hat.detach().cpu().numpy(), dtype=torch.float32, device=device
        )

        # ---- transition mechanisms (fit on the full dataset) ----------------
        S_all, A_all, SN_all = t_(state), t_(s["a"]).reshape(-1, 1), t_(s_next)
        pa_all = torch.cat([S_all, A_all], dim=1)
        rec = {
            "env": env_id,
            "seed": sd,
            "dataset_id": did,
            "ghash": ghash,
            "target_policy": f"{spec.generator_algo}-medium greedy (deterministic)",
            "gamma": GAMMA,
            "n_anchor_states": int(A_s.shape[0]),
            "anchor_ep_len_mean": float(np.mean(ep_lens)),
            "anchor_rtg_mean": float(A_rtg.mean()),
            "agent_rebuild_seconds": round(t_agent, 1),
            "estimator_fit_seconds": round(t_fit, 1),
            "buffer": int(S_buf.shape[0]),
            "mechanisms": {},
        }
        for name in ("linear_gaussian", "mdn"):
            torch.manual_seed(0)
            t2 = time.time()
            if name == "linear_gaussian":
                mech = LinearGaussianMechanism()
                mech.fit_local(SN_all, pa_all)
            else:
                mech = MDNMechanism(num_components=3, hidden=(64, 64))
                spe = max(1, int(np.ceil(S_all.shape[0] / BATCH)))
                mech.fit_local(
                    SN_all,
                    pa_all,
                    epochs=max(1, round(MDN_STEP_BUDGET / spe)),
                    lr=1e-3,
                    batch_size=BATCH,
                    consolidate=False,
                )
            t_mech = time.time() - t2

            # ---- sampling-based fitted V-iteration under pi -----------------
            torch.manual_seed(0)
            vnet = torch.nn.Sequential(
                torch.nn.Linear(S_buf.shape[1], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            ).to(device)
            opt = torch.optim.Adam(vnet.parameters(), lr=1e-3)
            pa_buf = torch.cat([S_buf, a_buf.reshape(-1, 1).float()], dim=1)
            n_sample_calls = 0
            sup_change = float("nan")
            t3 = time.time()
            v_prev = torch.zeros(S_buf.shape[0], device=device)
            for it in range(K_ITERS):
                with torch.no_grad():
                    # s' ~ P-hat(.|s, pi(s)), m draws; the do-channel sampling
                    # cost the pre-registration asks to measure
                    sn = mech.sample(pa_buf, M_SAMPLES)  # [n, m, D]
                    n_sample_calls += S_buf.shape[0] * M_SAMPLES
                    flat = sn.reshape(-1, sn.shape[-1])
                    # V(s') = 0 past a terminal state (the published
                    # predicate); V_k(s') already folds all future reward, so
                    # the target is the standard fitted-VI backup for V^pi
                    alive = (~_terminal(env_id, flat)).float()
                    vn = vnet(flat).reshape(-1)
                    v_next = (alive * vn).reshape(-1, M_SAMPLES).mean(dim=1)
                    target = r_hat + GAMMA * v_next
                for _ in range(V_EPOCHS):
                    perm = torch.randperm(S_buf.shape[0], device=device)
                    for j in range(0, S_buf.shape[0], BATCH):
                        b = perm[j : j + BATCH]
                        loss = torch.nn.functional.mse_loss(
                            vnet(S_buf[b]).reshape(-1), target[b]
                        )
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    v_now = vnet(S_buf).reshape(-1)
                    sup_change = float((v_now - v_prev).abs().max())
                    v_prev = v_now
            t_vi = time.time() - t3

            with torch.no_grad():
                v_anchor = vnet(A_s).reshape(-1)
                err = v_anchor - A_rtg
            rec["mechanisms"][name] = {
                "transition_fit_seconds": round(t_mech, 1),
                "vi_seconds": round(t_vi, 1),
                "vi_seconds_per_iter": round(t_vi / K_ITERS, 2),
                "n_transition_samples": int(n_sample_calls),
                "k_iters": K_ITERS,
                "final_sup_change": round(sup_change, 4),
                "v_vs_rtg_rmse": round(float((err**2).mean().sqrt()), 4),
                "v_vs_rtg_mae": round(float(err.abs().mean()), 4),
                "v_vs_rtg_bias": round(float(err.mean()), 4),
                "rtg_mean_abs": round(float(A_rtg.abs().mean()), 4),
            }
            m = rec["mechanisms"][name]
            print(
                f"  {name:<16} RMSE={m['v_vs_rtg_rmse']:.3f} "
                f"MAE={m['v_vs_rtg_mae']:.3f} bias={m['v_vs_rtg_bias']:+.3f} "
                f"|RTG|={m['rtg_mean_abs']:.2f} "
                f"vi={m['vi_seconds']:.0f}s supDelta={m['final_sup_change']:.3f}",
                flush=True,
            )
        rec["seconds"] = round(time.time() - t0, 1)
        out.append(rec)
        out_path.write_text(json.dumps(out, indent=1))

    print("Q2A D_A_NULL MACHINERY CHECK COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

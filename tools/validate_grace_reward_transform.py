"""Pre-launch validation of the REWARD-SUBSTITUTION serve (ruled 2026-08-31).

Two checks, both on objects that are actually deployed:

1. **Fitted iteration on the LOGGED tuples with r_hat** -- a validation, not
   the served path. On d_a_null the answer is known (~9) and r_hat is exactly
   1, so this tests "r_hat + exact recorded dynamics" with nothing else in
   play. Near 9 => the reward-substitution path is sound. Still diverging =>
   even plain fitted iteration has the OOD problem on this data, which argues
   FOR the base's conservatism being load-bearing.

2. **Positive control** (d100): the mean SUBSTITUTED reward on a_bad
   transitions must be LOWER than the mean LOGGED reward on those same
   transitions, by approximately M * tilt -- a predicted direction AND
   magnitude. Equal => passthrough; higher => the pessimism sign is wrong.
   (S14: for a seam whose failures are silent, "no error" is not evidence.)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

GAMMA, A_BAD, EPISODES, B = 0.99, 1, 3000, 5


def main() -> int:
    import minari
    import numpy as np
    import torch
    import torch.nn as nn
    from src.rl.off_policy.replay_buffer import ReplayBuffer
    from src.rl.offline.grace.serving import fit_reward_transform

    from tools.recertify_diagram_arms import rebuild_samples

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recert = json.loads(Path("results/vb_recertification/report.json").read_text())
    gen = {
        (r["cell"], r["env"], r["seed"]): r["dataset_id"]
        for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text())
    }
    jobs = [
        (
            "d_a_null",
            next(
                r["dataset_id"]
                for r in recert
                if r["cell"] == "d_a_null"
                and r["env"] == "CartPole-v1"
                and r["seed"] == 0
            ),
            (),
        ),
        ("d_d_sweep_d100", gen[("d_d_sweep_d100", "CartPole-v1", 0)], ("Z", "W", "V")),
    ]
    out = []
    for cell, did, proxies in jobs:
        print(f"\n=== {cell} ===", flush=True)
        s, blocks = rebuild_samples(minari.load_dataset(did), EPISODES)
        obs = np.concatenate([b.observations[:-1] for b in blocks], 0)
        nxt = np.concatenate([b.observations[1:] for b in blocks], 0)
        ep = s["episode"]
        dn = np.zeros(len(obs), dtype=np.float32)
        dn[np.flatnonzero(np.diff(ep) != 0)] = 1.0
        dn[-1] = 1.0
        cols = {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(s["a"], dtype=torch.long),
            "rewards": torch.tensor(s["r"], dtype=torch.float32),
            "next_obs": torch.tensor(nxt, dtype=torch.float32),
            "dones": torch.tensor(dn),
        }
        for nm in proxies:
            cols[f"proxy_{nm}"] = torch.tensor(s[nm.lower()], dtype=torch.float32)
        buf = ReplayBuffer(capacity=len(obs) + 10, device=dev)
        for i in range(len(obs)):
            buf.add({k: v[i] for k, v in cols.items()})
        logged = buf._data["rewards"][: len(obs)].clone()

        t0 = time.time()
        sv = fit_reward_transform(buf, proxy_names=proxies, b=B, device=dev)
        print(f"  fit {time.time()-t0:.0f}s | {sv.label()[:110]}", flush=True)
        if sv.abstained:
            out.append({"cell": cell, "abstained": True, "reason": sv.reason})
            continue

        acts = cols["actions"].numpy()
        new_r = sv.rewards.detach().cpu().numpy()
        bad = acts == A_BAD
        d_bad = float(new_r[bad].mean() - logged.numpy()[bad].mean())
        d_oth = float(new_r[~bad].mean() - logged.numpy()[~bad].mean())
        row = {
            "cell": cell,
            "abstained": False,
            "substituted_minus_logged_a_bad": d_bad,
            "substituted_minus_logged_other": d_oth,
            "pessimism_applied": sv.meta["pessimism_applied"],
            "contrast_point": sv.meta["contrast_point"],
            "lo": sv.lo,
            "hi": sv.hi,
        }
        if cell != "d_a_null":
            verdict = (
                "LOWER as predicted (GRACE removes the upward M*tilt bias)"
                if d_bad < 0
                else "NOT LOWER -- passthrough in disguise, or the sign is inverted"
            )
            print(
                f"  positive control: substituted - logged on a_bad = {d_bad:+.4f} "
                f"(other actions {d_oth:+.4f}) -> {verdict}",
                flush=True,
            )
            row["control_passes"] = d_bad < 0

        # ---- validation: fitted iteration on the LOGGED tuples with r_hat ----
        st = cols["obs"].to(dev)
        nx = cols["next_obs"].to(dev)
        dnv = cols["dones"].to(dev)
        av = cols["actions"].to(dev)
        rv = sv.rewards.to(dev).float()
        torch.manual_seed(0)
        q = nn.Sequential(
            nn.Linear(st.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        ).to(dev)
        opt = torch.optim.Adam(q.parameters(), lr=1e-3)
        for k in range(60):
            with torch.no_grad():
                tgt = rv + GAMMA * (1.0 - dnv) * q(nx).max(1).values
            for _ in range(3):
                perm = torch.randperm(st.shape[0], device=dev)
                for i in range(0, st.shape[0], 4096):
                    b_ = perm[i : i + 4096]
                    pred = q(st[b_]).gather(1, av[b_].reshape(-1, 1)).reshape(-1)
                    loss = nn.functional.mse_loss(pred, tgt[b_])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        with torch.no_grad():
            v = float(q(st).max(1).values.mean())
        print(
            f"  fitted-iteration on LOGGED tuples: mean V = {v:.2f}"
            + ("   (truth ~9 on this cell)" if cell == "d_a_null" else ""),
            flush=True,
        )
        row["mean_v_logged_backup"] = v
        out.append(row)
        Path("results/grace_reward_transform_validation.json").write_text(
            json.dumps(out, indent=1)
        )
    print("\nVALIDATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""L5 calibration — the report that gates any use of the Markov statistic.

Approval conditions served here:
  1. POWER, not only FPR: detection across effect size (full vs single-dim
     velocity mask), horizon (CartPole ~16-step episodes vs Acrobot up to
     500), and sample size (episode subsets).
  2. The SELECTION PROCEDURE calibrated as deployed: P(select k>0 | true MDP)
     measured by running `select_window` itself, never derived from the
     single-test alpha; plus the selected-k distribution on masked data.
  3. `k_max` treated as a budget: abstention counts reported.

Replicates are DISJOINT episode subsets of certified datasets — disjoint
episodes are independent draws from the arm, so the uniformity KS reads real
replicates, not resamples (S11: subsample the real data, never cut the fit).

Masking here is ANALYSIS-time column dropping (the S->A construction): valid
for the statistic's power — the observed process is non-Markov either way —
and distinct from the masked-behaviour generation the RL grid needs; the
distinction is recorded in the report.

CPU-only by design: the campaign owns the GPU, and the statistic is closed
form. Run:  uv run python tools/calibrate_l5.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

OUT = Path("results/l5_calibration")
ALPHA = 0.05  # stated, not tuned
K_MAX = 2
# The stated Delta-R^2 convention for the under-the-cut selector reading. A
# CONVENTION inside the measured gap, never derived from the rows: the report
# recomputes the gap (null max vs masked min) and records whether the cut
# lies inside it; if it does not, the under-cut reading is reported as
# invalid rather than silently computed.
DR2_CUT_CONVENTION = 1e-4

# (env tag, id substring filters, mask sets by effect size, chunk sizes, chunk episode cap)
PLANS = {
    "cartpole": dict(
        must=["cartpole"],
        must_not=[],
        masks={"full_velocity": (1, 3), "single_dim": (3,)},
        sizes=(50, 200),
        uniformity_size=500,
        max_chunks=4,
        limit=12,
    ),
    "acrobot": dict(
        must=["acrobot"],
        must_not=[],
        masks={"full_velocity": (4, 5), "single_dim": (5,)},
        sizes=(50, 100),
        uniformity_size=200,  # long episodes: 200 episodes is ~30-100k rows
        max_chunks=2,
        limit=8,
    ),
}

# Power cells run at a reduced draw budget (min p 0.02 < alpha) and on the
# FIRST chunk only: detection is a Bernoulli per (dataset, cell), so datasets
# supply the replication; uniformity needs many replicates and gets every
# chunk at the full budget.
_POWER_B = 49


def _chunks(n_total: int, size: int, cap: int):
    """Disjoint [start, end) episode ranges."""
    out = []
    start = 0
    while start + size <= n_total and len(out) < cap:
        out.append((start, start + size))
        start += size
    return out


def _episodes_slice(all_eps, lo, hi, mask):
    from src.rl.offline.grace.l5 import Episode

    out = []
    for e in all_eps[lo:hi]:
        obs = e.obs
        if mask:
            keep = [j for j in range(obs.shape[1]) if j not in set(mask)]
            obs = obs[:, keep]
        out.append(Episode(obs=obs, act=e.act, rew=e.rew))
    return out


def _stages(verdicts):
    """Every stage's (lag, p, Delta-R^2): what the under-the-cut reading of
    the selector needs post hoc for ANY cut, so a row never has to be
    recomputed to be re-read."""
    return [
        dict(lag=int(v.lag), p=float(v.p_value), stat=float(v.statistic))
        for v in verdicts
    ]


def _selector_readings(row, alpha, k_max, dr2_cut):
    """The two readings of one selector row, both stated explicitly.

    AS DEPLOYED (dr2_cut=None, what the sweep ran): ``k_selected`` is the
    first non-rejected stage; ``None`` means every stage 0..k_max rejected and
    the budget bound. ``k_tests`` = stages run, so stages_rejected =
    k_selected (stages before the pass) or k_tests (all of them).

    UNDER THE CUT: a stage is falsified iff p <= alpha AND stat > dr2_cut.
    Fully derivable when the row stores ``stages``; rows written before that
    field existed carry stage 0 only, so k = 0 is decidable (stage 0 passes)
    and anything else is reported as undetermined, never guessed.
    """
    k_sel = row["k_selected"]
    k_tests = int(row["k_tests"])
    budget_bound = k_sel is None
    stages_rejected = k_tests if budget_bound else int(k_sel)
    stages = row.get("stages")
    if stages is None:
        stages = [dict(lag=0, p=row["p"], stat=row["stat"])]
        complete = False
    else:
        complete = True
    k_cut = None
    undetermined = False
    for st in stages:
        falsified = st["p"] <= alpha and st["stat"] > dr2_cut
        if not falsified:
            k_cut = int(st["lag"])
            break
    else:
        # every stored stage falsified under the cut
        if complete and len(stages) == k_max + 1:
            k_cut = None  # budget-bound under the cut too
        elif complete:
            # the deployed run stopped early (it passed statistically at a
            # later stage than the cut would have needed) -- cannot happen:
            # the cut only makes passing EASIER, so the cut's pass is at or
            # before the deployed pass. Kept as a guard.
            undetermined = True
        else:
            undetermined = True
    return dict(
        stages_rejected=stages_rejected,
        budget_bound=budget_bound,
        k_under_cut=k_cut,
        k_under_cut_undetermined=undetermined,
    )


def _k_dist(values):
    return {
        str(k): int(sum(1 for v in values if v == k))
        for k in sorted(set(values), key=lambda x: (x is None, x))
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 dataset/env, b=99")
    ap.add_argument("--envs", nargs="*", default=list(PLANS))
    ap.add_argument(
        "--b", type=int, default=None, help="draw-budget override (smoke only)"
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild report.json from rows.jsonl without running any test",
    )
    ap.add_argument(
        "--dr2-cut",
        type=float,
        default=DR2_CUT_CONVENTION,
        help="the STATED cut for the under-the-cut selector reading; the "
        "report checks it against the measured gap and says so",
    )
    args = ap.parse_args()
    b = args.b if args.b else (99 if args.quick else 199)

    import minari
    from src.rl.offline.grace.l5 import episodes_from_minari, select_window

    OUT.mkdir(parents=True, exist_ok=True)
    all_ids = sorted(minari.list_local_datasets())
    # Incremental persistence (S7): a multi-day sweep that only writes at the
    # end has no surviving record on a crash. Every row is appended to
    # rows.jsonl the moment it is computed, and a restart SKIPS (env, dataset,
    # chunk, kind, n_ep) keys already on disk — resume is the default, and a
    # fresh sweep is `rm results/l5_calibration/rows.jsonl` first.
    rows_path = OUT / "rows.jsonl"
    done_keys = set()
    prior_rows = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prior_rows.append(r)
            done_keys.add(
                (
                    r.get("env"),
                    r.get("dataset"),
                    r.get("chunk"),
                    r.get("kind"),
                    r.get("n_ep"),
                )
            )
        print(f"[resume] {len(prior_rows)} rows already on disk", flush=True)
    report = {"alpha": ALPHA, "k_max": K_MAX, "b": b, "rows": list(prior_rows)}

    def _emit(row):
        report["rows"].append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    plans = [] if args.report_only else [(e, PLANS[e]) for e in args.envs]
    for env, plan in plans:
        ids = [
            i
            for i in all_ids
            if all(m in i for m in plan["must"])
            and not any(m in i for m in plan["must_not"])
        ]
        ids = ids[: (1 if args.quick else plan.get("limit", len(ids)))]
        print(f"[{env}] {len(ids)} datasets", flush=True)
        for did in ids:
            t0 = time.time()
            try:
                eps = episodes_from_minari(minari.load_dataset(did))
            except Exception as exc:  # a dataset that fails to load is recorded
                report["rows"].append(dict(env=env, dataset=did, error=str(exc)))
                continue
            n_total = len(eps)
            usize = plan["uniformity_size"]
            for ci, (lo, hi) in enumerate(_chunks(n_total, usize, plan["max_chunks"])):
                seed = 1000 * ci + 17
                # --- true-null: uniformity + the deployed selection procedure
                if (env, did, ci, "null", usize) in done_keys:
                    continue_null = False
                else:
                    continue_null = True
                sub = _episodes_slice(eps, lo, hi, mask=())
                # ONE selection run: its stage-0 verdict IS the lag-0 test
                # (same seed by construction), so nothing is computed twice.
                if continue_null:
                    k_sel, k_verdicts = select_window(
                        sub, alpha=ALPHA, k_max=K_MAX, b=b, seed=seed
                    )
                    v = k_verdicts[0]
                    _emit(
                        dict(
                            env=env,
                            dataset=did,
                            chunk=ci,
                            n_ep=usize,
                            kind="null",
                            p=v.p_value,
                            stat=v.statistic,
                            untestable=v.untestable,
                            base_r2_min=(
                                None
                                if v.base_r2 is None
                                else float(np.nanmin(v.base_r2[:-1]))
                            ),
                            reward_channel=v.reward_channel,
                            capacity=v.capacity,
                            k_selected=k_sel,
                            k_tests=len(k_verdicts),
                            stages=_stages(k_verdicts),
                        )
                    )
                # --- power grid: mask x size, first chunk only (see _POWER_B)
                if ci > 0:
                    continue
                for mname, mask in plan["masks"].items():
                    for size in plan["sizes"]:
                        if lo + size > hi:
                            continue
                        if (env, did, ci, f"masked:{mname}", size) in done_keys:
                            continue
                        subm = _episodes_slice(eps, lo, lo + size, mask=mask)
                        km, kv = select_window(
                            subm, alpha=ALPHA, k_max=K_MAX, b=_POWER_B, seed=seed + 7
                        )
                        vm = kv[0]
                        _emit(
                            dict(
                                env=env,
                                dataset=did,
                                chunk=ci,
                                n_ep=size,
                                kind=f"masked:{mname}",
                                p=vm.p_value,
                                stat=vm.statistic,
                                untestable=vm.untestable,
                                base_r2_min=(
                                    None
                                    if vm.base_r2 is None
                                    else float(np.nanmin(vm.base_r2[:-1]))
                                ),
                                reward_channel=vm.reward_channel,
                                capacity=vm.capacity,
                                k_selected=km,
                                k_tests=len(kv),
                                stages=_stages(kv),
                            )
                        )
            print(f"  {did.split('/')[1][:60]}: {time.time()-t0:.0f}s", flush=True)

    # ---- summary ---------------------------------------------------------
    rows = [r for r in report["rows"] if "error" not in r]
    summary = {}
    for env in args.envs:
        er = [r for r in rows if r["env"] == env]
        nulls = [r for r in er if r["kind"] == "null"]
        ps = np.array([r["p"] for r in nulls])
        # KS against uniform, plain formula (no scipy dependency needed).
        if ps.size:
            s = np.sort(ps)
            grid = np.arange(1, s.size + 1) / s.size
            ks = float(np.max(np.abs(s - grid)))
        else:
            ks = None
        sel_fpr = (
            float(np.mean([r["k_selected"] != 0 for r in nulls])) if nulls else None
        )
        masked_all = [r for r in er if r["kind"].startswith("masked")]
        cut = float(args.dr2_cut)
        null_max = max((r["stat"] for r in nulls), default=None)
        masked_min = min((r["stat"] for r in masked_all), default=None)
        cut_in_gap = (
            null_max is not None
            and masked_min is not None
            and null_max < cut < masked_min
        )

        def _selector_block(rr):
            rd = [_selector_readings(r, ALPHA, K_MAX, cut) for r in rr]
            under = [x["k_under_cut"] for x in rd if not x["k_under_cut_undetermined"]]
            return dict(
                n=len(rr),
                # reading 1: the as-deployed statistical selector (dr2_cut=None)
                as_deployed=dict(
                    k_dist=_k_dist([r["k_selected"] for r in rr]),
                    stages_rejected_dist=_k_dist([x["stages_rejected"] for x in rd]),
                    budget_bound_frac=(
                        float(np.mean([x["budget_bound"] for x in rd])) if rd else None
                    ),
                ),
                # reading 2: the same rows re-read under the stated cut
                under_cut=dict(
                    dr2_cut=cut,
                    cut_in_measured_gap=bool(cut_in_gap),
                    k_dist=_k_dist(under),
                    n_determined=len(under),
                    n_undetermined=len(rd) - len(under),
                    k0_frac=(
                        float(np.mean([k == 0 for k in under])) if under else None
                    ),
                ),
            )

        selector = dict(
            null=_selector_block(nulls),
            masked=_selector_block(masked_all),
            note=(
                "as_deployed: k_selected is the first stage NOT rejected at "
                "alpha (dr2_cut=None); None = every stage 0..k_max rejected = "
                "BUDGET-BOUND (stages_rejected = k_tests). under_cut: falsified "
                "iff p<=alpha AND stat>dr2_cut; rows without per-stage records "
                "decide k=0 from stage 0 alone and are otherwise undetermined. "
                "Contract row 2 (over-assumption is cheap) reads "
                "null.under_cut.k0_frac; cut_in_measured_gap must be true for "
                "the reading to be valid."
            ),
        )

        def _dist(rr):
            st = np.array([r["stat"] for r in rr], dtype=float)
            sh = [
                (r.get("capacity") or {}).get("shrink")
                for r in rr
                if (r.get("capacity") or {}).get("shrink") is not None
            ]
            return dict(
                n=len(rr),
                stat_min=float(st.min()) if st.size else None,
                stat_median=float(np.median(st)) if st.size else None,
                stat_max=float(st.max()) if st.size else None,
                shrink_median=(float(np.median(sh)) if sh else None),
            )

        masked_rows = [r for r in er if r["kind"].startswith("masked")]
        null_dist, masked_dist = _dist(nulls), _dist(masked_rows)
        separation = (
            float(masked_dist["stat_min"] / null_dist["stat_max"])
            if masked_rows
            and nulls
            and null_dist["stat_max"]
            and null_dist["stat_max"] > 0
            else None
        )
        env_summary = dict(
            n_null=len(nulls),
            null_p_min=float(ps.min()) if ps.size else None,
            null_frac_below_alpha=float(np.mean(ps <= ALPHA)) if ps.size else None,
            ks_stat=ks,
            selection_fpr=sel_fpr,  # as-deployed only; see selector below
            selector=selector,
            # THE RESULT the ruling names: the two Delta-R^2 distributions and
            # their separation. A cut in the gap is a stated convention; if
            # these overlap, no tolerance would have saved us.
            null_dr2=null_dist,
            masked_dr2=masked_dist,
            separation=separation,
            power={},
        )
        for mname in {r["kind"] for r in er if r["kind"].startswith("masked")}:
            for size in sorted({r["n_ep"] for r in er if r["kind"] == mname}):
                cell = [r for r in er if r["kind"] == mname and r["n_ep"] == size]
                env_summary["power"][f"{mname}@n{size}"] = dict(
                    n=len(cell),
                    detect=float(np.mean([r["p"] <= ALPHA for r in cell])),
                    k_dist={
                        str(k): int(sum(1 for r in cell if r["k_selected"] == k))
                        for k in sorted(
                            {r["k_selected"] for r in cell},
                            key=lambda x: (x is None, x),
                        )
                    },
                )
        summary[env] = env_summary
    report["summary"] = summary
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"-> {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

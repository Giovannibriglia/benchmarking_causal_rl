"""E1 — the GRACE deployment experiment. Purpose-built driver.

**Why not ``run_cell``.** The sweep derives a dataset id from
``(prefix, regime, env, beta, sigma, seed)`` -- the CELL NAME IS NOT IN IT. All
four E1 cells are ``offline_mdp`` at beta=0, sigma=0.25, so they collide onto
ONE id: the sweep would have generated fresh, uncertified data and trained
d100, d025 and d010asym on IDENTICAL datasets, with every comparison being
between identical arms, and it would all have completed without error. (Its
``arm_generator_kwargs`` also splats fields ``EnvConfig`` does not have, so a
diagram cell has never trained through that path at all -- recorded in the
handoff as a known gap; not fixed here.)

**So the id is READ, never reconstructed** -- from the same generation reports
that Q2-A, V4 and V-C1 resolve through. That single-construction-site
discipline is what makes E1's numbers commensurable with theirs, and it is the
same rule that fixed ``c_r``.

Two assertions run BEFORE any training, both aimed at the failure that nearly
happened:

1. every cell resolves to a DISTINCT dataset id (the collision above);
2. every id EXISTS in the Minari store and carries its certification stamp
   (a silently regenerated dataset is what the collision would have produced;
   requiring the stamp makes that unrepresentable).

The resolved id is written into each leaf, so a reader can verify from the
artifact which dataset a number came from without re-deriving anything.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

REGIME = "offline_mdp"
# Campaign prefix selects BOTH the YAML glob and the results root: e1 = the
# pilot cells (incl. incomplete-by-decision d025/d010asym — never re-entered
# unless this campaign is chosen), c1 = the observability-contract grid.
CAMPAIGN_ROOTS = {"e1": Path("results/e1"), "c1": Path("results/c1")}

# The campaign is DEFINED by reproducibility/rl_regimes/diagrams/e1_*.yaml
# (strict-parsed: an unknown key raises). The CELLS/SEEDS/ALGOS/PROXIES
# constants that used to live here were a second construction site for facts
# the YAMLs now carry — deleted 2026-09-03; `enumerate_plan` below is the one
# derivation, shared with the declaration tests.


def _scalar_meta(meta) -> dict:
    """The seam's scalar diagnostics, as VALID JSON.

    ``json.dumps`` writes bare ``Infinity``/``NaN`` for non-finite floats.
    Python reads those back, nothing else does, and a reader that does parse
    them sees "inf" where the honest value is "undefined" -- so the
    non-finite cases become ``null``. They arise for real: an undefined
    variance share (0/0 on a cell whose statistic cannot vary) and an
    unmeasurable optimiser arm both report non-finite by design.
    """
    import math

    out = {}
    for k, v in (meta or {}).items():
        if isinstance(v, bool) or isinstance(v, (int, str)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = v if math.isfinite(v) else None
    return out


GENERATION_REPORTS = (
    "results/vb_recertification/report.json",
    "results/dd_sweep_generation/report.json",
    "results/dd_asym_generation/report.json",
    # the sigma = 0 no-harm point, generated separately so the existing
    # sigma = 0.25 report stays the untouched record of what certified it
    "results/dd_sweep_sigma0_generation/report.json",
    # the TRUE-POMDP column (Phase 3): masked-behaviour generation (O->A, the
    # edge D-F declares), ids carrying -om13, report rows carrying the
    # behavior_information_set stamp — the contract grid's
    # source_cell: d_d_sweep_d100_om13 resolves through this.
    "results/dd_sweep_om13_generation/report.json",
)


def spec_points(spec) -> list:
    """The (beta, sigma) points a spec declares: the basic origin plus the
    confounded arm — the same L the sweep machinery derives."""
    pts = []
    if spec.include_basic:
        pts.append((0.0, 0.0))
    pts.extend((0.0, float(s)) for s in spec.sigma_arm)
    pts.extend((float(b), 0.0) for b in spec.beta_arm)
    return pts


def resolve_certified_ids_for_spec(spec) -> dict:
    """Certified ids for every (env, seed, point) a spec declares — READ from
    the generation reports, never reconstructed. ONE construction site shared
    by the driver and the declaration test, so 'the YAML defines the campaign'
    and 'the campaign runs on certified data' are the same fact.

    Returns {(env, seed, sigma): (dataset_id, stamp)}; raises when a declared
    point has no certified dataset."""
    srcs = []
    for f in GENERATION_REPORTS:
        p = Path(f)
        if p.exists():
            srcs.extend(json.loads(p.read_text()))
    out = {}
    for env in spec.envs:
        for sd in spec.seeds:
            for _b, sg in spec_points(spec):
                hits = [
                    r
                    for r in srcs
                    if r["cell"] == spec.source_cell
                    and r["env"] == env
                    and r["seed"] == sd
                    and (r.get("sigma") in (None, sg) or spec.source_cell == "d_a_null")
                ]
                if not hits:
                    raise SystemExit(
                        f"no certified dataset for {spec.source_cell} {env} "
                        f"s{sd} sigma={sg}"
                    )
                r = hits[0]
                out[(env, sd, sg)] = (
                    r["dataset_id"],
                    {
                        k: r.get(k)
                        for k in ("preflight_passed", "gate_passed", "ok")
                        if k in r
                    },
                )
    return out


def enumerate_plan(campaign: str = "e1") -> list:
    """One entry per (yaml arm) x env x dataset seed x TRAINING seed x algo —
    the whole campaign, derived from the {campaign}_*.yaml declarations alone
    through the SAME resolver the declaration tests pin (one construction
    site).

    ``train_seeds`` (contract cells) splits the two seed axes: the dataset
    seed selects the certified data (and the one GRACE fit, shared via the
    cache); the training seed is the RL run's initialisation. Cells without
    ``train_seeds`` keep the pilot's single-axis layout (ts == ds, plain seed
    segment) so the 42 pilot leaves stay addressable."""
    from src.benchmarking.regime_sweep import load_sweep_spec

    entries = []
    for f in sorted(
        Path("reproducibility/rl_regimes/diagrams").glob(f"{campaign}_*.yaml")
    ):
        spec = load_sweep_spec(f)
        ids = resolve_certified_ids_for_spec(spec)
        arm = "grace" if spec.grace_reward_transform else "base"
        pts = spec_points(spec)
        if len(pts) != 1:
            raise SystemExit(
                f"{f.name}: campaign cells declare exactly one point, got {pts}"
            )
        _b, sg = pts[0]
        train_seeds = getattr(spec, "train_seeds", None)
        for env in spec.envs:
            for sd in spec.seeds:
                did, stamp = ids[(env, sd, sg)]
                for ts in train_seeds or [sd]:
                    seg = f"ds{sd}_ts{ts}" if train_seeds else str(sd)
                    for algo in spec.algos:
                        entries.append(
                            dict(
                                spec=spec,
                                yaml=f.name,
                                tag=spec.e1_cell,
                                arm=arm,
                                env=env,
                                seed=sd,
                                train_seed=ts,
                                seed_segment=seg,
                                sigma=sg,
                                algo=algo,
                                dataset_id=did,
                                stamp=stamp,
                            )
                        )
    return entries


def assert_plan_safe(entries) -> None:
    """The two pre-flight assertions, now on the YAML-derived plan: (1) pairs
    share ids, distinct cells never do (the collision that would have made
    every comparison vacuous); (2) every id exists in the store and carries
    its certification stamp (a regenerated dataset is unrepresentable)."""
    import minari

    by_key: dict = {}
    for e in entries:
        k = (e["tag"], e["env"], e["seed"])  # dataset seed: pairing is per-ds
        if k in by_key and by_key[k] != e["dataset_id"]:
            raise SystemExit(
                f"paired arms of {k} resolve to different ids: "
                f"{by_key[k]!r} vs {e['dataset_id']!r}"
            )
        by_key[k] = e["dataset_id"]
    seen: dict = {}
    for (tag, env, sd), did in by_key.items():
        if did in seen and seen[did][0] != tag:
            raise SystemExit(
                f"DATASET COLLISION: {(tag, env, sd)} and {seen[did]} both "
                f"resolve to {did!r}."
            )
        seen[did] = (tag, env, sd)
    available = set(minari.list_local_datasets())
    for e in entries:
        if e["dataset_id"] not in available:
            raise SystemExit(
                f"{e['tag']} s{e['seed']}: dataset {e['dataset_id']!r} is not "
                "in the Minari store"
            )
        bad = [k for k, v in (e["stamp"] or {}).items() if v is False]
        if bad:
            raise SystemExit(
                f"{e['tag']} s{e['seed']}: dataset {e['dataset_id']!r} fails "
                f"its stamp {bad}"
            )
    print(
        f"  [assert] {len(by_key)} ids over {len(entries)} runs: pairs share, "
        "cells distinct, all present and stamped",
        flush=True,
    )


def _q1_truth(cell: str, sigma: float) -> float | None:
    """M = c_r * P(U=1), via arm_knobs -- the one construction site for c_r."""
    if cell == "d_a_null":
        return 0.0
    from src.benchmarking.regime_sweep import load_sweep_spec
    from src.envs.offline.diagram_arms import arm_knobs

    spec = load_sweep_spec(Path(f"reproducibility/rl_regimes/diagrams/{cell}.yaml"))
    k = arm_knobs(
        spec.diagram,
        sigma=sigma,
        confounder_c_r=(
            None
            if getattr(spec, "gate_mean_effect", None) is not None
            else spec.confounder_c_r
        ),
        proxy_strength=spec.proxy_strength,
        instrument_strength=spec.instrument_strength,
        u_drift=spec.u_drift,
        gate_probs=spec.gate_probs,
        gate_mean_effect=getattr(spec, "gate_mean_effect", None),
    )
    gp = k.gate_probs or (0.0, 0.0)
    return float(k.confounder_c_r or 0.0) * 0.5 * (float(gp[0]) + float(gp[1]))


def main() -> int:
    import torch
    from src.benchmarking.critic_ablation import CriticAblationConfig
    from src.benchmarking.regime_sweep import (
        _slice_critic_csv,
        arm_label,
        load_sweep_spec,
        results_leaf,
    )
    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.envs.offline.diagram_arms import arm_knobs
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    argv = sys.argv[1:]
    campaign = "e1"
    cache_dir = None
    for a in list(argv):
        if a.startswith("--campaign="):
            campaign = a.split("=", 1)[1]
            argv.remove(a)
        elif a.startswith("--cache-dir="):
            cache_dir = a.split("=", 1)[1] or None
            argv.remove(a)
    if campaign not in CAMPAIGN_ROOTS:
        raise SystemExit(f"--campaign must be one of {sorted(CAMPAIGN_ROOTS)}")
    results_root = CAMPAIGN_ROOTS[campaign]
    if cache_dir is None:
        cache_dir = str(results_root / "_transform_cache")
    only = argv or None  # optional smoke filter: tag algo seed arm

    entries = enumerate_plan(campaign)
    assert_plan_safe(entries)

    truths: dict = {}
    for e in entries:
        spec, tag, arm = e["spec"], e["tag"], e["arm"]
        env, sd, sg, algo, did = (
            e["env"],
            e["seed"],
            e["sigma"],
            e["algo"],
            e["dataset_id"],
        )
        ts, seg = e["train_seed"], e["seed_segment"]
        if tag not in truths:
            truths[tag] = _q1_truth(spec.source_cell, sg)
        truth = truths[tag]
        key = f"{tag}/{algo}/{arm}/{seg}"
        if only and not all(t in key for t in only):
            continue
        # e1 keeps the pilot's layout: the ARM occupies the critic slot, one
        # leaf per run, observational scoring only. c1 (Phase 6) runs the
        # cell's DECLARED critic set on the shared stream and explodes into
        # per-critic leaves the way run_cell does — the base algorithm's own
        # numbers are the observational leaf, and tpomdp excludes proximal
        # (L2: D-G q1 bounds-only), which the YAML declares, not the driver.
        if campaign == "e1":
            critics = ["observational"]
            slots = [(arm, None)]
        else:
            critics = list(spec.critics_for(arm_label(0.0, sg)))
            slots = [(c, c) for c in critics]
        leaves = {
            slot: results_leaf(
                results_root, f"{REGIME}_{tag}", 0.0, sg, env, algo, slot, seg
            )
            for slot, _ in slots
        }
        if all((lf / "eval_metrics.csv").exists() for lf in leaves.values()):
            print(f"  skip (done) {key}", flush=True)
            continue
        staging = results_root / "_staging" / key.replace("/", "_")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        # The learner's INFORMATION SET: the mask applies whenever the CELL
        # declares it — for base and grace alike, and regardless of the
        # DECLARED observability (the contract's true-POMDP cells keep
        # observability: mdp so cql/iql stay the memoryless learners; the
        # loader deletes the columns, so GRACE sees what the learner sees).
        _mask = tuple((spec.mask_indices or {}).get(env, ()) or ()) or None
        env_cfg = EnvConfig(
            env_id=env,
            n_train_envs=2,
            n_eval_envs=int(spec.n_eval_envs or 16),
            rollout_len=2,
            eval_rollout_len=int(spec.eval_rollout_len or 500),
            seed=ts,  # the TRAINING seed; the dataset seed rides in the id
            offline_dataset=did,  # PINNED, read not derived
            mask_indices=_mask,
            behavior_policy="bias_confounded_action",
            behavior_strength=sg,
            eval_confounded_reward=bool(spec.eval_confounded_reward),
            eval_confounded_mode=spec.eval_confounded_mode or "analytic",
            grace_reward_transform=(arm == "grace"),
            grace_proxy_names=(spec.grace_proxy_names if arm == "grace" else ()),
            grace_cache_dir=(getattr(spec, "grace_cache_dir", None) or cache_dir),
            declared_observability=spec.declared_observability,
            grace_k_max=spec.grace_k_max,
            grace_window_k=getattr(spec, "grace_window_k", None),
            grace_k_diagnostics=getattr(spec, "grace_k_diagnostics", True),
        )
        # the deployment reward's gate, from arm_knobs via the SOURCE cell
        if spec.source_cell != "d_a_null":
            sp = load_sweep_spec(
                Path(f"reproducibility/rl_regimes/diagrams/{spec.source_cell}.yaml")
            )
            k = arm_knobs(
                sp.diagram,
                sigma=sg,
                confounder_c_r=(
                    None
                    if getattr(sp, "gate_mean_effect", None) is not None
                    else sp.confounder_c_r
                ),
                proxy_strength=sp.proxy_strength,
                instrument_strength=sp.instrument_strength,
                u_drift=sp.u_drift,
                gate_probs=sp.gate_probs,
                gate_mean_effect=getattr(sp, "gate_mean_effect", None),
            )
            env_cfg.confounder_c_r = float(k.confounder_c_r or 0.0)
            env_cfg.gate_probs = k.gate_probs
        train_cfg = TrainingConfig(
            n_episodes=1,
            n_checkpoints=int(spec.budget("n_checkpoints", 25)),
            deterministic=True,
            device=device,
            algorithm=algo,
            aggregation="iqm",
            offline_grad_steps=(
                int(spec.budgets["offline_grad_steps"])
                if spec.budgets.get("offline_grad_steps")
                else None
            ),
            record_eval_video=False,
        )
        t0 = time.time()
        print(f"\n=== {key} | {did}", flush=True)
        runner = BenchmarkRunner(
            env_cfg,
            train_cfg,
            RunConfig(run_dir=str(staging), timestamp="e1"),
            registry.get(algo),
            critic_ablation_cfg=CriticAblationConfig(
                critics=critics,
                q1_truth=truth,
                a_bad=1,
            ),
        )
        runner.run()
        # The leaf MARKER the deployed report keys on (regime_report globs
        # for config.yaml; only main.py's CLI path writes one otherwise).
        import dataclasses

        import yaml

        (staging / "config.yaml").write_text(
            yaml.safe_dump(
                json.loads(
                    json.dumps(
                        {
                            "env": dataclasses.asdict(env_cfg),
                            "training": dataclasses.asdict(train_cfg),
                            "e1": {
                                "cell": tag,
                                "source_cell": spec.source_cell,
                                "arm": arm,
                                "dataset_id": did,
                                "declared_yaml": e["yaml"],
                            },
                        }
                    )
                ),
                default_flow_style=False,
            )
        )
        sv = getattr(runner, "grace_serving", None)
        for slot, critic_slice in slots:
            leaf = leaves[slot]
            leaf.mkdir(parents=True, exist_ok=True)
            for f in (
                "eval_metrics.csv",
                "eval_deployment.csv",
                "train_metrics.csv",
                "arm_diagnostics.csv",
                "config.yaml",
            ):
                src = staging / f
                if src.exists():
                    shutil.copy2(src, leaf / f)
            cam = staging / "critic_ablation_metrics.csv"
            if critic_slice is None:
                if cam.exists():
                    shutil.copy2(cam, leaf / "critic_ablation_metrics.csv")
            else:
                _slice_critic_csv(
                    cam, leaf / "critic_ablation_metrics.csv", critic_slice
                )
            (leaf / "e1_provenance.json").write_text(
                json.dumps(
                    {
                        "cell": tag,
                        "source_cell": spec.source_cell,
                        "declared_yaml": e["yaml"],
                        "arm": arm,
                        "critic": slot,
                        "algo": algo,
                        "env": env,
                        "seed": sd,
                        "dataset_seed": sd,
                        "train_seed": ts,
                        "dataset_id": did,
                        "stamp": e["stamp"],
                        "q1_truth": truth,
                        "grace": (
                            None
                            if sv is None
                            else {
                                "abstained": sv.abstained,
                                "label": sv.label(),
                                "lo": sv.lo,
                                "hi": sv.hi,
                                "meta": _scalar_meta(sv.meta),
                            }
                        ),
                        "seconds": round(time.time() - t0, 1),
                    },
                    indent=1,
                )
            )
        shutil.rmtree(staging, ignore_errors=True)
        print(
            f"  -> {len(slots)} leaf/leaves for {key}  ({time.time()-t0:.0f}s)",
            flush=True,
        )
    print("\nE1 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

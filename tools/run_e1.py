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
BETA, SIGMA = 0.0, 0.25
ENVS = ("CartPole-v1",)
SEEDS = (0, 1, 2)
ALGOS = ("cql", "iql")
PROXIES = ("Z", "W", "V")

# (e1 tag, source cell in the generation reports, declared proxy channels)
CELLS = (
    ("danull", "d_a_null", ()),
    ("d100", "d_d_sweep_d100", PROXIES),
    ("d025", "d_d_sweep_d025", PROXIES),
    ("d010asym", "d_d_sweep_d010_asym", PROXIES),
)
# The analytic q1 do-contrast truth per cell: M = c_r * P(U=1). Read from
# arm_knobs at runtime rather than tabulated here.
RESULTS_ROOT = Path("results/e1")


def _resolve_ids() -> dict:
    """Read certified ids from the generation reports. Never reconstruct."""
    srcs = []
    for f in (
        "results/vb_recertification/report.json",
        "results/dd_sweep_generation/report.json",
        "results/dd_asym_generation/report.json",
    ):
        p = Path(f)
        if p.exists():
            srcs.extend(json.loads(p.read_text()))
    out, stamps = {}, {}
    for tag, cell, _pn in CELLS:
        for env in ENVS:
            for sd in SEEDS:
                hits = [
                    r
                    for r in srcs
                    if r["cell"] == cell
                    and r["env"] == env
                    and r["seed"] == sd
                    and (r.get("sigma") in (None, SIGMA) or cell == "d_a_null")
                ]
                if not hits:
                    raise SystemExit(f"no certified dataset for {cell} {env} s{sd}")
                r = hits[0]
                out[(tag, env, sd)] = r["dataset_id"]
                stamps[(tag, env, sd)] = {
                    k: r.get(k)
                    for k in ("preflight_passed", "gate_passed", "ok")
                    if k in r
                }
    return out, stamps


def _assert_safe(ids: dict, stamps: dict) -> None:
    import minari

    # (1) DISTINCTNESS -- the collision that would have made every cell equal.
    seen = {}
    for key, did in ids.items():
        if did in seen:
            raise SystemExit(
                f"DATASET COLLISION: {key} and {seen[did]} both resolve to {did!r}. "
                "Two cells sharing data would make their comparison vacuous."
            )
        seen[did] = key
    # (2) EXISTENCE + CERTIFICATION -- a regenerated dataset is unrepresentable.
    available = set(minari.list_local_datasets())
    for key, did in ids.items():
        if did not in available:
            raise SystemExit(f"{key}: dataset {did!r} is not in the Minari store")
        st = stamps.get(key) or {}
        bad = [k for k, v in st.items() if v is False]
        if bad:
            raise SystemExit(f"{key}: dataset {did!r} fails its stamp {bad}")
    print(f"  [assert] {len(ids)} ids, all distinct, present and stamped", flush=True)


def _q1_truth(cell: str) -> float | None:
    """M = c_r * P(U=1), via arm_knobs -- the one construction site for c_r."""
    if cell == "d_a_null":
        return 0.0
    from src.benchmarking.regime_sweep import load_sweep_spec
    from src.envs.offline.diagram_arms import arm_knobs

    spec = load_sweep_spec(Path(f"reproducibility/rl_regimes/diagrams/{cell}.yaml"))
    k = arm_knobs(
        spec.diagram,
        sigma=SIGMA,
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
    from src.benchmarking.regime_sweep import load_sweep_spec, results_leaf
    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ids, stamps = _resolve_ids()
    _assert_safe(ids, stamps)

    only = sys.argv[1:] or None  # optional smoke filter: tag algo seed arm
    spec0 = load_sweep_spec(Path("reproducibility/rl_regimes/diagrams/e1_d100.yaml"))
    n_steps = spec0.budgets.get("offline_grad_steps")

    for tag, cell, proxies in CELLS:
        truth = _q1_truth(cell)
        for env in ENVS:
            for sd in SEEDS:
                did = ids[(tag, env, sd)]
                for algo in ALGOS:
                    for arm in ("base", "grace"):
                        key = f"{tag}/{algo}/{arm}/s{sd}"
                        if only and not all(t in key for t in only):
                            continue
                        leaf = results_leaf(
                            RESULTS_ROOT,
                            f"{REGIME}_{tag}",
                            BETA,
                            SIGMA,
                            env,
                            algo,
                            arm,
                            sd,
                        )
                        if (leaf / "eval_metrics.csv").exists():
                            print(f"  skip (done) {key}", flush=True)
                            continue
                        staging = Path("results/e1/_staging") / key.replace("/", "_")
                        if staging.exists():
                            shutil.rmtree(staging)
                        staging.mkdir(parents=True, exist_ok=True)
                        env_cfg = EnvConfig(
                            env_id=env,
                            n_train_envs=2,
                            n_eval_envs=2,
                            rollout_len=2,
                            seed=sd,
                            offline_dataset=did,  # PINNED, read not derived
                            behavior_policy="bias_confounded_action",
                            behavior_strength=SIGMA,
                            eval_confounded_reward=True,
                            eval_confounded_mode="analytic",
                            grace_reward_transform=(arm == "grace"),
                            grace_proxy_names=(proxies if arm == "grace" else ()),
                        )
                        # the deployment reward's gate, from arm_knobs
                        if cell != "d_a_null":
                            from src.envs.offline.diagram_arms import arm_knobs

                            sp = load_sweep_spec(
                                Path(f"reproducibility/rl_regimes/diagrams/{cell}.yaml")
                            )
                            k = arm_knobs(
                                sp.diagram,
                                sigma=SIGMA,
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
                            n_checkpoints=25,
                            deterministic=True,
                            device=device,
                            algorithm=algo,
                            aggregation="iqm",
                            offline_grad_steps=(int(n_steps) if n_steps else None),
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
                                critics=["observational"],
                                q1_truth=truth,
                                a_bad=1,
                            ),
                        )
                        runner.run()
                        leaf.mkdir(parents=True, exist_ok=True)
                        for f in (
                            "eval_metrics.csv",
                            "train_metrics.csv",
                            "critic_ablation_metrics.csv",
                            "arm_diagnostics.csv",
                            "config.yaml",
                        ):
                            src = staging / f
                            if src.exists():
                                shutil.copy2(src, leaf / f)
                        # THE PROVENANCE THE READER NEEDS, in the leaf itself.
                        sv = getattr(runner, "grace_serving", None)
                        (leaf / "e1_provenance.json").write_text(
                            json.dumps(
                                {
                                    "cell": tag,
                                    "source_cell": cell,
                                    "arm": arm,
                                    "algo": algo,
                                    "env": env,
                                    "seed": sd,
                                    "dataset_id": did,
                                    "stamp": stamps[(tag, env, sd)],
                                    "q1_truth": truth,
                                    "grace": (
                                        None
                                        if sv is None
                                        else {
                                            "abstained": sv.abstained,
                                            "label": sv.label(),
                                            "lo": sv.lo,
                                            "hi": sv.hi,
                                            "meta": {
                                                k: v
                                                for k, v in (sv.meta or {}).items()
                                                if isinstance(
                                                    v, (int, float, str, bool)
                                                )
                                            },
                                        }
                                    ),
                                    "seconds": round(time.time() - t0, 1),
                                },
                                indent=1,
                            )
                        )
                        shutil.rmtree(staging, ignore_errors=True)
                        print(f"  -> {leaf}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nE1 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PR 5 — the (regime × L-shaped-sweep) sweep driver.

ONE cell = one job. The 7 sweep points (an L: a shared ``basic`` origin + a
``biased`` arm + a ``confounded`` arm, NO cross-product) are the inner loop and
share ONE generator checkpoint per (env, seed), so every cross-arm delta is PAIRED
and never confounded by generator variance (the correctness core, CHANGE 1). We do
NOT pair across cells — different obs spaces make that impossible.

Results land in a PARALLEL ``results/`` tree whose PATH SEGMENTS carry the
parameters (CHANGE 3):

    results/{regime}/beta_{beta*100:03d}_sigma_{sigma*100:03d}/{env}/{algo}/{critic}/{seed}/

x100 zero-padded (the existing gamma_100 convention). Γ is a METHOD parameter and
does NOT enter the path (PR 4) — it is a logged column. Subcell labels
(basic/biased/confounded) are DERIVED from (beta, sigma) at reporting time, NEVER
stored in a path (store a label and you can never reslice). A leaf is an ORDINARY
run dir: it holds the same file set a run dir holds today (config.yaml,
train_metrics.csv, eval_metrics.csv, arm_diagnostics.csv,
critic_ablation_metrics.csv) — the runner's writer is unchanged; only the run_dir
it is handed differs. No current renderer reads this tree; PR 6 wires reporting.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

# --------------------------------------------------------------------------- #
# The L (CHANGE 2/3): two 1-D arms sharing an origin. NOT a cross-product.     #
# --------------------------------------------------------------------------- #
BETA_ARM: Tuple[float, ...] = (0.25, 0.50, 0.75)
SIGMA_ARM: Tuple[float, ...] = (0.25, 0.50, 1.00)

# The critic sets per arm (CHANGE 4). ``basic`` and ``confounded`` run the FULL
# strategy set (basic is the null-calibration run — it is what makes the gate
# meaningful, so it is not optional). ``biased`` (sigma=0, no backdoor path) runs
# observational only: the deconfounding critics have nothing to do there and the
# biased arm's metric is coverage. ``sensitivity`` REQUIRES ``observational`` in the
# set (PR 4) — the FULL set includes it, so the requirement holds by construction.
ADAPTIVE_CRITICS: Tuple[str, ...] = ("observational", "proximal", "oracle_u")
FULL_CRITICS: Tuple[str, ...] = ("observational", "proximal", "oracle_u", "sensitivity")


def sweep_points() -> List[Tuple[float, float]]:
    """The 7 (beta, sigma) points of the L: the shared origin, then the two arms."""
    pts = [(0.0, 0.0)]  # basic — ONE run, the shared reference for both arms
    pts += [(b, 0.0) for b in BETA_ARM]  # biased arm (sigma held at 0)
    pts += [(0.0, s) for s in SIGMA_ARM]  # confounded arm (beta held at 0)
    return pts


def arm_label(beta: float, sigma: float) -> str:
    """DERIVE the subcell from (beta, sigma). This is the ONLY source of the
    basic/biased/confounded label — it is never stored in a path, so any run can be
    resliced from its parameters alone (CHANGE 3, M3)."""
    b, s = float(beta), float(sigma)
    if b == 0.0 and s == 0.0:
        return "basic"
    if b > 0.0 and s == 0.0:
        return "biased"
    if b == 0.0 and s > 0.0:
        return "confounded"
    raise ValueError(
        f"(beta={b}, sigma={s}) is off the L: the sweep is two 1-D arms sharing an "
        "origin, never the (beta>0, sigma>0) cross-product (out of scope)."
    )


def critics_for_arm(arm: str) -> List[str]:
    """The critic set for an arm (CHANGE 4)."""
    if arm in ("basic", "confounded"):
        return list(FULL_CRITICS)
    if arm == "biased":
        return ["observational"]
    raise ValueError(f"unknown arm '{arm}'")


def arm_behavior(beta: float, sigma: float) -> Tuple[str, float]:
    """(behavior_policy, strength) for a sweep point — the PR-3 arm policies, all
    built on the SHARED pi_basic.

    basic (0,0) collects with ``bias_confounded_action`` at σ=0: its MARGINAL is
    exactly pi_basic (marginally matched) and A⊥U so it is unconfounded, but the
    nuisance U IS recorded — which is what lets the basic null-calibration run host
    the FULL critic set (oracle_u/proximal need the per-transition U). This is the
    established σ=0-anchor construction (test_sigma_zero_anchor). biased (β,0) uses
    the ``biased`` policy (no U — its critic set is observational only). confounded
    (0,σ) uses ``bias_confounded_action`` at strength σ."""
    arm = arm_label(beta, sigma)
    if arm == "biased":
        return "biased", float(beta)
    # basic AND confounded both use the action-dependent confounder policy; σ=0 for
    # basic makes it the unconfounded, U-recorded origin shared by both arms.
    return "bias_confounded_action", float(sigma)


def c_r_for(default_c_r: float, beta: float, sigma: float):
    """The reward-confounder magnitude for a sweep point. basic AND confounded use the
    cell's ``confounder_c_r`` (both collect via bias_confounded_action, which records
    U and gates the reward on a_bad — the action-dependent gate requires that gating,
    so c_r>0 is needed even at the σ=0 basic origin). At σ=0 the U-reward-noise is
    ACTION-INDEPENDENT (A⊥U), i.e. unbiased, so the adaptive critics still collapse at
    the origin (the null-calibration run). The biased arm injects no U at all (None)."""
    arm = arm_label(beta, sigma)
    if arm in ("basic", "confounded"):
        return float(default_c_r)
    return None


def _p3(x: float) -> str:
    """x100, zero-padded to 3 (matches the gamma_100 / sigma_050 conventions)."""
    return f"{int(round(float(x) * 100)):03d}"


def param_dirname(beta: float, sigma: float) -> str:
    """The single parameter segment: ``beta_{bbb}_sigma_{sss}``. Labels never here."""
    return f"beta_{_p3(beta)}_sigma_{_p3(sigma)}"


def results_leaf(
    root: str | Path,
    regime: str,
    beta: float,
    sigma: float,
    env: str,
    algo: str,
    critic: str,
    seed: int,
) -> Path:
    """The parameter-addressed run-dir leaf (CHANGE 3). Every segment is a parameter
    or an entity; no basic/biased/confounded label and no gamma anywhere."""
    return (
        Path(root)
        / regime
        / param_dirname(beta, sigma)
        / _safe(env)
        / _safe(algo)
        / _safe(critic)
        / str(seed)
    )


def _safe(name: str) -> str:
    return str(name).replace("/", "-")


# --------------------------------------------------------------------------- #
# Sweep spec (CHANGE 2: parsed from a cell's sweep.yaml + the _base fragments)  #
# --------------------------------------------------------------------------- #
@dataclass
class SweepSpec:
    regime: str  # offline_mdp | offline_pomdp | online_mdp | online_pomdp
    observability: str  # mdp | pomdp
    data_regime: str  # offline | online
    generator_algo: str
    envs: List[str]
    algos: List[str]
    seeds: List[int]
    pi_basic_epsilon: float
    confounder_c_r: float
    budgets: Dict[str, int] = field(default_factory=dict)
    discrete_only: bool = True
    # POMDP regimes mask these obs indices per env (the Cell-4/8 observability axis).
    mask_indices: Dict[str, List[int]] = field(default_factory=dict)
    # How many (env, seed) GROUPS the supervisor runs concurrently (regime-shared
    # ``_base/parallel.yaml``, overridable per sweep.yaml). DEFAULT 1 = the serial
    # in-process run_cell path (byte-identical to pre-supervisor). >=2 opts into the
    # subprocess pool (src/benchmarking/sweep_supervisor.py). run_cell itself is
    # untouched — it stays serial WITHIN a group; parallelism is across groups only.
    max_workers: int = 1

    def budget(self, key: str, default: int) -> int:
        return int(self.budgets.get(key, default))


def load_sweep_spec(sweep_yaml: str | Path) -> SweepSpec:
    """Load a cell's ``sweep.yaml``, merging the shared ``_base/*.yaml`` fragments
    (envs/algos/seeds/budgets) that sit two levels up. Explicit keys in sweep.yaml
    win over the _base defaults."""
    p = Path(sweep_yaml)
    cfg = yaml.safe_load(p.read_text()) or {}
    base_dir = p.parent.parent / "_base"
    base: Dict = {}
    if base_dir.is_dir():
        for frag in ("envs", "algos", "seeds", "budgets", "parallel"):
            fp = base_dir / f"{frag}.yaml"
            if fp.exists():
                loaded = yaml.safe_load(fp.read_text()) or {}
                base.update(loaded if isinstance(loaded, dict) else {frag: loaded})

    def pick(key, default=None):
        return cfg.get(key, base.get(key, default))

    return SweepSpec(
        regime=cfg["regime"],
        observability=pick("observability", "mdp"),
        data_regime=pick("data_regime", "offline"),
        generator_algo=pick("generator_algo", "dqn"),
        envs=list(pick("envs", [])),
        algos=list(pick("algos", [])),
        seeds=[int(s) for s in pick("seeds", [0])],
        pi_basic_epsilon=float(pick("pi_basic_epsilon", 0.5)),
        confounder_c_r=float(pick("confounder_c_r", 1.0)),
        budgets=dict(pick("budgets", {}) or {}),
        discrete_only=bool(pick("discrete_only", True)),
        mask_indices={
            k: [int(i) for i in v] for k, v in (pick("mask_indices", {}) or {}).items()
        },
        max_workers=int(pick("max_workers", 1)),
    )


# --------------------------------------------------------------------------- #
# The shared-generator guarantee (CHANGE 1, M1)                                #
# --------------------------------------------------------------------------- #
def assert_shared_generator(hashes: Dict[Tuple[float, float], str]) -> str:
    """Refuse a cell whose sweep points carry different generator-checkpoint hashes:
    that means the arms were collected under different pi_basic and EVERY cross-arm
    comparison is confounded by generator variance (the identifiability failure this
    whole driver exists to prevent). Returns the single shared hash on success."""
    uniq = sorted(set(hashes.values()))
    if len(uniq) != 1:
        detail = ", ".join(
            f"beta_{_p3(b)}_sigma_{_p3(s)}={h[:8]}"
            for (b, s), h in sorted(hashes.items())
        )
        raise ValueError(
            "shared-generator violation: the cell's sweep points carry "
            f"{len(uniq)} distinct generator-checkpoint hashes ({detail}). All arms "
            "MUST share one pi_basic; regenerate the cell from a single generator "
            "checkpoint (see build_generator_agent + generate_offline_dataset(agent=))."
        )
    return uniq[0]


# --------------------------------------------------------------------------- #
# Reporting-side derivation (CHANGE 3, M3) — reslice params -> subcell, no rerun #
# --------------------------------------------------------------------------- #
_PARAM_RE = None


def parse_param_dir(name: str) -> Tuple[float, float]:
    """Inverse of ``param_dirname``: ``beta_050_sigma_000`` -> (0.5, 0.0)."""
    import re

    m = re.fullmatch(r"beta_(\d{3})_sigma_(\d{3})", str(name))
    if not m:
        raise ValueError(f"not a parameter dir: {name!r}")
    return int(m.group(1)) / 100.0, int(m.group(2)) / 100.0


def reslice_results(results_root: str | Path, regime: str) -> List[dict]:
    """Walk a regime's parameter tree and DERIVE the subcell for each leaf from its
    (beta, sigma) path segment — no labels were stored, so the slice is recomputable
    without re-running anything (M3). Returns one record per leaf."""
    root = Path(results_root) / regime
    out: List[dict] = []
    if not root.is_dir():
        return out
    for param_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            beta, sigma = parse_param_dir(param_dir.name)
        except ValueError:
            continue
        arm = arm_label(beta, sigma)
        for leaf in sorted(param_dir.rglob("*")):
            if leaf.is_dir() and (leaf / "config.yaml").exists():
                rel = leaf.relative_to(param_dir).parts  # env / algo / critic / seed
                out.append(
                    {
                        "regime": regime,
                        "beta": beta,
                        "sigma": sigma,
                        "arm": arm,
                        "env": rel[0] if len(rel) > 0 else None,
                        "algo": rel[1] if len(rel) > 1 else None,
                        "critic": rel[2] if len(rel) > 2 else None,
                        "seed": rel[3] if len(rel) > 3 else None,
                        "path": str(leaf),
                    }
                )
    return out


# --------------------------------------------------------------------------- #
# Execution (CHANGE 5): one cell = one job; 7 paired sweep points inner loop.  #
# --------------------------------------------------------------------------- #
def _dataset_id(
    prefix: str, regime: str, env: str, beta: float, sigma: float, seed: int
) -> str:
    # Minari ids cannot contain dots; keep it lowercase-slug + the -vN suffix.
    return (
        f"{prefix}/{regime}/{_safe(env).lower()}-{param_dirname(beta, sigma)}"
        f"-seed{seed}-v0"
    )


def _write_run_metadata(
    run_dir: Path,
    spec: SweepSpec,
    env: str,
    algo: str,
    beta: float,
    sigma: float,
    seed: int,
    critics: List[str],
) -> None:
    """Write the two run-dir artifacts the RUNNER does not write (main.py does):
    ``config.yaml`` + ``metadata.json``, so a leaf holds the same file set a live
    run dir holds. Parameters go in the CONTENT here too (the path already carries
    them); labels are still derived, never stored."""
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "env": {"envs": [env], "seed": seed},
        "training": {
            "mode": "critic_ablation",
            "algos": [algo],
            "ablation": {"critics": list(critics)},
        },
        "sweep": {
            "regime": spec.regime,
            "beta": float(beta),
            "sigma": float(sigma),
            # arm is DERIVED, recorded for convenience but never a path segment.
            "arm": arm_label(beta, sigma),
            "pi_basic_epsilon": spec.pi_basic_epsilon,
        },
        "timestamp": "sweep",
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot))
    (run_dir / "metadata.json").write_text(json.dumps({"timestamp": "sweep"}, indent=2))


def _slice_critic_csv(src_csv: Path, dst_csv: Path, critic: str) -> None:
    """Copy ``critic_ablation_metrics.csv`` keeping the header + only ``critic``'s
    rows, so each per-critic leaf is a self-contained run dir sliced from the one
    shared ablation (the critics were fit on the SAME episode-grouped stream)."""
    import csv

    if not src_csv.exists():
        return
    with src_csv.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else None
    if fieldnames is None:
        shutil.copy2(src_csv, dst_csv)
        return
    with dst_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            if r.get("critic") == critic:
                w.writerow(r)


def _run_point(
    spec: SweepSpec,
    env: str,
    algo: str,
    seed: int,
    beta: float,
    sigma: float,
    dataset_id: str,
    results_root: str | Path,
    device: str | None,
) -> List[Path]:
    """Run ONE arm point (a single critic-ablation over the arm's critic set on the
    shared stream), then explode it into the per-``{critic}`` run-dir leaves."""
    import tempfile

    from src.benchmarking.critic_ablation import CriticAblationConfig
    from src.benchmarking.registry import registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    arm = arm_label(beta, sigma)
    critics = critics_for_arm(arm)
    bp, strength = arm_behavior(beta, sigma)
    recurrent = spec.observability == "pomdp"

    # Stage the ONE shared ablation run OUTSIDE the results tree, then explode it into
    # the per-critic leaves — so results/ holds only the parameter-addressed leaves
    # (no staging residue polluting the {algo}/ level).
    staging = Path(tempfile.mkdtemp(prefix="regime_sweep_"))

    env_cfg = EnvConfig(
        env_id=env,
        n_train_envs=spec.budget("n_train_envs", 2),
        n_eval_envs=spec.budget("n_eval_envs", 2),
        rollout_len=spec.budget("rollout_len", 2),
        seed=seed,
        offline_dataset=dataset_id,
        behavior_policy=bp,
        behavior_strength=strength,
        pi_basic_epsilon=spec.pi_basic_epsilon,
        confounder_c_r=c_r_for(spec.confounder_c_r, beta, sigma),
        mask_indices=(spec.mask_indices.get(env) if recurrent else None),
    )
    # offline_grad_steps (feat/offline-budget-key): the offline learner's total
    # optimiser-step count. None when a cell omits the key -> the runner warns and
    # falls back to the legacy n_episodes*rollout_len product (never silent).
    _ogs = spec.budgets.get("offline_grad_steps")
    train_cfg = TrainingConfig(
        n_episodes=spec.budget("n_episodes", 1),
        n_checkpoints=spec.budget("n_checkpoints", 2),
        deterministic=True,
        device=device or "cpu",
        algorithm=algo,
        aggregation="iqm",
        critic_network=("lstm" if recurrent else "mlp"),
        offline_grad_steps=(int(_ogs) if _ogs is not None else None),
    )
    _write_run_metadata(staging, spec, env, algo, beta, sigma, seed, critics)
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(staging), timestamp="sweep"),
        registry.get(algo),
        critic_ablation_cfg=CriticAblationConfig(critics=list(critics)),
    ).run()

    shared_files = (
        "config.yaml",
        "metadata.json",
        "train_metrics.csv",
        "eval_metrics.csv",
        "arm_diagnostics.csv",
    )
    leaves: List[Path] = []
    for critic in critics:
        leaf = results_leaf(
            results_root, spec.regime, beta, sigma, env, algo, critic, seed
        )
        leaf.mkdir(parents=True, exist_ok=True)
        for fn in shared_files:
            src = staging / fn
            if src.exists():
                shutil.copy2(src, leaf / fn)
        _slice_critic_csv(
            staging / "critic_ablation_metrics.csv",
            leaf / "critic_ablation_metrics.csv",
            critic,
        )
        leaves.append(leaf)
    shutil.rmtree(staging, ignore_errors=True)
    return leaves


def run_cell(
    sweep_yaml: str | Path,
    *,
    results_root: str | Path = "results",
    dataset_prefix: str = "sweep",
    device: str | None = None,
    envs: Sequence[str] | None = None,
    algos: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    budget_overrides: Dict[str, int] | None = None,
) -> List[Path]:
    """Run one cell (CHANGE 5). For each (env, seed): build ONE generator agent,
    generate all 7 sweep-point datasets from it, REFUSE the cell if their hashes
    differ (M1), then train each arm point into the parameter-addressed leaves. The
    optional envs/algos/seeds override the spec (used to shrink a cell for tests).
    Offline regimes only — online cells have no offline generator to share (their
    behavior policy IS the learner); use the online path for those."""
    from src.benchmarking.registry import register_default_algorithms
    from src.config.seeding import set_seed
    from src.envs.offline.generate import (
        build_generator_agent,
        generate_offline_dataset,
    )
    from src.envs.registry import register_default_env_wrappers

    spec = load_sweep_spec(sweep_yaml)
    if budget_overrides:
        spec.budgets = {**spec.budgets, **budget_overrides}
    if spec.data_regime != "offline":
        raise NotImplementedError(
            f"run_cell drives the OFFLINE path; regime '{spec.regime}' is online "
            "(no offline generator to share). The online arms run through the "
            "on-policy benchmark path — out of this driver's offline scope."
        )
    register_default_algorithms()
    register_default_env_wrappers()

    run_envs = list(envs) if envs is not None else spec.envs
    run_algos = list(algos) if algos is not None else spec.algos
    run_seeds = [int(s) for s in (seeds if seeds is not None else spec.seeds)]

    written: List[Path] = []
    for env in run_envs:
        for seed in run_seeds:
            agent, _hash = build_generator_agent(
                env, spec.generator_algo, "random", seed=seed, device=device
            )
            # 1) generate all 7 sweep points from the ONE shared agent
            datasets: Dict[Tuple[float, float], str] = {}
            point_hashes: Dict[Tuple[float, float], str] = {}
            for beta, sigma in sweep_points():
                bp, strength = arm_behavior(beta, sigma)
                did = _dataset_id(dataset_prefix, spec.regime, env, beta, sigma, seed)
                try:
                    import minari

                    minari.delete_dataset(did)
                except Exception:
                    pass
                # Seed EACH point's rollout independently so a dataset is reproducible
                # per (seed, point) regardless of how many points preceded it — the
                # realized confounding (and thus the gate outcome) must not depend on
                # generation order. The shared agent (pi_basic) is already fixed.
                set_seed(seed, deterministic=True)
                ds = generate_offline_dataset(
                    env_id=env,
                    generator_algo=spec.generator_algo,
                    tier="random",
                    behavior_policy=bp,
                    behavior_strength=strength,
                    pi_basic_epsilon=spec.pi_basic_epsilon,
                    confounder_c_r=c_r_for(spec.confounder_c_r, beta, sigma),
                    rollout_episodes=spec.budget("rollout_episodes", 30),
                    seed=seed,
                    dataset_id=did,
                    agent=agent,
                    device=device,
                )
                # The biased arm's ``biased`` policy is unconfounded, so its signature
                # leaves ``behavior_strength_sigma`` = None; the arm genuinely sits at
                # σ=0, so record that (the offline strategy path reads it as the
                # scoring σ and the σ=0 gate-bypass keys on it). basic / confounded
                # already carry their σ from the confounded signature.
                meta = ds.storage.metadata
                if meta.get("behavior_strength_sigma") is None:
                    ds.storage.update_metadata(
                        {"behavior_strength_sigma": float(sigma)}
                    )
                datasets[(beta, sigma)] = did
                point_hashes[(beta, sigma)] = ds.storage.metadata[
                    "generator_checkpoint_hash"
                ]
            # 2) M1: refuse a cell whose arms carry different generator hashes,
            #    BEFORE spending any training on a non-identified taxonomy.
            assert_shared_generator(point_hashes)
            # 3) train each arm point into its parameter-addressed leaves
            for beta, sigma in sweep_points():
                for algo in run_algos:
                    written += _run_point(
                        spec,
                        env,
                        algo,
                        seed,
                        beta,
                        sigma,
                        datasets[(beta, sigma)],
                        results_root,
                        device,
                    )
    return written


# The one-flag --smoke budget: tiny everything so a cell runs end-to-end in a couple
# of minutes (rollout_episodes=40 is deliberate — fewer makes the σ=1.0 gate flaky).
_SMOKE_BUDGET = {
    "n_episodes": 1,
    "n_checkpoints": 2,
    "n_train_envs": 2,
    "n_eval_envs": 2,
    "rollout_len": 2,
    "rollout_episodes": 40,
    # tiny offline budget so --smoke exercises the new offline path fast; without it
    # the merge with _base inherits the 50_000 production offline_grad_steps.
    "offline_grad_steps": 4,
}


def _main(argv: List[str] | None = None) -> int:
    import argparse

    from src.config.device import detect_device

    ap = argparse.ArgumentParser(
        description="Run one (regime × L-sweep) cell: ONE generator checkpoint per "
        "(env, seed), 7 paired sweep points, parameter-addressed results/ leaves. "
        "Offline cells only — online cells raise NotImplementedError (their behavior "
        "policy IS the learner, so there is no offline generator to share).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("sweep_yaml", help="path to a cell's sweep.yaml")
    ap.add_argument(
        "--results-root",
        default=None,
        help="results tree root (default 'results'; 'results_smoke' under --smoke)",
    )
    ap.add_argument(
        "--dataset-prefix",
        default=None,
        help="Minari dataset id prefix (default 'sweep'; 'smoke' under --smoke)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    ap.add_argument(
        "--envs", nargs="+", default=None, help="override the cell's env list"
    )
    ap.add_argument(
        "--algos", nargs="+", default=None, help="override the cell's algo list"
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="override the cell's seed list",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one-flag smoke run: tiny 1-episode budget + results_smoke/ + 'smoke' "
        "dataset prefix (confirm a cell runs before committing to the full budget)",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="how many (env, seed) GROUPS to run concurrently (overrides the cell's "
        "max_workers; default from _base/parallel.yaml = 1 = serial). 1 keeps the "
        "byte-identical in-process path; >=2 fans groups across subprocesses.",
    )
    args = ap.parse_args(argv)

    # --smoke sets the tiny budget AND the throwaway results_root/prefix, but an
    # explicit --results-root/--dataset-prefix still wins (None = not given).
    budget_overrides = dict(_SMOKE_BUDGET) if args.smoke else None
    results_root = args.results_root or ("results_smoke" if args.smoke else "results")
    dataset_prefix = args.dataset_prefix or ("smoke" if args.smoke else "sweep")
    device = args.device or str(detect_device())

    # Effective workers: --max-workers wins over the cell's max_workers (from
    # _base/parallel.yaml). >=2 hands off to the supervisor; 1 stays on the
    # byte-identical in-process run_cell path below.
    spec = load_sweep_spec(args.sweep_yaml)
    eff_workers = int(
        args.max_workers if args.max_workers is not None else spec.max_workers
    )

    if eff_workers >= 2:
        from src.benchmarking.sweep_supervisor import format_summary, run_sweep

        result = run_sweep(
            args.sweep_yaml,
            results_root=results_root,
            dataset_prefix=dataset_prefix,
            device=device,
            envs=args.envs,
            algos=args.algos,
            seeds=args.seeds,
            max_workers=eff_workers,
            smoke=args.smoke,
        )
        print(format_summary(result))
        # A failing group must surface: non-zero exit, never a silent drop.
        return 0 if result.ok else 1

    leaves = run_cell(
        args.sweep_yaml,
        results_root=results_root,
        dataset_prefix=dataset_prefix,
        device=device,
        envs=args.envs,
        algos=args.algos,
        seeds=args.seeds,
        budget_overrides=budget_overrides,
    )
    print(
        f"[regime_sweep] wrote {len(leaves)} run-dir leaves under {results_root}/ "
        f"(device={device}{'; SMOKE budget' if args.smoke else ''})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

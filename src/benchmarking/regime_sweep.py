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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

# --------------------------------------------------------------------------- #
# The L (CHANGE 2/3): two 1-D arms sharing an origin. NOT a cross-product.     #
# --------------------------------------------------------------------------- #
BETA_ARM: Tuple[float, ...] = (0.25, 0.50, 0.75)
SIGMA_ARM: Tuple[float, ...] = (0.25, 0.50, 1.00)

# The two simulation components a cell can run. ``critic_ablation`` is the
# historical (and default) one: every sweep point trains the arm's critic set on
# ONE shared stream and explodes into per-{critic} leaves. ``classical`` is the
# plain benchmark: every algo on every env at every sweep point, NO critic axis —
# its leaves live under a separate ``{regime}/classical/`` subtree so the two
# simulations never mix in one walker's path schema.
SIMULATIONS: Tuple[str, ...] = ("classical", "critic_ablation")

# Strategy names a ``critics:`` block may declare, per data regime. Offline runs
# them through the CriticAblationManager on one shared stream; online has no such
# manager (runner guard: "the online bias_confounded regime is deferred"), so the
# online ablation compares ALGO VARIANTS (dqn vs online_dqn_proximal) — one run
# per strategy, same leaf schema. oracle_u / sensitivity have no online variant.
KNOWN_STRATEGIES: Tuple[str, ...] = (
    "observational",
    "proximal",
    "oracle_u",
    "sensitivity",
)
ONLINE_STRATEGIES: Tuple[str, ...] = ("observational", "proximal")

# The critic sets per arm (CHANGE 4). ``basic`` and ``confounded`` run the FULL
# strategy set (basic is the null-calibration run — it is what makes the gate
# meaningful, so it is not optional). ``biased`` (sigma=0, no backdoor path) runs
# observational only: the deconfounding critics have nothing to do there and the
# biased arm's metric is coverage. ``sensitivity`` REQUIRES ``observational`` in the
# set (PR 4) — the FULL set includes it, so the requirement holds by construction.
ADAPTIVE_CRITICS: Tuple[str, ...] = ("observational", "proximal", "oracle_u")
FULL_CRITICS: Tuple[str, ...] = ("observational", "proximal", "oracle_u", "sensitivity")


def sweep_points(
    beta_arm: Sequence[float] = BETA_ARM,
    sigma_arm: Sequence[float] = SIGMA_ARM,
    include_basic: bool = True,
) -> List[Tuple[float, float]]:
    """The (beta, sigma) points of the L: the shared origin, then the two arms.
    Defaults reproduce the canonical 7-point L; a cell's ``sweep:`` block may
    declare different arm values or disable arms with ``false`` (parsed by
    load_sweep_spec, same L shape)."""
    pts: List[Tuple[float, float]] = []
    if include_basic:
        pts.append((0.0, 0.0))  # basic — ONE run, the shared reference for both arms
    pts += [(float(b), 0.0) for b in beta_arm]  # biased arm (sigma held at 0)
    pts += [(0.0, float(s)) for s in sigma_arm]  # confounded arm (beta held at 0)
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


def critics_for_arm(arm: str, data_regime: str = "offline") -> List[str]:
    """The DEFAULT critic set for an arm (CHANGE 4). Offline: the full strategy
    set on basic/confounded (basic is the null-calibration run), observational
    only on biased (sigma=0 has no backdoor path). Online: only the strategies
    with an online algo variant (observational/proximal)."""
    if arm == "biased":
        return ["observational"]
    if arm not in ("basic", "confounded"):
        raise ValueError(f"unknown arm '{arm}'")
    if data_regime == "online":
        return list(ONLINE_STRATEGIES)
    return list(FULL_CRITICS)


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


def _with_c_r(arm_kwargs: Dict, fallback_c_r: float) -> Dict:
    """Merge the arm kwargs with the cell's scalar c_r, the DERIVED one winning.

    Under the compensated gated-reward sweep, arm_generator_kwargs carries
    confounder_c_r = M / d and the scalar must yield; everywhere else the
    scalar is the value, exactly as before. One helper so the precedence rule
    exists in one place instead of five call sites.
    """
    out = dict(arm_kwargs)
    out.setdefault("confounder_c_r", fallback_c_r)
    return out


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


def parse_algo_entry(entry: str, observability: str) -> Tuple[str, str, str, str]:
    """Normalize one ``algos`` entry -> (name, actor_network, critic_network,
    algo_id).

    Two forms (both plain strings, so entries survive the supervisor's
    ``--algos`` CLI hand-off verbatim):

      * ``"cql"`` — the historical AUTO form: mlp trunks on an mdp cell, lstm
        critic on a pomdp cell (byte-identical to the pre-split driver). The
        leaf/path id is the bare name.
      * ``"dqn__mlp__mlp"`` — EXPLICIT ``name__actor__critic`` (the repo's
        canonical-id convention, PR #49): declares the trunks per row, e.g. a
        memoryless mlp baseline next to the recurrent learner in a pomdp cell.
        The leaf/path id is the entry VERBATIM, so two rows of the same base
        algo with different trunks never collide."""
    e = str(entry)
    if "__" in e:
        parts = e.split("__")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"algo entry {e!r} is not 'name' or 'name__actor__critic' "
                "(e.g. dqn__lstm__lstm, offline_dqn__mlp__mlp)."
            )
        name, actor, critic = parts
        return name, actor, critic, e
    critic = "lstm" if observability == "pomdp" else "mlp"
    return e, "mlp", critic, e


def classical_results_leaf(
    root: str | Path,
    regime: str,
    beta: float,
    sigma: float,
    env: str,
    algo: str,
    seed: int,
) -> Path:
    """The CLASSICAL simulation's run-dir leaf: same parameter addressing, NO
    critic segment, under a ``{regime}/classical/`` subtree. The subtree keeps the
    two simulations' path schemas apart: the ablation walkers (regime_report)
    require env/algo/critic/seed below the parameter dir and skip anything else,
    so classical leaves are invisible to them by construction."""
    return (
        Path(root)
        / regime
        / "classical"
        / param_dirname(beta, sigma)
        / _safe(env)
        / _safe(algo)
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
    # Rollout speed knobs (docs/dataset_generation_speedup.md), forwarded to
    # generate_offline_dataset. Generator TRAINING stays on the run device; only
    # the ROLLOUT moves. Defaults are the fast path (CPU stepping, 16 slots):
    # ~185x the pre-speedup CUDA-batch-1 rollout on CartPole. Set
    # legacy_rollout: true to regenerate historical dataset ids bit-for-bit.
    rollout_device: str = "cpu"
    rollout_n_envs: int = 16
    legacy_rollout: bool = False
    # S4 cross-simulation dataset reuse. Dataset ids carry no simulation
    # component, so a regime's classical and critic_ablation cells ask for the
    # SAME ids — the second cell would otherwise regenerate byte-equivalent data.
    # When True, a point whose existing dataset carries a matching
    # generation_fingerprint (every generation-determining input, incl. the
    # rollout mode) is reused instead of regenerated. Set False to always
    # regenerate.
    reuse_datasets: bool = True
    # WHICH simulation this cell runs: "critic_ablation" (default — the historical
    # behavior of every sweep.yaml) or "classical" (algo x env benchmark, no
    # critic axis, ``{regime}/classical/`` leaves).
    simulation: str = "critic_ablation"
    # Per-arm critic-set OVERRIDE parsed from the cell's ``critics:`` block
    # (previously documentation-only). Empty -> critics_for_arm defaults.
    critics: Dict[str, List[str]] = field(default_factory=dict)
    # The L's arm values parsed from the cell's ``sweep:`` block (previously
    # documentation-only). Defaults = the canonical 7-point L. An arm set to
    # ``false`` in the block is excluded (empty arm / include_basic False).
    beta_arm: Tuple[float, ...] = BETA_ARM
    sigma_arm: Tuple[float, ...] = SIGMA_ARM
    include_basic: bool = True
    # GRACE v2 diagram arm. None = the historical cells, which declare no
    # diagram and get exactly the generator kwargs they always did. Set to a
    # catalogue id (D-D, D-E, D-B-prime, D-A-null) to collect that diagram's
    # channels; WHICH channels exist is derived from the catalogue entry, and
    # only their strengths come from the YAML (see envs/offline/diagram_arms).
    diagram: Optional[str] = None
    proxy_strength: Optional[float] = None
    instrument_strength: Optional[float] = None
    u_drift: Optional[float] = None
    gate_probs: Optional[Sequence[float]] = None
    # Compensated gated-reward sweep (D-D revision 2026-08-21): M = c_r * d
    # held fixed, c_r DERIVED as M / d in arm_knobs -- the single construction
    # site. A YAML that sets this must NOT also set confounder_c_r (arm_knobs
    # raises on the contradiction).
    gate_mean_effect: Optional[float] = None

    def arm_generator_kwargs(self, sigma: float) -> Dict:
        """The diagram channels for a sweep point, or {} for a historical cell."""
        if self.diagram is None:
            return {}
        from src.envs.offline.diagram_arms import arm_knobs

        k = arm_knobs(
            self.diagram,
            sigma=sigma,
            # Under the compensated sweep c_r is DERIVED from M and d;
            # supplying the spec's scalar too would trip arm_knobs'
            # contradiction check, which is the intended failure for a YAML
            # that declares both.
            confounder_c_r=(
                None if self.gate_mean_effect is not None else self.confounder_c_r
            ),
            proxy_strength=self.proxy_strength,
            instrument_strength=self.instrument_strength,
            u_drift=self.u_drift,
            gate_probs=self.gate_probs,
            gate_mean_effect=self.gate_mean_effect,
        )
        out = {
            "proxy_strength": k.proxy_strength,
            "instrument_strength": k.instrument_strength,
            "u_drift": k.u_drift,
            "gate_probs": k.gate_probs,
            "n_proxies": k.n_proxies,
        }
        if self.gate_mean_effect is not None:
            # The DERIVED c_r rides in the kwargs; call sites yield to it
            # rather than passing their own (see the generation call sites).
            out["confounder_c_r"] = k.confounder_c_r
        return out

    def budget(self, key: str, default: int) -> int:
        return int(self.budgets.get(key, default))

    def points(self) -> List[Tuple[float, float]]:
        """This cell's sweep points (the declared L; canonical 7 by default)."""
        return sweep_points(self.beta_arm, self.sigma_arm, self.include_basic)

    def critics_for(self, arm: str) -> List[str]:
        """This cell's critic set for an arm: the ``critics:`` block when
        declared, else the data-regime-aware default."""
        declared = self.critics.get(arm)
        if declared is not None:
            return list(declared)
        return critics_for_arm(arm, self.data_regime)


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

    simulation = str(pick("simulation", "critic_ablation"))
    if simulation not in SIMULATIONS:
        raise ValueError(
            f"{p}: unknown simulation '{simulation}'; must be one of {SIMULATIONS}."
        )
    data_regime = str(pick("data_regime", "offline"))
    beta_arm, sigma_arm, include_basic = _parse_sweep_block(
        cfg.get("sweep"), source=str(p)
    )
    critics = _parse_critics_block(cfg.get("critics"), data_regime, source=str(p))

    return SweepSpec(
        regime=cfg["regime"],
        observability=pick("observability", "mdp"),
        data_regime=data_regime,
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
        rollout_device=str(pick("rollout_device", "cpu")),
        rollout_n_envs=int(pick("rollout_n_envs", 16)),
        legacy_rollout=bool(pick("legacy_rollout", False)),
        reuse_datasets=bool(pick("reuse_datasets", True)),
        simulation=simulation,
        critics=critics,
        beta_arm=beta_arm,
        sigma_arm=sigma_arm,
        include_basic=include_basic,
        diagram=pick("diagram", None),
        proxy_strength=_opt_float(pick("proxy_strength", None)),
        instrument_strength=_opt_float(pick("instrument_strength", None)),
        u_drift=_opt_float(pick("u_drift", None)),
        gate_probs=pick("gate_probs", None),
        gate_mean_effect=(
            None
            if pick("gate_mean_effect", None) is None
            else float(pick("gate_mean_effect", None))
        ),
    )


def _opt_float(v):
    return None if v is None else float(v)


def _as_float_list(val) -> List[float]:
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    return [float(val)]


def _arm_entry(block: dict, arm: str, default: dict, *, source: str):
    """One arm of the ``sweep:`` block. Absent or ``true`` -> the canonical
    default (backward compatible: commenting an arm OUT never shrinks the L);
    ``false`` -> None (the arm is EXCLUDED — the only way to drop one);
    a map -> itself (validated by the caller)."""
    val = block.get(arm, default)
    if val is True:
        return default
    if val is False:
        return None
    if not isinstance(val, dict):
        raise ValueError(
            f"{source}: sweep.{arm} must be a map (or false to exclude the arm), "
            f"got {val!r}."
        )
    return val


def _parse_sweep_block(
    block, *, source: str
) -> Tuple[Tuple[float, ...], Tuple[float, ...], bool]:
    """Parse a cell's ``sweep:`` block into (beta_arm, sigma_arm, include_basic),
    REFUSING any declaration off the L (the basic origin must be (0,0); each arm
    varies exactly one axis). Absent block -> the canonical arms; an ABSENT arm
    key also falls back to its canonical default (never a removal — this is what
    keeps legacy YAMLs byte-identical), so excluding an arm takes an EXPLICIT
    ``false``. ``basic: false`` drops the null-calibration anchor and warns:
    fine for a shrunk test run, wrong for a production/paper run."""
    if block is None:
        return BETA_ARM, SIGMA_ARM, True
    if not isinstance(block, dict):
        raise ValueError(f"{source}: 'sweep' must be a map of arms, got {block!r}.")
    unknown = set(block) - {"basic", "biased", "confounded"}
    if unknown:
        raise ValueError(
            f"{source}: unknown sweep arm(s) {sorted(unknown)}; the L has exactly "
            "basic/biased/confounded."
        )
    basic = _arm_entry(block, "basic", {"beta": 0.0, "sigma": 0.0}, source=source)
    include_basic = basic is not None
    if include_basic:
        if _as_float_list(basic.get("beta", 0.0)) != [0.0] or _as_float_list(
            basic.get("sigma", 0.0)
        ) != [0.0]:
            raise ValueError(
                f"{source}: sweep.basic must sit at the shared origin "
                "(beta=0, sigma=0), or be false to exclude it."
            )
    else:
        warnings.warn(
            f"{source}: sweep.basic is false — the basic origin is the "
            "null-calibration anchor; without it the cell cannot be "
            "null-calibrated. Fine for a shrunk test run, WRONG for a "
            "production/paper run.",
            stacklevel=2,
        )
    biased = _arm_entry(
        block, "biased", {"beta": list(BETA_ARM), "sigma": 0.0}, source=source
    )
    if biased is None:
        beta_arm: Tuple[float, ...] = ()
    else:
        if _as_float_list(biased.get("sigma", 0.0)) != [0.0]:
            raise ValueError(f"{source}: sweep.biased must hold sigma at 0 (the L).")
        beta_arm = tuple(_as_float_list(biased.get("beta", list(BETA_ARM))))
    confounded = _arm_entry(
        block, "confounded", {"beta": 0.0, "sigma": list(SIGMA_ARM)}, source=source
    )
    if confounded is None:
        sigma_arm: Tuple[float, ...] = ()
    else:
        if _as_float_list(confounded.get("beta", 0.0)) != [0.0]:
            raise ValueError(f"{source}: sweep.confounded must hold beta at 0 (the L).")
        sigma_arm = tuple(_as_float_list(confounded.get("sigma", list(SIGMA_ARM))))
    if any(b <= 0.0 for b in beta_arm) or any(s <= 0.0 for s in sigma_arm):
        raise ValueError(
            f"{source}: arm values must be > 0 (the origin is declared by basic)."
        )
    if not include_basic and not beta_arm and not sigma_arm:
        raise ValueError(
            f"{source}: every sweep arm is false — the cell has no points to run."
        )
    return beta_arm, sigma_arm, include_basic


def _parse_critics_block(
    block, data_regime: str, *, source: str
) -> Dict[str, List[str]]:
    """Parse a cell's ``critics:`` per-arm block (previously documentation-only).
    Validates strategy names, the sensitivity->observational requirement, and the
    online availability constraint. Absent block -> {} (critics_for_arm defaults)."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{source}: 'critics' must be a map of arms, got {block!r}.")
    unknown_arms = set(block) - {"basic", "biased", "confounded"}
    if unknown_arms:
        raise ValueError(f"{source}: unknown critics arm(s) {sorted(unknown_arms)}.")
    out: Dict[str, List[str]] = {}
    for arm, names in block.items():
        names = [str(n) for n in (names or [])]
        if not names:
            raise ValueError(f"{source}: critics.{arm} must be a non-empty list.")
        bad = [n for n in names if n not in KNOWN_STRATEGIES]
        if bad:
            raise ValueError(
                f"{source}: unknown critic strategy {bad}; known: {KNOWN_STRATEGIES}."
            )
        if data_regime == "online":
            offline_only = [n for n in names if n not in ONLINE_STRATEGIES]
            if offline_only:
                raise ValueError(
                    f"{source}: critics.{arm} declares {offline_only}, but only "
                    f"{ONLINE_STRATEGIES} have an online algo variant "
                    "(oracle_u/sensitivity are offline-only)."
                )
        if "sensitivity" in names and "observational" not in names:
            raise ValueError(
                f"{source}: critics.{arm} includes 'sensitivity', which requires "
                "'observational' in the same set (its pessimism_cost baseline)."
            )
        out[arm] = names
    return out


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
    critics: List[str] | None = None,
    mode: str = "critic_ablation",
) -> None:
    """Write the two run-dir artifacts the RUNNER does not write (main.py does):
    ``config.yaml`` + ``metadata.json``, so a leaf holds the same file set a live
    run dir holds. Parameters go in the CONTENT here too (the path already carries
    them); labels are still derived, never stored."""
    run_dir.mkdir(parents=True, exist_ok=True)
    training: Dict = {"mode": mode, "algos": [algo]}
    if critics is not None:
        training["ablation"] = {"critics": list(critics)}
    snapshot = {
        "env": {"envs": [env], "seed": seed},
        "training": training,
        "sweep": {
            "regime": spec.regime,
            "simulation": spec.simulation,
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
    critics = spec.critics_for(arm)
    bp, strength = arm_behavior(beta, sigma)
    recurrent = spec.observability == "pomdp"
    name, actor_net, critic_net, algo_id = parse_algo_entry(algo, spec.observability)

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
        **_with_c_r(
            spec.arm_generator_kwargs(sigma),
            c_r_for(spec.confounder_c_r, beta, sigma),
        ),
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
        algorithm=algo_id,
        aggregation="iqm",
        actor_network=actor_net,
        critic_network=critic_net,
        offline_grad_steps=(int(_ogs) if _ogs is not None else None),
        # videos land in the staging dir that is rmtree'd below — pure waste
        # (an ffmpeg spawn + rollout_len frame encodes per checkpoint).
        record_eval_video=False,
    )
    _write_run_metadata(staging, spec, env, algo_id, beta, sigma, seed, critics)
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(staging), timestamp="sweep"),
        registry.get(name),
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
            results_root, spec.regime, beta, sigma, env, algo_id, critic, seed
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


def _run_point_classical(
    spec: SweepSpec,
    env: str,
    algo: str,
    seed: int,
    beta: float,
    sigma: float,
    dataset_id: str | None,
    results_root: str | Path,
    device: str | None,
) -> List[Path]:
    """Run ONE classical (plain-benchmark) arm point: the algo trains on the
    point's data — the shared-generator offline dataset, or the online arm
    collection — with NO critic ablation. One leaf per (point, algo), written
    directly (no staging/explode: there is no critic axis to slice)."""
    from src.benchmarking.registry import registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    bp, strength = arm_behavior(beta, sigma)
    recurrent = spec.observability == "pomdp"
    name, actor_net, critic_net, algo_id = parse_algo_entry(algo, spec.observability)
    leaf = classical_results_leaf(
        results_root, spec.regime, beta, sigma, env, algo_id, seed
    )

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
        **_with_c_r(
            spec.arm_generator_kwargs(sigma),
            c_r_for(spec.confounder_c_r, beta, sigma),
        ),
        mask_indices=(spec.mask_indices.get(env) if recurrent else None),
    )
    _ogs = spec.budgets.get("offline_grad_steps")
    train_cfg = TrainingConfig(
        n_episodes=spec.budget("n_episodes", 1),
        n_checkpoints=spec.budget("n_checkpoints", 2),
        deterministic=True,
        device=device or "cpu",
        algorithm=algo_id,
        aggregation="iqm",
        actor_network=actor_net,
        critic_network=critic_net,
        offline_grad_steps=(
            int(_ogs) if (_ogs is not None and dataset_id is not None) else None
        ),
        record_eval_video=False,
    )
    _write_run_metadata(
        leaf, spec, env, algo_id, beta, sigma, seed, critics=None, mode="benchmark"
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(leaf), timestamp="sweep"),
        registry.get(name),
    ).run()
    return [leaf]


def resolve_online_strategy_algo(base: str, strategy: str) -> str:
    """Map (base online algo, id-strategy) -> the registered ALGO VARIANT that
    embodies that strategy online. ``observational`` is the base learner itself;
    other strategies resolve via the ``online_{base}_{strategy}`` /
    ``{base}_{strategy}`` naming conventions (e.g. dqn+proximal ->
    online_dqn_proximal, Gate B). Raises if no online variant exists — the online
    ablation compares algo variants because the CriticAblationManager's strategy
    path is offline-only by design (runner guard)."""
    from src.benchmarking.registry import registry

    if strategy == "observational":
        return base
    for cand in (f"online_{base}_{strategy}", f"{base}_{strategy}"):
        try:
            if registry.get(cand).data_regime == "online":
                return cand
        except KeyError:
            continue
    raise ValueError(
        f"no online algo variant for base '{base}' + strategy '{strategy}' "
        f"(looked for online_{base}_{strategy} / {base}_{strategy}); online "
        f"critic sets are limited to {ONLINE_STRATEGIES}."
    )


def _run_point_online_ablation(
    spec: SweepSpec,
    env: str,
    algo: str,
    seed: int,
    beta: float,
    sigma: float,
    results_root: str | Path,
    device: str | None,
) -> List[Path]:
    """Run ONE online critic-ablation arm point: one TRAINING RUN PER STRATEGY
    (dqn vs online_dqn_proximal, ...), each collecting its own arm stream —
    the online analog of the offline shared-stream ablation (Gate B's
    fixed-behavior construction lives inside the variant algo). Leaves reuse the
    ablation path schema: .../{env}/{base_algo}/{critic}/{seed}."""
    from src.benchmarking.registry import registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    arm = arm_label(beta, sigma)
    critics = spec.critics_for(arm)
    bp, strength = arm_behavior(beta, sigma)
    recurrent = spec.observability == "pomdp"
    name, actor_net, critic_net, algo_id = parse_algo_entry(algo, spec.observability)

    leaves: List[Path] = []
    for critic in critics:
        variant = resolve_online_strategy_algo(name, critic)
        leaf = results_leaf(
            results_root, spec.regime, beta, sigma, env, algo_id, critic, seed
        )
        env_cfg = EnvConfig(
            env_id=env,
            n_train_envs=spec.budget("n_train_envs", 2),
            n_eval_envs=spec.budget("n_eval_envs", 2),
            rollout_len=spec.budget("rollout_len", 2),
            seed=seed,
            behavior_policy=bp,
            behavior_strength=strength,
            pi_basic_epsilon=spec.pi_basic_epsilon,
            **_with_c_r(
                spec.arm_generator_kwargs(sigma),
                c_r_for(spec.confounder_c_r, beta, sigma),
            ),
            mask_indices=(spec.mask_indices.get(env) if recurrent else None),
        )
        train_cfg = TrainingConfig(
            n_episodes=spec.budget("n_episodes", 1),
            n_checkpoints=spec.budget("n_checkpoints", 2),
            deterministic=True,
            device=device or "cpu",
            algorithm=variant,
            aggregation="iqm",
            actor_network=actor_net,
            critic_network=critic_net,
            record_eval_video=False,
        )
        _write_run_metadata(
            leaf, spec, env, algo_id, beta, sigma, seed, critics=critics
        )
        BenchmarkRunner(
            env_cfg,
            train_cfg,
            RunConfig(run_dir=str(leaf), timestamp="sweep"),
            registry.get(variant),
        ).run()
        leaves.append(leaf)
    return leaves


def _validate_algos_for_regime(spec: SweepSpec, algos: Sequence[str]) -> None:
    """Refuse algorithms that are not DESIGNED for the cell's data regime — an
    offline learner in an online cell (or vice versa) would either crash deep in
    the runner or silently train on the wrong loop. The registry's
    ``data_regime`` is the source of truth. (Trunk compatibility — e.g. an lstm
    on a non-recurrent base — is enforced by the registry's builder guards with
    their own precise message.)"""
    from src.benchmarking.registry import registry

    for entry in algos:
        name, _, _, _ = parse_algo_entry(entry, spec.observability)
        try:
            algo_dr = registry.get(name).data_regime
        except KeyError:
            raise ValueError(
                f"unknown algorithm '{name}' (algos entry {entry!r}) in regime "
                f"'{spec.regime}'."
            )
        if algo_dr != spec.data_regime:
            raise ValueError(
                f"algorithm '{name}' (algos entry {entry!r}) is a "
                f"{algo_dr}-data learner and does not belong in regime "
                f"'{spec.regime}' (data_regime={spec.data_regime}); use only "
                "algorithms designed for the cell's data regime."
            )


def _reusable_dataset_hash(
    dataset_id: str,
    spec: "SweepSpec",
    env: str,
    seed: int,
    beta: float,
    sigma: float,
    behavior_policy: str,
    strength: float,
    generator_hash: str,
) -> str | None:
    """Return the generator hash of ``dataset_id`` when it is provably the dataset
    this sweep point would generate, else ``None`` (S4 reuse gate).

    Three independent checks, all of which must pass:

      1. the stored ``generation_fingerprint`` equals the one THIS point's inputs
         produce (arm params, epsilon, c_r, a_bad, rollout budget, seed,
         pi_basic hash, rollout device/slots/legacy);
      2. the episode count matches the configured rollout budget — catches a
         dataset truncated by an interrupted run, which the fingerprint (an
         input-only hash) cannot see;
      3. the confounding gate did not fail.

    Any error reading the store (absent, corrupt, older dataset without a
    fingerprint) returns ``None``, i.e. regenerate — reuse is never the fallback.
    """
    from src.envs.offline.generate import generation_fingerprint

    try:
        import minari

        if dataset_id not in minari.list_local_datasets():
            return None
        ds = minari.load_dataset(dataset_id)
        meta = ds.storage.metadata
        stored = meta.get("generation_fingerprint")
        if not stored:
            return None  # pre-S4 dataset: no proof available -> regenerate
        expected = generation_fingerprint(
            env_id=env,
            generator_algo=spec.generator_algo,
            tier="random",
            behavior_policy=behavior_policy,
            behavior_strength=strength,
            **_with_c_r(
                spec.arm_generator_kwargs(sigma),
                c_r_for(spec.confounder_c_r, beta, sigma),
            ),
            pi_basic_epsilon=spec.pi_basic_epsilon,
            a_bad=1,
            rollout_episodes=spec.budget("rollout_episodes", 30),
            seed=seed,
            generator_hash=generator_hash,
            rollout_device=spec.rollout_device,
            rollout_n_envs=spec.rollout_n_envs,
            legacy_rollout=spec.legacy_rollout,
        )
        if stored != expected:
            return None
        if int(ds.total_episodes) != int(spec.budget("rollout_episodes", 30)):
            return None
        if meta.get("gate_test_passed") is False:
            return None
        return meta.get("generator_checkpoint_hash")
    except Exception:
        return None


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
    phase: str = "all",
    points: Sequence[str] | None = None,
) -> List[Path]:
    """Run one cell (CHANGE 5), dispatching on (data_regime, simulation).

    OFFLINE: for each (env, seed) build ONE generator agent, generate every
    sweep-point dataset from it, REFUSE the cell if their hashes differ (M1),
    then train each arm point into the parameter-addressed leaves —
    per-{critic} leaves for the ``critic_ablation`` simulation, per-{algo}
    ``classical/`` leaves for the ``classical`` one (same shared-generator
    pairing either way).

    ONLINE: no offline generator exists (the behavior policy is fixed per arm,
    the learner trains on its own collection), so each arm point is an ordinary
    online run: ``classical`` = one benchmark run per (point, algo);
    ``critic_ablation`` = one run per (point, base algo, strategy VARIANT)
    (dqn vs online_dqn_proximal — the runner's strategy-ablation manager is
    offline-only, so online strategies are algo variants).

    The optional envs/algos/seeds override the spec (used to shrink a cell).

    ``phase``/``points`` are the supervisor's POINT-GRAIN seam (offline only):
    ``phase="generate"`` builds the shared generator, generates EVERY sweep-point
    dataset and runs the M1 hash gate, but trains nothing (writes no leaves);
    ``phase="train"`` assumes those datasets already sit in the ACTIVE Minari
    store (same ``MINARI_DATASETS_PATH``) and trains only the ``points`` given as
    ``param_dirname`` strings (e.g. ``b0.000_s0.500``). The shared-generator
    invariant is untouched: generation + M1 always happen whole-group in ONE
    process, and training is self-seeding (``BenchmarkRunner.run`` re-seeds, each
    point's rollout re-seeds), so the split is numerics-identical to a monolithic
    run. Defaults (``"all"``/None) are the historical byte-identical path."""
    from src.benchmarking.registry import register_default_algorithms
    from src.config.seeding import set_seed
    from src.envs.registry import register_default_env_wrappers

    if phase not in ("all", "generate", "train"):
        raise ValueError(f"unknown phase {phase!r} (expected all/generate/train)")

    spec = load_sweep_spec(sweep_yaml)
    if budget_overrides:
        spec.budgets = {**spec.budgets, **budget_overrides}
    register_default_algorithms()
    register_default_env_wrappers()

    run_envs = list(envs) if envs is not None else spec.envs
    run_algos = list(algos) if algos is not None else spec.algos
    run_seeds = [int(s) for s in (seeds if seeds is not None else spec.seeds)]
    _validate_algos_for_regime(spec, run_algos)

    if spec.data_regime != "offline":
        if phase != "all" or points is not None:
            raise ValueError("phase/points are offline-only (online has no datasets)")
        return _run_online_regime(
            spec, run_envs, run_algos, run_seeds, results_root, device
        )

    from src.envs.offline.generate import (
        build_generator_agent,
        generate_offline_dataset,
    )

    written: List[Path] = []
    all_points = spec.points()
    train_points = _select_points(points, all_points)
    for env in run_envs:
        for seed in run_seeds:
            if phase == "train":
                # Datasets were generated (and M1-gated) by a prior "generate"
                # phase into the SAME store — reconstruct their ids only.
                datasets = {
                    (beta, sigma): _dataset_id(
                        dataset_prefix, spec.regime, env, beta, sigma, seed
                    )
                    for beta, sigma in all_points
                }
                _train_points(
                    spec,
                    env,
                    seed,
                    train_points,
                    run_algos,
                    datasets,
                    results_root,
                    device,
                    written,
                )
                continue
            agent, _hash = build_generator_agent(
                env, spec.generator_algo, "random", seed=seed, device=device
            )
            # 1) generate all sweep points from the ONE shared agent
            datasets: Dict[Tuple[float, float], str] = {}
            point_hashes: Dict[Tuple[float, float], str] = {}
            for pi, (beta, sigma) in enumerate(all_points, start=1):
                bp, strength = arm_behavior(beta, sigma)
                did = _dataset_id(dataset_prefix, spec.regime, env, beta, sigma, seed)
                # Phase print (display-only): makes the generation phase visible
                # in the sweep-worker logs / supervisor mirror before training's
                # own tqdm bars take over.
                print(
                    f"[regime_sweep] {env} seed{seed}: dataset generation "
                    f"{pi}/{len(all_points)} — {param_dirname(beta, sigma)} "
                    f"(rollout_episodes={spec.budget('rollout_episodes', 30)})",
                    flush=True,
                )
                # S4: reuse an existing dataset whose generation fingerprint
                # matches what THIS point would generate — same inputs, same
                # deterministic pipeline, so regenerating could only reproduce
                # it. Chiefly this skips a regime's second simulation entirely
                # (classical and critic_ablation share dataset ids). Any changed
                # input (arm params, epsilon, c_r, rollout budget/mode, pi_basic)
                # misses and regenerates.
                if spec.reuse_datasets:
                    hit_hash = _reusable_dataset_hash(
                        did,
                        spec,
                        env,
                        seed,
                        beta,
                        sigma,
                        bp,
                        strength,
                        _hash,
                    )
                    if hit_hash is not None:
                        print(
                            f"[regime_sweep] {env} seed{seed}: reusing dataset "
                            f"{did} (fingerprint match — generation skipped)",
                            flush=True,
                        )
                        datasets[(beta, sigma)] = did
                        point_hashes[(beta, sigma)] = hit_hash
                        continue
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
                    **_with_c_r(
                        spec.arm_generator_kwargs(sigma),
                        c_r_for(spec.confounder_c_r, beta, sigma),
                    ),
                    rollout_episodes=spec.budget("rollout_episodes", 30),
                    seed=seed,
                    dataset_id=did,
                    agent=agent,
                    device=device,
                    rollout_device=spec.rollout_device,
                    rollout_n_envs=spec.rollout_n_envs,
                    legacy_rollout=spec.legacy_rollout,
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
            if phase == "generate":
                continue
            # 3) train each arm point into its parameter-addressed leaves
            _train_points(
                spec,
                env,
                seed,
                train_points,
                run_algos,
                datasets,
                results_root,
                device,
                written,
            )
    return written


def _select_points(
    names: Sequence[str] | None, all_points: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Resolve ``--points`` param-dirnames to (beta, sigma) pairs; None = all.
    Unknown names refuse loudly — a typo must never silently train nothing."""
    if names is None:
        return list(all_points)
    by_name = {param_dirname(b, s): (b, s) for (b, s) in all_points}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(
            f"unknown sweep point(s) {unknown}; this cell has {sorted(by_name)}"
        )
    return [by_name[n] for n in names]


def _train_points(
    spec: SweepSpec,
    env: str,
    seed: int,
    train_points: Sequence[Tuple[float, float]],
    run_algos: Sequence[str],
    datasets: Dict[Tuple[float, float], str],
    results_root: str | Path,
    device: str | None,
    written: List[Path],
) -> None:
    """The training inner loop of one offline (env, seed) group, over
    ``train_points`` (the full L, or a supervisor-assigned subset)."""
    for pi, (beta, sigma) in enumerate(train_points, start=1):
        print(
            f"[regime_sweep] {env} seed{seed}: TRAINING point "
            f"{pi}/{len(train_points)} — {param_dirname(beta, sigma)} "
            f"({arm_label(beta, sigma)} arm, {len(run_algos)} algo(s))",
            flush=True,
        )
        for algo in run_algos:
            if spec.simulation == "classical":
                written += _run_point_classical(
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
            else:
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


def _run_online_regime(
    spec: SweepSpec,
    run_envs: Sequence[str],
    run_algos: Sequence[str],
    run_seeds: Sequence[int],
    results_root: str | Path,
    device: str | None,
) -> List[Path]:
    """The ONLINE cell driver: no generator, no datasets — each arm point runs the
    online loop (the PR-3 arm policies collect during training; the online
    intervened gate applies). Group iteration mirrors the offline driver
    ((env, seed) outermost) so the supervisor's (env, seed) grain works verbatim."""
    written: List[Path] = []
    points = spec.points()
    for env in run_envs:
        for seed in run_seeds:
            for pi, (beta, sigma) in enumerate(points, start=1):
                print(
                    f"[regime_sweep] {env} seed{seed}: TRAINING point "
                    f"{pi}/{len(points)} — {param_dirname(beta, sigma)} "
                    f"({arm_label(beta, sigma)} arm, {len(run_algos)} algo(s))",
                    flush=True,
                )
                for algo in run_algos:
                    if spec.simulation == "classical":
                        written += _run_point_classical(
                            spec,
                            env,
                            algo,
                            seed,
                            beta,
                            sigma,
                            None,
                            results_root,
                            device,
                        )
                    else:
                        written += _run_point_online_ablation(
                            spec, env, algo, seed, beta, sigma, results_root, device
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
        description="Run one (regime × L-sweep) cell, offline or online, in either "
        "simulation (the YAML's `simulation:` key): classical (algo x env benchmark, "
        "{regime}/classical/ leaves) or critic_ablation (per-{critic} leaves). "
        "Offline cells share ONE generator checkpoint per (env, seed); online cells "
        "run the arm policies through the online loop (no generator).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "sweep_yaml", help="path to a cell YAML (classical.yaml / critic_ablation.yaml)"
    )
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
        help="how many sweep tasks to run concurrently (overrides the cell's "
        "max_workers; default from _base/parallel.yaml = 1 = serial). 1 keeps the "
        "byte-identical in-process path; >=2 fans tasks across subprocesses.",
    )
    ap.add_argument(
        "--phase",
        choices=["all", "generate", "train"],
        default="all",
        help="offline only: 'generate' builds the shared generator + every point "
        "dataset + the M1 gate (no training); 'train' trains on datasets already "
        "in the active Minari store. The supervisor's point-grain seam.",
    )
    ap.add_argument(
        "--points",
        nargs="+",
        default=None,
        help="restrict TRAINING to these sweep points (param dirnames, e.g. "
        "b0.000_s0.500); offline only, typically with --phase train",
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

    # A --phase/--points invocation IS a single supervisor work unit — it must
    # never re-enter the pool (the supervisor's children rely on this staying
    # serial via --max-workers 1; a human passing --phase gets the same).
    if args.phase != "all" or args.points is not None:
        eff_workers = 1

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
        phase=args.phase,
        points=args.points,
    )
    print(
        f"[regime_sweep] wrote {len(leaves)} run-dir leaves under {results_root}/ "
        f"(device={device}{'; SMOKE budget' if args.smoke else ''})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Generator knobs for a declared diagram — DERIVED from L1, not re-declared.

v2's one assumption is the declared causal diagram, and that has to bind the
data generator too. If a YAML could independently say "switch the proxies on",
the diagram would no longer be the single assumption surface: a config could
generate proxies for a diagram that declares none, or (worse) declare proxies
the generator never emits, and L2 would hand out a proximal verdict on data
that cannot support one.

So the direction here is one-way. **Which channels exist** comes from the
catalogue entry — ``proxy_nodes`` turns the proxies on, ``instrument_nodes``
turns the instrument on, ``persistent_latent`` turns drift on. **How strong
they are** comes from the config, because a magnitude is a sweep axis, not a
structural claim. ``arm_knobs`` refuses a strength for a channel the diagram
does not declare, and refuses a declared channel left without one.

The strengths themselves are not calibration constants in the sense v2 forbids:
nothing downstream reads them. They are properties of the generated world, in
the same family as ``confounder_c_r``, and the preflight measures what they
actually produced rather than trusting the number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.rl.offline.grace.cell_graph import catalogue_entry

__all__ = ["ArmKnobs", "arm_knobs", "declared_channels"]


@dataclass(frozen=True)
class ArmKnobs:
    """What a diagram's arm hands to ``generate_offline_dataset``."""

    diagram: str
    behavior_policy: str
    behavior_strength: float
    confounder_c_r: float
    proxy_strength: float | None = None
    instrument_strength: float | None = None
    u_drift: float = 0.0
    gate_probs: tuple | None = None

    def generator_kwargs(self) -> Dict:
        return {
            "behavior_policy": self.behavior_policy,
            "behavior_strength": self.behavior_strength,
            "confounder_c_r": self.confounder_c_r,
            "proxy_strength": self.proxy_strength,
            "instrument_strength": self.instrument_strength,
            "u_drift": self.u_drift,
            "gate_probs": self.gate_probs,
        }


def declared_channels(diagram: str) -> Dict[str, bool]:
    """Which generator channels the DIAGRAM says exist. Read off the graph."""
    g = catalogue_entry(diagram)
    return {
        "proxy": bool(g.proxy_nodes),
        "instrument": bool(g.instrument_nodes),
        "drift": bool(g.persistent_latent),
        "latent": any(not n.observed for n in g.nodes),
    }


def arm_knobs(
    diagram: str,
    *,
    sigma: float,
    confounder_c_r: float = 1.0,
    proxy_strength: float | None = None,
    instrument_strength: float | None = None,
    u_drift: float | None = None,
    gate_probs=None,
) -> ArmKnobs:
    """Resolve a diagram id plus config strengths into generator knobs.

    Raises when the config and the diagram disagree in either direction — an
    undeclared channel given a strength, or a declared one left without.
    """
    ch = declared_channels(diagram)
    supplied = {
        "proxy": proxy_strength,
        "instrument": instrument_strength,
        "drift": u_drift,
    }
    for name, val in supplied.items():
        if val is not None and not ch[name]:
            raise ValueError(
                f"{diagram} declares no {name} channel, but the config supplies a "
                f"{name} strength of {val}. The diagram is the assumption surface: "
                f"add the nodes to the catalogue entry, or drop the knob."
            )
        if val is None and ch[name]:
            raise ValueError(
                f"{diagram} declares a {name} channel, but the config supplies no "
                f"{name} strength. A declared channel the generator never emits "
                f"would give L2 a verdict the data cannot support."
            )

    # R2, route (a). An arm that declares an instrument MUST make its exclusion
    # restriction testable, because that restriction is what its whole verdict
    # rests on. Under the deterministic gate R is a function of (A, U), so
    # residualising on them leaves no variance and the check measures nothing
    # while reporting a pass. Requiring gate_probs here is what stops D-E being
    # declared without the property that lets it be checked.
    if ch["instrument"] and gate_probs is None:
        raise ValueError(
            f"{diagram} declares an instrument, so its reward must be stochastic "
            "given (A, U) or the exclusion restriction cannot be tested at all. "
            "Supply gate_probs = (q0, q1): U shifts the PROBABILITY of the gated "
            "bonus, which keeps R binary (so L4's Balke-Pearl anchor keeps its "
            "closed form) while giving the exclusion check real power."
        )
    if gate_probs is not None and not ch["instrument"]:
        raise ValueError(
            f"{diagram} declares no instrument; gate_probs exists to make the "
            "exclusion restriction testable and has no other arm to serve."
        )

    if not ch["latent"]:
        # No latent at all (D-A / D-A-null). Collect through the same
        # action-dependent policy so the code path is shared, but at sigma = 0
        # and c_r = 0: U is drawn and logged, and touches neither the action nor
        # the reward. That is what makes it the reference null -- L5's
        # false-positive rate is measured where there is genuinely nothing to
        # find, so a refutation there is a false alarm by construction and needs
        # no threshold to interpret.
        if sigma:
            raise ValueError(f"{diagram} has no latent; sigma must be 0, got {sigma}.")
        return ArmKnobs(diagram, "bias_confounded_action", 0.0, 0.0)

    return ArmKnobs(
        diagram=diagram,
        behavior_policy="bias_confounded_action",
        behavior_strength=float(sigma),
        confounder_c_r=float(confounder_c_r),
        proxy_strength=proxy_strength,
        instrument_strength=instrument_strength,
        u_drift=0.0 if u_drift is None else float(u_drift),
        gate_probs=None if gate_probs is None else tuple(map(float, gate_probs)),
    )

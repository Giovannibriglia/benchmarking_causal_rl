"""The YAML-driven driver's plan == the pilot's campaign, verified.

The CELLS/SEEDS/ALGOS/PROXIES constants were deleted 2026-09-03; the plan is
now derived from e1_*.yaml alone. These tests pin (1) the structural shape of
the derived plan and (2) — the equivalence that matters — that every leaf the
PILOT actually produced matches a plan entry with the SAME dataset id, read
from the leaf's own provenance. Ground truth over reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_e1 import assert_plan_safe, enumerate_plan


def test_plan_shape_matches_the_declared_campaign():
    entries = enumerate_plan()
    # 5 cells x 2 arms x 2 algos x 3 seeds x 1 env = 60 runs
    assert len(entries) == 60
    by_tag = {}
    for e in entries:
        by_tag.setdefault(e["tag"], []).append(e)
    assert set(by_tag) == {"danull", "d100s0", "d100", "d025", "d010asym"}
    for tag, es in by_tag.items():
        assert sorted({x["arm"] for x in es}) == ["base", "grace"]
        assert sorted({x["algo"] for x in es}) == ["cql", "iql"]
        seeds = sorted({x["seed"] for x in es})
        assert seeds == ([0, 2, 3] if tag == "d100s0" else [0, 1, 2]), (tag, seeds)
        sigma = {x["sigma"] for x in es}
        assert sigma == ({0.25} if tag in ("d100", "d025", "d010asym") else {0.0}), (
            tag,
            sigma,
        )
        proxies = {x["spec"].grace_proxy_names for x in es if x["arm"] == "grace"}
        assert proxies == ({()} if tag == "danull" else {("Z", "W", "V")}), tag
    assert_plan_safe(entries)  # pairs share, cells distinct, store + stamps


def test_plan_reproduces_every_pilot_leaf():
    """Every leaf the pilot wrote must appear in the derived plan with the
    SAME dataset id — the YAML rewrite may not silently re-map the campaign."""
    entries = enumerate_plan()
    plan = {
        (e["tag"], e["algo"], e["arm"], e["seed"]): e["dataset_id"] for e in entries
    }
    n = 0
    for prov in Path("results/e1").glob(
        "offline_mdp_*/beta_*/CartPole-v1/*/*/*/e1_provenance.json"
    ):
        d = json.loads(prov.read_text())
        key = (d["cell"], d["algo"], d["arm"], d["seed"])
        assert key in plan, f"pilot leaf {key} missing from the YAML-derived plan"
        assert plan[key] == d["dataset_id"], (key, plan[key], d["dataset_id"])
        n += 1
    assert n == 42, f"expected the pilot's 42 leaves, found {n}"

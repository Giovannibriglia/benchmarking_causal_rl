#!/usr/bin/env python3
"""Generate the cached data CSVs behind the Γ-calibration and horizon-ablation
figures, so ``plot_causal.py`` regenerates everything from CSVs alone.

Usage:
    python -m tools.make_plot_data <cell8_run_dir> <cell3_h50_run_dir> <cell3_run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from src.envs.registry import register_default_env_wrappers
from src.eval.regret import iqm


def gamma_calibration(cell8_run: Path, out: Path) -> None:
    from src.data.minari_io import to_offline_source
    from src.eval.kallus_zhou import kz_interval
    from src.eval.ope import StochasticPolicyAdapter
    from src.offline.bc import BehaviorCloning
    from src.rl.on_policy.policy import ActorCriticMLP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cell8_run / "causal_cells_metrics.csv")
    true_j = iqm(df[df.role == "basic"].J.to_numpy(float))

    src = to_offline_source(
        "causal/cartpole/cell7-b1-d0p5-v0",
        device,
        behavior_policy="unknown",
        mask_indices=[1, 3],
    )
    torch.manual_seed(0)
    bc = BehaviorCloning(ActorCriticMLP(2, 2, "discrete", device), device)
    bc.fit_source(src, 4000)
    target = StochasticPolicyAdapter(bc.policy)
    rows = []
    for gamma in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
        iv = kz_interval(src, target, gamma=gamma, clone_seed=0)
        rows.append(
            {"gamma": gamma, "lower": iv.lower, "upper": iv.upper, "true_j": true_j}
        )
        print(f"gamma={gamma}: [{iv.lower:.1f}, {iv.upper:.1f}]")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


def horizon_fixture_point() -> dict:
    """Exact-DP fixture numbers (H=6) via the test machinery, smaller N."""
    sys.path.insert(0, ".")
    from src.data.experience_source import OfflineDatasetSource
    from src.eval.ope import DoublyRobust, IPWEstimator
    from tests.test_ope import _collect, _dp_value, _world, TabularPolicy

    S, A, T, r_mean, p_plus = _world()
    g = torch.Generator().manual_seed(100)
    behavior_logits = torch.randn(S, A, generator=g)
    target_logits = 1.5 * behavior_logits + 0.5 * torch.randn(S, A, generator=g)
    behavior, target = TabularPolicy(behavior_logits), TabularPolicy(target_logits)
    episodes = _collect(behavior, T, p_plus, 2500, seed=7)
    source = OfflineDatasetSource(episodes, torch.device("cpu"), "known")
    j_t = _dp_value(torch.softmax(target_logits, dim=-1), T, r_mean, 6)
    ipw = IPWEstimator(behavior="known").estimate(source, target).value
    dr = DoublyRobust(behavior="known", gamma=1.0, seed=1).estimate(source, target)
    return {"regime": "H=6 (fixture)", "true_j": j_t, "ipw": ipw, "dr": dr.value}


def horizon_ablation(h50_run: Path, cell3_run: Path, out: Path) -> None:
    rows = [horizon_fixture_point()]
    for regime, run, tier in (
        ("H=50 (medium50)", h50_run, "medium50"),
        ("H=200+ (medium)", cell3_run, "medium"),
    ):
        df = pd.read_csv(run / "causal_cells_metrics.csv")
        bc = df[(df.role == "basic") & (df.tier == tier)]
        rows.append(
            {
                "regime": regime,
                # BC clones pi_b, so its true value ~= the logged mean return
                "true_j": iqm(bc.ope_naive.to_numpy(float)),
                "ipw": iqm(bc.ope_ipw.to_numpy(float)),
                "dr": iqm(bc.ope_dr.to_numpy(float)),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {out}")


def main() -> None:
    register_default_env_wrappers()
    cell8_run, h50_run, cell3_run = (Path(p) for p in sys.argv[1:4])
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    gamma_calibration(cell8_run, outdir / "gamma_calibration.csv")
    horizon_ablation(h50_run, cell3_run, outdir / "horizon_ablation.csv")


if __name__ == "__main__":
    main()

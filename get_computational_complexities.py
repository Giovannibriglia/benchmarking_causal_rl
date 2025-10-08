from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- MLP parameter counting (simple FC with biases) ----------
def mlp_param_count(input_dim: int, hidden: int, layers: int, output_dim: int) -> int:
    if layers < 1:
        return input_dim * output_dim + output_dim
    params = input_dim * hidden + hidden
    for _ in range(layers - 1):
        params += hidden * hidden + hidden
    params += hidden * output_dim + output_dim
    return params


# ---------- Configs for scaling study ----------
@dataclass
class NObsCfg:
    B: int = 64_000
    act_dim: int = 1
    actor_hidden: int = 256
    actor_layers: int = 2
    critic_hidden: int = 256
    critic_layers: int = 2
    ppo_epochs: int = 4
    trpo_K: int = 10
    trpo_L: int = 10


# TODO: trova dove hai trovato formula e indaga --> che unità di misura è: guarda se è "numero di aggiornamenti a parametri individuali",
# vado più veloce? a far cosa? poi scrivi su papero.


# ---------- Costs vs n_obs (up to proportional constants) ----------
def cost_vanilla_ac_vs_nobs(n_obs: int, cfg: NObsCfg) -> float:
    d_pi = mlp_param_count(n_obs, cfg.actor_hidden, cfg.actor_layers, cfg.act_dim)
    d_V = mlp_param_count(n_obs, cfg.critic_hidden, cfg.critic_layers, 1)
    return cfg.B * (d_pi + d_V)


def cost_a2c_vs_nobs(n_obs: int, cfg: NObsCfg) -> float:
    # same update cost model as vanilla AC
    return cost_vanilla_ac_vs_nobs(n_obs, cfg)


def cost_ppo_vs_nobs(n_obs: int, cfg: NObsCfg) -> float:
    d_pi = mlp_param_count(n_obs, cfg.actor_hidden, cfg.actor_layers, cfg.act_dim)
    d_V = mlp_param_count(n_obs, cfg.critic_hidden, cfg.critic_layers, 1)
    return cfg.ppo_epochs * cfg.B * (d_pi + d_V)


def cost_trpo_vs_nobs(n_obs: int, cfg: NObsCfg) -> float:
    d_pi = mlp_param_count(n_obs, cfg.actor_hidden, cfg.actor_layers, cfg.act_dim)
    d_V = mlp_param_count(n_obs, cfg.critic_hidden, cfg.critic_layers, 1)
    return cfg.B * d_V + (cfg.trpo_K + cfg.trpo_L) * d_pi


def cost_causal_ac_vs_nobs_inference(n_obs: int, cfg: NObsCfg) -> float:
    # vanilla_ac_cc / a2c_cc share same causal critic compute
    d_pi = mlp_param_count(n_obs, cfg.actor_hidden, cfg.actor_layers, cfg.act_dim)
    m = n_obs + cfg.act_dim
    inference = cfg.B * m
    return cfg.B * d_pi + inference


def cost_causal_ac_vs_nobs_refit(n_obs: int, cfg: NObsCfg) -> float:
    # include reward-node refit: O(B*m^2 + m^3)
    d_pi = mlp_param_count(n_obs, cfg.actor_hidden, cfg.actor_layers, cfg.act_dim)
    m = n_obs + cfg.act_dim
    inference = cfg.B * m
    refit = cfg.B * (m**2) + (m**3)
    return cfg.B * d_pi + inference + refit


# ---------- Plotting & CSV ----------
def plot_all(
    n_obs_vals,
    series: dict[str, np.ndarray],
    title: str,
    out_image: Path,
    deoverlap_eps: float = 0.0,
):
    """
    Plot all series together. If deoverlap_eps > 0, apply tiny plotting-only
    multiplicative offsets to visually separate identical curves (values in CSV remain exact).
    """
    colors = matplotlib.colormaps["Set1"]

    styles = {
        "vanilla": dict(
            linestyle="-", marker=None, linewidth=3.0, color=colors(0)
        ),  # red
        "a2c": dict(
            linestyle="--", marker=None, linewidth=3.0, color=colors(1)
        ),  # blue
        "ppo": dict(
            linestyle=":", marker=None, linewidth=3.0, color=colors(2)
        ),  # green
        "trpo": dict(
            linestyle="-",
            marker=None,
            linewidth=3.0,
            color=colors(3),
        ),  # purple
        "vanilla_cc": dict(
            linestyle="--", marker=None, linewidth=3.0, color=colors(4)
        ),  # orange
        "a2c_cc": dict(
            linestyle=":", marker=None, linewidth=3.0, color=colors(5)
        ),  # yellow
    }

    # Apply plotting-only epsilon to separate perfectly overlapping curves
    eps_map = {
        "vanilla": 0.0,
        "a2c": deoverlap_eps,  # tiny upward nudge
        "ppo": 0.0,
        "trpo": 0.0,
        "vanilla_cc": 2.0 * deoverlap_eps,  # tiny upward nudge
        "a2c_cc": 3.0 * deoverlap_eps,  # tiny upward nudge
    }

    plt.figure(dpi=500, figsize=(3, 3))
    for name, ys in series.items():
        ys_plot = ys * (1.0 + eps_map.get(name, 0.0))  # plotting-only adjustment
        st = styles.get(name, {})
        plt.plot(n_obs_vals, ys_plot, label=name, **st)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("#obs features")
    plt.ylabel("complexity")
    # plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(out_image)
    plt.close()


def to_csv(n_obs_vals, series: dict[str, np.ndarray], out_csv: Path):
    df = pd.DataFrame({"n_obs": n_obs_vals.astype(int)})
    for name, ys in series.items():
        df[name] = ys
    df.to_csv(out_csv, index=False)


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-update cost vs n_obs for all variants."
    )
    ap.add_argument("--outdir", type=str, default="computational_complexities")
    ap.add_argument("--B", type=int, default=64000)
    ap.add_argument("--act-dim", type=int, default=2, choices=[1, 2])
    ap.add_argument("--actor-hidden", type=int, default=256)
    ap.add_argument("--actor-layers", type=int, default=2)
    ap.add_argument("--critic-hidden", type=int, default=256)
    ap.add_argument("--critic-layers", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--trpo-K", type=int, default=10)
    ap.add_argument("--trpo-L", type=int, default=10)
    ap.add_argument("--min-nobs", type=float, default=1)
    ap.add_argument("--max-nobs", type=float, default=10000)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--no-refit", action="store_true", help="Skip the REFIT plot/CSV.")
    ap.add_argument(
        "--deoverlap",
        type=float,
        default=0.0,
        help="Tiny plotting-only epsilon to separate identical curves (e.g., 1e-6).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = NObsCfg(
        B=args.B,
        act_dim=args.act_dim,
        actor_hidden=args.actor_hidden,
        actor_layers=args.actor_layers,
        critic_hidden=args.critic_hidden,
        critic_layers=args.critic_layers,
        ppo_epochs=args.ppo_epochs,
        trpo_K=args.trpo_K,
        trpo_L=args.trpo_L,
    )

    n_obs_vals = np.unique(
        np.round(
            np.logspace(np.log10(args.min_nobs), np.log10(args.max_nobs), args.points)
        )
    ).astype(int)

    # Inference-only series
    series_inf = {
        "vanilla": np.array(
            [cost_vanilla_ac_vs_nobs(n, cfg) for n in n_obs_vals], dtype=float
        ),
        "a2c": np.array([cost_a2c_vs_nobs(n, cfg) for n in n_obs_vals], dtype=float),
        # "ppo": np.array([cost_ppo_vs_nobs(n, cfg) for n in n_obs_vals], dtype=float),
        # "trpo": np.array([cost_trpo_vs_nobs(n, cfg) for n in n_obs_vals], dtype=float),
        "vanilla_cc": np.array(
            [cost_causal_ac_vs_nobs_inference(n, cfg) for n in n_obs_vals], dtype=float
        ),
        "a2c_cc": np.array(
            [cost_causal_ac_vs_nobs_inference(n, cfg) for n in n_obs_vals], dtype=float
        ),
    }
    plot_all(
        n_obs_vals,
        series_inf,
        title="inference only",
        out_image=outdir / "cost_vs_nobs_ALL_inference.pdf",
        deoverlap_eps=args.deoverlap,
    )
    to_csv(n_obs_vals, series_inf, outdir / "table_vs_nobs_ALL_inference.csv")

    # REFIT series (unless disabled)
    if not args.no_refit:
        series_refit = {
            "vanilla": series_inf["vanilla"],
            "a2c": series_inf["a2c"],
            # "ppo": series_inf["ppo"],
            # "trpo": series_inf["trpo"],
            "vanilla_cc": np.array(
                [cost_causal_ac_vs_nobs_refit(n, cfg) for n in n_obs_vals], dtype=float
            ),
            "a2c_cc": np.array(
                [cost_causal_ac_vs_nobs_refit(n, cfg) for n in n_obs_vals], dtype=float
            ),
        }
        plot_all(
            n_obs_vals,
            series_refit,
            title="learning + inference",
            out_image=outdir / "cost_vs_nobs_ALL_refit.pdf",
            deoverlap_eps=args.deoverlap,
        )
        to_csv(n_obs_vals, series_refit, outdir / "table_vs_nobs_ALL_refit.csv")

    print(f"[ok] wrote plots & tables to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

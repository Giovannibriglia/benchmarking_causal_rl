from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx

import torch

from .core import BNMeta, DiscreteCPDTable, LGParams


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _as_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def _ensure_1d(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1)


def _parent_config_grid(parent_cards: List[int]) -> torch.Tensor:
    """
    Return a grid of parent configurations as [P, len(parent_cards)] where
    P = product(parent_cards). Values are 0..card-1.
    """
    if not parent_cards:
        return torch.zeros(1, 0, dtype=torch.long)
    ranges = [torch.arange(k, dtype=torch.long) for k in parent_cards]
    meshes = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack([m.reshape(-1) for m in meshes], dim=1)  # [P, n_par]
    return grid


def _cov_ellipse_points(
    mu: torch.Tensor, Sigma: torch.Tensor, n_std: float = 2.0, n_pts: int = 200
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ellipse points (x,y) for 2D Gaussian with mean mu[2], cov Sigma[2,2].
    """
    vals, vecs = torch.linalg.eigh(Sigma)
    vals = torch.clamp(vals, min=1e-16)
    axes_len = n_std * torch.sqrt(vals)
    theta = torch.atan2(vecs[1, 0], vecs[0, 0])
    t = torch.linspace(0, 2 * math.pi, n_pts)
    circ = torch.stack([torch.cos(t), torch.sin(t)], dim=0)  # [2, n_pts]
    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=Sigma.dtype,
    )
    pts = (R @ (axes_len.unsqueeze(1) * circ)) + mu.unsqueeze(1)
    return pts[0], pts[1]


# ──────────────────────────────────────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────────────────────────────────────


def draw_graph(
    meta: BNMeta,
    pos: Optional[Dict[str, Tuple[float, float]]] = None,
    with_labels: bool = True,
    node_size: int = 900,
    figsize: Tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Draw the DAG using networkx. Discrete nodes are drawn as squares, continuous as circles.
    """
    G = meta.G
    if pos is None:
        pos = nx.spring_layout(G, seed=3)

    fig, ax = plt.subplots(figsize=figsize)
    disc = [n for n in G.nodes if meta.types[n] == "discrete"]
    cont = [n for n in G.nodes if meta.types[n] == "continuous"]

    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        arrows=True,
        arrowstyle="->",
        min_source_margin=10,
        min_target_margin=10,
    )
    if disc:
        nx.draw_networkx_nodes(
            G, pos=pos, nodelist=disc, ax=ax, node_shape="s", node_size=node_size
        )
    if cont:
        nx.draw_networkx_nodes(
            G, pos=pos, nodelist=cont, ax=ax, node_shape="o", node_size=node_size
        )
    if with_labels:
        nx.draw_networkx_labels(G, pos=pos, ax=ax)

    ax.axis("off")
    ax.set_title("Bayesian Network")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Discrete CPDs
# ──────────────────────────────────────────────────────────────────────────────


def plot_discrete_table(
    table: DiscreteCPDTable, var_name: str, figsize: Tuple[int, int] = (6, 4)
) -> plt.Figure:
    """
    Show a discrete CPD as either bars (no parents) or heatmap (with parents).
    probs shape is [P, C] where P=#parent configs (1 if no parents).
    """
    P, C = table.probs.shape
    fig, ax = plt.subplots(figsize=figsize)
    probs = _as_cpu(table.probs)

    if P == 1:
        ax.bar(range(C), _ensure_1d(probs[0]))
        ax.set_xlabel(var_name)
        ax.set_ylabel("P({}=k)".format(var_name))
        ax.set_xticks(range(C))
        ax.set_title(f"CPD: P({var_name})")
    else:
        im = ax.imshow(probs, aspect="auto", interpolation="nearest")
        ax.set_xlabel(f"{var_name} = k (0..{C - 1})")
        ax.set_ylabel("Parent config index")
        ax.set_title(f"CPD: P({var_name} | parents)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Parent legend (optional): show first few configs as ticklabels
        grid = _parent_config_grid(table.parent_cards)
        if grid.shape[0] == P and P <= 20:  # avoid overcrowding
            labels = [",".join(str(v.item()) for v in row) for row in grid]
            ax.set_yticks(range(P))
            ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    return fig


def plot_discrete_marginal(
    probs: torch.Tensor, var_name: str, figsize: Tuple[int, int] = (6, 4)
) -> plt.Figure:
    """
    Plot a marginal distribution over a discrete variable.
    Accepts shape [C] or [B, C] (batched). If batched, plots mean ± band across batches.
    """
    p = _as_cpu(probs)
    fig, ax = plt.subplots(figsize=figsize)
    if p.ndim == 1:
        ax.bar(range(p.numel()), p)
    else:
        # summarize across batches
        mean = p.mean(dim=0)
        std = p.std(dim=0)
        xs = torch.arange(mean.numel())
        ax.bar(xs, mean)
        ax.errorbar(xs, mean, yerr=std, fmt="none", capsize=3, linewidth=1)

    ax.set_title(f"Marginal P({var_name})")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Probability")
    ax.set_xticks(range(p.shape[-1]))
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Continuous Gaussian posterior (Linear-Gaussian inference output)
# ──────────────────────────────────────────────────────────────────────────────


def plot_continuous_gaussian(
    mu: Dict[str, torch.Tensor],
    Sigma: torch.Tensor,
    query_order: Sequence[str],
    dims: Optional[Sequence[str]] = None,
    n_std: float = 2.0,
    grid_points: int = 400,
    figsize: Tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Visualize Gaussian posterior:
      - If len(dims)==1: plots the 1D pdf along that dimension.
      - If len(dims)==2: plots 2D covariance ellipse (n_std).
    `mu` is {name: tensor()}, `Sigma` is [k,k] aligned with `query_order`.
    """
    # Build mean vector in the order
    k = len(query_order)
    if k == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty query", ha="center", va="center")
        ax.axis("off")
        return fig

    mu_vec = torch.stack([mu[n].reshape(()) for n in query_order])
    Sigma = Sigma

    if dims is None:
        dims = [query_order[0]]

    if len(dims) == 1:
        i = query_order.index(dims[0])
        m = mu_vec[i].item()
        v = Sigma[i, i].item()
        x = torch.linspace(
            m - 4.0 * math.sqrt(v + 1e-12), m + 4.0 * math.sqrt(v + 1e-12), grid_points
        )
        y = torch.exp(-0.5 * (x - m) ** 2 / max(v, 1e-12)) / math.sqrt(
            2 * math.pi * max(v, 1e-12)
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(_as_cpu(x), _as_cpu(y))
        ax.set_title(f"Posterior N({dims[0]})")
        ax.set_xlabel(dims[0])
        ax.set_ylabel("density")
        fig.tight_layout()
        return fig

    if len(dims) == 2:
        i, j = query_order.index(dims[0]), query_order.index(dims[1])
        mu2 = mu_vec[[i, j]]
        S2 = Sigma[[i, j]][:, [i, j]]
        fig, ax = plt.subplots(figsize=figsize)
        ex, ey = _cov_ellipse_points(mu2, S2, n_std=n_std)
        ax.plot(_as_cpu(ex), _as_cpu(ey))
        ax.scatter([mu2[0].item()], [mu2[1].item()], marker="x")
        ax.set_title(f"Posterior N({dims[0]}, {dims[1]}) – {n_std}σ ellipse")
        ax.set_xlabel(dims[0])
        ax.set_ylabel(dims[1])
        fig.tight_layout()
        return fig

    # If more than 2 dims specified, show pair (first two)
    return plot_continuous_gaussian(
        mu, Sigma, query_order, dims=dims[:2], n_std=n_std, figsize=figsize
    )


# ──────────────────────────────────────────────────────────────────────────────
# Linear-Gaussian params inspection
# ──────────────────────────────────────────────────────────────────────────────


def plot_lg_params(
    lg: LGParams,
    figsize_W: Tuple[int, int] = (5, 4),
    figsize_sigma: Tuple[int, int] = (5, 3),
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Heatmap of W and bar plot of sigma^2 for the continuous subgraph.
    """
    W = _as_cpu(lg.W)
    s2 = _as_cpu(lg.sigma2)
    names = lg.order

    figW, axW = plt.subplots(figsize=figsize_W)
    im = axW.imshow(W, aspect="auto", interpolation="nearest")
    axW.set_title("Linear-Gaussian weights W (child row, parent col)")
    axW.set_xlabel("Parent index")
    axW.set_ylabel("Child index")
    axW.set_xticks(range(len(names)))
    axW.set_xticklabels(names, rotation=90, fontsize=8)
    axW.set_yticks(range(len(names)))
    axW.set_yticklabels(names, fontsize=8)
    figW.colorbar(im, ax=axW, fraction=0.046, pad=0.04)
    figW.tight_layout()

    figS, axS = plt.subplots(figsize=figsize_sigma)
    axS.bar(range(len(s2)), s2)
    axS.set_title("Node noise variances σ²")
    axS.set_xlabel("Node")
    axS.set_ylabel("σ²")
    axS.set_xticks(range(len(names)))
    axS.set_xticklabels(names, rotation=90, fontsize=8)
    figS.tight_layout()

    return figW, figS


# ──────────────────────────────────────────────────────────────────────────────
# Samples & approximate posterior diagnostics
# ──────────────────────────────────────────────────────────────────────────────


def plot_continuous_samples(
    samples: torch.Tensor, title: str = "Samples", figsize: Tuple[int, int] = (6, 4)
) -> plt.Figure:
    """
    Plot a simple trace (if 1D) or scatter (if 2D) of samples.
    `samples` shape: [S] or [S,2]. If [B,S] or [B,S,2], only the first batch is used.
    """
    x = _as_cpu(samples)
    if x.ndim == 2 and x.shape[0] > 1:  # [B,S]
        x = x[0]
    if x.ndim == 3 and x.shape[0] > 1:  # [B,S,2]
        x = x[0]

    fig, ax = plt.subplots(figsize=figsize)
    if x.ndim == 1:
        ax.plot(range(len(x)), x)
        ax.set_xlabel("sample idx")
        ax.set_ylabel("value")
    elif x.ndim == 2 and x.shape[1] == 2:
        ax.scatter(x[:, 0], x[:, 1], s=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    else:
        ax.plot(range(x.shape[-2]), x[..., 0])
        ax.set_xlabel("sample idx")
        ax.set_ylabel("value[0]")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_weight_histogram(
    weights: torch.Tensor,
    title: str = "Importance weights",
    figsize: Tuple[int, int] = (5, 3),
) -> plt.Figure:
    """
    Histogram of (possibly batched) importance weights.
    weights: [S] or [B,S] tensor.
    """
    w = _as_cpu(weights)
    if w.ndim == 2:
        w = w[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(w.numpy(), bins=30)
    ax.set_title(title)
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Discrete posteriors (from exact/approx)
# ──────────────────────────────────────────────────────────────────────────────


def plot_discrete_posteriors(
    out: Dict[str, torch.Tensor], figsize: Tuple[int, int] = (6, 4)
) -> Dict[str, plt.Figure]:
    """
    Plot per-variable posterior marginals returned by discrete exact/approx:
      out[var] is [C] or [B,C]. Returns {var: fig}.
    """
    figs: Dict[str, plt.Figure] = {}
    for var, p in out.items():
        figs[var] = plot_discrete_marginal(p, var, figsize=figsize)
    return figs

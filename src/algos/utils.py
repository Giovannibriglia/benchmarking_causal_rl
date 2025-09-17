from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def set_learning_and_inference_objects(if_discrete: bool, approximate: bool):

    if if_discrete:
        if approximate:
            fit_method = "discrete_mlp"
            inf_method = "discrete_approx"
        else:
            fit_method = "discrete_mle"
            inf_method = "discrete_exact"
    else:
        if approximate:
            fit_method = "continuous_gaussian"
            inf_method = "continuous_gaussian"
        else:
            fit_method = "continuous_mlp_gaussian"
            inf_method = "continuous_approx"

    return fit_method, inf_method


def _to_1d(x: torch.Tensor) -> torch.Tensor:
    """Return a contiguous 1D float tensor on CPU (no grad)."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().float().cpu()
    # If it's a square [B,B] (pairwise) matrix, take diagonal
    if x.ndim == 2 and x.shape[0] == x.shape[1]:
        x = torch.diagonal(x, dim1=0, dim2=1)
    # Squeeze singleton dims, then flatten to 1D
    x = x.squeeze()
    if x.ndim > 1:
        x = x.reshape(-1)
    return x.contiguous()


def _align_xy(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Coerce x,y to 1D with same length; truncate to min length if needed."""
    x = _to_1d(x)
    y = _to_1d(y)
    n = min(x.numel(), y.numel())
    if n == 0:
        return x.new_empty(0), y.new_empty(0)
    return x[:n], y[:n]


# ---------- metrics (shape-safe) ----------


def explained_variance(target: torch.Tensor, pred: torch.Tensor) -> float:
    y, p = _align_xy(target, pred)
    if y.numel() == 0:
        return float("nan")
    var_y = torch.var(y, unbiased=False)
    if var_y <= 0:
        return 0.0
    ev = 1.0 - torch.var(y - p, unbiased=False) / (var_y + 1e-12)
    return float(ev.item())


def mse(target: torch.Tensor, pred: torch.Tensor) -> float:
    y, p = _align_xy(target, pred)
    if y.numel() == 0:
        return float("nan")
    return float(torch.mean((y - p) ** 2).item())


def pearson_corr(target: torch.Tensor, pred: torch.Tensor) -> float:
    """Pearson r (shape-safe)."""
    y, p = _align_xy(target, pred)
    if y.numel() < 2:
        return float("nan")
    y = (y - y.mean()) / (y.std(unbiased=False) + 1e-12)
    p = (p - p.mean()) / (p.std(unbiased=False) + 1e-12)
    r = torch.mean(y * p)
    return float(r.item())


def spearman_corr(target: torch.Tensor, pred: torch.Tensor) -> float:
    """Spearman rho via rank-transform (ties handled by average rank)."""
    y, p = _align_xy(target, pred)
    if y.numel() < 2:
        return float("nan")
    y_np = y.numpy()
    p_np = p.numpy()
    # average-rank for ties
    y_rank = _average_rank(y_np)
    p_rank = _average_rank(p_np)
    yr = torch.from_numpy(y_rank).float()
    pr = torch.from_numpy(p_rank).float()
    return pearson_corr(yr, pr)


def _average_rank(arr: np.ndarray) -> np.ndarray:
    order = arr.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    # tie groups -> average
    vals = arr[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order][i] + ranks[order][j])
            ranks[order][i : j + 1] = avg
        i = j + 1
    return ranks


def _to_1d_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
    return x


def _diag_if_pairwise(arr, other_len=None):
    """
    If 'arr' is:
      • 2D square  -> return its diagonal
      • 1D but length == other_len**2 (likely flattened [B,B]) -> reshape to [B,B] and take diagonal
      • otherwise   -> flatten to 1D
    """
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        return np.diag(a)
    if a.ndim == 1 and other_len is not None:
        n = int(round(other_len**0.5))
        # if a looks like flattened [B,B] and matches other's B
        if n * n == a.size and n == other_len:
            return np.diag(a.reshape(n, n))
    return a.reshape(-1)


def _align_xy_numpy(x, y):
    """
    Make x,y 1D, fix pairwise diagonals, drop non-finite, and align lengths.
    """
    x = _to_1d_numpy(x)
    y = _to_1d_numpy(y)

    # Heuristic: try to recover diagonals if one looks like B^2
    x = _diag_if_pairwise(x, other_len=y.size)
    y = _diag_if_pairwise(y, other_len=x.size)

    # Now ensure 1D
    x = x.reshape(-1).astype(np.float64, copy=False)
    y = y.reshape(-1).astype(np.float64, copy=False)

    # Align lengths
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    # Filter non-finite jointly
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return x[m], y[m]


def mutual_information(
    x,
    y,
    *,
    n_bins: int = 20,
    strategy: str = "quantile",  # "quantile" or "uniform"
    normalized: bool = True,
    eps: float = 1e-12,
) -> float:
    """
    Plug-in MI with 2D histogram. Broadcast-safe, shape-safe, and robust to ties.
    Returns NMI in [0,1] if normalized=True; MI in nats otherwise.
    """
    x, y = _align_xy_numpy(x, y)
    if x.size == 0:
        return float("nan")

    # Degenerate variables -> MI = 0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0

    # Build bin edges
    def _edges(z):
        if strategy == "quantile":
            qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
            e = np.quantile(z, qs)
            e = np.unique(e)  # drop duplicates from ties
            if e.size < 2:
                zmin = float(np.min(z))
                return np.array([zmin - 1e-6, zmin + 1e-6], dtype=np.float64)
            return e.astype(np.float64)
        else:
            zmin, zmax = float(np.min(z)), float(np.max(z))
            if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin + 1e-12:
                zmin = zmin if np.isfinite(zmin) else 0.0
                return np.array([zmin - 1e-6, zmin + 1e-6], dtype=np.float64)
            return np.linspace(zmin, zmax, int(n_bins) + 1, dtype=np.float64)

    bx = _edges(x)
    by = _edges(y)

    # Joint histogram -> probabilities
    H, _, _ = np.histogram2d(x, y, bins=[bx, by])
    total = H.sum()
    if total <= 0:
        return 0.0
    Pxy = H / total  # (Kx, Ky)
    Px = Pxy.sum(axis=1, keepdims=True)  # (Kx, 1)
    Py = Pxy.sum(axis=0, keepdims=True)  # (1, Ky)

    # Logs with safe zeros; pure broadcasting
    logPxy = np.where(Pxy > 0, np.log(Pxy + eps), 0.0)
    logPx = np.where(Px > 0, np.log(Px + eps), 0.0)
    logPy = np.where(Py > 0, np.log(Py + eps), 0.0)

    mi = float((Pxy * (logPxy - logPx - logPy)).sum())

    if not normalized:
        return mi

    # Normalized MI in [0,1] via sqrt(Hx * Hy)
    Hx = float(-(Px[Px > 0] * np.log(Px[Px > 0])).sum())
    Hy = float(-(Py[Py > 0] * np.log(Py[Py > 0])).sum())
    denom = np.sqrt(max(Hx, eps) * max(Hy, eps))
    nmi = mi / denom if denom > eps else float("nan")
    return float(np.clip(nmi, 0.0, 1.0))


def wasserstein_1d(
    x,
    y,
    *,
    n_quantiles: int = 200,
) -> float:
    """
    1D Wasserstein-1 (Earth Mover's) distance between empirical distributions of x and y.
    Quantile-based implementation (robust & SciPy-free):
        W1 = ∫_0^1 |Qx(u) - Qy(u)| du  ≈ mean_u |Qx(u) - Qy(u)|
    Units: same as x/y (e.g., returns).
    """
    x = _to_1d_numpy(x)
    y = _to_1d_numpy(y)
    mx = np.isfinite(x)
    my = np.isfinite(y)
    if not np.any(mx) or not np.any(my):
        return float("nan")
    x = x[mx]
    y = y[my]
    # handle degenerate (constant) cases
    if np.allclose(x, x[0]) and np.allclose(y, y[0]):
        return float(abs(float(x[0]) - float(y[0])))
    u = np.linspace(0.0, 1.0, int(n_quantiles), endpoint=True)
    qx = np.quantile(x, u)
    qy = np.quantile(y, u)
    return float(np.mean(np.abs(qx - qy)))


def kl_divergence_hist(
    p_samples,
    q_samples,
    *,
    n_bins: int = 30,
    strategy: str = "quantile",  # "quantile" or "uniform"
    eps: float = 1e-10,
) -> float:
    """
    KL(P || Q) between empirical 1D distributions via shared-bin histograms.
    Returns KL in **nats**.
    - Use the same bins for P and Q (built from pooled samples).
    - Adds eps smoothing to avoid log(0).
    NOTE: For continuous scalars, this is a *distribution-level* alignment metric,
    not a pointwise calibration metric.
    """
    p = _to_1d_numpy(p_samples)
    q = _to_1d_numpy(q_samples)
    mp = np.isfinite(p)
    mq = np.isfinite(q)
    if not np.any(mp) or not np.any(mq):
        return float("nan")
    p = p[mp]
    q = q[mq]

    # Build shared edges from pooled data
    z = np.concatenate([p, q], axis=0)
    if strategy == "quantile":
        edges = np.quantile(z, np.linspace(0, 1, int(n_bins) + 1))
        edges = np.unique(edges)
        if edges.size < 2:
            # widen tiny span
            zmin = float(z.min())
            edges = np.array([zmin - 1e-6, zmin + 1e-6])
    else:
        zmin, zmax = float(z.min()), float(z.max())
        if zmax <= zmin + 1e-12:
            edges = np.array([zmin - 1e-6, zmin + 1e-6])
        else:
            edges = np.linspace(zmin, zmax, int(n_bins) + 1)

    Hp, _ = np.histogram(p, bins=edges)
    Hq, _ = np.histogram(q, bins=edges)
    P = Hp.astype(np.float64)
    Q = Hq.astype(np.float64)

    P = P / (P.sum() + eps)
    Q = Q / (Q.sum() + eps)

    # KL(P||Q) = sum P * log(P/Q); safe with eps
    kl = P * (np.log(P + eps) - np.log(Q + eps))
    return float(np.sum(kl))


def js_divergence_hist(
    p_samples,
    q_samples,
    *,
    n_bins: int = 30,
    strategy: str = "quantile",
    eps: float = 1e-10,
) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded in [0, ln 2]) using hist bins.
    Often nicer to report than raw KL due to symmetry + finiteness.
    """
    p = _to_1d_numpy(p_samples)
    q = _to_1d_numpy(q_samples)
    mp = np.isfinite(p)
    mq = np.isfinite(q)
    if not np.any(mp) or not np.any(mq):
        return float("nan")
    p = p[mp]
    q = q[mq]

    z = np.concatenate([p, q], axis=0)
    if strategy == "quantile":
        edges = np.quantile(z, np.linspace(0, 1, int(n_bins) + 1))
        edges = np.unique(edges)
        if edges.size < 2:
            zmin = float(z.min())
            edges = np.array([zmin - 1e-6, zmin + 1e-6])
    else:
        zmin, zmax = float(z.min()), float(z.max())
        if zmax <= zmin + 1e-12:
            edges = np.array([zmin - 1e-6, zmin + 1e-6])
        else:
            edges = np.linspace(zmin, zmax, int(n_bins) + 1)

    Hp, _ = np.histogram(p, bins=edges)
    Hq, _ = np.histogram(q, bins=edges)
    P = Hp.astype(np.float64)
    Q = Hq.astype(np.float64)
    P = P / (P.sum() + eps)
    Q = Q / (Q.sum() + eps)

    M = 0.5 * (P + Q)
    kl_p = np.sum(P * (np.log(P + eps) - np.log(M + eps)))
    kl_q = np.sum(Q * (np.log(Q + eps) - np.log(M + eps)))
    return float(0.5 * (kl_p + kl_q))  # nats

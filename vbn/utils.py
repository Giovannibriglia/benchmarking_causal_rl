from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ───────── general helpers ─────────


def make_strides(cards: List[int]) -> List[int]:
    if not cards:
        return []
    strides = [1]
    for k in cards[:-1]:
        strides.append(strides[-1] * k)
    return strides


def bincount_fixed(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.bincount(x, minlength=minlength).to(torch.float32)


def stack_float(xs: List[torch.Tensor], device, dtype) -> torch.Tensor:
    return torch.stack([x.to(device=device, dtype=dtype).flatten() for x in xs], dim=1)


# ───────── discrete factor ops for VE ─────────


def align_to_axes(
    t: torch.Tensor,
    axes: List[str],
    target_axes: List[str],
    batch_dim: Optional[int],
) -> torch.Tensor:
    """
    Return a view of `t` that has dims [B?] + target_axes in order.
    - If `batch_dim` is not None, the first dim is treated as batch.
      We will insert a leading singleton batch dim if it's missing.
    - We permute existing variable dims to their target positions.
    - We insert singleton dims for any variables in `target_axes` that
      are not present in `axes`.
    """
    # 0) Ensure leading batch dim if in batched mode
    base = 1 if batch_dim is not None else 0
    if batch_dim is not None and t.ndim == len(axes):  # no batch dim yet
        t = t.unsqueeze(0)  # [1, ...]
    # Recompute base after potential unsqueeze (still 1, but keeps logic clear)
    base = 1 if batch_dim is not None else 0

    if axes == target_axes:
        # If we're in batched mode but t somehow still lacks batch, we already unsqueezed above.
        return t

    # 1) Move existing axes into the order of target_axes where present
    # positions of variables that are present in both
    present_positions_target = [j for j, a in enumerate(target_axes) if a in axes]
    old_positions_axes = [axes.index(target_axes[j]) for j in present_positions_target]

    if len(old_positions_axes) > 0:
        # Map [base + old_pos] -> [base + new_pos]
        src = [base + i for i in old_positions_axes]
        dst = [base + j for j in present_positions_target]
        # When t has fewer dims than max(dst)+1, something is off; but the batch fix above should prevent it.
        t = t.movedim(src, dst)

    # 2) Insert singleton dims for missing variables (those in target_axes not in axes)
    missing_positions = [j for j, a in enumerate(target_axes) if a not in axes]
    # Insert left-to-right; account for prior insertions shifting indices
    inserted = 0
    for j in missing_positions:
        t = t.unsqueeze(base + j + inserted)
        inserted += 1

    return t


def product_factors(
    fs: List[Tuple[torch.Tensor, List[str]]],
    batch_dim: Optional[int],
) -> Tuple[torch.Tensor, List[str]]:
    """
    Multiply a list of factors; each factor as (tensor, axes-names in order).
    Returns fused (tensor, axes) with axes in the union (existing order preserved + new).
    Supports optional leading batch dim at 0.
    """
    if not fs:
        return torch.tensor(1.0, device="cpu"), []

    t, axes = fs[0]
    for u, ax_u in fs[1:]:
        union = axes + [a for a in ax_u if a not in axes]
        # Align both to union — this will insert batch dim if needed and add singleton variable dims
        t = align_to_axes(t, axes, union, batch_dim)
        u = align_to_axes(u, ax_u, union, batch_dim)
        t = t * u  # broadcast multiply
        axes = union
    return t, axes


# ───────── pivots / evidence utilities ─────────


def pivot_from_data(
    order: List[str],
    types: Dict[str, str],
    cards: Optional[Dict[str, int]],
    data: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Mean for continuous, mode for discrete."""
    pivot = {}
    for n in order:
        if types[n] == "continuous":
            pivot[n] = data[n].mean()
        else:
            x = to_long(data[n])
            k = int(cards[n])
            counts = torch.bincount(x, minlength=k)
            pivot[n] = torch.argmax(counts).to(x.device)
    return pivot


def normal_logpdf(
    x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    var = var.clamp_min(eps)
    return -0.5 * (torch.log(2 * torch.pi * var) + (x - mean) ** 2 / var)


def normal_pdf(
    x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    return torch.exp(normal_logpdf(x, mean, var, eps))


def to_long(x):
    # Accept torch.Tensor, pandas.Series, numpy arrays, lists
    if not isinstance(x, torch.Tensor):
        # pandas Series / numpy arrays / lists → numpy array → tensor
        x = torch.as_tensor(np.asarray(x))
    return x.long()


def to_float(x, dtype: torch.dtype = torch.float32):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(np.asarray(x))
    # Keep device if already a tensor with device; otherwise default cpu
    return x.to(dtype)


def unpack_gaussian(result, var_name: str):
    """
    Supports two shapes:
      1) (mean_dict, cov_matrix)   – your current return
      2) {var: {"mean": ..., "std": ...}} – future/dict style
    Returns: (mean, std) as 1D tensors (squeezed).
    """
    # Case 2: dict-of-dicts
    if (
        isinstance(result, dict)
        and var_name in result
        and isinstance(result[var_name], dict)
    ):
        mu = result[var_name]["mean"].squeeze()
        std = result[var_name]["std"].squeeze()
        return mu, std

    # Case 1: tuple -> (mean_dict, cov)
    mean_dict, cov = result  # means is a dict[str, Tensor]; cov is a Tensor or dict
    mu = mean_dict[var_name].squeeze()

    # cov can be:
    #  - a full matrix (… x k x k) for the queried vars in the same order you passed
    #  - a dict[var] -> variance
    if isinstance(cov, dict):
        var_tensor = cov[var_name].squeeze()
    else:
        # If you queried a single variable like ["A"], it's the [0,0] entry.
        # Handle possible batch dims with ellipsis.
        var_tensor = cov[..., 0, 0].squeeze()
    std = var_tensor.clamp_min(0).sqrt()
    return mu, std


def infer_types_and_cards(data, threshold_unique=20):
    """
    Infer variable types and cardinalities from data.

    Args:
        data (dict | pd.DataFrame):
            Mapping from variable name -> tensor/series/array.
        threshold_unique (int):
            If a variable has <= threshold_unique unique values, treat it as discrete.
            Otherwise, continuous.

    Returns:
        types (dict): {var: "discrete"/"continuous"}
        cards (dict): {var: cardinality (only for discrete vars)}
    """
    types, cards = {}, {}

    for var, values in data.items():
        # Convert pandas -> numpy
        if isinstance(values, pd.Series):
            values = values.values
        # Convert to torch tensor if not already
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)

        unique_vals = torch.unique(values)

        if values.dtype in (torch.float32, torch.float64):
            # Heuristic: treat small set of unique values as categorical
            if unique_vals.numel() <= threshold_unique:
                types[var] = "discrete"
                cards[var] = unique_vals.numel()
            else:
                types[var] = "continuous"
        else:
            # For integer/bool types → always discrete
            types[var] = "discrete"
            cards[var] = unique_vals.numel()

    return types, cards

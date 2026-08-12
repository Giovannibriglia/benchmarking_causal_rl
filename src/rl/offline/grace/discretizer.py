"""Per-dimension quantile discretizer for the tabular GRACE slice.

The benchmark's wired envs (CartPole-v1, Acrobot-v1) have continuous Box
observations and the repo has NO discretization path (verified in the GRACE
Phase-1 audit), so GRACE carries its own: per-dimension quantile bin edges are
FIT ON THE OFFLINE DATASET (state-conditional by construction — no marginal
statistic anywhere), then observations encode to either a joint bin index
(small spaces, exact joint transition CPT) or per-dimension indices (large
spaces, the factored-transition approximation approved as amendment A8).

Rewards get the same treatment via ``RewardCodec``: the wired confounders keep
reward support tiny ({1, 1+c_r} on CartPole, {-1, 0} on Acrobot), so distinct
values become classes directly; a quantile fallback guards dense-reward envs.

Determinism: fitting uses only sorted quantiles of the dataset — no RNG.
"""

from __future__ import annotations

import torch

__all__ = ["Discretizer", "RewardCodec"]


class Discretizer:
    """Per-dim quantile bins over a ``(N, D)`` observation tensor.

    ``encode`` returns the JOINT bin index ``(N,)`` (row-major over dims);
    ``encode_dims`` returns the per-dim indices ``(N, D)`` for the factored
    path. Both are pure tensor ops (bucketize), safe on any device.
    """

    def __init__(self, n_bins: int) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = int(n_bins)
        self.edges: torch.Tensor | None = None  # (D, n_bins-1) inner edges
        self.obs_dim: int | None = None

    @property
    def joint_size(self) -> int:
        if self.obs_dim is None:
            raise RuntimeError("Discretizer.fit must run before joint_size")
        return self.n_bins**self.obs_dim

    def fit(self, obs: torch.Tensor) -> "Discretizer":
        """Fit inner edges at the per-dim quantiles ``i/n_bins`` of the data.

        Degenerate dims (constant feature) get strictly increasing synthetic
        edges so bucketize stays well-defined and every row lands in bin 0.
        """
        if obs.dim() != 2:
            raise ValueError(f"expected (N, D) observations, got {tuple(obs.shape)}")
        obs = obs.detach().float()
        self.obs_dim = int(obs.shape[1])
        qs = torch.linspace(0.0, 1.0, self.n_bins + 1, device=obs.device)[1:-1]
        edges = torch.quantile(obs, qs, dim=0).T.contiguous()  # (D, n_bins-1)
        # Enforce strictly increasing edges per dim (quantile ties on discrete
        # or constant features collapse edges -> nudge by a tiny epsilon).
        eps = 1e-6 + 1e-6 * obs.abs().amax(dim=0, keepdim=True).T
        for j in range(1, edges.shape[1]):
            edges[:, j] = torch.maximum(edges[:, j], edges[:, j - 1] + eps.squeeze(1))
        self.edges = edges
        return self

    def _check(self, obs: torch.Tensor) -> torch.Tensor:
        if self.edges is None:
            raise RuntimeError("Discretizer.fit must run before encode")
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs.detach().float()

    def encode_dims(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-dim bin indices ``(N, D)`` in ``[0, n_bins)``."""
        obs = self._check(obs)
        edges = self.edges.to(obs.device)
        cols = [
            torch.bucketize(obs[:, d].contiguous(), edges[d].contiguous())
            for d in range(self.obs_dim)
        ]
        return torch.stack(cols, dim=1).long()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Joint (row-major) bin index ``(N,)`` in ``[0, n_bins**D)``."""
        dims = self.encode_dims(obs)
        idx = dims[:, 0]
        for d in range(1, self.obs_dim):
            idx = idx * self.n_bins + dims[:, d]
        return idx


class RewardCodec:
    """Map rewards to a small class alphabet, remembering per-class means.

    Distinct-value codec when the support is small (the wired confounders keep
    it at 2-3 values); quantile-bin fallback otherwise. ``class_values`` are
    the per-class MEANS, so ``E[R | class-dist]`` is exact for the distinct
    case and a documented approximation for the binned fallback.
    """

    def __init__(self, max_classes: int = 12) -> None:
        self.max_classes = int(max_classes)
        self.boundaries: torch.Tensor | None = None  # sorted midpoints
        self.class_values: torch.Tensor | None = None  # (C,) per-class mean

    @property
    def n_classes(self) -> int:
        if self.class_values is None:
            raise RuntimeError("RewardCodec.fit must run before n_classes")
        return int(self.class_values.numel())

    def fit(self, rewards: torch.Tensor) -> "RewardCodec":
        r = rewards.detach().float().reshape(-1)
        uniq = torch.unique(r)
        if uniq.numel() <= self.max_classes:
            vals = uniq.sort().values
        else:
            qs = torch.linspace(0.0, 1.0, self.max_classes + 1, device=r.device)
            edges = torch.quantile(r, qs)[1:-1]
            idx = torch.bucketize(r, edges.contiguous())
            vals = torch.stack(
                [
                    r[idx == c].mean() if bool((idx == c).any()) else r.mean()
                    for c in range(self.max_classes)
                ]
            )
        self.class_values = vals
        self.boundaries = (vals[1:] + vals[:-1]) / 2.0 if vals.numel() > 1 else vals[:0]
        return self

    def encode(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.class_values is None:
            raise RuntimeError("RewardCodec.fit must run before encode")
        r = rewards.detach().float()
        return torch.bucketize(r, self.boundaries.to(r.device).contiguous())

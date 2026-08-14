# NOTICE — vendored NeuralBayesianNetworks (NBN)

This directory is a vendored snapshot of the NeuralBayesianNetworks library.

- **Source:** https://github.com/Giovannibriglia/NeuralBayesianNetworks
- **Upstream commit:** `926fa62b8db6` ("fix(learning): respect mechanism-designed
  training budgets; make EWC consolidation opt-out (#254)")
- **Version string:** `0.1.dev30+g926fa62b8` (see `_version.py`)
- **Synced into this repository:** 2026-08-14 (GRACE v2, branch `feat/grace-v2`)
- **Author:** Giovanni Briglia (author of both this repository and the upstream
  library)

## Licence

**Apache-2.0** — see `nbn/LICENSE`.

Upstream declares Apache-2.0 in its `pyproject.toml` but ships no LICENSE file,
so the canonical Apache-2.0 text is vendored here on its behalf. The host
repository is GPLv3; Apache-2.0 is one-way compatible with GPLv3, so including
this directory is fine, but **the vendored licence must state the code's own
licence, not the host's**.

> Correction of record: an earlier snapshot on `feat/grace-critic` vendored a
> **GPLv3** `LICENSE` here, copied from the host repo. That misstated the
> upstream licence and is corrected by this commit.

## Version-string caveat

`0.1.dev30` is **not** older than the previously vendored `0.7.dev37`. Upstream
carries no PEP 440 tags, so setuptools-scm derives different strings from the
same tag history; the `0.7.dev37` snapshot came from a truncated non-PEP440 tag.
The **upstream commit id is the only reliable provenance** — hence it is pinned
above.

## Local conventions

- The snapshot is **pinned**: excluded from this repository's pre-commit
  formatters/linters (`exclude: ^nbn/`) so the tree stays byte-diffable against
  upstream. Do not reformat or "fix" files here.
- **Do not upgrade torch to satisfy upstream.** Upstream's floor is `torch>=2.2`
  and a fresh install resolves to 2.13; this repository hard-pins **2.10.0** for
  golden-bitwise reasons. The pin wins.
- Upstream's packaging would also install a top-level `benchmarking` package and
  an `nbn-bench` script; only `nbn/` is vendored here.

## Known sharp edges (verified at this commit)

GRACE works around these; it does **not** patch this directory.

| edge | consequence for callers |
|---|---|
| `TensorVariableElimination` silently ignores `do=` | `model.query(do=…)` on an all-discrete net returns the **prior**, no warning. Never route interventions through VE. |
| `intervene()` severs the caller's gradient (`nn.Parameter` copy) | Use `model.sample(n, do=…)` when you need ∂/∂θ — that path *is* differentiable (verified: grad 1.0051 vs analytic 1.0). |
| Batched `do` is unsupported | `sample`/`query` derive the batch from evidence only; a batched do-value raises. Loop over intervention values. |
| Engine caches keyed on `id(model)`, never invalidated on refit | An externally-held engine serves stale posteriors after `fit()`. Invalidate explicitly. |
| `LinearGaussianMechanism` crashes on long (discrete) parents | Float-cast discrete parent columns before `fit` — discrete action → continuous reward is exactly this topology. |
| `fit()`'s "held-out LL" is train LL; no early stopping, no validation split | Callers must own their held-out machinery. |
| `is_fitted` is `False` for every continuous mechanism after a successful fit | Never gate on it. |
| `save`/`load` cannot round-trip (`load()` discards the state dict) | Persist via `torch.save` of the module instead. |

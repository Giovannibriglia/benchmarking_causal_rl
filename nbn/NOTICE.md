# NOTICE — vendored NeuralBayesianNetworks (NBN)

This directory is a vendored snapshot of the NeuralBayesianNetworks library.

- **Source:** https://github.com/Giovannibriglia/NeuralBayesianNetworks
- **Upstream commit:** `9b5c6b7c6d22` ("fix(bench): raise the per-cell memory
  floor above torch's startup cost; pin xdist threads; harden the LG SEM
  (#256)"). The `nbn/` subtree is identical at its parent `a91d8f9`
  ("fix(core): honour do= in VE, mutilate the graph in intervene(), repair
  from_bif, and close eight lifecycle defects (#255)") — that is the commit
  that matters for callers.
- **Version string:** `0.1.dev32+g9b5c6b7c6` (see `_version.py`)
- **Synced into this repository:** 2026-08-15 (GRACE v2, branch `feat/grace-v2`).
  Supersedes the 2026-08-14 sync to `926fa62b8db6`.
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

## Known sharp edges — RE-AUDITED at this commit

Upstream `a91d8f9` fixed most of what the previous snapshot's audit recorded.
Verified empirically against this drop, not taken on trust:

| edge | status at this commit |
|---|---|
| `TensorVariableElimination` silently ignored `do=` | **FIXED** — VE implements the do-operator directly. Verified: prior `[0.491, 0.508]`, `do(A=1)` gives `[0.102, 0.898]` under VE and `[0.099, 0.901]` under LW. |
| `intervene()` broke exact VE on discrete nets | **FIXED** — `DeterministicMechanism` gained the tabular interface. Verified: `intervene().query(engine=VE)` returns `[0.102, 0.898]`. |
| Batched `do` raised an opaque shape error | **FIXED for the engines** — `query_batch(do=…)` spans evidence and do. Verified: `E[R|do(B)] = [0.00, 0.97, 1.94, 2.91]` against truth `[0, 1, 2, 3]`. Ancestral sampling still has no batch axis but now raises a clear `ValueError` instead of an `expand()` error. |
| Engine caches keyed on bare `id(model)` served stale posteriors | **FIXED** — weakref + `_cache_version` bumped by fit/update/set_mechanism. Verified: a held engine returns `[0.102, 0.898]` before a refit and `[0.507, 0.493]` after. |
| `LinearGaussianMechanism` crashed on long (discrete) parents | **FIXED** — verified: a discrete-parent → continuous-child fit completes. This is the discrete-action → continuous-reward topology. |
| `is_fitted` stayed `False` after a successful fit | **FIXED** for LG/MDN/flow — verified `True`. (Our earlier claim said "every continuous mechanism"; kde/knn/flexcode always implemented it. Upstream corrected us.) |
| `save`/`load` could not round-trip | **FIXED** — format 2 carries the fitted mechanism modules. Verified: reloaded model reproduces `E[R|do(H=1)] = 1.994` vs `1.997`. |
| `fit()`'s "held-out LL" was in-sample | **FIXED** — relabelled in-sample, and the absence of any split or early stopping is now stated. |
| `intervene()` severs the caller's gradient | **STILL PRESENT, and inherent** — it returns a deep-copied model. Documented upstream rather than altered. Use `model.sample(n, do=…)`, which is the differentiable path (verified grad ≈ 1.0 vs analytic 1.0). |

Remaining gaps that matter to this repository (see `docs/nbn_requirements.md`):
**no sample weights in fitting** and **no latent/EM/mixture support**. Both are
worked around GRACE-side.

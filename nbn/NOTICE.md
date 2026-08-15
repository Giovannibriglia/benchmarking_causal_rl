# NOTICE — vendored NeuralBayesianNetworks (NBN)

This directory is a vendored snapshot of the NeuralBayesianNetworks library.

- **Source:** https://github.com/Giovannibriglia/NeuralBayesianNetworks
- **Release:** **v0.14.0** (annotated tag) — the repository's first PEP 440 tag.
- **Upstream commit:** `4784b8e6a09e` ("feat: parent-gradient contract,
  per-sample weighted fitting, model.log_prob, licence + provenance (#257)")
- **Version string:** `0.14.0` (see `_version.py`)
- **Synced into this repository:** 2026-08-15 (GRACE v2, branch `feat/grace-v2`).
  Supersedes the syncs to `9b5c6b7c6d22` and `926fa62b8db6`.
- **Author:** Giovanni Briglia (author of both this repository and the upstream
  library)

## Licence

**Apache-2.0** — see `nbn/LICENSE`.

Upstream now **ships its own `LICENSE`** at its repository root; `nbn/LICENSE`
here is a byte-identical copy of that file, placed inside `nbn/` because only
the `nbn/` subtree is vendored. Earlier syncs had to supply the text on
upstream's behalf, since it declared Apache-2.0 in `pyproject.toml` without
shipping the file.

The host repository is GPLv3; Apache-2.0 is one-way compatible with GPLv3, so
including this directory is fine, but **the vendored licence states the code's
own licence, not the host's**.

> Correction of record: an early snapshot on `feat/grace-critic` vendored a
> **GPLv3** `LICENSE` here, copied from the host repo. That misstated the
> upstream licence and was corrected on `feat/grace-v2`.

## Version-string provenance

**Correction of record — our earlier explanation was wrong.** We attributed the
`0.1.dev30` vs `0.7.dev37` mismatch to setuptools-scm deriving different
strings from the same tag history. The real cause was **clone depth**: the
`0.1.dev30` string came from a shallow, tagless clone falling back, not from an
older commit. The two snapshots were never out of order.

With `v0.14.0` tagged and `fallback_version` configured upstream, the string is
now trustworthy and orderable, and the untrustworthy cases announce themselves:

| how the source was obtained | resolves to |
|---|---|
| full clone with tags | `0.14.0` |
| shallow / tagless clone | `0.14.0.dev1+unknown.g4784b8e6a` |
| `.git`-less tree (a file drop, as vendored here) | `0.14.0.dev0+unknown` |

The `+unknown` local segment is the marker: a version carrying it was derived
without tag history and should not be trusted for ordering. Because upstream
generates `_version.py` at build time and gitignores it, the vendored copy is
**hand-pinned** to `0.14.0` with the commit id recorded above.

## Local conventions

- The snapshot is **pinned**: excluded from this repository's pre-commit
  formatters/linters (`exclude: ^nbn/`) so the tree stays byte-diffable against
  upstream. Do not reformat or "fix" files here.
- **Do not upgrade torch to satisfy upstream.** Upstream's floor is `torch>=2.2`
  and a fresh install resolves to 2.13; this repository hard-pins **2.10.0** for
  golden-bitwise reasons. The pin wins.
- Upstream's packaging would also install a top-level `benchmarking` package and
  an `nbn-bench` script; only `nbn/` is vendored here.

## Known sharp edges — RE-AUDITED at v0.14.0

Eight of the nine edges the original audit recorded were fixed at `a91d8f9`;
the ninth is now **documented contract, not defect**. Re-verified on this
vendored copy.

| edge | status at v0.14.0 |
|---|---|
| VE silently ignored `do=` | **FIXED** — VE implements the do-operator; verified prior `[0.491,0.508]` vs `do(A=1)` `[0.102,0.898]`, matching LW. |
| `intervene()` broke exact VE on discrete nets | **FIXED** — `DeterministicMechanism` gained the tabular interface. |
| Batched `do` raised an opaque shape error | **FIXED for the engines** — `query_batch(do=…)` verified `[0.00,0.97,1.94,2.91]` vs truth `[0,1,2,3]`. Ancestral sampling has no batch axis and now says so. |
| Engine caches keyed on bare `id(model)` | **FIXED** — weakref + `_cache_version`; a held engine re-reads after refit. |
| `LinearGaussianMechanism` crashed on long parents | **FIXED** — the discrete-action → continuous-reward topology fits. |
| `is_fitted` stayed `False` after fitting | **FIXED** for LG/MDN/flow. (Our claim said "every continuous mechanism"; kde/knn/flexcode always implemented it — upstream corrected us.) |
| `save`/`load` could not round-trip | **FIXED** — format 2 carries the fitted mechanisms. |
| `fit()`'s "held-out LL" was in-sample | **FIXED** — relabelled, and the absence of a split/early stopping stated. |
| `intervene()` severs the caller's gradient | **CONTRACT, not defect** — inherent to returning a deep-copied model. `model.sample(n, do=…)` is the differentiable interventional path (verified grad **1.0000** vs analytic 1.0). |

### Differentiability contract (v0.14.0, pinned upstream by tests)

| path | differentiable through caller-supplied tensors? |
|---|---|
| `mechanism.log_prob(x, parents)` | **yes** |
| `model.log_prob(data[, per_node])` | **yes** |
| `model.sample(n[, do=…])` | **yes** |
| `query` / `query_batch` | **no — by design** (they run under `inference_mode`) |
| `intervene()` | **no** — severs; use `sample(do=…)` |

### Remaining constraints (not defects)

* **`update_local` refuses weights** (`NotImplementedError`, verified) — there
  is no incremental *weighted* update, so a weighted EM refresh must refit.
* **KDE's bandwidth rule is unweighted** — the weighted Nadaraya-Watson mixture
  is exact, but Scott/Silverman still use the unweighted spread and row count.
* **No latent / EM / mixture support** — by design; callers implement it.

Both constraints and their consequences for this repository are recorded in
`docs/grace_v2.md`.

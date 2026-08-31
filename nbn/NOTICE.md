# NOTICE — vendored NeuralBayesianNetworks (NBN)

This directory is a vendored snapshot of the NeuralBayesianNetworks library.

- **Source:** https://github.com/Giovannibriglia/NeuralBayesianNetworks
- **Release:** **v0.15.0** (annotated tag).
- **Upstream commit:** `3f134126921e` ("feat(learning): warm_start on
  fit_local, so an M-step is a step (#260)" — the squash commit; delivers R3
  plus the FlexCode root-branch weights fix)
- **Version string:** `0.15.0` (see `_version.py`)
- **Synced into this repository:** 2026-08-19 (GRACE v2, branch `feat/grace-v2`).
  Supersedes the v0.14.0 sync to `4784b8e6a09e` (and, before it,
  `9b5c6b7c6d22` and `926fa62b8db6`).
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
| full clone with tags | `0.15.0` |
| shallow / tagless clone | `0.15.0.dev1+unknown.g3f1341269` |
| `.git`-less tree (a file drop, as vendored here) | `0.15.0.dev0+unknown` |

The `+unknown` local segment is the marker: a version carrying it was derived
without tag history and should not be trusted for ordering. Because upstream
generates `_version.py` at build time and gitignores it, the vendored copy is
**hand-pinned** to `0.15.0` with the commit id recorded above.

## Local conventions

- The snapshot is **pinned**: excluded from this repository's pre-commit
  formatters/linters (`exclude: ^nbn/`) so the tree stays byte-diffable against
  upstream. Do not reformat or "fix" files here.
- **Do not upgrade torch to satisfy upstream.** Upstream's floor is `torch>=2.2`
  and a fresh install resolves to 2.13; this repository hard-pins **2.10.0** for
  golden-bitwise reasons. The pin wins.
- Upstream's packaging would also install a top-level `benchmarking` package and
  an `nbn-bench` script; only `nbn/` is vendored here.

## Known sharp edges — RE-AUDITED at v0.15.0

**How this audit was carried at the v0.15.0 sync** (and should be at every
sync): `tools/audit_nbn_sharp_edges.py`, run against THIS vendored copy under
this repository's torch pin — 10/10 checks pass. Two tiers: subtrees
byte-identical to the previously audited tag keep their rows without
re-measurement (`inference/`, `sampling/`, `update/`, `core/dag.py` are
untouched v0.14.0 → v0.15.0, so every engine row below carries over); the
touched subtrees (`learning/`, `mechanisms/`, the fit-threading kwarg in
`core/network.py`) were re-verified empirically.

### GAINED at v0.15.0

| edge | status |
|---|---|
| `fit_local` rebuilds the network with a fresh init **on every call** (default) | **DOCUMENTED CONTRACT with an opt-out** — `warm_start=True` continues the existing parameters. The rebuild default is byte-compatible with v0.14.0 behaviour; an EM caller must pass the flag or its M-step is an independent refit (this was R3, and it is what made GRACE restart-EM). |
| the warm-start contract | **DELIVERED and audited on this copy**: fresh Adam over existing parameters (moments deliberately not carried — they are absent from `state_dict()`, so a snapshot/restore backtrack could not revert them); standardisation buffers freeze (`_pa_mean`/`_pa_std`, FlexCode's `_y_min`/`_y_max`); shape mismatch **raises** (never a silent rebuild); never-fitted **cold-builds**, observable via `warm_started: bool` in the metrics dict; closed-form branches are accepted no-ops declared by `Mechanism.warm_start_is_noop`, with **root** branches of MDN/neural-categorical/FlexCode always recomputing so they keep responding to the caller's weights. |
| FlexCode **root branch dropped per-sample weights** | **FIXED at v0.15.0** (`346f073`, kept as its own commit in the PR): `weighted_moments(targets, w_vec)` replaces the dropped vector. Audited here: zeroed-weight half of a bimodal target is excluded from the fitted density. Present-but-unknown at v0.14.0 — weighted EM through a FlexCode ROOT node was silently unweighted there. |
| `fit_local` draws a fresh `randperm` **every epoch, even at full batch** | **SHARP EDGE for bitwise consumers, not a defect**: rows are permuted per call, and a permuted batch changes the floating-point reduction order, so two otherwise-identical calls differ in ulps platform-dependently. Found via CI on upstream #260. Any bitwise comparison across `fit_local` calls must pin the RNG immediately before each call. |

### LOST at v0.15.0

None — all nine v0.14.0 rows below stand (engine rows by byte-identity;
mechanism/fit rows re-verified).

### Carried over from the v0.14.0 audit

Eight of the nine edges the original audit recorded were fixed at `a91d8f9`;
the ninth is now **documented contract, not defect**.

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

### Differentiability contract (since v0.14.0, pinned upstream by tests; re-verified at v0.15.0: `sample(do=)` grad 0.9998 vs analytic 1.0)

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

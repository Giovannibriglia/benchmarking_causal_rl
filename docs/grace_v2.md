# GRACE v2 — architecture and design constraints

**Status: the estimator is not yet implemented.** L1 (the declared-diagram
layer) is in `src/rl/offline/grace/cell_graph.py`, documented in
`docs/diagram_catalogue.md`. L2–L5 are specified but unwritten. This document
records the design constraints that bind that implementation; it grows into the
as-built description as the layers land.

Companion documents: `docs/diagram_catalogue.md` (the assumption surface),
`docs/grace_v2_conditions.md` (identification derivations, C1/C2/C5),
`docs/nbn_requirements.md` (what the library owes us, and what it now
delivers).

---

## The assumption surface

v2 asserts exactly one thing per scenario: **the declared causal diagram**.
Everything else must be derived from it, learned from data, selected by a
held-out criterion, or named as an assumption in the catalogue. See
`diagram_catalogue.md` — every verdict carries the assumptions it rests on,
and `Verdict.label()` makes that travel with each number produced.

---

## Constraints imposed by NBN v0.14.0

These are properties of the library, verified on the vendored copy. Each has a
consequence GRACE must respect; none is a defect.

### N1 — Differentiability contract: route targets through `sample(do=)`

| path | differentiable through caller tensors? |
|---|---|
| `mechanism.log_prob`, `model.log_prob`, `model.sample(±do)` | **yes** |
| `query` / `query_batch` | **no — by design** (`inference_mode`) |
| `intervene()` | **no** — returns a deep copy, severing the caller's graph |

**Consequence.** L3's interventional target computation and L4's bound
optimiser must route through **`model.sample(n, do=…)`**, never `query` or
`intervene()`. Verified: `sample(do=)` gradient **1.0000** against an analytic
1.0. This is why the target computation sits behind one function boundary —
that boundary is where the differentiable path is guaranteed.

`query`/`query_batch` remain the right tool for *diagnostics* that need no
gradient (ESS, PSIS-k̂, batched dose-response readouts).

### N2 — `update_local` refuses weights: the online refresh must **refit**

`update_local(..., weights=…)` raises `NotImplementedError` (verified). There
is no incremental *weighted* update, and the EM M-step is inherently weighted
by per-episode responsibilities.

**Consequence for the online blocks.** The rolling-buffer EM refresh must
**refit** on the current episode view rather than incrementally update. This is
a design constraint recorded *before* that work starts, not a bug to discover
mid-block. Cost is bounded and already measured: a full EM fit is ≈ 22 s at
production scale, so a periodic refit is affordable at the cadences the online
cells use.

### N3 — KDE's bandwidth rule is unweighted: prefer parametric mechanisms in EM

The weighted Nadaraya–Watson mixture is exact, but the Scott/Silverman
bandwidth rule still uses the **unweighted** spread and row count. Under EM a
stratum's bandwidth would therefore reflect the pooled data rather than that
stratum's, biasing the component densities toward each other — precisely the
direction that would make strata look *less* separable than they are.

**Consequence.** In the weighted EM path, use **MDN, normalizing flow, or
LinearGaussian** for continuous nodes. `supports_weights` is `True` for all
three (verified), and `False` for KNN — so a mistaken choice fails fast in
`fit()` rather than silently biasing. If KDE ever becomes load-bearing, making
it exact requires a weighted quantile, which would be a new NBN request.

---

## What the library now provides (v0.14.0), and how v2 uses it

| capability | v2 use |
|---|---|
| `model.log_prob(data, per_node=False)` | L3's complete-data likelihood, and — with `per_node=True` — **L5's channel-split diagnostic** (action channel vs reward channel) directly, instead of re-deriving the decomposition. Raises on a missing node, so a silently-marginalised variable cannot masquerade as a fitted one. |
| `fit(..., weights=)` with `supports_weights` | The EM M-step's per-episode responsibilities, applied to whole episodes (the latent is episode-static, so one responsibility broadcasts across that episode's rows). Verified exact against replication: weighted == replicated to 4.8e-07. |
| Parent-gradient transparency (pinned upstream) | Lets **all recurrence live GRACE-side** (Hypothesis B) — a GRACE encoder's output is passed as an ordinary parent and still receives gradient. No recurrent mechanism class is needed in NBN. |

### The do-semantics obligation this leaves with GRACE

Because recurrence is GRACE-side, GRACE owns a correctness trap that no library
API can enforce: **under `do(A_t = a)`, a recurrent hidden state must advance on
the *realized* trajectory including the intervened action**, not on what the
un-mutilated mechanism would have emitted. An implementation that advances the
state from the observational action produces plausible, wrong interventional
values. Keeping recurrence GRACE-side is what makes this tractable — the
component performing the intervention is the one that owns the state update —
but it must be written deliberately and tested.

---

## Superseded workarounds

Recorded so the implementation is not written against the old plan:

* **A GRACE-side weighted M-step loop is no longer needed.** The earlier plan
  had GRACE bypass `fit_local` and run its own Adam loop over `log_prob`
  (measured 1.08 s/M-step) because sample weights were absent. Use
  `fit(..., weights=)` instead wherever `supports_weights` is `True`, and keep
  a GRACE-side loop only for a mechanism that refuses.
* **A manual per-node log-likelihood loop is no longer needed.** Use
  `model.log_prob(..., per_node=True)`.

Neither workaround was ever implemented — the estimator did not exist when the
capabilities landed — so this is a change of plan, not a refactor.

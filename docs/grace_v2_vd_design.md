# V-D — the falsification experiment: declaration matrix and cost

**Status: design, not implementation.** L5 is unstarted; this document exists
because the V-D cost projection needs the declaration structure and a scalar
guess for it brackets a range wider than the real answer.

## What V-D varies is the DECLARATION, not the data

L5 tests a *declared* diagram against data. A misspecification arm is therefore a
**wrong declaration of an existing dataset**, not a new dataset: M1 is "declare
D-A where the data is D-B". The same dataset is tested under several
declarations, and **each declaration gets its own null computed under its own
declared model** — the null is a property of the (dataset, declaration) pair.

A projection that assumes one declaration per dataset therefore **under-counts
for exactly the experiment V-D is**. The first projection did assume that.

## The matrix — a misspecification applies only where the thing it misspecifies exists

The count is **cell-dependent**, not flat. Declaring an omission of an edge that
was never declared is not a misspecification, it is the same model.

| cell | declaration | declared diagram | constraints | applies? |
|---|---|---|---|---|
| **d_a_null** | true | D-A-null | 2 | always — this is where the FALSE-POSITIVE RATE comes from |
| | M1 omit `U→A` | — | — | **no**: D-A-null declares no `U→A` edge, so the omission is a no-op |
| | M3 invalid proxy | — | — | **no**: no proxies declared |
| **d_b_prime** | true | D-B-prime | 3 | always |
| | M1 omit `U→A` | D-A | 2 | **yes** |
| | M3 invalid proxy | — | — | **no**: D-B′ uses lagged views, not declared proxies |
| **d_e** | true | D-E | 1 | always |
| | M1 omit `U→A` | D-A | 2 | **yes** |
| | M3 invalid proxy | — | — | **no**: no declared proxies |
| **d_d** | true | D-D | 2 | always |
| | M1 omit `U→A` | D-A | 2 | **yes** |
| | M3 invalid proxy | D-D | 2 | **yes** — the only cell with declared proxies to invalidate |

Declarations per cell: **d_a_null 1, d_b_prime 2, d_e 2, d_d 3** — mean ≈ 2, not
the scalar 4 the projection was carrying.

## M2 is a DEMONSTRATION, and must be scoped as one

**M2 (a spurious edge) is undetectable in principle**, and its purpose is to
*demonstrate that limitation*: detection is indistinguishable from the
false-positive rate, because a model with an extra edge is a strictly larger
model that cannot be refuted by data the smaller one fits.

Running it across 130 datasets buys **a repeated null result at full price**. It
needs **one configuration**, reported as a demonstration with the reason stated,
and the limitation belongs in the paper's limitations section next to the
untestable assumptions rather than in the detection curve.

*Chosen subset:* `d_d / CartPole-v1 / σ = 1.0`, 5 seeds — the cell with the
richest declared structure (so a spurious edge has the most room to look
plausible) and the cheapest fits, since the result is known in advance and the
run exists to document it.

## Cost consequence

Projected by `tools/project_vd_cost.py --declaration-matrix`, which reads the
per-cell counts above rather than a scalar. The two things that dominate remain
the **grid size** and the **fit cost**; declarations multiply, and the levers
(pooling across seeds, reduced B) divide.

# Diagram catalogue (GRACE v2)

**Approved as Gate A2.** The machine-readable declarations live in
`src/rl/offline/grace/cell_graph.py` (`CATALOGUE`), and
`tests/test_diagram_catalogue.py` asserts this document and that code agree
entry-for-entry, so neither can drift from the other.

**Entry ids as used in code:** `D-A`, `D-A-null`, `D-C`, `D-B`, `D-B-prime`,
`D-D`, `D-E`, `D-F`, `D-G` (`D-B′` is spelled `D-B-prime` in code).

**Shipped verdicts are the *effective* ones.** `D-B`'s point-ID rests on a
derivation under review, so it is declared `gated_off_by_default=True` and
`Verdict.effective_status` reports `bounds_only` unless the gate is explicitly
opened. `D-B-prime` ships `bounds_only` outright. The conservative reading is
what runs.

The declared diagram is v2's *only* assumption, so this catalogue **is** the method's assumption surface. Each entry is simultaneously: L2's input, gate V2's ground truth, L5's source of testable implications, and the identifiability axis of the taxonomy.

## Conventions

Two-slice template, unrolled over an episode of length `T`. Core MDP edges present in every entry: `S_t → A_t`, `S_t → R_t`, `A_t → R_t`, `S_t → S_{t+1}`, `A_t → S_{t+1}`.

Three structural facts verified in code, on which every derivation below depends:

1. **The behaviour policy reads only the observation** (`behavior_policy.py:24` — `act(self, obs)`). No past reward or action enters the policy.
2. **Reward is a sink**: nothing downstream reads `R_t` (`confounded.py` perturbs the reward and passes `obs` through untouched). So `R_t` has no children.
3. **`U` never enters the dynamics** — there is no `U → S_{t+1}` edge in any wired cell.

Facts 1–3 are what make lagged rewards usable as negative-control outcomes (D-B) and are themselves assumptions the catalogue owns.

Two queries per entry, because **they can differ**:
- **Q1 (per-step)**: `E[R_t | do(A_t = a), S_t = s]`.
- **Q2 (sequential)**: `V^π = E[Σ_t γ^t R_t | do(π)]` for a target policy `π`. This is what the critic actually needs.

---

## ⚠ Correction to the proposed table: the basic arm does not assert D-A

The addendum lists D-A as "asserted by both the basic and biased arms". **The basic arm asserts D-C, not D-A.** Verified at `regime_sweep.c_r_for`:

> "basic AND confounded use the cell's `confounder_c_r` … so `c_r>0` is needed even at the σ=0 basic origin. At σ=0 the U-reward-noise is ACTION-INDEPENDENT (A⊥U) … The biased arm injects no U at all (None)."

So at the basic point `U` **is sampled and does shift the reward** on `a_bad` transitions; only the `U → A` edge is absent. That is exactly D-C. Only the **biased** arm has no `U` at all (D-A).

The verdict is unchanged (both point-ID), but the *diagram* differs, and that matters twice over: the two imply **different** testable implications (§D-A vs §D-C), and the taxonomy's "clean" reference cell — v1's null-calibration anchor — turns out to carry a latent confounder that is merely action-independent. That is a sharper version of the cross-axis observation the addendum asks for, and it is worth a sentence in the paper.

---

## D-A — MDP, no latent

**Asserted by:** the **biased** arm (β>0, `confounder_c_r=None`). *Not* the basic arm.

| field | value |
|---|---|
| Nodes | `S_t` (observed, continuous, lag 0), `A_t` (observed, discrete, 0), `R_t` (observed, continuous, 0), `S_{t+1}` (observed, continuous, +1) |
| Edges | core only |
| Intervention target | `A_t` |

**Q1 — point-ID.** `A_t`'s only parent is the observed `S_t`; there is no back-door path. `E[R_t | do(a), s] = E[R_t | a, s]`.
**Q2 — point-ID.** Rewards and transitions are both identified; `V^π` follows by g-computation.

**Testable implications.** Rich, because a latent-free Markov model is a strong claim:
- `A_t ⫫ (R_{t-1}, A_{t-1}, S_{t-1}, …) | S_t` — the behaviour policy is memoryless given the state.
- `R_t ⫫ R_{t'} | (S_t, A_t, S_{t'}, A_{t'})` for `t ≠ t'` — **no cross-step reward dependence**. This is the constraint that separates D-A from D-C.

**Does not assume.** Any reward-support shape, exploration adequacy, or overlap. **Coverage is not a graphical property**: the biased arm degrades overlap without changing this diagram, so it shows up only as a wider L4 interval — this is the cleanest demonstration of why v2 needs no coverage gate.

---

## D-C — MDP, action-independent latent (reward-only)

**Asserted by:** the **basic** arm (σ=0, `c_r=1.0`). Also the historical additive cells 7/8.

| field | value |
|---|---|
| Nodes | D-A's, plus `U` (**latent**, discrete, **episode-static**) |
| Edges | core, plus `U → R_t` for all `t`. **No `U → A_t`.** |

**Q1 — point-ID.** `U` is not a parent of `A_t`, so no back-door path exists: `E[R_t | do(a), s] = E[R_t | a, s]`. `U` is a nuisance that inflates reward *variance*, not a confounder.
**Q2 — point-ID**, same argument, plus unconfounded dynamics.

**Validation value.** This entry predicts the known empirical finding that an action-independent confounder produces **no return-level gap** even for an oracle — recorded in the project's cell-7/8 history. If v2's L2 labels D-C point-ID and its estimate matches the observational floor, that reproduces the finding from the diagram alone.

**Testable implications.**
- `A_t ⫫ (R_{t-1}, …) | S_t` — still implied (no `U → A`).
- **Not** implied: cross-step reward independence. `R_t ⫫̸ R_{t'} | (S, A)` because both load on `U`. So D-A's second constraint is exactly what D-C drops, making the two **empirically distinguishable**.

**Does not assume.** `|U|`, the U-distribution, or the reward-shift magnitude — all learned.

---

## D-B — MDP, episode-static U, action-gated reward ⭐ the priority question

**Asserted by:** the **confounded** arm (σ>0, `c_r=1.0`), i.e. `r = r_clean(S,A) + c_r·U·1[A = a_bad]`.

| field | value |
|---|---|
| Edges | core, plus `U → A_t` (strength σ) and `U → R_t` (gated on `A_t = a_bad`) for all `t`; `U` episode-static (no `U_{t-1} → U_t` edge; ρ=0) |

### The derivation the addendum asked for

**Verdict: case (3) — the proximal conditions hold, but only under stated extra conditions, and only for a specific lag assignment. The natural assignment fails.**

Proximal causal inference needs a treatment-inducing proxy `Z` and an outcome-inducing proxy `W` with, given latent `U` and observed `X`:
(i) `W ⫫ (Z, A_t) | (U, X)`, (ii) `Z ⫫ R_t | (A_t, U, X)`, (iii) completeness.

**The natural assignment fails.** Take `Z = A_{t-1}`, `W = R_{t-1}`. Requirement (i) demands `W ⫫ Z | (U, X)` — but there is a **direct edge `A_{t-1} → R_{t-1}`**. Violated, badly. Any "use the previous transition as a proxy pair" scheme is invalid.

**A valid assignment exists**, separating the proxies across *different* transitions:

> `W = R_{t-1}`,  `Z = A_{t-2}`,  `X = (S_{t-2}, S_{t-1}, S_t)`

- (i) `R_{t-1} ⫫ A_t | (U, X)`: given `S_t` and `U`, `A_t` is fresh policy noise (fact 1). ✓
  `R_{t-1} ⫫ A_{t-2} | (U, X)`: the paths `A_{t-2} → S_{t-1} → R_{t-1}` and `A_{t-2} → S_{t-1} → A_{t-1} → R_{t-1}` are both blocked by conditioning on `S_{t-1}`. ✓
- (ii) `A_{t-2} ⫫ R_t | (A_t, U, X)`: given `(S_t, A_t, U)`, `R_t` is determined up to fresh noise. ✓
- `W` is a valid negative-control **outcome** precisely because reward is a sink (fact 2) — nothing downstream of `R_{t-1}` can leak back.

**The extra conditions this rests on** — these become part of the D-B entry and are assumptions the catalogue owns:

1. **`U` is truly episode-static** (ρ = 0). With persistence, `U_{t-2} ≠ U_t` and the proxies measure a different latent than the one confounding step `t`.
2. **Episode length ≥ 3**, so `t-2` exists. CartPole at random tier averages ~13 steps ✓; short episodes lose the earliest steps.
3. **Non-degenerate measurement (completeness)**, which here needs *both*: σ > 0 (else `A_{t-2}` carries no `U` information — `Z` degenerates) **and** `P(A_{t-1} = a_bad)` bounded away from 0 (else `R_{t-1}` carries no `U` information — `W` degenerates, because the reward shift is *gated* on `a_bad`). Completeness itself is **untestable** and must be declared.
4. **Overlap under the target policy** for Q2.

**Q1 — point-ID via proximal**, under 1–4, by the nonparametric outcome-bridge argument. No finite-`|U|` assumption is needed for Q1.

### Q2 (sequential) — derived separately, per R1. Answer: **point-ID, but on a strictly stronger assumption than Q1.**

Per-step identification does **not** compose by itself. The derivation:

**Step 1 — the occupancy is U-independent.** Under `do(π)` applied at every step, `U` is exogenous, the dynamics do not depend on `U` (**fact 3**), and the target policy does not read `U`. Hence the whole state–action trajectory is independent of `U`:
```
V^π = Σ_t γ^t · E_{(s,a) ~ d^π_t} [ g(s,a) ],    g(s,a) := E_{U ~ P(U)} [ E[R | s, a, U] ]
```
and `d^π_t` is identified because the transition kernel is unconfounded. **This is the step that would fail in a dynamics-confounding variant** (`U → S_{t+1}`), which would put us in the genuine longitudinal-proximal setting. The benchmark's reward-only confounder is what keeps Q2 tractable — a design property worth naming.

**Step 2 — the gap.** `g(s,a)` marginalises `U` over its **exogenous marginal** `P(U)`. Per-step proximal delivers the **X-conditional** effect `E[R^{(a)} | X = x] = E_{U ~ P(U|X=x)}[·]`. These differ, and here they genuinely do differ: `S_t` is a **descendant of past actions**, which depend on `U`, so `P(U | S_t) ≠ P(U)`. Composing per-step estimates therefore integrates the *wrong* U-distribution.

**Step 3 — closing it.** Recovering `g` from the bridge requires re-marginalising `U` from `P(U|X)` to `P(U)`, which means recovering the **latent-class structure itself** — `P(U)` and `P(R | s, a, U)` — not merely a bridge function. Under the **finite `|U| = K` mixture with completeness** that v2 assumes anyway, that model is identifiable (this is exactly what the rank-≤K constraints encode), so `g` is recoverable and **Q2 is point-ID**.

**Verdict, stated precisely:**

| query | verdict | rests on |
|---|---|---|
| Q1 | point-ID | nonparametric proximal bridge; **no** finite-K needed |
| Q2 | point-ID | proximal **plus finite-`|U|` latent-class identifiability**; **bounds-only** if one declines to assume finite K |

So D-B's sequential answer is *not* "Q1 point-ID, Q2 bounds-only" — but it is also not a free composition. It is point-ID **on a stronger assumption**, and that asymmetry must be visible in L2's output (two verdicts, two assumption sets) rather than collapsed into one label.

**Written proof required in Phase 2** (per R5), covering both the completeness step and this sequential argument. **Literature to check rather than reconstruct** — all `TODO-verify` (venue/year unverified from this repo): Ying, Miao, Shi & Tchetgen Tchetgen on proximal causal inference for complex longitudinal studies; and the confounded-POMDP OPE line (Bennett & Kallus; Shi, Uehara, Huang & Jiang; Uehara et al.), whose premise — that the sequential case needs bridge machinery rather than per-step adjustment — is precisely what Step 2 above reproduces from this diagram.

**What opening the gate would mean.** Not "point-ID under proxy
informativeness" alone. The lagged views are covariate-conditional, so
Kruskal applies per `(s, a)` and identifies the latent only up to a
relabelling at each configuration; the labels are linked by the **shared
mechanism family**, a model-class assumption declared as
`cross_stratum_label_linking` (untestable). D-D needs no such assumption — its
proxies are covariate-free. So the honest reading is: **point-ID under proxy
informativeness *plus* a model-class linking assumption.** Still defensible;
just not free. See `docs/grace_v2_conditions.md`.

**Consequence if this survives review:** the existing confounded cells were **identifiable all along**, and the taxonomy's identifiability axis must relabel them from "non-ID" to "point-ID (proximal, conditional on completeness)". As the addendum says, this strengthens the paper. It also means D-B and D-D differ not in *whether* point-ID is available but in whether the proxies are **explicit and clean** (D-D) or **implicit and conditional** (D-B) — which is itself the more interesting scientific statement.

**Fallback if review rejects the derivation:** D-B is bounds-only and D-D is the sole point-ID cell. **I recommend implementing L2 so that D-B's point-ID verdict is gated on a `proximal_lagged` declaration that is off by default**, so the conservative reading ships unless the derivation is explicitly accepted.

**Testable implications.** Few independences (latents remove constraints). The usable one is a **rank constraint**: an episode-static `|U| = K` latent implies rank ≤ K on cross-time moment matrices of the observables. This simultaneously (a) tests the diagram and (b) informs `u_card` selection — the two are the same statistic.

---

## D-B′ — MDP, **persistent** U (ρ > 0) — added per R2

Identical to D-B plus the persistence edge `U_{t-1} → U_t`. `U` is no longer episode-static.

**Derivation — the surprise is *which* condition breaks.** I expected the exclusion restrictions to fail; they do not.

- `W = R_{t-1} ⫫ A_t | (U_t, X)`: the path `R_{t-1} ← U_{t-1} → U_t → A_t` is **blocked by conditioning on `U_t`**. ✓ survives.
- `Z = A_{t-2} ⫫ R_t | (A_t, U_t, X)`: the path `A_{t-2} ← U_{t-2} → U_{t-1} → U_t → R_t` is likewise **blocked by `U_t`**. ✓ survives.

So the *structural* proximal conditions are robust to drift. What degrades is **completeness**: `W` now measures `U_{t-1}`, which is only *correlated* with the `U_t` that confounds step `t`. For binary `U` with per-step flip probability `p`, the latent transition matrix is invertible iff `p ≠ 0.5`, so:

| drift | latent transition | verdict |
|---|---|---|
| `p = 0` (D-B, static) | identity | point-ID, maximal proxy signal |
| `0 < p < 0.5` | invertible, increasingly ill-conditioned | **point-ID in principle**, variance and interval width growing as `p → 0.5` |
| `p = 0.5` (full refresh) | singular — `U_t ⊥ U_{t-1}` | **non-ID**: the proxy carries zero information |

**Verdict: point-ID degrading continuously to non-ID**, with the failure appearing as an ill-conditioned completeness condition rather than a broken exclusion. That is a *graceful* degradation, which makes it an unusually good empirical probe.

**Experiment (R2), and it is the sharpest test of the catalogue's own conditions:** sweep `ρ` from static to full refresh; plot the point estimate's error against the L4 interval. The prediction is specific — the point estimate should drift away from truth while the interval widens to cover it, and the two should cross near `p = 0.5`. If the interval *fails* to widen as the point estimate degrades, L4 is broken, and this sweep detects that in a way no fixed-cell experiment does.

`ρ` is currently a config switch that raises `NotImplementedError` in v1. Per the v2 constraint ("`rho` is an *edge* in the diagram, not a config switch"), D-B′ is a **separate catalogue entry**, and selecting it is selecting a diagram — not tuning a knob.

---

## D-A-null — clean D-A instantiation — added per R3

**Problem this solves.** With `basic` reassigned to D-C, D-A is asserted only by the `biased` arm — which carries a coverage defect by construction. L5 would then be measuring its false-positive rate on a statistically awkward cell.

**Arm:** `β = 0`, `σ = 0`, **`confounder_c_r = 0`**. `U` is still drawn but influences nothing, so the diagram is exactly D-A: no latent edges, no coverage defect, no confounding.

**Cost:** one config point (`confounder_c_r: 0.0` at the basic point in a scoped cell). **Role:** the reference null for *every* falsification test — where V3's false-positive rate is measured, and the only cell where a `refuted` verdict is unambiguously a Type-I error.

One Phase-2 check: `enforce_confounding_gate` may object to a σ=0 arm whose signature reports no confounding — the exemption path exists for the additive σ=0 anchor and may need extending. Flagged, not assumed.

---

## D-D — MDP + explicit negative-control proxies (new proximal cell)

| field | value |
|---|---|
| Nodes | D-B's, plus `Z` (observed, continuous, episode-level), `W` (observed, continuous, episode-level) |
| Edges | D-B's, plus `U → Z`, `U → W`. **Exclusions:** no `Z → R`, no `A → W`, no `Z ↔ W` beyond `U` |

Construction: two conditionally-independent noisy measurements of `U` emitted into `infos` (`W = U + ε_W`, `Z = U + ε_Z`, independent noise). Because neither *causes* anything, requirements (i) and (ii) hold **by construction** rather than by the delicate lag argument of D-B.

**Q1, Q2 — point-ID via proximal**, under completeness (non-degenerate noise scale). The one cell where point-ID is clean and uncontested; keep it regardless of how D-B resolves.

**Testable implications.** `P(Z, W | A, S)` has **rank ≤ |U|** — a genuine observable shadow of the latent-class structure, and the constraint that misspecification arm M3 violates.

**Does not assume.** The noise scale, `|U|`, or the U-distribution — learned/selected. **Assumes** completeness (irreducible).

---

## D-E — MDP + instrument (new IV cell)

| field | value |
|---|---|
| Nodes | D-B's, plus `I` (observed, episode-level) |
| Edges | D-B's, plus `I → A_t`. **Exclusions:** `I ⫫ U`, no `I → R` except through `A` |

**Q1, Q2 — bounds-only (Balke–Pearl).** Confirming F4: a valid instrument does **not** point-identify `E[R | do(a)]` without extra assumptions (monotonicity/homogeneity). Its role is to exercise L4's bound engine and L2's bounds verdict with a genuinely informative, sub-trivial interval.

**Testable implications.** The **instrumental inequalities** — a real, checkable constraint that an invalid instrument violates. Note these are *refutation-only*: satisfying them does not validate the instrument.

**Value for L4 validation.** Binary action + bounded reward gives closed-form Balke–Pearl bounds, so this cell is the **correctness anchor** for the approximate min/max-over-compatible-models optimizer — I would not trust that optimizer without it.

---

## D-F — POMDP, latent state, no U

**Asserted by:** the POMDP basic/biased arms (velocity dims masked).

| field | value |
|---|---|
| Nodes | `S_t` (**latent**), `O_t` (observed, `O = mask(S)`), `A_t`, `R_t` |
| Edges | `S_t → O_t`, `O_t → A_t` (the policy sees `O`), `S_t → R_t`, `A_t → R_t`, `S_t, A_t → S_{t+1}` |

**The addendum's warning is correct — the two queries diverge here.**

**Q1 — point-ID.** The back-door path `A_t ← O_t ← S_t → R_t` is a chain through the observed `O_t`; conditioning on `O_t` blocks it. Adjustment on `O_t` suffices for the per-step estimand.

**Q2 — NOT identified in general.** A history-dependent target policy's value requires the joint law of latent-state trajectories under `do(π)`, and `S` is latent with partially-observed dynamics. Per-step adjustment does not compose into `V^π`: this is the known hardness of OPE in POMDPs. Identification requires additional proxy structure (past/future observations as proxies for the latent state).

This entry alone justifies the addendum's insistence on stating both queries — a catalogue that reported only Q1 would license a point estimate the critic is not entitled to.

**Testable implications.** The observable process is HMM-like: rank constraints on cross-time observation matrices bound the latent-state cardinality.

---

## D-G — POMDP + U

**Asserted by:** the POMDP confounded arm. Nodes/edges = D-F ∪ D-B (latent state **and** latent confounder).

**Q1 — bounds-only** (adjusting `O_t` blocks the state path but not `A_t ← U → R_t`); point-ID only under a proximal argument analogous to D-B, now complicated by `O` being a noisy view of `S`.
**Q2 — not identified** without proxy structure; the hardest entry.

**Recommendation:** label D-G **bounds-only for Q1, non-ID for Q2** in the shipped catalogue, and treat any stronger claim as a research extension. This is the entry where over-claiming would be easiest and least defensible.

---

## Misspecified diagrams (drive V3 / V6)

| Id | Declaration | Truth | Detectable? |
|---|---|---|---|
| **M1** | D-A (or D-C) | D-B | **Yes.** D-A/D-C imply `A_t ⫫ R_{t-1} | S_t`; under D-B both load on `U`, so the path `A_t ← U → R_{t-1}` is open and the constraint is violated. **This is the principled replacement for v1's `δ_a`** — same signal, but derived from the diagram and tested against a within-dataset bootstrap null rather than an external threshold. |
| **M2** | D-B + spurious edge | D-B | **No — undetectable in principle.** Adding an edge *removes* implications; the data satisfies everything the enlarged model requires. Report as the demonstration of L5's stated limitation, measured through V6, not scored as a V3 miss. |
| **M3** | D-D with an invalid proxy (`W` actually affected by `A`) | D-D-with-`A→W` | **Yes, partially.** The declared exclusion `A ⫫ W | U` has an observable shadow: the **rank ≤ |U| constraint** on `P(Z, W | A, S)` fails when `A → W` exists. Detection power depends on the strength of the spurious edge, so V3 must report a **curve**, not a single rate. Where the edge is weak the entry degrades to undetectable — state that honestly. |

---

## Proxy informativeness degrades as the logged policy improves — elevated per R4

Condition 3 of D-B is not merely a side condition; it has a consequence that deserves to be a **reported result**.

The reward shift is **action-gated**: `r += c_r·U·1[A = a_bad]`. So `W = R_{t-1}` carries information about `U` **only on transitions where `a_bad` was actually taken**. Informativeness therefore scales with `P(A_{t-1} = a_bad)` under the behaviour policy. Symmetrically, `Z = A_{t-2}` carries information about `U` only in proportion to σ, the strength of the `U → A` edge.

Both terms shrink as the logged policy improves: a good policy avoids `a_bad`, and a policy that is *less* driven by `U` has a weaker `U → A` edge. Hence:

> **Lagged-proxy identification in RL is strongest exactly when the logged policy is worst, and degrades toward the regime one actually wants to deploy in.**

This is a structural tension, not an artefact of this benchmark: any action-gated confounder makes the proxy signal proportional to how often the bad action is tried.

**Report as a measured curve, not a stated condition** (R4): sweep behaviour-policy quality (the generator's tiers and `pi_basic_epsilon`), and plot against it (a) proxy informativeness — the mutual information between `W` and the logged `U`, evaluation-side only, plus the estimated rank gap — and (b) the L4 interval width and point-estimate error. The predicted shape is monotone decay of (a) with a matching widening of (b).

If it holds, it is a genuine finding about lagged-proxy identification in RL, and it bears directly on the practical question of when proximal methods are worth reaching for offline. It also implies a caution for D-B's use in practice: the cells where proximal identification is *easiest* to demonstrate are the ones whose data is least like deployment data.

---

## Cross-axis observations for the taxonomy paper

The identifiability axis (what the declared diagram licenses) is **not** aligned with the defect axis. The sharpest framing (per D2, replacing the original "basic and biased share D-A"):

> **Three distinct points on the identifiability axis across only two defect labels.**
> **biased → D-A** (no latent at all; coverage defect only) · **basic → D-C** (latent present, still identified) · **confounded → D-B** (latent *plus* `U → A`).

That is the figure: the defect axis has two non-trivial labels, the identifiability axis has three distinct structures underneath them, and they do not line up.

Supporting observations:

1. **Coverage is not a graphical property.** The biased arm degrades overlap without changing D-A — it surfaces as interval width, never as an identification verdict. This is the argument for deleting the coverage gate.
2. **D-C is confounded-but-identified**, the sharpest single counterexample to equating "has a latent confounder" with "not identified" — and the entry that predicts the project's known no-return-gap finding from the diagram alone.
3. **D-F is unconfounded-but-unidentified for `V^π`** — the mirror image of D-C, and the reason the sequential query must be stated separately.
4. **D-B′ shows identification is not binary**: it degrades continuously with latent drift, failing through an ill-conditioned completeness condition rather than a broken exclusion.

Points 3 and 4 together are, in my view, the catalogue's main contribution to the paper: identifiability and confounding are orthogonal, and the taxonomy can now show both off-diagonal cells.

**Paper cross-references:** figure numbers to be filled from the taxonomy/compass manuscript, which is not in this repository — flagged `TODO-verify` per entry rather than guessed.

---

## Summary

| Id | Q1 (per-step) | Q2 (sequential) | Testable implications | Asserted by |
|---|---|---|---|---|
| D-A | point-ID | point-ID | rich (incl. no cross-step reward dependence) | biased arm |
| **D-A-null** | point-ID | point-ID | same as D-A | **new clean-null arm** (`c_r=0`) |
| D-C | point-ID | point-ID | `A_t ⫫ past \| S_t`; **not** cross-step reward indep. | **basic arm** (corrected) |
| D-B | point-ID* | point-ID*, **+finite-K** | rank ≤ K | confounded arm |
| **D-B′** | point-ID → non-ID as drift → full refresh | same | rank ≤ K, ill-conditioned with drift | **new ρ-sweep arm** |
| D-D | point-ID | point-ID | rank ≤ \|U\| on `P(Z,W\|A,S)` | new proximal cell |
| D-E | bounds-only | bounds-only | instrumental inequalities | new IV cell |
| D-F | point-ID | **not identified** | HMM rank constraints | POMDP basic/biased |
| D-G | bounds-only | not identified | HMM rank constraints | POMDP confounded |

\* conditional on the four stated conditions and on review accepting the lagged-proxy derivation; **recommended to ship gated off by default**. Q2 additionally requires finite-`|U|` latent-class identifiability — a strictly stronger assumption than Q1's, and L2 must surface the two verdicts with their two assumption sets rather than collapsing them.

**Design consequence for L2:** every entry carries **two verdicts** (Q1, Q2) and, per verdict, the assumption set it rests on. A single-verdict L2 would silently license a sequential point estimate from a per-step argument — exactly the error D-F and D-B Step 2 expose.

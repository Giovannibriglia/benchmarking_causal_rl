# GRACE v2 — derivation notes: identification conditions (C1, C2, C5)

Working notes backing the catalogue's D-B entry and the review conditions.
These are derivations, not implementation docs; `docs/grace_v2.md` will document
the as-built system.

---

## C2 — the finite-mixture identifiability step, and what it actually requires

D-B's **Q2** verdict rests on recovering the latent-class model (`P(U)` and
`P(R | s, a, U)`), not merely a bridge function — see the catalogue's Step 3.
That recovery is the classical finite-mixture identifiability result: a mixture
with `K` components is identifiable up to label swapping from **three or more
conditionally independent measurements**, under a Kruskal-rank condition on the
three component-conditional matrices.

**References (verified against primary sources; do not re-derive):**

- Kruskal, J. B. (1977). "Three-way arrays: rank and uniqueness of trilinear
  decompositions, with application to arithmetic complexity and statistics."
  *Linear Algebra and its Applications* **18**(2): 95–138.
- Allman, E. S., Matias, C., & Rhodes, J. A. (2009). "Identifiability of
  parameters in latent structure models with many observed variables." *The
  Annals of Statistics* **37**(6A): 3099–3132. doi:10.1214/09-AOS689
  (arXiv:0809.5032).

AMR's framing matches Step 3 directly: models in which some observed variables
are conditionally independent given the hidden ones, with Kruskal's theorem for
a finite-state latent-class model at the core.

### The measurement triple for D-B — with a correction

Three rewards at distinct times inside one episode:

> `M₁ = R_{t₁}`, `M₂ = R_{t₂}`, `M₃ = R_{t₃}`

**Correction to the earlier statement.** These are *not* conditionally
independent given `U` alone: `R_{t₁}` and `R_{t₂}` remain coupled through the
observed state–action chain (`S_{t₁} → … → S_{t₂}`, and `A_t` depends on `S_t`).
The correct statement is that they are conditionally independent given
**`U` together with the `(S_t, A_t)` at the measurement times**, because each
reward's noise is fresh and reward is a sink. The view matrices are therefore
`Mᵢ[u, ·] = P(R_{tᵢ} | U = u, S = s, A = a)` — the standard conditional-on-
covariates reading of AMR, not the bare marginal one.

### The k-rank condition, evaluated

Kruskal's sufficient condition for essential uniqueness is that the **sum of the
three views' k-ranks is at least `2R + 2`**, for `R` latent classes. A view's
k-rank is capped at `R`, and reaches `R` only if the reward law differs across
*all* `R` strata at that `(s, a)` — which, under the action-gated confounder
`r += c_r·U·1[A = a_bad]`, requires `a = a_bad`. A non-`a_bad` view has
identical laws across strata and k-rank 1.

| `R` | required sum | 3 informative | 2 informative | 1 informative |
|---|---|---|---|---|
| 2 | 6 | **6 — OK (exactly tight)** | 5 — fails | 4 — fails |
| 4 | 10 | **12 — OK** | 9 — fails | 6 — fails |

**Consequence 1 — at `|U| = 2` the condition is exactly tight.** All three views
must have full k-rank 2; there is no slack. This is `proxy_informativeness`
arriving from the theorem rather than from intuition, and it is why
`P(A = a_bad)` bounded away from zero is load-bearing: one view in which the
gate never fires drops the sum to 5 and the condition fails.

**Consequence 2 — raising `u_card` makes identification harder, not merely more
expensive.** At `|U| = 4` the requirement rises to 10 against views capped at 4,
so the views must be strictly richer. A `K` chosen purely by held-out likelihood
may therefore sit outside the range Kruskal's condition licenses. **`u_card`
selection must report the estimated view k-ranks alongside the likelihood
curve**, so the two criteria are visible together and a likelihood-preferred `K`
that violates the rank condition is caught rather than silently adopted.

**Verdict for the D-B entry: the condition HOLDS ONLY UNDER THE STATED
PROXY-INFORMATIVENESS** — concretely, at least three `a_bad` transitions per
episode. That is checkable per episode, so D-B's q2 gate stays **conditional and
shut by default** rather than permanently shut: GRACE counts informative
transitions and degrades the estimate to bounds when the count is short.

### Consequence for the implementation

The per-episode informative-transition count is cheap to compute and should be
reported alongside every D-B estimate, with the estimate degraded to bounds
when the count is short. This is a condition v2 can *check*, not merely assume —
which is the whole point of the design.

---

## C1 — sample splitting for the dual-use rank statistic

The rank statistic is used twice: to **select** `u_card = K`, and to **test**
the diagram in L5. Using one split for both invalidates the test, because the
bootstrap null would be computed under a model chosen to maximise the very
statistic being tested; the reported false-positive rate would be optimistic by
an unquantified amount and V3's detection curve would inherit the bias.

**Adopted fix: episode-level sample splitting.**

| split | size | used for |
|---|---|---|
| `select` | 50 % of episodes | choose `K` by held-out predictive likelihood; the rank/selection curve is reported from here |
| `test` | 50 % of episodes | compute the L5 statistic and its parametric-bootstrap null, with `K` held **fixed** at the selected value |

Splitting is at **episode** granularity, never transition, for the same reason
the bootstrap resamples episodes: a persistent `U` induces exactly the
cross-step dependence under test, and a transition-level split would leak the
same episode's `U` across both halves.

Both split sizes are reported with every falsification verdict, and the test
suite asserts the two consumers never see the same episode ids — the assertion
the review requires, implemented as a property of the splitter rather than a
convention.

Rejected alternatives, for the record: a **nested bootstrap** (re-selecting `K`
inside every bootstrap replicate) is the more principled correction but
multiplies the measured cost of L5 by the number of candidate `K` values,
turning a ~7.7 min per-constraint budget into ~30 min; a **selection-aware
null** would require characterising the selection event's effect on the null
distribution, which is not tractable here. Sample splitting costs statistical
power — the honest trade — and that cost is visible, not hidden.

---

## C5 — the `c_r = 0` null arm is rejected by the generation gate (verified)

**Finding: the reference null arm would fail preflight.** Confirmed by reading
`enforce_confounding_gate` and `_action_dependent_signature`:

- Check **A4** requires `corr_r_u_gated > 0.0` **strictly**. With `c_r = 0` the
  reward carries no `U`-dependence whatsoever, so this correlation is noise
  around zero and is negative roughly half the time.
- `gate_test_passed = a2 and a3 and a4 and a5`, so A4's failure fails the gate.
- The σ=0 escape hatch applies **only** to the additive gate (`gate_type is
  None`); the action-dependent gate is documented as authoritative "with NO
  exemption", precisely because it *can* validate its own σ=0 baseline.

The other checks pass: A2's target is `sigma × entropy = 0` and the observed
statistic is ≈ 0; A3 holds (`pi_basic` has real entropy); A5 holds offline.

**Fix (its own commit, per C5):** the gate is already declarative, so extend the
gate config rather than special-casing a policy name. Add
`expect_gated_reward: bool = True`; the generator sets it `False` when
`confounder_c_r` is 0 or `None` for an action-gated arm. When it is `False`, A4
is **not evaluated** — because there is no `U → R` edge to detect — and the
metadata stamps `gated_reward_expected: False` so a later reader can see the
check was skipped by declaration rather than passed by luck. `corr_r_u_gated`
is still recorded as a diagnostic.

This keeps the invariant that matters: a dataset that *claims* a confounding
signature must exhibit one, while a dataset that declares itself signature-free
is not asked to prove a signature it does not have.


---

## Addendum (2026-08-15) — NBN v0.14.0 supersedes the M-step workaround

The C1 splitting design above is unchanged. One implementation note attached to
it is now obsolete: the plan assumed GRACE would own a hand-written weighted
M-step loop because NBN had no sample weights. **v0.14.0 delivers
`fit(..., weights=)`** with a `supports_weights` capability flag, verified exact
against replication (weighted == replicated to 4.8e-07 for LinearGaussian;
zero-weighted rows fully excluded for MDN). The EM M-step should therefore call
the library, keeping a GRACE-side loop only for a mechanism that refuses.

Two constraints this introduces are recorded in `docs/grace_v2.md`: weighted
*incremental* updates are unsupported (so the online refresh refits), and KDE's
bandwidth rule stays unweighted (so the weighted EM path should use MDN, flow,
or LinearGaussian).

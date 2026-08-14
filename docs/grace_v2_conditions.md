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

> **Citation `TODO-verify`.** The result is Kruskal's trilinear-decomposition
> theorem as applied to latent structure models by Allman, Matias & Rhodes.
> Exact venue, year, and theorem number must be checked against the source
> before this appears in the paper — deliberately not reconstructed from
> memory.

### The measurement triple for D-B

Three measurements conditionally independent given `U`, all available inside one
episode with `U` static:

> `M₁ = R_{t₁}`, `M₂ = R_{t₂}`, `M₃ = R_{t₃}` at three distinct times,
> conditionally independent given `(U, S_{tᵢ}, A_{tᵢ})` because each reward's
> noise is fresh and reward is a sink (nothing downstream reads it).

### The condition, sharpened — and it is *not* the one first stated

Kruskal's condition needs the three measurement matrices' Kruskal ranks to sum
to at least `2K + 2`. For binary `U` (`K = 2`), each measurement contributes
Kruskal rank 2 **iff that measurement's distribution actually differs between
`U = 0` and `U = 1`**, i.e. `P(R_{tᵢ} | U=0) ≠ P(R_{tᵢ} | U=1)`. Three
informative measurements give `2 + 2 + 2 = 6 ≥ 2(2) + 2 = 6` — satisfied, but
**exactly at the boundary**, so no measurement may be uninformative.

Now the specific structure bites. The reward shift is **action-gated**:
`r += c_r·U·1[A = a_bad]`. A transition where `A ≠ a_bad` carries **no**
`U`-signal at all, so its measurement matrix has Kruskal rank 1, and the sum
drops below the threshold.

**Therefore the real condition is not "episode length ≥ 3" but:**

> **at least three transitions per episode on which `a_bad` was actually taken.**

The catalogue's `episode_length_ge_3` assumption is a necessary consequence of
this, not the condition itself, and the entry should be read accordingly. Three
consequences:

1. **It is directly measurable** — count `a_bad` transitions per episode. That
   makes it an assumption with an *observable shadow*, unlike completeness.
2. **It ties to the R4 finding.** With `P(A = a_bad) = p` and episode length
   `T`, the expected count is `pT`. At the wired operating point (CartPole,
   `T ≈ 13` at random tier, `pi_basic_epsilon = 0.5`, so `p ≈ 0.5`) the
   expectation is ≈ 6.5 — comfortable. As the logged policy improves and `p`
   falls, episodes stop supplying three informative measurements and Q2's
   identification fails *before* Q1's degrades, because Q1 needs two
   informative proxies and Q2 needs three.
3. **Q2 fails earlier than Q1 along the policy-quality axis.** That is a
   prediction the R4 sweep can check directly, and a sharper statement than
   "identification degrades".

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

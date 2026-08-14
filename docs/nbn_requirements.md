# NBN requirements from GRACE v2

**Audience:** the NBN library author. **Scope:** what GRACE v2 needs from NBN
that it cannot get GRACE-side without compromising correctness.

**Headline: nothing here is blocking.** After upstream `a91d8f9`, every
GRACE-side need is either satisfied or has a measured workaround. The list
below is two requirements, one of which is a *guarantee about existing
behaviour* rather than new code.

This is deliberately shorter than anticipated. Requirements that were expected
and are **not** being asked for — a recurrent mechanism class, sequence-shaped
fitting, do-semantics under recurrence, batched interventions, serialization —
are listed in [§ Not requested](#not-requested) with the evidence that retired
each one.

---

## R1 — Pin parent-tensor gradient transparency as a tested contract

**Statement.** Guarantee, and test, that mechanisms do not detach their
`parents` argument: gradients must flow from `log_prob(x, parents)` and from
`sample(n, do={node: value})` back into whatever produced those tensors.

**Motivation.** This single fact is what lets GRACE v2 keep *all* recurrence on
its own side (Hypothesis B, below), which in turn is why no recurrent mechanism
class is requested. It is load-bearing for: L3's EM M-step (a
responsibility-weighted NLL optimised through `log_prob`), the POMDP behaviour
channel `P_b(A_t | o_{1:t})` in catalogue entries **D-F/D-G**, and L4's bound
optimiser (which differentiates the interventional value through
`sample(do=)`). It currently **works** — measured below — but nothing upstream
tests it, so it could regress silently, and the failure mode is a silently
untrained encoder rather than an exception.

**Why not GRACE-side.** GRACE cannot enforce a property of NBN's internals.

**Proposed API.** No API change. Add tests; state the contract in the
`Mechanism` docstring: *"`parents` is used as supplied and never detached;
callers may pass tensors requiring grad and expect gradients to propagate."*

**Acceptance test.**

```python
import torch, torch.nn as nn
from nbn.mechanisms.parametric.mdn import MDNMechanism

enc = nn.Linear(3, 1)                       # stand-in for a caller-side encoder
raw = torch.randn(512, 3)
y   = torch.randn(512, 1)
mech = MDNMechanism(num_components=3)
mech.fit_local(y, enc(raw).detach(), epochs=3, batch_size=256)

loss = -mech.log_prob(y, enc(raw)).mean()   # parents carry grad
loss.backward()
assert enc.weight.grad is not None and enc.weight.grad.abs().sum() > 0
```

and the interventional counterpart, which must also hold:

```python
v = torch.tensor([2.0], requires_grad=True)
model.sample(n=512, do={"B": v})["R"].mean().backward()
assert abs(float(v.grad) - 1.0) < 0.1      # analytic dR/dB = 1 for R = B + A
```

**Measured today:** `log_prob` → encoder grad-norm `0.0103` (flows);
`sample(do=)` → grad `0.971` vs analytic `1.0`. Both pass.

**Priority.** Not blocking (it already works) — but the **highest-value** item
here, because it is the assumption the rest of the design rests on.

**Fallback if dropped.** None needed while the behaviour holds. If it ever
regressed, GRACE would have to reimplement conditional densities outside NBN,
which would eliminate most of the reason to use the library.

---

## R2 — Per-sample weights in mechanism fitting

**Statement.** Accept an optional `weights: Tensor | None` of shape `(N,)` in
`Mechanism.fit_local` and in `fit(...)`, contributing a weighted objective
`-(w * log_prob).sum() / w.sum()`.

**Motivation.** L3 fits a finite-mixture latent by EM. The M-step needs each
row weighted by its **per-episode responsibility** `q(U = k | episode)`. Note
the shape subtlety: the latent is *episode-static*, so a single responsibility
is broadcast across every transition of that episode — the weights vector is
per-row, but its values are constant within an episode. GRACE builds it.

**Why not GRACE-side.** It *is* doable GRACE-side and GRACE will do it: bypass
`fit_local` and run an Adam loop over `mech.log_prob`. Measured cost: **1.08 s
per M-step** (30 epochs, N = 45 000, K = 2, CUDA), against **5.6 s** for
`fit_local`'s own unweighted loop. So this is a convenience and a
correctness-by-default win, not a capability gap.

**Proposed API.**

```python
def fit_local(self, x, parents, *, weights=None, epochs=..., lr=..., batch_size=...) -> dict
def fit(model, data, *, weights=None, ...) -> TrainHistory   # dict[node] -> (N,) or a shared (N,)
```

Semantics: `weights=None` is exactly today's behaviour (bitwise). Negative
weights are an error; all-zero is an error. Weights participate in minibatching
by being indexed alongside `x`/`parents`.

**Acceptance test** — weighting must reproduce a known conditional mean:

```python
import torch
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism

x = torch.cat([torch.full((500, 1), 0.0), torch.full((500, 1), 10.0)])
pa = torch.zeros(1000, 1)
w  = torch.cat([torch.ones(500), torch.zeros(500)])   # keep only the first group
m = LinearGaussianMechanism(); m.fit_local(x, pa, weights=w)
mean = m(pa[:1]).mean.item()
assert abs(mean - 0.0) < 0.1        # 0.0, NOT the unweighted 5.0
```

**Priority.** Nice-to-have.

**Fallback.** GRACE owns the M-step loop (already measured and planned). Cost:
GRACE duplicates minibatching and optimiser setup, and mechanisms whose
`fit_local` does non-gradient work — `LinearGaussianMechanism`'s closed-form
ridge solve, the KDE/kNN memorisation — cannot be weighted at all by that
route. **That is the real limitation:** without R2, the EM M-step is restricted
to gradient-trained mechanisms (MDN, flow, neural-categorical). GRACE's slice
uses exactly those, so nothing is lost today.

---

## Not requested

Each of these was anticipated; the evidence retired it.

| candidate | why not |
|---|---|
| **Recurrent mechanism class** | Hypothesis B holds (below). A GRACE-side GRU passed as an ordinary parent recovers history dependence: NLL **0.4197** vs **0.6385** for a memoryless control on a policy that depends on a running sum of observations, with gradient reaching the GRU. Adding recurrence inside NBN would duplicate this and introduce the do-semantics trap below. |
| **Sequence-shaped fitting with episode masks** | Unnecessary once recurrence is GRACE-side. GRACE computes the per-step agent state itself, respecting episode boundaries, and hands NBN a flat `(N, d)` parent matrix. Episode structure never crosses the API. |
| **Do-semantics under recurrence** | Moot for NBN, but **still a live correctness trap for GRACE**, recorded here so it is not lost: under `do(A_t = a)`, a recurrent hidden state must be advanced using the **realized** trajectory including the intervened action — not what the un-mutilated mechanism would have emitted. Keeping recurrence GRACE-side makes this GRACE's rollout loop to get right, which is the safer place for it: the component that performs the intervention is the one that owns the state update. |
| **Batched interventions** | **Fixed** in `a91d8f9`. Verified: `query_batch(do=…)` gives `E[R|do(B)] = [0.00, 0.97, 1.94, 2.91]` against truth `[0, 1, 2, 3]`. Ancestral sampling still has no batch axis, but now says so explicitly, and GRACE's action sweep costs **1.5 ms/action** by looping. |
| **Serialization round-trip** | **Fixed** (format 2). Verified: reloaded model reproduces `E[R|do(H=1)] = 1.994` vs `1.997`. |
| **Sequential-IS diagnostics / stability at horizon 1024** | GRACE does **exact** filtering over its discrete latent block, so it never runs NBN's importance sampler over a 1024-step sequence. ESS/PSIS are confirmed available on continuous nets (`k̂` returns `nan` on a near-deterministic net, which is the documented degenerate case). Revisit only if the continuous-latent POMDP slice (Phase 3) adopts particle methods. |
| **Latent / EM / mixture support** | Still absent upstream, and **not requested**. GRACE implements EM over its own discrete latent block; a general latent-variable framework in NBN would be a much larger change than v2 needs, and v2's exact enumeration is cheap (full EM fit measured at 22 s). |

---

## Evidence: which hypothesis holds

Tested in the order the brief specified, cheapest first.

**Hypothesis B — recurrence lives entirely in GRACE. ✅ HOLDS.** Both
preconditions verified against the new drop:

1. *No parent detachment.* `log_prob` → encoder gradient flows (norm `0.0103`);
   `sample(do=…)` → encoder gradient flows.
2. *Sequence handling.* Not needed from NBN: GRACE computes per-step states
   and passes flat rows, so episode boundaries are respected on GRACE's side
   by construction.

The decisive test was the strongest case for a recurrent mechanism — the
**POMDP behaviour policy** `P_b(A_t | o_{1:t})`, where the generator is
`offline_dqn_recurrent` and the policy is genuinely history-dependent. Against
a ground-truth policy driven by a running sum (so a memoryless model provably
cannot fit it), a GRACE-side GRU whose output is an ordinary parent reached NLL
**0.4197** versus **0.6385** memoryless, with gradient reaching the GRU. B
covers the case that would have forced A.

**Hypothesis C — recurrence as an inference concern.** Not applicable. The
generative POMDP model is Markov in the latent, and GRACE filters exactly over
its discrete latent block rather than using an amortized proposal.

**Hypothesis A — a genuine recurrent mechanism class.** **Not needed**, on the
strength of the B result above.

---

## Status of the previously reported edges

All re-verified empirically against the new drop; the table lives in
`nbn/NOTICE.md`. Summary: eight of nine **fixed**; the ninth —
`intervene()` severing the caller's gradient — is inherent to returning a
deep-copied model and is now documented, with `sample(n, do=…)` as the
differentiable path. One of our own claims was corrected upstream:
`is_fitted` was already implemented by kde/knn/flexcode; only LG/MDN/flow were
affected.

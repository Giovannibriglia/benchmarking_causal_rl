# Diagram arms — one cell per declared diagram

These cells are **not** points on the (beta, sigma) L. The L varies the
*severity* of one fixed scenario (how biased, how confounded); a diagram arm
varies the *scenario itself*, and each one exists to exercise a different
branch of L2's decision procedure end to end on real data.

| cell | diagram | L2 q1 verdict | what it is for |
|---|---|---|---|
| `d_a_null.yaml` | D-A-null | point-ID, back-door | the reference null: **no latent at all**. L5's false-positive rate is measured here, where a refutation is a false alarm by construction. |
| `d_d.yaml` | D-D | point-ID, proximal | the clean proximal cell: two **covariate-free** negative-control proxies. The one arm where point identification is uncontested. |
| `d_e.yaml` | D-E | **bounds-only**, IV | a valid instrument bounds the interventional value and does not point-identify it. The arm that proves L2 does not launder bounds into points. |
| `d_b_prime.yaml` | D-B' | bounds-only | a **persistent** latent (rho > 0). The lagged-proxy argument needs a static U; this is the arm where it is supposed to fail. |

Which channels each generator emits is derived from the diagram itself
(`src/envs/offline/diagram_arms.py`), never from the YAML — the YAML supplies
only the strengths. Declaring a strength for a channel the diagram does not
have, or omitting one it does, is an error at resolve time.

Every arm's generated data is checked against **ground truth** before use:
`src/envs/offline/arm_preflight.py` measures the proxy, instrument and drift
properties against the logged `U` and the declared parameters. It never
consults GRACE's estimator or L5 — L5 is validated against the generator
afterwards, and never the reverse, so a misconception shared by both cannot
pass in silence.

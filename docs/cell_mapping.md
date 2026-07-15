# Cell mapping: flat `cell_N` → `(regime, β, σ)` layout

The legacy flat `cell_1 … cell_9` tree is **frozen** under
`reproducibility/rl_regimes/_legacy/` (git-moved verbatim, read-only). New work uses
the four `{offline,online}×{mdp,pomdp}` cells, each an **L-shaped sweep** of two 1-D
arms sharing an origin:

```
basic      = (β=0,   σ=0)                    # ONE run, the shared reference for both arms
biased     = (β∈{0.25,0.50,0.75}, σ=0)       # β = behavior-policy bias
confounded = (β=0,   σ∈{0.25,0.50,1.00})     # σ = confounding strength (action-dependent)
```

`(β>0, σ>0)` is out of scope (no cross-product). Confounding is **action-dependent**
(`r += c_r·U·1[a==a_bad]`), the cell_9 mechanism — not the additive `r += c_r·U` of
the legacy cell_7/cell_8. Derived from `docs/rl_regimes_restructure.md` §2.

## Old → new

| Legacy | New regime | Arm | β | σ | Notes |
|---|---|---|---|---|---|
| `cell_1` base | `online_mdp` | basic | 0 | 0 | clean anchor |
| `cell_1` anti_reward / curiosity `{025..100}` | `online_mdp` | biased | >0 | 0 | β via `behavior_strength`; anti_reward/curiosity are exploration/pessimism shapers, not classic πb-bias (ambiguous dial) |
| `cell_2` base | `online_pomdp` | basic | 0 | 0 | |
| `cell_2` anti_reward / curiosity | `online_pomdp` | biased | >0 | 0 | same ambiguity |
| `cell_3` (random/medium/expert tiers) | `offline_mdp` | basic | 0 | 0 | coverage tier is a THIRD axis with no (β,σ) home — folds into `basic` |
| `cell_4` (tiers, masked) | `offline_pomdp` | basic | 0 | 0 | same coverage-tier fold |
| `cell_5` (πb unknown) | `offline_mdp` | — | — | — | no clean home: πb-unknown is a data-mode the new schema drops (paper-text only) |
| `cell_6` (πb unknown, masked) | `offline_pomdp` | — | — | — | same as cell_5 |
| `cell_7` `confounded_sigma_*` | `offline_mdp` | confounded | 0 | >0 | **additive** confounder (legacy); new confounded is action-dependent |
| `cell_7` `online_confounded_*` | `online_mdp` | confounded | 0 | >0 | additive (legacy) |
| `cell_7` `sensitivity_sweep_gamma_*` | `offline_mdp` | confounded | 0 | 0.5 | Γ is a METHOD axis (a logged column), not β/σ; never a path segment |
| `cell_7` `*_deconfounded` triad | `offline_mdp` | confounded | 0 | ≥0 | method comparison (floor/proximal/oracle), not a grid point |
| `cell_8` `confounded_sigma_*_masked_*` | `offline_pomdp` | confounded | 0 | >0 | additive (legacy) |
| `cell_8` `online_confounded_*_masked_*` | `online_pomdp` | confounded | 0 | >0 | additive (legacy) |
| `cell_8` `sensitivity_*_recurrent` | `offline_pomdp` | confounded | 0 | 0.5 | Γ-axis, recurrent |
| `cell_9` `action_gated_sigma_100` | `offline_mdp` | confounded | 0 | 1.0 | the ONLY action-dependent legacy config — matches the new confounder definition |

## Legacy cells with no clean new home

- **`cell_5`, `cell_6`** — πb-unknown data mode is not representable in `(regime, β, σ)`.
- **`cell_3`/`cell_4` coverage tiers** — `random/medium/expert` is a genuine data axis
  orthogonal to β and σ; the new schema folds it into `basic` (random tier).
- **`cell_7`/`cell_8` additive confounder** — byte-frozen and kept in `_legacy/`; the new
  `confounded` arm is action-dependent, so they are not re-pointed, only preserved.

## New points with no legacy equivalent

- `offline_pomdp / confounded (action-dependent)` — cell_9 is MDP-only.

## What did NOT move

The reporting layer (`plotting.py`, `table_formatting.py`) still reads the legacy
`runs/rl_regimes/cell_N/…` layout and serves `_legacy/` runs unchanged; it is wired to
read the new parameter-addressed `results/` tree in PR 6. `render_sweep_tables` degrading
to warn-and-skip on a legacy run whose YAML now lives under `_legacy/` is expected and
acceptable — PR 6 retires it.

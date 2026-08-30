# Consolidation + first RL experiment — PHASE 1 PLAN (2026-08-30)

**Status: a plan. Nothing here is executed.** No merge, no worktree removal,
no seam code. Phase 2 on approval.

Giovanni's direction: `benchmarking_causal_rl` becomes the official repo with
everything GRACE in it; `bcrl-grace-v2` is retired; GRACE variants become
launchable from configs exactly like `cql`/`iql`; then real RL experiments —
learned policies, deployment returns.

---

## Phase 0 — campaign state (gates everything below)

**NOT complete.** One of four re-run priorities has landed, and a defect found
while validating it invalidated part of that one and blocks the rest. Detail
in the handoff's S1c section; summary:

| priority | state |
|---|---|
| 1. D-D sweep headline | **Re-run complete**, except Acrobot s1, re-running now after the criterion fix (`5412283`) |
| 2. V-C1 V1 + V5 | **Running** (`results/vc1_s1c/`) |
| 3. V4 interval block | Not started — waits for a free GPU |
| 4. Q2-A | Not started |
| deferred: `d ≤ 0.10` estimand-drift examination | Not started |

**Nothing structural may proceed until these report.** The worktree is the
live write target; removing or merging under a running fit destroys it.

---

## Phase 1a — consolidation checklist

Investigated 2026-08-30; every claim below is measured, not assumed.

### The git picture is simpler than expected

```
origin   = github.com/Giovannibriglia/benchmarking_causal_rl        -> master ONLY
backup   = github.com/Giovannibriglia/benchmarking_causal_rl_backup -> master,
                                            feat/grace-critic, feat/grace-v2
master           c8f9aa9   the official main branch (origin/HEAD -> origin/master)
feat/grace-v2    (HEAD)    144 commits AHEAD of master, 0 master-only commits
feat/grace-critic b513ba3  frozen v1 record, checked out in benchmarking_causal_rl
```

* **`master` is an ANCESTOR of `feat/grace-v2`.** Zero commits exist on master
  that are not on the branch, so the merge is conflict-free by construction
  and a fast-forward is technically possible.
* **⚠ `feat/grace-v2` has NEVER been pushed to `origin`.** Only `backup` has
  it, and `backup/feat/grace-v2` is **24 commits behind** local. All GRACE v2
  work exists on ONE disk. This is the highest-priority item in the whole
  plan and should be done before anything else, campaign or no campaign —
  pushing is read-only with respect to the running fits.
* `feat/grace-critic` is **not** an ancestor of `feat/grace-v2` (they diverged
  from master independently), so retiring the worktree does not touch it. It
  stays frozen, as directed.

### The steps, in order

1. **Push first, merge later.** `git push origin feat/grace-v2` and
   `git push backup feat/grace-v2`. Never `--force` (nothing to overwrite:
   backup is strictly behind). Safe to do during the campaign.
2. **Campaign completes**, verdicts reported, working tree clean, everything
   committed. Re-verify `git status` in BOTH checkouts.
3. **Merge into master with `--no-ff`.** A fast-forward would also preserve
   all 144 commits, but `--no-ff` records an explicit integration point for
   "GRACE v2 landed here", which is worth one commit. **No disagreement with
   the instruction**: squash would destroy a history that is itself the
   record — the S10/S1b/S1c rulings are legible only as a sequence of
   measured corrections, and several commit messages are the only place a
   falsified justification is recorded next to the measurement that killed it.
   ```
   cd ~/PycharmProjects/benchmarking_causal_rl
   git checkout master && git merge --no-ff feat/grace-v2
   ```
   Note this changes the frozen checkout's branch from `feat/grace-critic` to
   `master`; `feat/grace-critic` is untouched as a ref.
4. **Verify from the main checkout before removing anything**: `git log
   --oneline -5`, `git log --oneline | wc -l`, and a spot diff of
   `src/rl/offline/grace/estimator.py` and `docs/grace_v2_handoff.md` against
   the worktree copies. Push master to origin and backup.
5. **Only then** `git worktree remove ../bcrl-grace-v2`. Shared object store,
   so nothing is copied and nothing is lost. Follow with `git worktree prune`
   and a `git status` in the surviving checkout (the standing rule).

### What lives outside git and must survive

| artifact | size | status | action |
|---|---|---|---|
| `~/.minari-grace-v2` | **16 GB**, 163 certified datasets | outside the repo, referenced only by env var | keep in place; make it a **declared config value** (below) |
| `results/` | 50 MB | 29 files tracked (force-added past `.gitignore`), 21 untracked/ignored | audit before removal; force-add the ones a claim rests on |
| `/tmp/claude-*/scratchpad` | small | scratch only | nothing to keep; the one script that mattered (`rerun_dd_sweep.sh`) is superseded |
| `results/*/generator/` checkpoints | in `results/` | untracked | needed to rebuild target policies — Q2-A rebuilds them reproducibly from seed, so **not** required to survive |

**The env-var dependency is a real gap, not a formality.** `MINARI_DATASETS_PATH`
is read in ~12 tools and in `sweep_supervisor.py` (which sets it per worker),
but **no config declares it** — so every launch depends on shell state, which
is exactly the kind of undeclared channel A1 exists to forbid. Proposal: a
`minari_datasets_path` key in the cell spec (defaulting to
`~/.minari-grace-v2`), exported by the driver, with the env var retained as an
explicit override. One-line change per launcher; removes the session
dependence the user called out.

### Doc pass for the official repo

`README.md` gains: what GRACE is (one paragraph + the layer table), where the
authoritative records live (`docs/grace_v2_handoff.md` = standing rules and
current validity; `docs/diagram_catalogue.md` = the assumption surface), and
how to launch every algorithm **including the variants** once Phase 1b lands.

---

## Phase 1b — the seam design

### What already exists (investigated, not assumed)

**The sockets are in v2; the wire is cut.** `CRITIC_LIBRARY` already declares
`grace` and `grace_no_router` specs (`src/benchmarking/critic_ablation.py:211`,
`:218`) and `CriticAblationConfig.grace` exists (`:257`) — but
`_build_strategy_critic` has branches only for
`observational/proximal/oracle_u/sensitivity` and raises on anything else
(`:495`), `StrategyCritic.__init__` never forwards `spec.grace` (`:517`), and
`KNOWN_STRATEGIES` (`src/benchmarking/regime_sweep.py:54`) omits `grace`, so
`critics: [grace]` is **rejected at config-parse time today**.

**The v1 pattern is the template**, readable in the frozen checkout:
`_install_grace` (`builders.py:76`) swaps `agent.q_network`/`target_network`
for a serving head (`heads.py:30`) and binds
`agent.set_sequence_buffer = machinery.fit_from_buffer`.

**`set_sequence_buffer` is exactly the fit-once-then-serve point**: called at
`runner.py:1661` (and `:1666` for the ablation) immediately after the Minari
fill and **before any gradient step**. Fit there, serve thereafter — no
cadence refit, as ruled.

**Training-time Q flows through one seam already**: `IdentificationStrategy.
critic_value(net, x, batch)` (`src/rl/off_policy/identification.py:39`), with
`Observational/OracleU/Proximal/SensitivityBounds` implementations. A `Grace`
strategy joins that list; **nothing in the runner changes**.

### The wiring, minimal and enumerated

1. `src/rl/offline/grace/serving.py` — the serving head + a `Grace`
   `IdentificationStrategy`, built on Q2-A's fitted iteration.
2. `_build_strategy_critic`: a `builder == "grace"` branch taking
   `grace_options` (`critic_ablation.py:495`).
3. Forward `spec.grace` / `config.grace` from `StrategyCritic.__init__`
   (`:517`) and `CriticAblationManager` (`:626`).
4. Pass `grace=` in `regime_sweep.py:793`; add `"grace"` to
   `KNOWN_STRATEGIES` (`regime_sweep.py:54`).

### Config layout — RECOMMENDATION: the critic axis, not new algorithm names

`critics: [observational, grace]` rather than `grace_cql` / `grace_iql`.

**The argument is the experiment's control requirement.** `_run_ablation_point`
(`regime_sweep.py:735-800`) builds **one shared run per (point, env, algo,
seed)** and explodes it into per-critic leaves (`_slice_critic_csv`, `:700`).
So baseline and variant share the same dataset, the same seeds, the same
budgets and the same base agent **by construction** — not by convention. A
separate `grace_cql` registry entry would be a *different run* whose matching
we would have to maintain by discipline, which is precisely the kind of thing
that drifts silently. The axis also gives `cql+grace` and `iql+grace` for free
from the existing `algos:` list, which satisfies "launchable from configs
exactly like `cql`/`iql`".

If a standalone name is still wanted for one-off launches through `main.py`,
register `cql_grace`/`iql_grace` as **thin aliases delegating to the same
builder** — one construction site (S6), never a second implementation.

### The serving rule (L4's, as ruled)

| L4 verdict | served offline |
|---|---|
| `interval` | `Q⁻`, the pessimistic end |
| `bounds` | `Q⁻`, pessimism over the identified set |
| `abstain` | base algorithm's own critic, run labelled **`GRACE-ABSTAINED`** |

A silent fallback would make every comparison meaningless, so the label is
load-bearing: abstained runs are reported **separately**, never pooled into a
variant-vs-base average. **C3 labels travel into the run artifacts**: each run
records the fit's conditions (`converged`/`monotone`/`degenerate_mechanism`/
`reached_tau_one`, the resolved reward mechanism) and the serving mode per
cell.

**No per-environment GRACE parameters in any config.** One method
configuration; the binding audit enforces it.

**Deferred, noted not built**: L2's verdict does not select estimator or
channels. The variant may hand-wire the D-D path exactly as D-E's bounds were
hand-wired.

### ⚠ TWO GATES BEFORE THIS CAN RUN

**1. The Q2-A entry ticket is not yet satisfied for the experiment's cells.**
The variant is only enabled where Q2-A passed, and Q2-A has passed **only on
`d_a_null` CartPole** (RMSE 0.36–2.86 on |RTG| 6.5–56.6, ~5%). `d_d` at
d = 1.0 / 0.25 / asym have **no Q2-A validation yet** — that is Q2-A step 3,
and it is inside this entry ticket rather than skippable.

**2. The deployment environment does NOT currently do what the brief says.**
The brief specifies "U→R intact, U→A severed". Measured: `eval_env` is built
clean at `runner.py:354` and **never wrapped** — `ConfoundedCollectionWrapper`
wraps the TRAIN env only, and says so in its own docstring
(`src/envs/wrappers/confounded.py:20`). So evaluation today severs **both**
edges: there is no `U` at evaluation at all.

This is not a detail. On `d_d` the reward channel is **gated**
(`r += c_r·U·1[a = a_bad]`), so with no `U` at evaluation the gate can never
fire and the entire reward structure the cell is built around is absent from
the metric. Two coherent options:

* **(A) Keep the clean eval** (no code change). The experiment then asks *"is
  the critic fooled by the spurious signal?"* — return is survival time, and a
  baseline that overvalues `a_bad` survives less. Still a valid, falsifiable
  test, and P-E1 still applies (bigger marginal bias → more overvaluation).
* **(B) Wrap the eval env with `c_a = 0, c_r = c_r`** — U→R intact, U→A
  severed, exactly the brief. Small and well-scoped: one wrap at
  `runner.py:354` behind a config flag. This is the only option under which
  the *gate* is part of the deployment return.

**Recommendation: (B), with (A) reported alongside** — they answer different
questions and (A) costs nothing extra once (B) exists. Flagging rather than
silently building either, since it changes what the headline number means.

---

## Phase 1c — the first experiment (design, PRE-REGISTERED)

**CartPole only.** The Acrobot transition mechanism is unproven (the
single-component MDN ruling is implemented but unmeasured), so Acrobot q2 is
not ready and the first experiment is not coupled to it.

### Cells and why each is in

| cell | dataset ids | role |
|---|---|---|
| `d_d_sweep_d100` (d = 1.0) | CartPole s0/s1/s2 | symmetric gate, STRONG R — proxies decorative |
| `d_d_sweep_d025` (d = 0.25) | CartPole s0/s1/s2 | symmetric gate, past the located transition |
| `d_d_sweep_d010_asym` | CartPole s0/s1/s2 | **the asymmetric point — the one expected to win** |
| `d_a_null` | CartPole s0/s1/s2 | no-harm control: no latent at all |

All twelve datasets exist and are certified in `~/.minari-grace-v2`.

### The mechanism behind the prediction, recorded because it is the point

`gate_probs` is the gate's firing probability under `U = 0` / `U = 1`:

* symmetric `d010`: **[0.45, 0.55]** — ratio 1.22. The marginal gate rate is
  ~0.5 whatever `U` does, so a `U`-blind (observational) critic's reward model
  is close to the interventional one. **Confounding is present but nearly
  invisible in the marginal**, so there is little for a causal critic to
  recover at the value level.
* asymmetric `d010_asym`: **[0.05, 0.15]** — ratio 3.0. The gate rate depends
  strongly on `U` *relative to its own size*, so the `U`-blind reward model is
  materially biased and the action a policy should prefer differs from the one
  the observational critic scores highest.

### ⚠ THE FIRST REGISTRATION WAS WITHDRAWN BEFORE ANY RUN (2026-08-30)

The original P-E1/P-E2/P-E3 predicted gains **largest at the asymmetric
point** and parity at `d = 1.0`. **Withdrawn by the author as wrong, and the
reason is recorded because it is instructive**: it conflated two different
contrasts.

* **with-vs-without proxies** (what the ABLATION measures) — governed by the
  symmetric marginal's near-latent-independence, so the proxy differential is
  small at symmetric points. True, measured, and *irrelevant to this
  experiment*.
* **GRACE-vs-base** (what the EXPERIMENT measures) — governed by the naive
  bias and GRACE's correction of it. The naive bias is `M · tilt`, flat in `d`
  and independent of gate symmetry (the tilt enters through `P(U|a_bad)`;
  symmetry only moves `q̄`).

The two predict **opposite orderings**. The replacement below follows the
value-level measurements instead. This is the second prediction in the project
corrected by its own prior measurements before a run — the discipline of
writing predictions down first is load-bearing in both directions, and that is
why the withdrawn version stays on the page rather than being deleted.

### The amended registration

* **P1 — no harm.** Parity with base within seed noise on `d_a_null`. A
  variant that "wins" there is a BUG: there is no latent to exploit.
* **P2 — safety.** Grace variants never worse than base beyond seed noise on
  any enabled cell.
* **P3 — ordering.** Where return gaps appear, they order by the measured
  value-correction share: **`d = 1.0` ≥ `d = 0.25` ≥ `d010asym`.**
* **P4 — no silent abstention.** `GRACE-ABSTAINED` never fires on an enabled
  cell (the entry ticket is what makes a cell enabled).
* **P5 — the null outcome is a result.** The bias may not cross any decision
  boundary on CartPole, giving parity everywhere while the CRITICS differ.
  Informative, not a failure — provided the experiment can see it, which is
  what the secondary endpoint below exists for.

**What P3 actually rests on, stated so it is not over-read.** Correction share
`1 − err_grace/err_naive_tr` from V-C1 (CartPole, pre-fix run, 3 seeds/cell):

| cell | d100 | d050 | d025 | d010 | d005 |
|---|---|---|---|---|---|
| pre-fix | 91.6% | 92.1% | 91.1% | 68.1% | 19.6% |
| **post-fix (final)** | **93.3%** | **88.7%** | **86.0%** | **72.8%** | **48.0%** |

**RE-QUOTED from the corrected-likelihood run (2026-08-30), and it improves
P3's standing.** The near-tie at the top resolved into a monotone ordering —
93.3 > 88.7 > 86.0 > 72.8 > 48.0 — so `d = 1.0 ≥ d = 0.25` is now a visible
gap (7.3 points) rather than a coin-flip, and the weak end is corrected far
better than the pre-fix numbers suggested (d005 19.6% → 48.0%). P3's
predicted ordering is exactly the measured one.

One caveat still travels: **`d010_asym` has never been measured in V-C1**.
Its expected value is inferred from the symmetric `d010` (72.8%), whose gate
strength is the closest analogue — so P3's third term is a prediction proper,
not a restatement of a measurement.

### Secondary endpoint (required by P5): critic Q-accuracy

Return parity and "GRACE corrected nothing" produce the **same headline
number and opposite conclusions**, so returns alone are unreadable. For both
variants and bases, log **Q-error against the analytic q1 truth** (and MC
return-to-go where sequential), recorded during and after training. Then every
outcome yields a claim: return gains where decision boundaries are crossed,
Q-level gains where they are not.

### Evaluation

* **Deployment environment**: `U → R` intact, `U → A` **severed**. Deployment
  breaks the confounding link by construction, which is precisely the regime
  the benchmark's evaluation path was built for. The learned policy chooses
  actions; `U` still perturbs the reward.
* **Metric**: episodic return of the learned policy, per (dataset seed,
  training seed), reported with the **MC noise band** from the evaluation
  rollouts — never a single number, and never averaged across dataset seeds
  (the D-D reporting constraint applies here too: CartPole s1 behaves
  differently from s0/s2 and a mean would hide it).
* **Baselines**: `cql` and `iql` **unmodified** — same datasets, same seeds,
  same budgets, same evaluation. The only difference is the critic.

### Reading the result

A win is: variant return > base return at `d010_asym`, per seed, outside the
MC band, with the base algorithms matched on `d_a_null`. Anything else is
reported as measured. Because the variants abstain rather than serve when a
fit's C3 conditions fail, an `GRACE-ABSTAINED` run is **not** evidence about
the method — it is evidence the cell was out of scope, and is reported
separately rather than pooled.

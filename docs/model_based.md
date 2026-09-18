# Model-based value iteration (`src/rl/model_based`)

Two planners that learn a model of the environment from the replay buffer and
plan on it, registered as first-class algorithms next to `dqn` / `ppo`:

| name | observations | actions | regime |
|---|---|---|---|
| `tabular_vi` | `Discrete(n)` (arrives one-hot) | `Discrete` | online |
| `offline_tabular_vi` | same | same | offline (Minari dataset) |
| `mlp_vi` | flat vector (`Box`) | `Discrete` | online |
| `offline_mlp_vi` | same | same | offline |

Both are `BaseOffPolicy` agents that own a `ReplayBuffer`, so they ride the
existing loops **unchanged**: online = `collect -> buffer.add -> agent.update(batch)`;
offline = Minari fill -> `agent.update(buffer.sample(B))` for `offline_grad_steps`.
Nothing in the runner knows they are model-based.

## `TabularVI`

* **State index** = argmax of the one-hot observation (the env wrapper
  flattens every gymnasium space; `Discrete(n)` becomes a one-hot of length
  `n`, so `n_states == obs_dim`). The first batch is checked to be one-hot;
  anything else raises with a pointer to `mlp_vi`.
* **Model** from *every* transition in the buffer (never from the sampled
  batch — sampling with replacement across calls would double count):
  `N(s,a,s')`, `R_sum(s,a)`, `D(s,a,s')` (terminations) via `bincount`;
  maximum likelihood `P(s'|s,a)`, `R(s,a)`, `d(s,a,s')`. Done transitions are
  not bootstrapped through, so terminal states never need a value.
* **Planning**: exact value iteration
  `Q(s,a) = R(s,a) + γ Σ_s' P(s'|s,a)(1-d(s,a,s')) V(s')` to `vi_tol`
  (default 1e-6) or `vi_max_iters`. Unvisited `(s,a)` have no model and are
  pinned at the two bounds the data implies — no tuned constant:
  the **exploitation** table (`q_table`: greedy eval, the offline policy)
  uses the R-min bound `min_r_seen/(1−γ)` (never trust an untried action; a
  fixed 0 is *optimistic* on negative-reward tasks), the **exploration**
  table (`q_explore`, `optimistic=True` = the online builder's default) uses
  the R-max bound `max_r_seen/(1−γ)` so training is driven to untried pairs.
  Before any reward is seen both are `unvisited_value` (0.0).
* **Schedule**: `learn` is a tick; every `plan_every` ticks (50) it re-counts
  and re-plans *if the buffer changed*. Offline that is exactly one plan;
  online a periodic replan on the growing buffer. `train_metrics.csv` carries
  the Bellman residual as `loss`, plus `coverage` (visited fraction of
  `(s,a)`), `n_visited_sa`, `vi_iters`.
* Acting is ε-greedy with the same RNG pattern as `DQN.act`, on `q_explore`
  during training and on `q_table` when greedy (`deterministic` / `epsilon=0`,
  i.e. evaluation); ties among maximal actions break uniformly at random
  (an all-equal row is a uniform-random step, not "action 0").
* The online buffer holds the whole history (1M rows): the counts are rebuilt
  from the buffer, and a wrapped buffer would re-open visited pairs.

## `MLPValueIteration`

* **Model**: `DynamicsEnsemble` — `K` (default 3) ReLU MLPs `(200, 200)`
  mapping `(obs, one_hot(a)) -> (Δobs, r, done_logit)` in normalised
  coordinates (running mean/std of obs and Δobs, Welford, stored as buffers).
  Loss = MSE(Δ) + MSE(r) + BCE(done), one Adam step per `learn`.
* **Planning** (after `model_warmup` = 500 model steps): for every state in
  the batch and **every** action, the ensemble mean gives `(s', r, d)` and the
  target `y(s,a) = r − c·u(s,a) + γ(1−d) max_a' Q_target(s',a')`; the Q-net
  regresses onto all `|A|` targets at once (MSE) and the target net tracks
  with Polyak `tau`. This is a full Bellman backup on the batch's states —
  fitted value iteration on the buffer's state sample. `u` is the ensemble
  disagreement (std of Δ across members); `c = penalty_coef` is 0 online and
  1.0 in `offline_mlp_vi` (MOPO-style: don't chase model error off-support).
* Q never sees a real reward or next state; the real transition only trains
  the model. Acting is ε-greedy on `Q`, evaluation greedy.

Builder kwargs (via the YAML `networks:` map, like the other algos):
`gamma`, `epsilon`, `plan_every`, `unvisited_value` (tabular); `gamma`,
`epsilon`, `tau`, `model_warmup`, `lr_model`, `lr_q`, `penalty_coef`,
`n_members`, `model_hidden_dims` (MLP).

## Scope and known limits

* Discrete actions only — value iteration maxes over actions. Continuous
  actions need an actor (that is the next step, not a flag).
* The MLP planner does 1-step model backups; no multi-step imagined rollouts
  yet (MBPO-style). One-step is enough to be a real value-iteration method and
  keeps model error from compounding, at the price of relying on the buffer's
  state coverage for the "grid".
* Under gymnasium's NEXT_STEP autoreset the step after a done stores a
  spurious `(terminal_obs -> reset_obs, r=0, done=0)` transition; DQN lives
  with the same quirk. Tabular: harmless (terminal states are never
  bootstrapped into). MLP: a small amount of label noise on terminal-looking
  states.
* Eval on sparse tasks needs `eval_count_terminal_reward=True` (the legacy
  eval drops the terminal step's reward and reads 0.0 on FrozenLake).

## The first trial

`tools/run_model_based_trial.py` — the grid, the datasets, the report:

```
uv run python tools/run_model_based_trial.py --datasets-only   # tabular datasets
uv run python tools/run_model_based_trial.py --shard i/N       # N CPU workers
uv run python tools/run_model_based_trial.py --report          # report.md + report.png
```

* Online: FrozenLake-v1, Taxi-v3, CliffWalking200-v1 (a 200-step-limited
  registration; the stock `CliffWalking-v1` has no TimeLimit) with
  `tabular_vi` / `dqn` / `ppo`; CartPole-v1, Acrobot-v1, MountainCar-v0,
  LunarLander-v3 with `mlp_vi` / `dqn` / `ppo`. Same env-step budget for all
  (8 envs × 256 steps × 50 rounds ≈ 102k steps).
* Offline: tabular envs on `uniform` (ε = 1 agent: uniform-random actions,
  full coverage) and `medium` (the DQN generator's 1/3-of-expert checkpoint)
  datasets; vector envs on the existing `generated/<env>/medium-v0`.
  `offline_tabular_vi` / `offline_mlp_vi` vs `offline_dqn` vs `cql` (PPO has
  no offline form). 10k gradient steps each.
* Results: `results/model_based_trial/report.md` and `report.png`.

## First-trial results (2026-09-18, 2 seeds, CPU)

Full tables and curves: `results/model_based_trial/report.md` / `report.png`.
Metric = summed reward over a fixed eval window per env (256 steps online, 500
offline) with autoreset, so on CliffWalking the optimal 13-step loop reads
−238 (online) / −465 (offline) and "never reaches the goal" reads −255 / −498.

**Online, same ≈102k env steps each:**

| env | model-based | dqn | ppo |
|---|---|---|---|
| FrozenLake-v1 | 3.8 | 4.3 | 4.0 |
| Taxi-v3 | **+126** | −255 | −255 |
| CliffWalking200-v1 | **−238 (optimal)** | −255 | −255 |
| CartPole-v1 | 254 | 253 | 256 |
| Acrobot-v1 | −254 | −256 | −251 |
| MountainCar-v0 | −255 | −255 | −255 |
| LunarLander-v3 | −3 ± 58 | −16 ± 9 | −47 ± 28 |

**Offline, 10k gradient steps each (offline_tabular_vi / offline_mlp_vi vs offline_dqn vs cql):**

| env · tier | model-based | offline_dqn | cql |
|---|---|---|---|
| FrozenLake · uniform | **6.4** | 5.2 | 1.6 |
| FrozenLake · medium | 3.2 | 0.9 | 3.8 |
| Taxi · uniform | **+278** | −1324 | −497 |
| Taxi · medium | −506 | −692 | −498 |
| CliffWalking · uniform | **−465 (optimal)** | −498 | −498 |
| CliffWalking · medium | −498 | −498 | −498 |
| CartPole · medium | **498** | 462 | 470 |
| Acrobot · medium | −500 | −500 | −500 |
| LunarLander · medium | −3250 | −2936 | −1243 |

Reading:

* **Tabular VI is the right tool for small discrete MDPs.** Online it solves
  Taxi and CliffWalking within the budget where DQN/PPO have not started;
  offline on uniform-coverage data it recovers the optimal policy (Taxi +278,
  CliffWalking optimal) where the model-free learners fail outright. On
  *medium* (narrow) data every method is at the floor on Taxi/CliffWalking
  and the R-min rule keeps the planner from doing worse than "stay safe".
* **MLP VI holds its own on dense, smooth dynamics** (CartPole: best offline
  at 498/500; online on par) and is the best online on LunarLander at this
  budget, with high seed variance. It does nothing on Acrobot/MountainCar —
  neither do DQN/PPO at 102k steps; those need exploration bonuses or more
  steps, not a better planner.
* **Offline LunarLander is the model-based failure case**: one-step backups
  through a 3-member ensemble at 10k steps exploit model error (−3250 vs
  CQL −1243). Multi-step rollouts with a stronger penalty, or a larger
  ensemble, are the obvious next levers.

Three things the trial fixed on the way, each measured before it was fixed:
an all-zero table's argmax is a fixed action (FrozenLake 0.0 → uniform
tie-breaking); a circular buffer evicts early visits and re-opens optimism
(Taxi +134 → −1011 → whole-history buffer + a separate exploitation table);
a fixed 0 for unvisited pairs is optimistic on negative-reward tasks (Taxi
−2244, CliffWalking −4876 on medium data → the R-min bound).

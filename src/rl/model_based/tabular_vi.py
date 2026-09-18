from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..base import ActionOutput
from ..off_policy.base_off_policy import BaseOffPolicy


class QTable(nn.Module):
    """The planner's ``Q`` as a module, so the runner checkpoints it like any
    policy network (``policy_state``) and ``act`` reads it in place."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()
        self.register_buffer("q", torch.zeros(n_states, n_actions))

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.q[state_index]


class TabularVI(BaseOffPolicy):
    """Count-based model + exact value iteration for discrete (obs, action).

    **Observations.** Gymnasium ``Discrete(n)`` observations reach the agent
    ONE-HOT (the env wrapper flattens every space), so the state index is the
    argmax and ``n_states == obs_dim``. The first batch is checked to be
    one-hot; anything else is a clear error pointing at ``mlp_vi``.

    **Model.** From every transition in the buffer (never from the sampled
    batch -- sampling with replacement across calls would double count):
    ``N(s,a,s')``, ``R_sum(s,a)`` and ``D(s,a,s')`` (terminations). Maximum
    likelihood: ``P(s'|s,a) = N/N(s,a)``, ``R(s,a) = R_sum/N(s,a)``,
    ``d(s,a,s') = D/N(s,a,s')``. Transitions flagged done are not
    bootstrapped through, so terminal states never need a value.

    **Planning.** ``Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a)(1-d(s,a,s')) V(s')``
    iterated to ``vi_tol`` (or ``vi_max_iters``). Unvisited ``(s,a)`` have
    no model; the two tables pin them at the two BOUNDS the data implies:

    * the EXPLOITATION table (``q_table``, what a greedy eval and the offline
      policy use) pins them at the R-min bound ``min_r_seen / (1 - gamma)``,
      a lower bound on any Q: never trust an action the data never tried.
      A fixed 0 is not neutral -- on negative-reward tasks (Taxi, CliffWalking)
      it is OPTIMISTIC, and the offline planner walked into untried illegal /
      cliff actions (measured: Taxi -2244, CliffWalking -4876 on medium data).
    * the EXPLORATION table (``q_explore``, ``optimistic=True``, the ONLINE
      builder's default) pins them at the R-max bound ``max_r_seen / (1 - gamma)``,
      an upper bound, so acting during training is DRIVEN to unvisited pairs
      until they are visited -- the exploration a count-based model needs;
      epsilon-greedy from an all-zero table walks into action 0 forever
      (measured: FrozenLake 0.0 return).

    Before any reward has been seen both bounds are ``unvisited_value`` (0).
    No constant is tuned: the bounds are the data's own extreme reward.

    **Explore vs exploit.** Optimism is for ACTING DURING TRAINING. A greedy
    evaluation (``deterministic`` or ``epsilon == 0``) reads the PLAIN table
    (unvisited = ``unvisited_value``): an eval rollout cannot learn, so an
    optimistic greedy policy that meets an unvisited pair repeats it forever
    (measured on Taxi: +134 mid-run -> -1011 once the buffer wrapped and early
    visits fell out of the counts). Hence also: the ONLINE buffer must hold
    the whole history (the builder gives it 1M rows; tabular rows are tiny).

    **Ties.** ``act`` breaks ties among maximal actions UNIFORMLY at random
    (an all-equal row is a uniform-random step, not a fixed action); a
    ``torch.argmax`` first-index rule is not a policy, it is an artefact.

    **Schedule.** ``learn`` is called once per sampled batch by both loops; it
    re-counts and re-plans every ``plan_every`` calls IF the buffer changed.
    Offline (static buffer) that is exactly one plan; online it is a
    periodic replan on the growing buffer.
    """

    action_type = "discrete"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        buffer,
        device: torch.device,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        plan_every: int = 50,
        vi_tol: float = 1e-6,
        vi_max_iters: int = 10_000,
        unvisited_value: float = 0.0,
        optimistic: bool = False,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.buffer = buffer
        self.epsilon = float(epsilon)
        self.plan_every = int(plan_every)
        self.vi_tol = float(vi_tol)
        self.vi_max_iters = int(vi_max_iters)
        self.unvisited_value = float(unvisited_value)
        self.optimistic = bool(optimistic)
        self._max_r_seen = float("-inf")
        self._min_r_seen = float("inf")
        self.q_table = QTable(self.n_states, self.n_actions).to(device)  # exploit
        self.q_explore = QTable(self.n_states, self.n_actions).to(device)  # optimistic
        self.is_recurrent = False
        self._ticks = 0
        self._counted = -1  # buffer size at the last (re)plan
        self._checked_one_hot = False
        self._last_metrics: Dict[str, float] = {}
        # The fitted tables, kept for inspection/tests (None until a plan ran).
        self.P: Optional[torch.Tensor] = None
        self.R: Optional[torch.Tensor] = None
        self.N_sa: Optional[torch.Tensor] = None

    # -- observation -> state index ---------------------------------------
    def _state_index(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if not self._checked_one_hot:
            is_one_hot = bool(
                (obs.sum(dim=1) == 1).all() and (obs.max(dim=1).values == 1).all()
            )
            if obs.shape[1] != self.n_states or not is_one_hot:
                raise ValueError(
                    "tabular_vi needs a Discrete observation space (delivered "
                    f"one-hot with obs_dim == n_states == {self.n_states}); got "
                    f"a batch of shape {tuple(obs.shape)} that is not one-hot. "
                    "Use mlp_vi for continuous observations."
                )
            self._checked_one_hot = True
        return obs.argmax(dim=1)

    # -- acting -------------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> ActionOutput:
        if deterministic:
            epsilon = 0.0
        eps = self.epsilon if epsilon is None else epsilon
        s = self._state_index(obs)
        # Same RNG pattern as DQN: one torch.rand draw per call, then either a
        # uniform action batch or the greedy one.
        if torch.rand(1).item() < eps:
            action = torch.randint(0, self.n_actions, (s.shape[0],), device=obs.device)
            return ActionOutput(action=action, state=state)
        exploit = deterministic or eps == 0.0
        with torch.no_grad():
            table = self.q_table if (exploit or not self.optimistic) else self.q_explore
            action = self._greedy(table(s))
        return ActionOutput(action=action, state=state)

    @staticmethod
    def _greedy(q: torch.Tensor) -> torch.Tensor:
        """Argmax with UNIFORM tie-breaking: a random draw ranks the maximal
        entries, so an all-equal row yields a uniform-random action."""
        is_max = q == q.max(dim=1, keepdim=True).values
        noise = torch.rand(q.shape, device=q.device)
        return torch.argmax(is_max.float() + noise * is_max.float(), dim=1)

    # -- model ----------------------------------------------------------------
    def fit_model(self, rows: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Count the tables from a dict of transition columns and store
        ``P``, ``R``, ``N_sa`` and the termination fractions."""
        S, A = self.n_states, self.n_actions
        s = self._state_index(rows["obs"].to(self.device))
        s2 = rows["next_obs"].to(self.device).argmax(dim=1)
        a = rows["actions"].to(self.device).long().reshape(-1)
        r = rows["rewards"].to(self.device).float().reshape(-1)
        d = rows["dones"].to(self.device).float().reshape(-1)
        sa = s * A + a
        sas = sa * S + s2
        N = torch.bincount(sas, minlength=S * A * S).float().reshape(S, A, S)
        D = torch.bincount(sas, weights=d, minlength=S * A * S).reshape(S, A, S)
        R_sum = torch.bincount(sa, weights=r, minlength=S * A).reshape(S, A)
        N_sa = N.sum(dim=2)
        visited = N_sa > 0
        P = torch.where(visited.unsqueeze(-1), N / N_sa.clamp(min=1).unsqueeze(-1), 0.0)
        R = torch.where(visited, R_sum / N_sa.clamp(min=1), 0.0)
        done_frac = torch.where(N > 0, D / N.clamp(min=1), 0.0)
        # Fold the termination mask into the transition kernel once: the
        # continuation kernel C(s,a,s') = P(s'|s,a) * (1 - d(s,a,s')).
        self.P, self.R, self.N_sa = P, R, N_sa
        self._C = P * (1.0 - done_frac)
        self._visited = visited
        if r.numel():
            self._max_r_seen = max(self._max_r_seen, float(r.max().item()))
            self._min_r_seen = min(self._min_r_seen, float(r.min().item()))
        return {
            "n_transitions": float(s.shape[0]),
            "n_visited_sa": float(visited.sum().item()),
            "coverage": float(visited.float().mean().item()),
        }

    # -- planning -------------------------------------------------------------
    def _solve(self, fill: float) -> Tuple[torch.Tensor, int, float]:
        """Exact VI on the fitted tables with unvisited pairs pinned at ``fill``."""
        V = torch.zeros(self.n_states, device=self.device)
        unvisited_q = torch.full_like(self.R, fill)
        iters, residual = 0, float("inf")
        for iters in range(1, self.vi_max_iters + 1):
            Q = self.R + self.gamma * torch.einsum("ijk,k->ij", self._C, V)
            Q = torch.where(self._visited, Q, unvisited_q)
            V_new = Q.max(dim=1).values
            residual = float((V_new - V).abs().max().item())
            V = V_new
            if residual < self.vi_tol:
                break
        return Q, iters, residual

    def value_iteration(self) -> Dict[str, float]:
        """Writes the exploitation table ``q_table.q`` (unvisited = the R-min
        bound) and, when optimistic, the exploration table ``q_explore.q``
        (unvisited = the R-max bound)."""
        if self.R is None:
            raise RuntimeError("value_iteration before fit_model")
        seen = self._max_r_seen > float("-inf")
        lo = self._min_r_seen / (1.0 - self.gamma) if seen else self.unvisited_value
        Q, iters, residual = self._solve(lo)
        self.q_table.q.copy_(Q)
        if self.optimistic:
            hi = self._max_r_seen / (1.0 - self.gamma) if seen else self.unvisited_value
            Q_opt, _, _ = self._solve(hi)
            self.q_explore.q.copy_(Q_opt)
        return {"vi_iters": float(iters), "vi_residual": residual}

    def plan(self) -> Dict[str, float]:
        """Recount from the WHOLE buffer and replan."""
        n = len(self.buffer)
        rows = self.buffer.gather(range(n))
        metrics = self.fit_model(rows)
        metrics.update(self.value_iteration())
        self._counted = n
        return metrics

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._ticks += 1
        due = self._counted < 0 or (self._ticks % self.plan_every == 0)
        if due and len(self.buffer) != self._counted and len(self.buffer) > 0:
            m = self.plan()
            # The Bellman residual is the planner's "loss": the runner's
            # train_metrics schema reads loss / critic_loss / q_loss.
            m.update(
                loss=m["vi_residual"],
                critic_loss=m["vi_residual"],
                q_loss=m["vi_residual"],
            )
            self._last_metrics = m
        return dict(self._last_metrics)

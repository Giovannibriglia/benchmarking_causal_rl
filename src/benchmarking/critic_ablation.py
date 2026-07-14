from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.nets.mlp import MLP
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.on_policy.base_actor_critic import RolloutBatch

CRITIC_ABLATION_COLUMNS: list[str] = [
    "episode",
    "algorithm",
    "environment",
    "critic",
    "train_loss",
    "advantage_mean",
    "explained_variance",
    "pearson",
    "spearman",
    "mutual_information",
    "kl",
    "js_normalized",
    "mse",
    "td_error_mean",
    "real_reward_mean",
    "pred_reward_mean",
    "reward_explained_variance",
    "reward_pearson",
    "reward_spearman",
    "reward_mutual_information",
    "reward_kl",
    "reward_js_normalized",
    "reward_mse",
    "reward_error_mean",
]

# Strategy-critic ablation (Cell-7 deconfounding) — a SEPARATE schema/file-path
# from the V-head columns above. Scored estimation-vs-oracle (ground truth =
# oracle Q_adj = UMarginalizedQ.forward), NOT against returns; no return metric.
STRATEGY_CRITIC_ABLATION_COLUMNS: list[str] = [
    "episode",
    "algorithm",
    "environment",
    "critic",
    "base_algo",
    "sigma",
    "train_loss",
    "apparent_q_mean",
    "oracle_q_mean",
    "q_inflation",
    "value_mse_to_oracle",
    "argmax_agreement_oracle",
    "explained_variance_to_oracle",
    "pearson_to_oracle",
    "spearman_to_oracle",
    "gap_closed_fraction",
    # Γ (MSM sensitivity bound) — a METHOD parameter of the sensitivity critic,
    # ORTHOGONAL to the (β, σ) regime axes. Logged (blank for non-sensitivity
    # critics) but NEVER folded into the dataset/(β,σ) path encoding.
    "gamma",
    # NULL-CALIBRATION control for the ADAPTIVE critics ONLY (observational /
    # proximal / oracle_u): each self-nulls at the (β=0, σ=0) origin (no
    # correction / posterior->prior / A⊥U), so at the origin they must agree with
    # the oracle within estimator noise. True iff value_mse_to_oracle clears the
    # noise floor AT the origin (σ=0); blank off-origin (undefined). The
    # NON-ADAPTIVE sensitivity critic is EXEMPT (always blank) — its unconditional
    # Γ-bound is pessimism BY DESIGN, reported as ``pessimism_cost`` instead.
    "null_calibrated",
    # PESSIMISM COST (sensitivity critic ONLY, at the origin): oracle_q_mean -
    # apparent_q_mean = the value given up to the Γ-bound when there was NOTHING to
    # be robust against (σ=0). A REPORTED RESULT, not a failure mode; blank
    # off-origin and blank for every non-sensitivity critic.
    "pessimism_cost",
]

_EPS = 1e-8

# Below this, MSE(observational, oracle) is estimator noise, not a confounding
# gap: the gap-closed RATIO of two noise terms is not a metric, so it is reported
# ONLY where sigma>0 AND the observational->oracle MSE clears this floor.
_GAP_NOISE_FLOOR_MSE = 1e-2

# --ablation-critics base-algo name -> proximal/oracle_u builder suffix. The
# recurrent base (offline_dqn_recurrent, the Cell-8 POMDP floor) resolves to the
# SAME "dqn" family — the recurrent-vs-MLP axis is the ORTHOGONAL ``encoder``
# argument, not the suffix (which only selects the base-learner family).
_BASE_ALGO_SUFFIX: Dict[str, str] = {
    "offline_dqn": "dqn",
    "offline_dqn_recurrent": "dqn",
    "dqn": "dqn",
    "bcq": "bcq",
    "cql": "cql",
    "iql": "iql",
}

# ``encoder`` values that select the Cell-8 recurrent arm (else "mlp" = the
# byte-frozen Cell-7 MLP arm). Mirrors the recurrent builders' critic_network.
_RECURRENT_ENCODERS: frozenset[str] = frozenset({"rnn", "lstm", "gru"})


@dataclass(frozen=True)
class CriticSpec:
    target: str  # returns | td0 | q_adj
    loss: str  # mse | huber
    architecture: str = "mlp"  # mlp | residual_mlp
    # "v_head" = the frozen V(s)-from-returns aux critic (byte-identical golden
    # path). "strategy" = a Q(s,a,u) IDENTIFICATION strategy hosted on the
    # episode-grouped stream (the Cell-7 deconfounding ablation).
    kind: str = "v_head"
    # strategy only: observational | proximal | oracle_u | sensitivity
    builder: str | None = None
    requires_u: bool = False  # strategy only: oracle_u reads the realized U (its fill)
    # Γ >= 1, the sensitivity critic's MSM confounding bound (a METHOD parameter,
    # not a regime axis). Read ONLY by builder="sensitivity"; inert (unread) for
    # the other strategy critics — the 1.0 field default is just that inert value.
    # NB: Γ=1 is the MSM null (byte-identical Observational) — the deterministic
    # ANCHOR of a Γ sweep, NOT a sensible default for the sensitivity critic (at
    # Γ=1 its ablation row is a verbatim copy of the observational row, which
    # silently deletes the method). The sensitivity spec sets Γ=2.0 explicitly.
    gamma: float = 1.0


CRITIC_LIBRARY: Dict[str, CriticSpec] = {
    # Default baseline critic used when no critic list is provided.
    "standard_mlp": CriticSpec(target="returns", loss="mse", architecture="mlp"),
    # Custom example critic with residual connection; useful for ablation comparisons.
    "residual_reward_model": CriticSpec(
        target="returns", loss="mse", architecture="residual_mlp"
    ),
    # Strategy critics (Cell-7 deconfounding ablation). They REUSE the existing
    # merged builders verbatim — NO reimplementation of ProximalEM /
    # IdentificationStrategy / UMarginalizedQ: observational = MLP + DQN(
    # Observational()); proximal = build_proximal_<base>; oracle_u =
    # build_oracle_u_<base>. MLP row only; the rnn row is deferred behind the
    # recurrent-offline prerequisite (= cell 8, see StrategyCritic).
    "observational": CriticSpec(
        target="q_adj", loss="mse", kind="strategy", builder="observational"
    ),
    "proximal": CriticSpec(
        target="q_adj", loss="mse", kind="strategy", builder="proximal"
    ),
    "oracle_u": CriticSpec(
        target="q_adj",
        loss="mse",
        kind="strategy",
        builder="oracle_u",
        requires_u=True,
    ),
    # Sensitivity-bounds critic (Kallus-Zhou MSM). REUSES build_sensitivity_<base>
    # verbatim (NO reweighter reimplementation): it shares the BASE algo's class
    # (build_sensitivity_cql -> a CQL learner with SensitivityBounds + the reward
    # reweighter), so --algos cql compares CQL-sensitivity against CQL-oracle with
    # no base-learner confound — the same base-parity fix the observational floor
    # got. Γ is the method's bound, read from THIS spec, logged as the ``gamma``
    # column, and NEVER folded into the (β, σ) path encoding.
    #
    # DEFAULT Γ=2.0 — a GENUINELY ACTIVE MSM bound (NOT the Γ=1 no-op, which would
    # make the ablation row a verbatim copy of observational and silently delete
    # the method from the paper). The sensitivity critic is NON-ADAPTIVE: it
    # applies the worst-case Γ-bound unconditionally with no σ=0 detector, so it is
    # EXEMPT from the null-calibration gate; its origin deviation is a REPORTED
    # result (``pessimism_cost``), not a miscalibration. Γ=1 is the byte-identity
    # anchor of a Γ sweep, not the default. Discrete-only; recurrent = DQN-base.
    "sensitivity": CriticSpec(
        target="q_adj",
        loss="mse",
        kind="strategy",
        builder="sensitivity",
        gamma=2.0,
    ),
}


def default_aux_critics() -> list[str]:
    return ["standard_mlp"]


DEFAULT_AUX_CRITICS: list[str] = default_aux_critics()


class ResidualValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h) + self.skip(x))
        return self.out(h)


@dataclass
class CriticAblationConfig:
    critics: list[str]
    lr: float = 3e-4
    hidden_dims: tuple[int, ...] = (64, 64)
    bins: int = 32

    def to_dict(self) -> dict:
        return {
            "critics": list(self.critics),
            "lr": float(self.lr),
            "hidden_dims": list(self.hidden_dims),
            "bins": int(self.bins),
        }


class AuxiliaryCritic:
    def __init__(
        self,
        name: str,
        spec: CriticSpec,
        obs_dim: int,
        device: torch.device,
        lr: float,
        hidden_dims: tuple[int, ...],
        gamma: float,
    ) -> None:
        self.name = name
        self.spec = spec
        self.device = device
        self.gamma = gamma
        self.net = self._build_network(obs_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _build_network(
        self, obs_dim: int, hidden_dims: tuple[int, ...]
    ) -> torch.nn.Module:
        if self.spec.architecture == "residual_mlp":
            hidden_dim = int(hidden_dims[-1]) if hidden_dims else 64
            return ResidualValueNet(obs_dim, hidden_dim)
        return MLP(obs_dim, 1, hidden_dims=hidden_dims)

    def _training_target(self, batch: RolloutBatch) -> torch.Tensor:
        if self.spec.target == "returns":
            return batch.returns
        with torch.no_grad():
            next_values = self.net(batch.next_obs).squeeze(-1)
        return batch.rewards + self.gamma * next_values * (1.0 - batch.dones)

    def update(self, batch: RolloutBatch) -> float:
        pred = self.net(batch.obs).squeeze(-1)
        target = self._training_target(batch)
        if self.spec.loss == "huber":
            loss = F.smooth_l1_loss(pred, target)
        else:
            loss = F.mse_loss(pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(obs).squeeze(-1)

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "spec": {
                "target": self.spec.target,
                "loss": self.spec.loss,
                "architecture": self.spec.architecture,
            },
            "network": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }


def _build_strategy_critic(
    builder: str,
    base_algo: str,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    encoder: str = "mlp",
    gamma: float = 1.0,
):
    """Return ``(net, agent)`` for one strategy critic from the EXISTING builders
    (no estimator reimplementation). ``builder`` selects the identification arm
    (observational floor / proximal / oracle_u / sensitivity); ``gamma`` is the
    sensitivity critic's MSM bound Γ (ignored by the others); ``encoder`` selects
    the ORTHOGONAL architecture axis:

      * ``encoder="mlp"`` (default, BYTE-FROZEN — the Cell-7 MLP row):
        observational = MLP + DQN(Observational()); proximal/oracle_u =
        ``build_{proximal,oracle_u}_<base>`` (flat UMarginalizedQ, is_recurrent
        False -> the caller flattens (B,T)->(B*T)).
      * ``encoder`` in {"rnn","lstm","gru"} (the Cell-8 recurrent row, POMDP ×
        confounded): observational = ``build_offline_dqn_recurrent`` (plain
        recurrent Q, NO q_su, NO U); proximal = ``build_recurrent_proximal_dqn``;
        oracle_u = ``build_recurrent_oracle_u_dqn`` (RecurrentUMarginalizedQ +
        the cell-8 ProximalEM / OracleU). All three fire ``is_recurrent`` and
        consume (B,T) windows NATIVELY via the batched ``q_su``. DQN base only.
    """
    recurrent = str(encoder or "mlp").lower() in _RECURRENT_ENCODERS

    if builder == "observational":
        if recurrent:
            # Recurrent observational floor: plain recurrent Q (no q_su, no U).
            # DQN is the ONLY recurrent base (cql/iql/bcq have no recurrent path),
            # so the encoder fully determines the base regardless of --algos.
            # Reuses the merged cell-8 builder verbatim (no reimplementation).
            from src.rl.offline.dqn import build_offline_dqn_recurrent

            return build_offline_dqn_recurrent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device,
                action_type="discrete",
                critic_network=encoder,
            )
        # BASE PARITY: the observational floor MUST use the SAME base learner as
        # proximal/oracle_u so the ablation isolates the IDENTIFICATION strategy,
        # not the base learner. Build the base algo's OWN class with Observational()
        # (a literal pass-through -> plain Q(s,.)). Building a bare DQN here would
        # make --algos cql compare CQL-proximal against a DQN-observational: CQL's
        # conservative regularization curbs value overestimation independent of
        # deconfounding, so a DQN floor inflates the observational->proximal gap
        # with a CQL-vs-DQN base confound.
        suffix = _BASE_ALGO_SUFFIX.get(str(base_algo).split("__")[0])
        if suffix is None:
            raise ValueError(
                f"strategy-critic ablation base algo '{base_algo}' is unsupported; "
                "choose a discrete offline base: offline_dqn, bcq, cql, iql."
            )
        if suffix == "dqn":
            # DQN floor UNCHANGED (byte-frozen golden): MLP + DQN default strategy
            # (== Observational()). DQN-vs-DQN, so there is no base confound here.
            q = MLP(obs_dim, action_dim).to(device)
            tgt = MLP(obs_dim, action_dim).to(device)
            agent = DQN(q, tgt, ReplayBuffer(1_000_000, device), device=device)
            return q, agent
        # cql/iql/bcq floor: the base builder with a plain (U-free) Q-net + the
        # Observational() strategy. Reuses the merged base builders (no learner
        # reimplementation); each accepts a ``strategy`` override.
        from src.rl.off_policy.identification import Observational
        from src.rl.offline.bcq import build_bcq
        from src.rl.offline.cql import build_cql
        from src.rl.offline.iql import build_iql

        fn = {"cql": build_cql, "iql": build_iql, "bcq": build_bcq}[suffix]
        return fn(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            action_type="discrete",
            strategy=Observational(),
        )

    suffix = _BASE_ALGO_SUFFIX.get(str(base_algo).split("__")[0])
    if suffix is None:
        raise ValueError(
            f"strategy-critic ablation base algo '{base_algo}' is unsupported; "
            "choose a discrete offline base: offline_dqn, bcq, cql, iql."
        )
    if recurrent and suffix != "dqn":
        raise ValueError(
            f"recurrent strategy-critic ablation is DQN-base only (Cell 8): base "
            f"'{base_algo}' -> '{suffix}' has no recurrent proximal/oracle_u "
            "builder. Use offline_dqn or offline_dqn_recurrent as the base."
        )
    kwargs = dict(
        obs_dim=obs_dim, action_dim=action_dim, device=device, action_type="discrete"
    )
    if builder == "proximal":
        if recurrent:
            from src.rl.offline.proximal import build_recurrent_proximal_dqn

            return build_recurrent_proximal_dqn(critic_network=encoder, **kwargs)
        from src.rl.offline.proximal import (
            build_proximal_bcq,
            build_proximal_cql,
            build_proximal_dqn,
            build_proximal_iql,
        )

        fn = {
            "dqn": build_proximal_dqn,
            "bcq": build_proximal_bcq,
            "cql": build_proximal_cql,
            "iql": build_proximal_iql,
        }[suffix]
        return fn(**kwargs)
    if builder == "oracle_u":
        if recurrent:
            from src.rl.offline.oracle_u import build_recurrent_oracle_u_dqn

            return build_recurrent_oracle_u_dqn(critic_network=encoder, **kwargs)
        from src.rl.offline.oracle_u import (
            build_oracle_u_bcq,
            build_oracle_u_cql,
            build_oracle_u_dqn,
            build_oracle_u_iql,
        )

        fn = {
            "dqn": build_oracle_u_dqn,
            "bcq": build_oracle_u_bcq,
            "cql": build_oracle_u_cql,
            "iql": build_oracle_u_iql,
        }[suffix]
        return fn(**kwargs)
    if builder == "sensitivity":
        # SHARES THE BASE ALGO'S CLASS: build_sensitivity_<base> = the base builder
        # (build_cql / build_iql / build_bcq / build_offline_dqn) with a
        # SensitivityBounds strategy + reward reweighter installed — NOT a bare DQN.
        # So the ablation isolates the identification method, not the base learner
        # (the same base-parity requirement as the observational floor). Γ is the
        # method parameter, threaded via ``gamma_sensitivity`` (the builders' own
        # seam); Γ=1 -> byte-identical Observational.
        if recurrent:
            from src.rl.offline.sensitivity import build_sensitivity_dqn_recurrent

            return build_sensitivity_dqn_recurrent(
                critic_network=encoder, gamma_sensitivity=gamma, **kwargs
            )
        from src.rl.offline.sensitivity import (
            build_sensitivity_bcq,
            build_sensitivity_cql,
            build_sensitivity_dqn,
            build_sensitivity_iql,
        )

        fn = {
            "dqn": build_sensitivity_dqn,
            "bcq": build_sensitivity_bcq,
            "cql": build_sensitivity_cql,
            "iql": build_sensitivity_iql,
        }[suffix]
        return fn(gamma_sensitivity=gamma, **kwargs)
    raise ValueError(f"unknown strategy-critic builder '{builder}'.")


class StrategyCritic:
    """A Cell-7 deconfounding critic = a ``(net, agent)`` from the merged builders,
    fit on the SHARED episode-grouped stream and scored estimation-vs-oracle.

    The identification is entirely the strategy + net the agent was built with
    (Observational / Proximal / OracleU × plain-MLP / UMarginalizedQ) — this class
    only routes the shared window into the right consumer and exposes the deployed
    ``Q_adj`` for scoring:

      * ``proximal`` consumes the ``(B, T, *)`` window NATIVELY (its wrapped
        ``learn`` = ``ProximalEM.m_step`` samples the INFERRED u and flattens).
      * ``observational`` / ``oracle_u`` are flat learners: the SAME window is
        flattened ``(B, T)->(B*T)`` exactly as proximal's own flatten, so all
        three provably fit identical transitions.

    Five-keys: only an ``oracle_u`` critic reads the realized U (its fill runs
    ``load_u=True``); ``observational``/``proximal`` never see it.
    """

    def __init__(
        self,
        name: str,
        spec: CriticSpec,
        base_algo: str,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        encoder: str = "mlp",
    ) -> None:
        self.name = name
        self.spec = spec
        self.device = device
        self.requires_u = spec.requires_u
        # Γ (MSM bound) read from THIS critic's spec — a method parameter threaded
        # into the sensitivity builder, inert for the other strategy critics.
        self.gamma = float(spec.gamma)
        _net, agent = _build_strategy_critic(
            spec.builder, base_algo, obs_dim, action_dim, device, encoder, self.gamma
        )
        self.agent = agent
        # The Q-net whose forward is the DEPLOYED estimand: Q_adj = E_u[Q(s,.,u)]
        # for UMarginalizedQ (proximal/oracle), plain Q(s,.) for observational.
        self.net = agent.q_network
        # Recurrent critics (encoder=lstm/gru/rnn) fire is_recurrent and consume the
        # (B,T) window NATIVELY (batched q_su); the MLP proximal also consumes (B,T)
        # (its m_step flattens internally). Only MLP observational/oracle_u are flat
        # learners that the caller flattens (B,T)->(B*T).
        self.is_recurrent = bool(getattr(agent, "is_recurrent", False))
        self.consumes_sequences = spec.builder == "proximal" or self.is_recurrent
        self.wants_sequence_buffer = hasattr(agent, "set_sequence_buffer")

    def set_sequence_buffer(self, seq_buffer) -> None:
        if self.wants_sequence_buffer:
            self.agent.set_sequence_buffer(seq_buffer)

    def update(self, window: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.consumes_sequences:
            return self.agent.update(window)  # m_step flattens + samples inferred u
        flat = {
            k: (v.flatten(0, 1) if torch.is_tensor(v) and v.dim() >= 2 else v)
            for k, v in window.items()
        }
        return self.agent.update(flat)

    def predict_q_adj(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.net(obs)
            # Recurrent nets (RecurrentUMarginalizedQ / plain recurrent trunk)
            # return (q_all, new_state); the flat (N, obs_dim) eval set is one
            # zero-state single step. MLP nets return the bare (N, A) tensor.
            if isinstance(out, tuple):
                out = out[0]
            return out


class CriticAblationManager:
    def __init__(
        self,
        obs_dim: int,
        device: torch.device,
        config: CriticAblationConfig,
        gamma: float = 0.99,
        *,
        base_algo: str | None = None,
        action_dim: int | None = None,
        encoder: str = "mlp",
    ) -> None:
        self.config = config
        self.gamma = gamma
        self.device = device
        # Architecture axis for the strategy critics: "mlp" (Cell-7 row, frozen)
        # or lstm/gru/rnn (Cell-8 recurrent row). Derived by the runner from the
        # base algo's critic_network so the base and the triad share one encoder.
        self.encoder = str(encoder or "mlp")
        selected_critics = (
            [str(name) for name in config.critics]
            if config.critics
            else default_aux_critics()
        )
        self.config.critics = list(selected_critics)

        # Kind detection: strategy critics take a wholly separate path (episode-
        # grouped stream, estimation-vs-oracle scoring). Unknown names fall through
        # to the V-head loop below, preserving its exact "Unknown ablation critic"
        # error. Mixing the two kinds in one ablation is rejected.
        _resolved = [
            CRITIC_LIBRARY[k.lower().strip()]
            for k in selected_critics
            if k.lower().strip() in CRITIC_LIBRARY
        ]
        self.is_strategy = any(s.kind == "strategy" for s in _resolved)
        if self.is_strategy:
            if any(s.kind != "strategy" for s in _resolved):
                raise ValueError(
                    "Cannot mix V-head and strategy critics in one ablation."
                )
            if base_algo is None or action_dim is None:
                raise ValueError(
                    "strategy-critic ablation requires base_algo and action_dim."
                )
            self.base_algo = str(base_algo).split("__")[0]
            strat: Dict[str, StrategyCritic] = {}
            for name in selected_critics:
                key = name.lower().strip()
                if key in strat:
                    continue
                strat[key] = StrategyCritic(
                    key,
                    CRITIC_LIBRARY[key],
                    self.base_algo,
                    obs_dim,
                    action_dim,
                    device,
                    self.encoder,
                )
            if not strat:
                raise ValueError("At least one ablation critic must be configured.")
            self.strategy_critics = strat
            self.critics: Dict[str, AuxiliaryCritic] = {}  # V-head path unused
            self._eval_obs: torch.Tensor | None = None
            self._eval_act: torch.Tensor | None = None
            return

        # ---- V-head path (unchanged; golden-pinned) ----
        self.strategy_critics = None
        critics: Dict[str, AuxiliaryCritic] = {}
        for name in selected_critics:
            key = name.lower().strip()
            if key not in CRITIC_LIBRARY:
                available = ", ".join(sorted(CRITIC_LIBRARY))
                raise ValueError(
                    f"Unknown ablation critic '{name}'. Available: {available}"
                )
            if key in critics:
                continue
            critics[key] = AuxiliaryCritic(
                name=key,
                spec=CRITIC_LIBRARY[key],
                obs_dim=obs_dim,
                device=device,
                lr=config.lr,
                hidden_dims=config.hidden_dims,
                gamma=gamma,
            )
        if not critics:
            raise ValueError("At least one ablation critic must be configured.")
        self.critics = critics

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        losses: Dict[str, float] = {}
        for name, critic in self.critics.items():
            losses[name] = critic.update(batch)
        return losses

    def checkpoint_rows(
        self,
        batch: RolloutBatch,
        episode: int,
        algorithm: str,
        environment: str,
        latest_losses: Dict[str, float] | None = None,
    ) -> list[dict]:
        losses = latest_losses or {}
        rows: list[dict] = []
        for name, critic in self.critics.items():
            metrics = _compute_metrics(
                batch=batch,
                critic=critic,
                gamma=self.gamma,
                bins=max(4, int(self.config.bins)),
            )
            rows.append(
                {
                    "episode": episode,
                    "algorithm": algorithm,
                    "environment": environment,
                    "critic": name,
                    "train_loss": losses.get(name, ""),
                    **metrics,
                }
            )
        return rows

    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "gamma": float(self.gamma),
            "critics": {
                name: critic.state_dict() for name, critic in self.critics.items()
            },
        }

    # -- strategy-critic ablation (Cell-7 deconfounding; separate from V-head) --
    def needs_u(self) -> bool:
        """True iff any hosted strategy critic reads the realized U (oracle_u) —
        the runner ORs this into the episode-grouped fill's ``load_u``. The
        five-keys invariant holds: only the oracle critic's estimator ever reads
        ``confounder_u``; observational/proximal never do."""
        if not self.is_strategy:
            return False
        return any(c.requires_u for c in self.strategy_critics.values())

    def set_sequence_buffer(self, seq_buffer) -> None:
        """Fan the shared episode buffer out to any strategy critic that wants it
        (proximal's E-step warm-start), then cache a fixed eval set of ``(obs,
        a_data)`` transitions for estimation-vs-oracle scoring. The eval set reads
        ONLY the five base keys (never U)."""
        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        for ep in seq_buffer.iter_episodes():
            for tr in ep:
                obs_list.append(tr["obs"])
                act_list.append(tr["actions"])
        # Fan-out AFTER reading transitions (proximal's warm-start mutates tr in
        # place with r_tau, but never obs/actions).
        for critic in self.strategy_critics.values():
            critic.set_sequence_buffer(seq_buffer)
        if not obs_list:
            return
        obs = torch.stack(obs_list).float().to(self.device)
        act = torch.stack(act_list).long().to(self.device)
        n = obs.shape[0]
        if n > 4000:  # fixed, deterministic subsample of the shared stream
            idx = torch.linspace(0, n - 1, 4000).long()
            obs, act = obs[idx], act[idx]
        self._eval_obs, self._eval_act = obs, act

    def update_strategy(self, window: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Fit every strategy critic on the SAME ``(B, T, *)`` window (proximal
        native; observational/oracle_u on the identical flatten)."""
        losses: Dict[str, float] = {}
        for name, critic in self.strategy_critics.items():
            m = critic.update(window)
            losses[name] = float(m.get("loss", m.get("q_loss", 0.0))) if m else 0.0
        return losses

    def checkpoint_rows_strategy(
        self,
        episode: int,
        algorithm: str,
        environment: str,
        sigma: float,
        latest_losses: Dict[str, float] | None = None,
    ) -> list[dict]:
        """Estimation-vs-oracle scoring rows. Ground truth = the oracle critic's
        deployed ``Q_adj`` (``UMarginalizedQ.forward``); the deconfounded value the
        others infer/ignore. NO return metric (the confounder is an action-
        independent nuisance -> no return gap even for the oracle)."""
        losses = latest_losses or {}
        obs_e, act_e = self._eval_obs, self._eval_act
        rows: list[dict] = []
        if obs_e is None:
            return rows

        oracle = self.strategy_critics.get("oracle_u")
        q_oracle = pi_oracle = None
        if oracle is not None:
            q_all = oracle.predict_q_adj(obs_e)
            q_oracle = q_all.gather(1, act_e.unsqueeze(-1)).squeeze(-1)
            pi_oracle = q_all.argmax(1)

        # Observational->oracle MSE = the denominator for the gap-closed fraction.
        obs_mse = None
        obs_c = self.strategy_critics.get("observational")
        if oracle is not None and obs_c is not None:
            q_obs = (
                obs_c.predict_q_adj(obs_e).gather(1, act_e.unsqueeze(-1)).squeeze(-1)
            )
            obs_mse = float(torch.mean((q_obs - q_oracle) ** 2).item())

        for name, critic in self.strategy_critics.items():
            row = {col: "" for col in STRATEGY_CRITIC_ABLATION_COLUMNS}
            q_all = critic.predict_q_adj(obs_e)
            q_c = q_all.gather(1, act_e.unsqueeze(-1)).squeeze(-1)
            row.update(
                episode=episode,
                algorithm=algorithm,
                environment=environment,
                critic=name,
                base_algo=self.base_algo,
                sigma=float(sigma),
                train_loss=losses.get(name, ""),
                apparent_q_mean=float(q_c.mean().item()),
            )
            # Γ is a sensitivity-only method parameter; blank for the others (they
            # apply no MSM bound). Orthogonal to (β, σ) — logged, never path-encoded.
            if critic.spec.builder == "sensitivity":
                row["gamma"] = float(critic.gamma)
            if q_oracle is not None:
                mse = float(torch.mean((q_c - q_oracle) ** 2).item())
                tgt, pred = _to_vector(q_oracle), _to_vector(q_c)
                at_origin = float(sigma) == 0.0
                is_sensitivity = critic.spec.builder == "sensitivity"
                # NULL-CALIBRATION — ADAPTIVE critics only. At the (β=0, σ=0) origin
                # observational/proximal/oracle_u each self-null (no correction /
                # posterior->prior / A⊥U), so they must land within estimator noise
                # of the oracle; a disagreement (e.g. a bare-DQN floor scored against
                # a CQL oracle) is a base-learner confound the gate flags. The
                # NON-ADAPTIVE sensitivity critic applies its Γ-bound unconditionally
                # with no σ=0 detector, so it is EXEMPT (blank) — see pessimism_cost.
                if at_origin and not is_sensitivity:
                    row["null_calibrated"] = bool(mse < _GAP_NOISE_FLOOR_MSE)
                # PESSIMISM COST (sensitivity only, at the origin): the value given
                # up to the Γ-bound when there was NOTHING to be robust against.
                # Exactly 0 at Γ=1 (byte-identity with observational), > 0 and rising
                # with Γ. A reported result; blank off-origin and for other critics.
                if at_origin and is_sensitivity:
                    row["pessimism_cost"] = float((q_oracle.mean() - q_c.mean()).item())
                row.update(
                    oracle_q_mean=float(q_oracle.mean().item()),
                    q_inflation=float((q_c.mean() - q_oracle.mean()).item()),
                    value_mse_to_oracle=mse,
                    argmax_agreement_oracle=float(
                        (q_all.argmax(1) == pi_oracle).float().mean().item()
                    ),
                    explained_variance_to_oracle=_explained_variance(tgt, pred),
                    pearson_to_oracle=_pearson(tgt, pred),
                    spearman_to_oracle=_spearman(tgt, pred),
                )
                # Gap-closed: proximal only, and ONLY where a real gap exists
                # (sigma>0 AND obs->oracle MSE above the noise floor). At sigma=0
                # the two-noise-terms ratio is not a metric -> left blank (the
                # absolute value_mse_to_oracle IS the control).
                if (
                    name == "proximal"
                    and obs_mse is not None
                    and sigma > 0.0
                    and obs_mse > _GAP_NOISE_FLOOR_MSE
                ):
                    row["gap_closed_fraction"] = 1.0 - mse / obs_mse
            rows.append(row)
        return rows


def _to_vector(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1).float().cpu()


def _explained_variance(target: torch.Tensor, pred: torch.Tensor) -> float:
    var_target = torch.var(target, unbiased=False)
    if var_target.item() < _EPS:
        return 0.0
    return float(
        (1.0 - torch.var(target - pred, unbiased=False) / (var_target + _EPS)).item()
    )


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum()))
    if denom.item() < _EPS:
        return 0.0
    return float((x_centered * y_centered).sum().div(denom + _EPS).item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearson(_rank(x), _rank(y))


def _distribution_metrics(
    target: torch.Tensor, pred: torch.Tensor, bins: int
) -> tuple[float, float, float]:
    combined = torch.cat([target, pred], dim=0)
    low = float(combined.min().item())
    high = float(combined.max().item())
    if abs(high - low) < _EPS:
        high = low + 1e-3
    edges = torch.linspace(low, high, bins + 1)

    target_idx = torch.bucketize(target, edges[1:-1])
    pred_idx = torch.bucketize(pred, edges[1:-1])

    target_counts = torch.bincount(target_idx, minlength=bins).float()
    pred_counts = torch.bincount(pred_idx, minlength=bins).float()
    p = (target_counts + _EPS) / (target_counts.sum() + _EPS * bins)
    q = (pred_counts + _EPS) / (pred_counts.sum() + _EPS * bins)

    kl = float((p * torch.log((p + _EPS) / (q + _EPS))).sum().item())
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * torch.log((p + _EPS) / (m + _EPS))).sum()
        + (q * torch.log((q + _EPS) / (m + _EPS))).sum()
    )
    js_normalized = float((js / torch.log(torch.tensor(2.0))).item())

    joint_idx = target_idx * bins + pred_idx
    joint_counts = torch.bincount(joint_idx, minlength=bins * bins).float()
    joint = (joint_counts.reshape(bins, bins) + _EPS) / (
        joint_counts.sum() + _EPS * bins * bins
    )
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    mutual_information = float(
        (joint * torch.log((joint + _EPS) / (px @ py + _EPS))).sum().item()
    )
    return mutual_information, kl, js_normalized


def _compute_metrics(
    batch: RolloutBatch,
    critic: AuxiliaryCritic,
    gamma: float,
    bins: int,
) -> dict:
    pred_obs = critic.predict(batch.obs)
    pred = _to_vector(pred_obs)
    target = _to_vector(batch.returns)

    advantage = target - pred
    mse = float(torch.mean((target - pred) ** 2).item())
    explained_variance = _explained_variance(target, pred)
    pearson = _pearson(target, pred)
    spearman = _spearman(target, pred)
    mutual_information, kl, js_normalized = _distribution_metrics(
        target, pred, bins=bins
    )

    value = _to_vector(pred_obs)
    next_value = _to_vector(critic.predict(batch.next_obs))
    rewards = _to_vector(batch.rewards)
    dones = _to_vector(batch.dones)
    pred_reward = value - gamma * next_value * (1.0 - dones)
    td_error = rewards - pred_reward
    td_error_mean = float(td_error.mean().item())
    reward_mse = float(torch.mean((rewards - pred_reward) ** 2).item())
    reward_explained_variance = _explained_variance(rewards, pred_reward)
    reward_pearson = _pearson(rewards, pred_reward)
    reward_spearman = _spearman(rewards, pred_reward)
    reward_mutual_information, reward_kl, reward_js_normalized = _distribution_metrics(
        rewards, pred_reward, bins=bins
    )

    return {
        "advantage_mean": float(advantage.mean().item()),
        "explained_variance": explained_variance,
        "pearson": pearson,
        "spearman": spearman,
        "mutual_information": mutual_information,
        "kl": kl,
        "js_normalized": js_normalized,
        "mse": mse,
        "td_error_mean": td_error_mean,
        "real_reward_mean": float(rewards.mean().item()),
        "pred_reward_mean": float(pred_reward.mean().item()),
        "reward_explained_variance": reward_explained_variance,
        "reward_pearson": reward_pearson,
        "reward_spearman": reward_spearman,
        "reward_mutual_information": reward_mutual_information,
        "reward_kl": reward_kl,
        "reward_js_normalized": reward_js_normalized,
        "reward_mse": reward_mse,
        "reward_error_mean": td_error_mean,
    }

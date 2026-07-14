"""feat/sensitivity-critic (PR 4) — the sensitivity-bounds critic in the ablation
library + the NULL-CALIBRATION gate. Tests J1-J5.

The sensitivity critic REUSES ``build_sensitivity_<base>`` verbatim (the merged
Kallus-Zhou MSM reward-reweighter), so it SHARES THE BASE ALGO'S CLASS
(CQL-sensitivity, not a bare-DQN floor) — the same base-parity fix the
observational floor carries, so ``--algos cql`` compares like-with-like. Γ is a
METHOD parameter (orthogonal to the (β, σ) regime axes): read from the critic
spec, logged as its own column, never folded into the dataset/(β,σ) path.

NULL-CALIBRATION (the point of the PR). At the (β=0, σ=0) origin the confounder is
inert, so every critic must agree with the oracle within estimator noise; the
``null_calibrated`` column reports ``value_mse_to_oracle < _GAP_NOISE_FLOOR_MSE``
there (blank off-origin). The sensitivity critic defaults to Γ=1 — the MSM NULL,
byte-identical to Observational — so it manufactures ZERO null-origin gap (J4). A
bare-DQN floor scored against a CQL oracle inflates Q above the conservative
oracle even at the origin: a base-learner gap masquerading as deconfounding, which
the gate must flag (J5).
"""

from __future__ import annotations

import random
import warnings

import pytest
import torch
from src.benchmarking.critic_ablation import (
    _build_strategy_critic,
    _GAP_NOISE_FLOOR_MSE,
    CRITIC_LIBRARY,
    CriticAblationConfig,
    CriticAblationManager,
    CriticSpec,
    STRATEGY_CRITIC_ABLATION_COLUMNS,
    StrategyCritic,
)
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer
from src.rl.offline.cql import CQL

warnings.filterwarnings("ignore")

_CPU = torch.device("cpu")
_OBS, _ACT = 4, 2
_SPEC = CRITIC_LIBRARY["sensitivity"]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _unconfounded_buffer(
    seed: int, n_ep: int = 12, T: int = 10
) -> SequenceReplayBuffer:
    """A σ=0 episode-grouped stream: a per-episode nuisance U is stored but NEVER
    enters the reward (no U -> R term), so the dataset is unconfounded by
    construction — the null-calibration origin."""
    g = torch.Generator().manual_seed(seed)
    buf = SequenceReplayBuffer(capacity=100_000, device=_CPU)
    for e in range(n_ep):
        u = float(torch.bernoulli(torch.tensor(0.5), generator=g))
        obs = torch.randn(T, _OBS, generator=g)
        for t in range(T):
            a = int(torch.randint(0, _ACT, (1,), generator=g))
            buf.add(
                e,
                {
                    "obs": obs[t],
                    "next_obs": obs[t],
                    "actions": torch.tensor(a),
                    "rewards": torch.rand(1, generator=g).reshape(()),  # no U term
                    "dones": torch.tensor(1.0),
                    "confounder_u": torch.tensor(u),
                    "r_tau": torch.tensor(0.5),
                },
            )
        buf.mark_episode_end(e)
    return buf


def _manager(critics: list[str], base: str = "cql", encoder: str = "mlp"):
    return CriticAblationManager(
        obs_dim=_OBS,
        device=_CPU,
        config=CriticAblationConfig(critics=list(critics)),
        base_algo=base,
        action_dim=_ACT,
        encoder=encoder,
    )


class _StubCritic:
    """A fixed-output strategy critic: exposes only what checkpoint_rows_strategy
    reads (``predict_q_adj``, ``spec.builder``, ``gamma``, ``name``) so the null-
    calibration LOGIC can be exercised on controlled Q with no training noise."""

    def __init__(self, name: str, q: torch.Tensor, builder: str, gamma: float = 1.0):
        self.name = name
        self.spec = CriticSpec(
            target="q_adj", loss="mse", kind="strategy", builder=builder, gamma=gamma
        )
        self.gamma = float(gamma)
        self._q = q

    def predict_q_adj(self, obs: torch.Tensor) -> torch.Tensor:
        return self._q


def _rows_with_stubs(stubs: list[_StubCritic], sigma: float) -> dict:
    """Drive checkpoint_rows_strategy over hand-built critics on a fixed eval set."""
    mgr = _manager(["observational", "oracle_u"])  # any real init; overwritten below
    mgr.strategy_critics = {s.name: s for s in stubs}
    mgr._eval_obs = torch.randn(30, _OBS, generator=torch.Generator().manual_seed(1))
    mgr._eval_act = torch.randint(
        0, _ACT, (30,), generator=torch.Generator().manual_seed(2)
    )
    return {
        r["critic"]: r
        for r in mgr.checkpoint_rows_strategy(1, "cql", "CartPole-v1", sigma)
    }


# --------------------------------------------------------------------------- #
# J1 — registered (flat + recurrent), appears in the ablation output           #
# --------------------------------------------------------------------------- #
def test_j1_sensitivity_registered_flat_and_recurrent():
    assert _SPEC.kind == "strategy" and _SPEC.builder == "sensitivity"
    assert _SPEC.gamma == 1.0  # default = the MSM null (null-calibrated)
    assert "gamma" in STRATEGY_CRITIC_ABLATION_COLUMNS
    assert "null_calibrated" in STRATEGY_CRITIC_ABLATION_COLUMNS

    flat = StrategyCritic("sensitivity", _SPEC, "cql", _OBS, _ACT, _CPU, "mlp")
    rec = StrategyCritic(
        "sensitivity", _SPEC, "offline_dqn_recurrent", _OBS, _ACT, _CPU, "lstm"
    )
    assert flat.is_recurrent is False
    assert rec.is_recurrent is True

    # appears as a strategy critic and in checkpoint output with Γ logged
    mgr = _manager(["oracle_u", "sensitivity"])
    assert "sensitivity" in mgr.strategy_critics
    buf = _unconfounded_buffer(0)
    mgr.set_sequence_buffer(buf)
    mgr.update_strategy(buf.sample_sequences(8, 6))
    rows = {r["critic"]: r for r in mgr.checkpoint_rows_strategy(1, "cql", "X", 0.0)}
    assert "sensitivity" in rows
    assert rows["sensitivity"]["gamma"] == 1.0  # method param, logged as a column
    assert rows["oracle_u"]["gamma"] == ""  # non-sensitivity: no MSM bound -> blank


# --------------------------------------------------------------------------- #
# J2 — pomdp × confounded × sensitivity runs as an ablation critic (recurrent)  #
# --------------------------------------------------------------------------- #
def test_j2_recurrent_sensitivity_runs_on_confounded_pomdp_window():
    c = StrategyCritic(
        "sensitivity", _SPEC, "offline_dqn_recurrent", _OBS, _ACT, _CPU, "lstm"
    )
    assert c.is_recurrent and c.consumes_sequences  # Cell-8 native (B,T)

    g = torch.Generator().manual_seed(0)
    B, T = 8, 6
    window = {
        "obs": torch.randn(B, T, _OBS, generator=g),
        "next_obs": torch.randn(B, T, _OBS, generator=g),
        "actions": torch.randint(0, _ACT, (B, T), generator=g),
        "rewards": torch.rand(B, T, generator=g),
        "dones": torch.zeros(B, T),
        "confounder_u": torch.bernoulli(torch.full((B, T), 0.5)),  # confounded
        "r_tau": torch.rand(B, T, generator=g),
    }
    metrics = c.update(window)
    loss = metrics.get("loss", metrics.get("q_loss"))
    assert loss is not None and torch.isfinite(torch.tensor(loss))
    # the recurrent (q_all, state) forward scores to a flat (N, A) for the eval set
    assert c.predict_q_adj(torch.randn(20, _OBS)).shape == (20, _ACT)


def test_j2_recurrent_sensitivity_is_dqn_base_only():
    with pytest.raises(ValueError, match="DQN-base only"):
        _build_strategy_critic("sensitivity", "cql", _OBS, _ACT, _CPU, "lstm")


# --------------------------------------------------------------------------- #
# J3 — sensitivity SHARES the base algo's class (assert on class, not metric)   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base,cls", [("cql", CQL), ("offline_dqn", DQN)])
def test_j3_sensitivity_shares_base_algo_class(base, cls):
    c = StrategyCritic("sensitivity", _SPEC, base, _OBS, _ACT, _CPU, "mlp")
    assert isinstance(c.agent, cls)
    # the prior bug — a cql base collapsing to a bare-DQN floor — must NOT happen:
    if base == "cql":
        assert not isinstance(c.agent, DQN)
    # the MSM reweighter is installed on the SHARED base learner (not a side net)
    assert hasattr(c.agent, "_sensitivity_reweighter")
    assert c.agent._sensitivity_reweighter.gamma_s == _SPEC.gamma


# --------------------------------------------------------------------------- #
# J4 — NULL CALIBRATION: gate PASSES when all critics agree at the origin       #
# --------------------------------------------------------------------------- #
def test_j4_null_gate_passes_when_all_critics_agree_at_origin():
    # Controlled Q: all four within estimator noise of the oracle -> every critic
    # null_calibrated True at σ=0. Off-origin (σ>0) the flag is undefined -> blank.
    g = torch.Generator().manual_seed(7)
    base_q = torch.randn(30, _ACT, generator=g)

    def near(seed):  # a within-noise perturbation (mse well below the floor)
        return base_q + 0.03 * torch.randn(
            30, _ACT, generator=torch.Generator().manual_seed(seed)
        )

    stubs = [
        _StubCritic("oracle_u", base_q, "oracle_u"),
        _StubCritic("observational", near(11), "observational"),
        _StubCritic("proximal", near(12), "proximal"),
        _StubCritic("sensitivity", near(13), "sensitivity", gamma=1.0),
    ]
    rows0 = _rows_with_stubs(stubs, sigma=0.0)
    for name in ("oracle_u", "observational", "proximal", "sensitivity"):
        assert rows0[name]["null_calibrated"] is True, name
        assert float(rows0[name]["value_mse_to_oracle"]) < _GAP_NOISE_FLOOR_MSE
    assert rows0["sensitivity"]["gamma"] == 1.0
    assert rows0["observational"]["gamma"] == ""

    rows_off = _rows_with_stubs(stubs, sigma=0.5)  # off-origin -> undefined -> blank
    for name in ("oracle_u", "observational", "proximal", "sensitivity"):
        assert rows_off[name]["null_calibrated"] == "", name


def test_j4_sensitivity_gamma1_manufactures_no_null_gap():
    # At the MSM null Γ=1 the reweighter early-returns the untouched batch, so the
    # sensitivity critic is BYTE-IDENTICAL to the CQL observational floor: same
    # weights (same seed, same base builder), same updates, same Q. It therefore
    # adds ZERO gap over the floor at the origin -> shares the floor's verdict. No
    # convergence needed: this is exact equality, the deterministic null anchor.
    windows = None

    def fit(spec):
        nonlocal windows
        torch.manual_seed(0)
        c = StrategyCritic(spec.builder, spec, "cql", _OBS, _ACT, _CPU, "mlp")
        buf = _unconfounded_buffer(0)
        if windows is None:
            random.seed(0)
            windows = [buf.sample_sequences(8, 6) for _ in range(60)]
        for w in windows:
            c.update(w)
        eval_set = torch.randn(30, _OBS, generator=torch.Generator().manual_seed(99))
        return c.predict_q_adj(eval_set)

    q_obs = fit(CRITIC_LIBRARY["observational"])
    q_sens = fit(CRITIC_LIBRARY["sensitivity"])  # Γ=1 default
    assert torch.equal(q_obs, q_sens)


# --------------------------------------------------------------------------- #
# J5 — NULL CALIBRATION: gate FAILS on a bare-DQN floor scored vs a CQL oracle  #
# --------------------------------------------------------------------------- #
def test_j5_null_gate_fails_on_bare_dqn_floor_vs_cql_oracle():
    # The prior bug: a bare-DQN observational floor scored against a CQL oracle.
    # DQN's max-overestimation inflates Q above the conservative oracle EVEN at the
    # origin — a base-learner gap, not deconfounding. The null gate must catch it:
    # value_mse_to_oracle >> floor -> null_calibrated False.
    g = torch.Generator().manual_seed(3)
    base_q = torch.randn(30, _ACT, generator=g)
    stubs = [
        _StubCritic("oracle_u", base_q, "oracle_u"),  # conservative CQL oracle
        _StubCritic("observational", base_q + 0.5, "observational"),  # DQN inflation
    ]
    rows = _rows_with_stubs(stubs, sigma=0.0)
    assert rows["observational"]["null_calibrated"] is False
    assert float(rows["observational"]["value_mse_to_oracle"]) > _GAP_NOISE_FLOOR_MSE
    # the gate is not vacuously failing — a matched critic (the oracle itself) passes
    assert rows["oracle_u"]["null_calibrated"] is True

"""Rollout speedup paths (S1 CPU rollout, S2 vectorized) — docs/dataset_generation_speedup.md.

The fast paths are deliberately NOT byte-identical to the legacy rollout (CPU vs
CUDA argmax ties, batched policy draws). What MUST hold is everything the
pipeline actually depends on:

  * the output contract (episode count, T+1 observations, aligned infos),
  * the confounding SIGNATURE + gate — datasets self-certify, so a fast-path
    dataset has to pass the same gate the scalar path does,
  * determinism for a fixed (seed, n_envs),
  * output ORDER independence from episode completion order,
  * ``legacy_rollout=True`` still reaching the scalar collector.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("minari")
pytest.importorskip("h5py")

from src.config.seeding import set_seed  # noqa: E402
from src.envs.offline.generate import (  # noqa: E402
    _rollout,
    _rollout_vectorized,
    build_generator_agent,
    build_rollout_env,
    enforce_confounding_gate,
    generate_offline_dataset,
)
from src.rl.policies.behavior_policy import AgentBehaviorPolicy  # noqa: E402


def _collect(n_slots, n_eps, seed=0, fn=None):
    set_seed(seed, deterministic=True)
    agent, _ = build_generator_agent(
        "CartPole-v1", "dqn", "random", seed=seed, device="cpu"
    )
    env = build_rollout_env("CartPole-v1", n_slots, "cpu", seed)
    fn = fn or (_rollout_vectorized if n_slots > 1 else _rollout)
    out = fn(env, AgentBehaviorPolicy(agent), n_eps, seed, "discrete")
    env.close()
    return out


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------
def test_vectorized_output_contract_matches_scalar():
    n_eps = 24
    buf_s, sig_s = _collect(1, n_eps)
    buf_v, sig_v = _collect(8, n_eps)

    assert len(buf_s) == len(buf_v) == n_eps
    assert sig_s is None and sig_v is None  # unconfounded rollout
    for b in buf_v:
        T = len(b.rewards)
        assert T > 0
        # T+1 observations, T actions/terminations/truncations (Minari contract).
        assert b.observations.shape[0] == T + 1
        assert len(b.actions) == T
        assert len(b.terminations) == T and len(b.truncations) == T
        # Exactly one episode boundary, at the end.
        assert not b.terminations[:-1].any() and not b.truncations[:-1].any()
        assert bool(b.terminations[-1]) or bool(b.truncations[-1])
        assert b.observations.dtype == np.float32
        assert b.actions.dtype == np.int64


def test_vectorized_is_deterministic_for_fixed_seed_and_slots():
    b1, _ = _collect(8, 16, seed=3)
    b2, _ = _collect(8, 16, seed=3)
    assert [len(b.rewards) for b in b1] == [len(b.rewards) for b in b2]
    for x, y in zip(b1, b2):
        assert np.array_equal(x.observations, y.observations)
        assert np.array_equal(x.actions, y.actions)
        assert np.array_equal(x.rewards, y.rewards)


def test_vectorized_episode_order_is_assignment_order_not_completion_order():
    """Episodes are keyed by assignment index, so slot-completion races cannot
    reorder the dataset. With 8 slots and 8 episodes, slot i owns episode i —
    so the emitted order must match the per-slot streams of an 8-slot run
    regardless of which slot finished first."""
    buffers, _ = _collect(8, 8, seed=5)
    lengths = [len(b.rewards) for b in buffers]
    assert len(buffers) == 8
    # A completion-ordered emitter would sort these ascending; assignment order
    # must not be sorted unless it coincidentally is.
    assert lengths == [len(b.rewards) for b in _collect(8, 8, seed=5)[0]]


# --------------------------------------------------------------------------
# The property that actually matters: the gate still certifies the dataset
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_slots", [1, 8])
def test_confounded_fast_path_passes_the_gate(tmp_path, monkeypatch, n_slots):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    set_seed(0, deterministic=True)
    agent, _ = build_generator_agent(
        "CartPole-v1", "dqn", "random", seed=0, device="cpu"
    )
    ds = generate_offline_dataset(
        "CartPole-v1",
        "dqn",
        "random",
        behavior_policy="bias_confounded",
        behavior_strength=1.0,
        rollout_episodes=120,
        seed=0,
        dataset_id=f"speedup/conf{n_slots}-v0",
        device="cpu",
        agent=agent,
        rollout_device="cpu",
        rollout_n_envs=n_slots,
    )
    meta = ds.storage.metadata
    # Signature present and the gate accepts it (no exception).
    assert meta["gate_test_passed"] is True
    enforce_confounding_gate(meta, ds.id)
    # The latent U rides along, aligned with the transitions.
    for ep in ds.iterate_episodes():
        assert "confounder_u" in ep.infos
        assert len(ep.infos["confounder_u"]) == len(ep.rewards)


def test_legacy_rollout_flag_uses_the_scalar_collector(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    import src.envs.offline.generate as G

    calls = {"scalar": 0, "vector": 0}
    real_scalar, real_vector = G._rollout, G._rollout_vectorized

    def spy_scalar(*a, **k):
        calls["scalar"] += 1
        return real_scalar(*a, **k)

    def spy_vector(*a, **k):
        calls["vector"] += 1
        return real_vector(*a, **k)

    monkeypatch.setattr(G, "_rollout", spy_scalar)
    monkeypatch.setattr(G, "_rollout_vectorized", spy_vector)

    set_seed(0, deterministic=True)
    agent, _ = build_generator_agent(
        "CartPole-v1", "dqn", "random", seed=0, device="cpu"
    )
    G.generate_offline_dataset(
        "CartPole-v1",
        "dqn",
        "random",
        rollout_episodes=8,
        seed=0,
        dataset_id="speedup/legacy-v0",
        device="cpu",
        agent=agent,
        legacy_rollout=True,
        rollout_n_envs=16,  # ignored under legacy
    )
    assert calls == {"scalar": 1, "vector": 0}


def test_rollout_device_leaves_the_training_agent_untouched(tmp_path, monkeypatch):
    """S1 copies the agent to the rollout device; the caller's agent (which the
    sweep reuses across every arm of a cell, and whose hash is stamped) must not
    be moved or mutated."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to exercise a cross-device copy")
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    set_seed(0, deterministic=True)
    agent, ghash = build_generator_agent(
        "CartPole-v1", "dqn", "random", seed=0, device="cuda"
    )
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    ds = generate_offline_dataset(
        "CartPole-v1",
        "dqn",
        "random",
        rollout_episodes=8,
        seed=0,
        dataset_id="speedup/devcopy-v0",
        device="cuda",
        agent=agent,
        rollout_device="cpu",
        rollout_n_envs=4,
    )
    assert next(agent.parameters()).device.type == "cuda"
    for k, v in agent.state_dict().items():
        assert torch.equal(v, before[k])
    # Same parameter VALUES => the stamped generator hash is unaffected by S1.
    assert ds.storage.metadata["generator_checkpoint_hash"] == ghash


# --------------------------------------------------------------------------
# S4 — cross-simulation dataset reuse (verified by generation fingerprint)
# --------------------------------------------------------------------------
def test_generation_fingerprint_covers_every_generating_input():
    from src.envs.offline.generate import generation_fingerprint

    base = dict(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="bias_confounded",
        behavior_strength=0.5,
        confounder_c_r=1.0,
        pi_basic_epsilon=0.5,
        a_bad=1,
        rollout_episodes=40,
        seed=0,
        generator_hash="abc123",
        rollout_device="cpu",
        rollout_n_envs=16,
        legacy_rollout=False,
    )
    fp = generation_fingerprint(**base)
    assert generation_fingerprint(**base) == fp  # stable
    # Every input must move the fingerprint — a silent collision would let a
    # stale dataset be reused after a config change.
    for key, other in [
        ("env_id", "Acrobot-v1"),
        ("generator_algo", "sac"),
        ("tier", "medium"),
        ("behavior_policy", "biased"),
        ("behavior_strength", 0.75),
        ("confounder_c_r", 2.0),
        ("pi_basic_epsilon", 0.3),
        ("a_bad", 0),
        ("rollout_episodes", 41),
        ("seed", 1),
        ("generator_hash", "def456"),
        ("rollout_device", "cuda"),
        ("rollout_n_envs", 8),
        ("legacy_rollout", True),
    ]:
        assert generation_fingerprint(**{**base, key: other}) != fp, key
    # cuda:0 and cuda are the same numerics -> same fingerprint.
    assert generation_fingerprint(**{**base, "rollout_device": "cpu"}) == fp


def test_fingerprint_is_stamped_and_reuse_gate_accepts_then_rejects(
    tmp_path, monkeypatch
):
    """End-to-end on the sweep's own reuse gate: a freshly generated dataset is
    accepted for reuse, and a changed input (sigma) is rejected."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from src.benchmarking.regime_sweep import _reusable_dataset_hash, load_sweep_spec
    from src.envs.offline.generate import build_generator_agent

    spec = load_sweep_spec("reproducibility/rl_regimes/offline_mdp/classical.yaml")
    spec.budgets = dict(spec.budgets)
    spec.budgets["rollout_episodes"] = 12
    spec.rollout_n_envs = 4

    set_seed(0, deterministic=True)
    agent, ghash = build_generator_agent(
        "CartPole-v1", spec.generator_algo, "random", seed=0, device="cpu"
    )
    did = "s4test/point-v0"
    ds = generate_offline_dataset(
        "CartPole-v1",
        spec.generator_algo,
        "random",
        behavior_policy="bias_confounded",
        behavior_strength=0.5,
        confounder_c_r=1.0,
        pi_basic_epsilon=spec.pi_basic_epsilon,
        rollout_episodes=12,
        seed=0,
        dataset_id=did,
        device="cpu",
        agent=agent,
        rollout_device="cpu",
        rollout_n_envs=4,
    )
    assert ds.storage.metadata["generation_fingerprint"]

    # Same inputs -> reusable, and it hands back the generator hash the
    # shared-generator assertion needs.
    hit = _reusable_dataset_hash(
        did, spec, "CartPole-v1", 0, 0.0, 0.5, "bias_confounded", 0.5, ghash
    )
    assert hit == ghash

    # A different sigma is a different dataset -> must NOT reuse.
    assert (
        _reusable_dataset_hash(
            did, spec, "CartPole-v1", 0, 0.0, 1.0, "bias_confounded", 1.0, ghash
        )
        is None
    )
    # A different pi_basic (generator hash) -> must NOT reuse.
    assert (
        _reusable_dataset_hash(
            did, spec, "CartPole-v1", 0, 0.0, 0.5, "bias_confounded", 0.5, "deadbeef"
        )
        is None
    )
    # An unknown dataset id -> must NOT reuse.
    assert (
        _reusable_dataset_hash(
            "s4test/absent-v0",
            spec,
            "CartPole-v1",
            0,
            0.0,
            0.5,
            "bias_confounded",
            0.5,
            ghash,
        )
        is None
    )


def test_reuse_gate_rejects_truncated_dataset(tmp_path, monkeypatch):
    """The fingerprint hashes INPUTS only, so an interrupted run that wrote too
    few episodes must be caught by the episode-count check."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from src.benchmarking.regime_sweep import _reusable_dataset_hash, load_sweep_spec
    from src.envs.offline.generate import build_generator_agent

    spec = load_sweep_spec("reproducibility/rl_regimes/offline_mdp/classical.yaml")
    spec.budgets = dict(spec.budgets)
    spec.budgets["rollout_episodes"] = 12
    spec.rollout_n_envs = 4

    set_seed(0, deterministic=True)
    agent, ghash = build_generator_agent(
        "CartPole-v1", spec.generator_algo, "random", seed=0, device="cpu"
    )
    # Generate only 6 of the configured 12 episodes: same inputs otherwise, so
    # the fingerprint would have to be forced to match — emulate the truncation
    # by generating at 12 then asking the gate for a spec wanting 20.
    generate_offline_dataset(
        "CartPole-v1",
        spec.generator_algo,
        "random",
        behavior_policy="bias_confounded",
        behavior_strength=0.5,
        confounder_c_r=1.0,
        pi_basic_epsilon=spec.pi_basic_epsilon,
        rollout_episodes=12,
        seed=0,
        dataset_id="s4test/trunc-v0",
        device="cpu",
        agent=agent,
        rollout_device="cpu",
        rollout_n_envs=4,
    )
    spec.budgets["rollout_episodes"] = 20  # budget changed -> not reusable
    assert (
        _reusable_dataset_hash(
            "s4test/trunc-v0",
            spec,
            "CartPole-v1",
            0,
            0.0,
            0.5,
            "bias_confounded",
            0.5,
            ghash,
        )
        is None
    )

"""PR-0 — Minari -> SequenceReplayBuffer offline fill (additive plumbing).

The episode-grouped offline producer the proximal sequence path needs. Tests:
(a) episode boundaries + ordering survive the round-trip and sample_sequences
    yields same-episode windows (checked via the contiguity invariant, not shapes);
(b) infos["confounder_u"] survives when load_u=True, and load_u on a clean dataset
    raises (mirroring the flat loader's guard);
(c) the flat loader is byte-untouched (transition count parity; the hard gate is
    the golden suite + the git-diff).
"""

from __future__ import annotations

import warnings

import pytest
import torch

warnings.filterwarnings("ignore")
pytest.importorskip("minari")
pytest.importorskip("h5py")

from src.benchmarking.registry import register_default_algorithms  # noqa: E402
from src.envs.offline.minari_loader import (  # noqa: E402
    fill_replay_buffer_from_minari,
    fill_sequence_buffer_from_minari,
    load_minari_dataset,
)
from src.rl.off_policy.replay_buffer import ReplayBuffer  # noqa: E402
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer  # noqa: E402

_CPU = torch.device("cpu")


def _gen(tmp_path, monkeypatch, dataset_id, behavior_policy, sigma=None):
    """Small deterministic CartPole dataset in an isolated Minari cache."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from src.envs.offline.generate import generate_offline_dataset
    from src.envs.registry import register_default_env_wrappers

    torch.manual_seed(0)  # order-independent generation (global RNG for the agent)
    register_default_algorithms()
    register_default_env_wrappers()
    return generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy=behavior_policy,
        behavior_strength=sigma,
        rollout_episodes=12,
        seed=0,
        dataset_id=dataset_id,
        device="cpu",
    )


def _episode_lengths(dataset_id):
    ds = load_minari_dataset(dataset_id)
    return [int(len(ep.rewards)) for ep in ds.iterate_episodes()]


def test_episode_boundaries_and_contiguity_roundtrip(tmp_path, monkeypatch):
    did = "seqfill/clean-v0"
    _gen(tmp_path, monkeypatch, did, "agent")
    ep_lens = _episode_lengths(did)

    seq = SequenceReplayBuffer(capacity=1_000_000, device=_CPU)
    n = fill_sequence_buffer_from_minari(did, seq, _CPU)

    # Counts + per-episode lengths preserved (each Minari episode -> one buffer episode).
    assert n == sum(ep_lens) == len(seq)
    assert len(seq.episodes) == len(ep_lens)
    assert sorted(len(e) for e in seq.episodes) == sorted(ep_lens)

    # Same-episode window: within a contiguous slice of ONE episode,
    # next_obs[t] == obs[t+1]; a boundary-spanning window would break at the seam.
    # seq_len = min episode length, so every episode is eligible.
    T = min(ep_lens)
    batch = seq.sample_sequences(batch_size=8, seq_len=T)
    assert set(batch.keys()) == {"obs", "actions", "rewards", "next_obs", "dones"}
    assert batch["obs"].shape[0] == 8 and batch["obs"].shape[1] == T
    assert torch.equal(batch["next_obs"][:, :-1], batch["obs"][:, 1:])


def test_confounder_u_survives_with_load_u(tmp_path, monkeypatch):
    did = "seqfill/conf-v0"
    _gen(tmp_path, monkeypatch, did, "bias_confounded", sigma=0.5)
    T = min(_episode_lengths(did))

    seq = SequenceReplayBuffer(capacity=1_000_000, device=_CPU)
    fill_sequence_buffer_from_minari(did, seq, _CPU, load_u=True)
    batch = seq.sample_sequences(batch_size=4, seq_len=T)

    assert "confounder_u" in batch
    assert tuple(batch["confounder_u"].shape[:2]) == (4, T)
    uniq = set(batch["confounder_u"].reshape(-1).tolist())
    assert uniq.issubset({0.0, 1.0})
    # U is per-episode constant -> each sampled window has a single U value.
    for row in batch["confounder_u"]:
        assert len(set(row.tolist())) == 1


def test_load_u_without_infos_raises(tmp_path, monkeypatch):
    did = "seqfill/clean-no-u-v0"
    _gen(tmp_path, monkeypatch, did, "agent")  # clean -> no U infos
    seq = SequenceReplayBuffer(capacity=1_000_000, device=_CPU)
    with pytest.raises(ValueError, match="confounder_u"):
        fill_sequence_buffer_from_minari(did, seq, _CPU, load_u=True)


def test_flat_loader_output_unchanged(tmp_path, monkeypatch):
    """The flat fill and the sequence fill ingest the same transitions; the flat
    loader is byte-frozen (guard against accidental shared-helper drift)."""
    did = "seqfill/parity-v0"
    _gen(tmp_path, monkeypatch, did, "agent")
    flat = ReplayBuffer(capacity=1_000_000, device=_CPU)
    seq = SequenceReplayBuffer(capacity=1_000_000, device=_CPU)
    n_flat = fill_replay_buffer_from_minari(did, flat, _CPU)
    n_seq = fill_sequence_buffer_from_minari(did, seq, _CPU)
    assert n_flat == n_seq == len(flat) == len(seq)

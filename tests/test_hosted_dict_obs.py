"""Hosted Dict-obs (MiniGrid/BabyAI family) adapter — fills, wrapper, dispatch.

Covers the three seams added for hosted Minari datasets whose observations are
``Dict(direction, image, ...)`` (docs/minari_adoption_report.md §2.1):

  * ``hosted_dict_obs`` fills: symbolic ``image`` flattened to float32, 5-key
    transitions, episode grouping in the sequence twin;
  * the ``minigrid_symbolic`` env wrapper: flattened symbolic obs matching the
    fill's encoding (partial and full_obs variants);
  * runner dispatch: Dict-obs dataset -> adapter fill end-to-end, and the clear
    rejections (mask_indices / load_u never silently compose with hosted data).

The dataset fixture is LOCAL (hand-built EpisodeBuffers, no network), mirroring
the shape of the hosted MiniGrid family: Dict(direction, image (V,V,3) uint8).
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

pytest.importorskip("minari")
pytest.importorskip("h5py")
pytest.importorskip("minigrid")

VIEW = 7  # default MiniGrid agent_view_size -> (7, 7, 3) symbolic image
OBS_DIM = VIEW * VIEW * 3


def _make_dict_dataset(
    tmp_path, monkeypatch, dataset_id: str, n_episodes: int = 3, ep_len: int = 5
):
    """A local Dict-obs Minari dataset shaped like the hosted MiniGrid family."""
    import gymnasium as gym
    import minari
    from minari.data_collector.episode_buffer import EpisodeBuffer

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    rng = np.random.default_rng(0)
    buffers = []
    for _ in range(n_episodes):
        buffers.append(
            EpisodeBuffer(
                observations={
                    "direction": rng.integers(0, 4, size=ep_len + 1, dtype=np.int64),
                    "image": rng.integers(
                        0, 11, size=(ep_len + 1, VIEW, VIEW, 3), dtype=np.uint8
                    ),
                },
                actions=rng.integers(0, 7, size=ep_len, dtype=np.int64),
                rewards=rng.random(ep_len).astype(np.float32),
                terminations=np.array([False] * (ep_len - 1) + [True]),
                truncations=np.zeros(ep_len, dtype=bool),
            )
        )
    return minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        buffer=buffers,
        observation_space=gym.spaces.Dict(
            {
                "direction": gym.spaces.Discrete(4),
                "image": gym.spaces.Box(0, 255, (VIEW, VIEW, 3), np.uint8),
            }
        ),
        action_space=gym.spaces.Discrete(7),
    )


# --------------------------------------------------------------------------
# Adapter fills
# --------------------------------------------------------------------------
def test_flat_dict_fill_shapes_and_values(tmp_path, monkeypatch):
    ds = _make_dict_dataset(tmp_path, monkeypatch, "hosted_test/flat-v0")
    from src.envs.offline.hosted_dict_obs import fill_replay_buffer_from_minari_dict
    from src.rl.off_policy.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=1000, device=torch.device("cpu"))
    n = fill_replay_buffer_from_minari_dict(ds.id, buffer, torch.device("cpu"))
    assert n == 3 * 5 == len(buffer)
    batch = buffer.sample(8)
    assert batch["obs"].shape == (8, OBS_DIM)
    assert batch["obs"].dtype == torch.float32
    assert batch["next_obs"].shape == (8, OBS_DIM)
    # No /255 rescaling: values stay in the stored uint8 range. (The fixture's
    # values round-trip through minari 0.5.x's lossy small-image JPEG write —
    # see _flat_symbolic_obs — so exact [0, 10] bounds don't survive; hosted
    # MiniGrid datasets are stored raw and unaffected.)
    assert 0.0 <= batch["obs"].min() and batch["obs"].max() <= 255.0
    # First transition of the first episode matches the decoded episode row.
    from src.envs.offline.hosted_dict_obs import _flat_symbolic_obs

    ep = next(ds.iterate_episodes())
    expected = _flat_symbolic_obs(ep, ds.id)[0]
    first = buffer.storage[0]
    assert torch.equal(first["obs"], expected)
    assert float(first["rewards"]) == pytest.approx(float(ep.rewards[0]))


def test_sequence_dict_fill_groups_episodes(tmp_path, monkeypatch):
    ds = _make_dict_dataset(tmp_path, monkeypatch, "hosted_test/seq-v0")
    from src.envs.offline.hosted_dict_obs import fill_sequence_buffer_from_minari_dict
    from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

    seq = SequenceReplayBuffer(capacity=1000, device=torch.device("cpu"))
    n = fill_sequence_buffer_from_minari_dict(ds.id, seq, torch.device("cpu"))
    assert n == 3 * 5
    episodes = list(seq.iter_episodes())
    assert len(episodes) == 3
    assert all(len(ep) == 5 for ep in episodes)
    assert episodes[0][0]["obs"].shape == (OBS_DIM,)


def test_dict_fill_rejects_box_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(dataset_id="hosted_test/box-v0", n_episodes=2, seed=0)
    from src.envs.offline.hosted_dict_obs import fill_replay_buffer_from_minari_dict
    from src.rl.off_policy.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="'image' key"):
        fill_replay_buffer_from_minari_dict(
            "hosted_test/box-v0", buffer, torch.device("cpu")
        )


# --------------------------------------------------------------------------
# minigrid_symbolic env wrapper
# --------------------------------------------------------------------------
def test_symbolic_env_obs_contract():
    from src.envs.wrappers.minigrid import make_minigrid_symbolic_env

    env = make_minigrid_symbolic_env("MiniGrid-Empty-5x5-v0")
    assert env.observation_space.shape == (OBS_DIM,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.uint8
    env.close()
    # full_obs: the whole 5x5 grid instead of the 7x7 egocentric view.
    env_full = make_minigrid_symbolic_env("MiniGrid-Empty-5x5-v0", full_obs=True)
    assert env_full.observation_space.shape == (5 * 5 * 3,)
    env_full.close()


def test_symbolic_wrapper_registered_and_vectorized():
    from src.envs.registry import build_env, register_default_env_wrappers

    register_default_env_wrappers()
    env = build_env(
        env_id="MiniGrid-Empty-5x5-v0",
        n_envs=2,
        device=torch.device("cpu"),
        seed=0,
        env_wrapper="minigrid_symbolic",
    )
    assert env.obs_space.shape == (OBS_DIM,)
    # BaseEnv.reset returns (obs_tensor, info) or obs_tensor depending on impl.
    out = env.reset(seed=0)
    obs_tensor = out[0] if isinstance(out, tuple) else out
    assert obs_tensor.shape == (2, OBS_DIM)
    assert obs_tensor.dtype == torch.float32


# --------------------------------------------------------------------------
# Runner dispatch (end-to-end on the local Dict fixture)
# --------------------------------------------------------------------------
def _run(tmp_path, dataset_id, algo, mask_indices=None, ep_len=5, n_episodes=3):
    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id="MiniGrid-Empty-5x5-v0",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=4,
        seed=0,
        env_wrapper="minigrid_symbolic",
        offline_dataset=dataset_id,
        mask_indices=mask_indices,
    )
    train_cfg = TrainingConfig(
        n_episodes=2,
        n_checkpoints=2,
        device="cpu",
        algorithm=algo,
        aggregation="mean",
        offline_grad_steps=8,
        # The recurrent builder rejects an mlp critic (the TrainingConfig
        # default); real configs pin this via the algos networks block.
        critic_network="lstm" if "recurrent" in algo else "mlp",
    )
    run_dir = tmp_path / f"run_{algo}"
    run_dir.mkdir()
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get(algo),
    )
    runner.run()
    return run_dir


def test_runner_dispatches_dict_fill_end_to_end(tmp_path, monkeypatch):
    _make_dict_dataset(tmp_path, monkeypatch, "hosted_test/e2e-v0")
    run_dir = _run(tmp_path, "hosted_test/e2e-v0", "offline_dqn")
    with (run_dir / "eval_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "eval_metrics.csv is empty"
    assert float(rows[-1]["eval_return_mean"]) == float(rows[-1]["eval_return_mean"])


def test_runner_dispatches_grouped_dict_fill(tmp_path, monkeypatch):
    # Recurrent learner -> episode-grouped path -> sequence dict fill. Episodes
    # longer than offpolicy_seq_len (8) so sample_sequences can draw windows.
    _make_dict_dataset(
        tmp_path, monkeypatch, "hosted_test/rec-v0", n_episodes=3, ep_len=12
    )
    run_dir = _run(tmp_path, "hosted_test/rec-v0", "offline_dqn_recurrent")
    assert (run_dir / "eval_metrics.csv").exists()


def test_runner_rejects_mask_indices_with_dict_dataset(tmp_path, monkeypatch):
    _make_dict_dataset(tmp_path, monkeypatch, "hosted_test/mask-v0")
    with pytest.raises(ValueError, match="mask_indices is not supported"):
        _run(tmp_path, "hosted_test/mask-v0", "offline_dqn", mask_indices=(0,))


def test_runner_rejects_oracle_u_with_dict_dataset(tmp_path, monkeypatch):
    _make_dict_dataset(tmp_path, monkeypatch, "hosted_test/oracle-v0")
    with pytest.raises(ValueError, match="no\\s+infos\\['confounder_u'\\]"):
        _run(tmp_path, "hosted_test/oracle-v0", "offline_dqn_oracle_u")


# --------------------------------------------------------------------------
# eval_count_terminal_reward (the sparse-reward eval fix the hosted cells need)
# --------------------------------------------------------------------------
def test_eval_count_terminal_reward_recovers_done_step_reward(tmp_path, monkeypatch):
    """The legacy eval accumulation drops the done-step reward (`* (~done)`) —
    invisible on dense CartPole, fatal on sparse MiniGrid (return reads 0.0 for
    a goal-reaching policy). Two identical runners (same seeds, same net init,
    greedy eval => identical trajectories) must differ by exactly the recovered
    terminal +1s: flag-on strictly greater."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(dataset_id="hosted_test/evalflag-v0", n_episodes=2, seed=0)

    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()

    def eval_return(count_terminal: bool) -> float:
        torch.manual_seed(0)
        env_cfg = EnvConfig(
            env_id="CartPole-v1",
            n_train_envs=2,
            n_eval_envs=4,
            rollout_len=64,
            seed=0,
            offline_dataset="hosted_test/evalflag-v0",
        )
        train_cfg = TrainingConfig(
            n_episodes=2,
            n_checkpoints=2,
            device="cpu",
            algorithm="offline_dqn",
            aggregation="mean",
            record_eval_video=False,
            eval_count_terminal_reward=count_terminal,
        )
        run_dir = tmp_path / f"evalflag_{count_terminal}"
        run_dir.mkdir()
        runner = BenchmarkRunner(
            env_cfg,
            train_cfg,
            RunConfig(run_dir=str(run_dir), timestamp="t"),
            registry.get("offline_dqn"),
        )
        return float(runner.evaluate(0)["eval_return_mean"])

    r_legacy = eval_return(False)
    r_fixed = eval_return(True)
    # Untrained greedy policy fails CartPole well inside the 64-step window, so
    # at least one termination per env: the fixed path must recover >= +1/env.
    assert r_fixed > r_legacy

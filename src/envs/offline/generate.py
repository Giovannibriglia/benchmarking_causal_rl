"""B2 offline GENERATE pipeline: train -> snapshot-by-return -> rollout -> write.

Produces tiered (random/medium/expert), optionally provenance-varied, Minari
datasets for unhosted domains by reusing the online training loop
(``BenchmarkRunner``), the checkpoint machinery, the A-series collection-policy
seam (so a dataset is characterized by ``tier x behavior_policy``), and the
``make_*_offline`` Minari-write path. Consumption stays through B1's load path
(``--offline-dataset`` -> ``load_minari_dataset``).

Purely additive: no edits to the online/offline/load paths. The generator is an
OFF-POLICY online algo (dqn/sac/ddpg) — the rollout drives ``agent.act`` and the
provenance policies reach into the critic/buffer; on-policy generators (a
``policy.act`` adapter) are deferred.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch

_DISCRETE_ONLY = {"dqn"}
_CONTINUOUS_ONLY = {"sac", "ddpg"}


# --------------------------------------------------------------------------
# Tier selection (pure / deterministic) — performance-defined, sign-robust
# --------------------------------------------------------------------------
def select_tier_episode(
    returns: dict[int, float], tier: str, fraction: float = 1.0 / 3.0
):
    """Select the checkpoint episode for ``tier`` from ``{episode: eval_return}``.

    * ``expert`` -> the argmax-return checkpoint (earliest, on ties).
    * ``medium`` -> the FIRST checkpoint reaching
      ``R_random + fraction*(R_expert - R_random)`` (the range-based
      generalization of D4RL's "1/3 of expert return"; sign-robust for
      negative-return envs like Pendulum). ``R_random`` is the lowest
      checkpoint return.
    * ``random`` -> ``None`` (signals a fresh untrained agent, no checkpoint).
    """
    if tier == "random":
        return None
    if not returns:
        raise ValueError("no eval returns recorded; cannot select a tier")
    items = sorted(returns.items())  # by episode
    r_expert = max(returns.values())
    if tier == "expert":
        return min(ep for ep, r in items if r == r_expert)
    if tier == "medium":
        r_random = min(returns.values())
        target = r_random + fraction * (r_expert - r_random)
        for ep, r in items:
            if r >= target:
                return ep
        return items[-1][0]
    raise ValueError(f"unknown tier '{tier}' (expected random/medium/expert)")


# --------------------------------------------------------------------------
# Guards (both fire BEFORE any training)
# --------------------------------------------------------------------------
def assert_online_generator(algo: str) -> None:
    """Reject generating WITH an offline algo (the category error)."""
    from src.benchmarking.registry import registry

    if registry.get(algo).data_regime != "online":
        raise ValueError(
            f"cannot generate with offline algo '{algo}'; the generator must be "
            "an online algo (dqn/sac/ddpg)."
        )


def assert_action_space_match(algo: str, env_action_type: str) -> None:
    """Reject a generator whose action type can't match the env's."""
    if algo in _DISCRETE_ONLY and env_action_type != "discrete":
        raise ValueError(
            f"generator '{algo}' is discrete-only but the env action space is "
            f"{env_action_type}; use sac/ddpg for continuous envs."
        )
    if algo in _CONTINUOUS_ONLY and env_action_type != "continuous":
        raise ValueError(
            f"generator '{algo}' is continuous-only but the env action space is "
            f"{env_action_type}; use dqn for discrete envs."
        )


# --------------------------------------------------------------------------
# Naming + rollout env (provenance: confounded wraps the rollout env)
# --------------------------------------------------------------------------
def dataset_name(env_id: str, tier: str, behavior_policy: str = "agent") -> str:
    """``generated/{env_slug}/{tier}[-{behavior}]-v0`` (behavior omitted for
    the clean 'agent' rollout)."""
    slug = env_id.split("-v")[0].lower().replace("/", "-")
    suffix = "" if behavior_policy == "agent" else f"-{behavior_policy}"
    return f"generated/{slug}/{tier}{suffix}-v0"


def build_rollout_env(
    env_id, n_envs, device, seed, behavior_policy="agent", strength=None
):
    """Build the rollout env, wrapped in the confounder iff bias_confounded."""
    from src.envs.registry import build_env

    env = build_env(env_id=env_id, n_envs=n_envs, device=device, seed=seed)
    if behavior_policy == "bias_confounded":
        from src.envs.wrappers.confounded import ConfoundedCollectionWrapper

        sig = 1.0 if strength is None else float(strength)
        env = ConfoundedCollectionWrapper(env, c_a=sig, c_r=sig)
    return env


def _env_dims(env):
    if len(env.obs_space.shape) == 0:
        obs_dim = 1
    else:
        obs_dim = int(torch.tensor(env.obs_space.shape).prod().item())
    obs_shape = tuple(env.obs_space.shape)
    act_space = env.act_space
    if hasattr(act_space, "n"):
        return obs_dim, obs_shape, "discrete", int(act_space.n), act_space
    return obs_dim, obs_shape, "continuous", int(act_space.shape[0]), act_space


def _to_np(obs):
    return obs.reshape(obs.shape[0], -1)[0].detach().cpu().numpy()


def _rollout(env, collection_policy, n_episodes, seed, action_type, max_steps=1000):
    """Roll out ``n_episodes`` (n_envs=1) into Minari EpisodeBuffers. Explicit
    per-episode reset + break-on-done keeps clean episode boundaries (and the
    confounder's per-episode U resamples at each reset)."""
    from minari.data_collector.episode_buffer import EpisodeBuffer

    buffers = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        obs_list = [_to_np(obs)]
        acts, rews, terms, truncs = [], [], [], []
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = collection_policy.act(obs).action
            obs, reward, term, trunc, _ = env.step(action)
            obs_list.append(_to_np(obs))
            a = action.reshape(action.shape[0], -1)[0].detach().cpu().numpy()
            acts.append(
                int(a[0]) if action_type == "discrete" else a.astype(np.float32)
            )
            rews.append(float(reward.reshape(-1)[0].item()))
            terms.append(bool(term.reshape(-1)[0].item()))
            truncs.append(bool(trunc.reshape(-1)[0].item()))
            done = terms[-1] or truncs[-1]
            steps += 1
        adt = np.int64 if action_type == "discrete" else np.float32
        buffers.append(
            EpisodeBuffer(
                observations=np.asarray(obs_list, dtype=np.float32),
                actions=np.asarray(acts, dtype=adt),
                rewards=np.asarray(rews, dtype=np.float32),
                terminations=np.asarray(terms, dtype=bool),
                truncations=np.asarray(truncs, dtype=bool),
            )
        )
    return buffers


def _read_eval_returns(run_dir: str) -> dict[int, float]:
    with open(os.path.join(run_dir, "eval_metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    return {int(r["episode"]): float(r["eval_return_mean"]) for r in rows}


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
def generate_offline_dataset(
    env_id: str,
    generator_algo: str,
    tier: str,
    *,
    behavior_policy: str = "agent",
    behavior_strength: float | None = None,
    fraction: float = 1.0 / 3.0,
    train_episodes: int = 50,
    n_checkpoints: int = 10,
    rollout_episodes: int = 20,
    seed: int = 0,
    dataset_id: str | None = None,
    run_dir: str | None = None,
    device: str | None = None,
):
    """Train an online generator, snapshot the ``tier`` policy by return, roll it
    out (optionally via a collection policy), and write a Minari dataset to the
    local cache. Returns the created MinariDataset."""
    import gymnasium as gym

    from src.benchmarking.registry import register_default_algorithms, registry
    from src.config.device import detect_device
    from src.envs.registry import register_default_env_wrappers
    from src.rl.policies.behavior_policy import (
        AgentBehaviorPolicy,
        build_collection_policy,
    )

    register_default_algorithms()
    register_default_env_wrappers()

    # --- guards (before any training) ---
    assert_online_generator(generator_algo)
    probe = gym.make(env_id)
    env_action_type = "discrete" if hasattr(probe.action_space, "n") else "continuous"
    probe.close()
    assert_action_space_match(generator_algo, env_action_type)

    dev = torch.device(device) if device else detect_device()

    # --- train (skipped for the random tier, which uses a fresh agent) ---
    sel_ep = None
    if tier != "random":
        if run_dir is None:
            raise ValueError("non-random tiers require run_dir for the generator")
        _train_generator(
            env_id, generator_algo, train_episodes, n_checkpoints, seed, run_dir, dev
        )
        sel_ep = select_tier_episode(_read_eval_returns(run_dir), tier, fraction)

    # --- rollout env + agent ---
    rollout_env = build_rollout_env(
        env_id, 1, dev, seed, behavior_policy, behavior_strength
    )
    obs_dim, obs_shape, action_type, action_dim, action_space = _env_dims(rollout_env)
    _, agent = registry.get(generator_algo).builder(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        device=dev,
        action_space=action_space,
        obs_shape=obs_shape,
    )
    if sel_ep is not None:
        from src.benchmarking.checkpoints import load_checkpoint

        tag = env_id.replace("/", "-")
        ckpt = load_checkpoint(
            os.path.join(
                run_dir,
                "checkpoints",
                f"{tag}_{generator_algo}_seed{seed}",
                f"ckpt_ep{sel_ep:04d}.pt",
            )
        )
        agent.load_state_dict(ckpt["agent_state"])

    if behavior_policy == "agent":
        collection_policy = AgentBehaviorPolicy(agent)
    else:
        collection_policy = build_collection_policy(
            behavior_policy,
            agent,
            action_type,
            action_space,
            behavior_strength,
            env=rollout_env,
        )

    buffers = _rollout(
        rollout_env, collection_policy, rollout_episodes, seed, action_type
    )
    rollout_env.close()

    import minari

    name = dataset_id or dataset_name(env_id, tier, behavior_policy)
    return minari.create_dataset_from_buffers(
        dataset_id=name,
        buffer=buffers,
        env=gym.make(env_id),
        algorithm_name=f"{generator_algo}-{tier}-{behavior_policy}",
    )


def _train_generator(env_id, algo, train_episodes, n_checkpoints, seed, run_dir, dev):
    from src.benchmarking.registry import registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    env_cfg = EnvConfig(
        env_id=env_id, n_train_envs=4, n_eval_envs=4, rollout_len=64, seed=seed
    )
    train_cfg = TrainingConfig(
        n_episodes=train_episodes,
        n_checkpoints=n_checkpoints,
        device=str(dev),
        algorithm=algo,
        aggregation="mean",
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=run_dir, timestamp="t"),
        registry.get(algo),
    ).run()

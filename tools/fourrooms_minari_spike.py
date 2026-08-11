"""Spike: hosted Minari dataset (D4RL/minigrid/fourrooms-v0) through our offline stack.

Part 1 — probe the UNTOUCHED load path (assert_dataset_matches_algo + the flat fill)
         and record exactly where the Dict-obs dataset breaks it.
Part 2 — adapter chain: flatten the symbolic 7x7x3 image to a 147-dim vector,
         fill the real ReplayBuffer, train the real offline_dqn (build_offline_dqn +
         DQN.learn, batch 128 like the runner), eval greedy on a matching
         ImgObsWrapper-flattened FourRooms env vs a random-policy baseline.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch

DATASET_ID = "D4RL/minigrid/fourrooms-v0"
OBS_SCALE = 10.0  # symbolic indices are small ints; same scaling train + eval
GRAD_STEPS = 8000
BATCH_SIZE = 128  # runner.offpolicy_batch_size
EVAL_EPISODES = 50
SEED = 0


def flatten_obs(img: np.ndarray) -> np.ndarray:
    return (
        (img.astype(np.float32) / OBS_SCALE).reshape(img.shape[0], -1)
        if img.ndim == 4
        else (img.astype(np.float32) / OBS_SCALE).reshape(-1)
    )


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import minari

    ds = minari.load_dataset(DATASET_ID)

    # ---------------- Part 1: untouched load path ----------------
    print("=== Part 1: untouched load path ===")
    from src.envs.offline.minari_loader import (
        assert_dataset_matches_algo,
        dataset_action_type,
        fill_replay_buffer_from_minari,
    )

    print("dataset_action_type:", dataset_action_type(ds.action_space))
    try:
        assert_dataset_matches_algo(ds, "discrete", DATASET_ID, "offline_dqn")
        print("assert_dataset_matches_algo: PASS (discrete/discrete)")
    except Exception as e:  # noqa: BLE001
        print("assert_dataset_matches_algo: FAIL:", e)

    from src.rl.off_policy.replay_buffer import ReplayBuffer

    probe_buffer = ReplayBuffer(capacity=20_000, device=device)
    try:
        n = fill_replay_buffer_from_minari(DATASET_ID, probe_buffer, device)
        print(f"fill_replay_buffer_from_minari: unexpectedly OK, {n} transitions")
    except Exception:  # noqa: BLE001
        exc_type, exc, tb = sys.exc_info()
        last = traceback.extract_tb(tb)[-1]
        print(
            f"fill_replay_buffer_from_minari: FAILS as expected -> "
            f"{exc_type.__name__}: {exc} (at {last.filename.split('/')[-1]}:{last.lineno})"
        )

    # ---------------- Part 2: adapter chain ----------------
    print("\n=== Part 2: adapter chain ===")
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    n_transitions = 0
    behavior_returns = []
    for ep in ds.iterate_episodes():
        img = ep.observations["image"]  # (T+1, 7, 7, 3) uint8 symbolic
        obs = flatten_obs(img)  # (T+1, 147) float32
        dones = np.logical_or(ep.terminations, ep.truncations)
        behavior_returns.append(float(ep.rewards.sum()))
        for t in range(len(ep.actions)):
            buffer.add(
                {
                    "obs": torch.as_tensor(obs[t]),
                    "actions": torch.as_tensor(np.int64(ep.actions[t])),
                    "rewards": torch.as_tensor(np.float32(ep.rewards[t])),
                    "next_obs": torch.as_tensor(obs[t + 1]),
                    "dones": torch.as_tensor(np.float32(dones[t])),
                }
            )
            n_transitions += 1
    obs_dim = obs.shape[-1]
    print(
        f"adapter fill: {n_transitions} transitions, obs_dim={obs_dim}, "
        f"behavior return mean={np.mean(behavior_returns):.3f} "
        f"over {len(behavior_returns)} episodes"
    )

    from src.rl.offline.cql import build_cql
    from src.rl.offline.dqn import build_offline_dqn

    agents = {}
    for name, builder, step_attr in (
        ("offline_dqn", build_offline_dqn, "learn"),
        ("cql", build_cql, "update"),
    ):
        torch.manual_seed(SEED)
        kwargs = dict(
            obs_dim=obs_dim, action_dim=7, action_type="discrete", device=device
        )
        if name == "cql":
            kwargs["action_space"] = None
        _, agent = builder(**kwargs)
        step_fn = getattr(agent, step_attr)
        for step in range(GRAD_STEPS):
            metrics = step_fn(buffer.sample(BATCH_SIZE))
            if (step + 1) % 4000 == 0:
                loss = metrics.get("q_loss", metrics.get("loss"))
                print(f"  [{name}] step {step + 1}/{GRAD_STEPS} loss={loss:.5f}")
        agents[name] = agent

    # Eval env producing the SAME symbolic obs encoding as the dataset.
    import gymnasium as gym
    import minigrid  # noqa: F401  (registers MiniGrid-* ids)
    from minigrid.wrappers import ImgObsWrapper

    env = ImgObsWrapper(gym.make("MiniGrid-FourRooms-v0"))

    def rollout(policy) -> float:
        obs_img, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = policy(obs_img)
            obs_img, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
        return total

    def greedy(agent):
        def policy(obs_img):
            with torch.no_grad():
                x = torch.as_tensor(flatten_obs(obs_img), device=device).unsqueeze(0)
                return int(agent.q_network(x).argmax(dim=1).item())

        return policy

    def summ(xs):
        return f"mean={np.mean(xs):.3f} sd={np.std(xs):.3f} success={np.mean([x > 0 for x in xs]):.2f}"

    print()
    for name, agent in agents.items():
        env.reset(seed=SEED)
        returns = [rollout(greedy(agent)) for _ in range(EVAL_EPISODES)]
        print(f"eval ({EVAL_EPISODES} eps)  {name:12s}: {summ(returns)}")
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    random_ = [
        rollout(lambda _o: int(rng.integers(0, 7))) for _ in range(EVAL_EPISODES)
    ]
    print(f"eval ({EVAL_EPISODES} eps)  {'random':12s}: {summ(random_)}")
    print(f"dataset behavior policy:  mean={np.mean(behavior_returns):.3f}")


if __name__ == "__main__":
    main()

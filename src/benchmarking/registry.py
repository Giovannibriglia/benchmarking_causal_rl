from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn

from src.rl.models.backbone import build_trunk, select_backbone
from src.rl.off_policy.ddpg import DDPG
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.on_policy.a2c import A2C
from src.rl.on_policy.actor_critic import ActorCritic
from src.rl.on_policy.ppo import PPO
from src.rl.on_policy.trpo import TRPO
from src.rl.on_policy.vanilla import VanillaPolicyGradient
from .runner import AlgorithmSpec


class Registry:
    def __init__(self):
        self._algos: Dict[str, AlgorithmSpec] = {}

    def register(self, name: str, spec: AlgorithmSpec) -> None:
        self._algos[name.lower()] = spec

    def get(self, name: str) -> AlgorithmSpec:
        key = name.lower()
        if key not in self._algos:
            raise KeyError(f"Algorithm {name} not registered")
        return self._algos[key]


registry = Registry()


# Central env set registry. Values can be static lists or callables returning a list.
ENV_SETS: Dict[str, List[str] | Callable[[], List[str]]] = {
    # Dynamically enumerates all installed Gymnasium env ids.
    "gymnasium": [
        "Blackjack-v1",
        # "CarRacing-v3",
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Acrobot-v1",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "FrozenLake-v1",
        "FrozenLake8x8-v1",
        "CliffWalking-v1",
        "Taxi-v3",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
    ],
    "mujoco": [
        "Ant-v5",
        "Reacher-v5",
        "Pusher-v5",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Swimmer-v5",
        "Walker2d-v5",
        "Humanoid-v5",
        "HumanoidStandup-v5",
    ],
    # Image-observation Atari (discrete); needs `uv sync --extra atari`.
    # Run with a CNN-capable algo, e.g. --env-set atari --algos dqn.
    "atari": [
        "ALE/Pong-v5",
        "ALE/Breakout-v5",
    ],
    # Image-observation MiniGrid (discrete); needs `uv sync --extra minigrid`.
    # RGB partial render -> (3, 84, 84), runs through the same CNN path as Atari.
    # Run with a CNN-capable algo, e.g. --env-set minigrid --algos ppo.
    "minigrid": [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
    ],
    "gymnasium-robotics": [
        # Fetch (multi-goal, Dict obs)
        "FetchReach-v4",
        "FetchPush-v4",
        "FetchSlide-v4",
        "FetchPickAndPlace-v4",
        # Shadow Dexterous Hand (multi-goal, Dict obs)
        "HandReach-v3",
        "HandManipulateBlock-v1",
        "HandManipulateEgg-v1",
        "HandManipulatePen-v1",
        # Touch-sensor variants (optional; only if you want the bigger obs)
        # "HandManipulateEgg_BooleanTouchSensors-v1",
        # "HandManipulateEgg_ContinuousTouchSensors-v1",
    ],
}


def expand_env_set(name: str) -> List[str]:
    key = name.lower()
    if key not in ENV_SETS:
        raise KeyError(
            f"Unknown env_set '{name}'. Add it to ENV_SETS in src/benchmarking/registry.py."
        )
    entries = ENV_SETS[key]
    envs = entries() if callable(entries) else list(entries)
    if len(envs) == 0:
        raise ValueError(
            f"Environment set '{name}' is defined but empty. Populate ENV_SETS or install the provider."
        )
    return envs


def _ac_trunk_kwargs(kwargs) -> dict:
    """Optional recurrent-trunk kwargs (hidden_dim, num_layers) from the builder
    call; absent/None entries are dropped so ActorCritic's defaults apply."""
    out = {}
    for k in ("hidden_dim", "num_layers"):
        if kwargs.get(k) is not None:
            out[k] = kwargs[k]
    return out


def _build_actor_critic(kwargs) -> ActorCritic:
    return ActorCritic(
        kwargs.get("obs_shape", (kwargs["obs_dim"],)),
        kwargs["obs_dim"],
        kwargs["action_dim"],
        kwargs["action_type"],
        kwargs["device"],
        actor_network=kwargs.get("actor_network", "mlp"),
        critic_network=kwargs.get("critic_network", "mlp"),
        **_ac_trunk_kwargs(kwargs),
    )


def _reject_recurrent(algo_name: str, kwargs) -> None:
    """Non-PPO on-policy algos get the separate-trunk architecture but NOT
    recurrent support in this PR — recurrent BPTT is wired for PPO only."""
    if kwargs.get("actor_network", "mlp") != "mlp" or (
        kwargs.get("critic_network", "mlp") != "mlp"
    ):
        raise ValueError(
            f"Recurrent networks for {algo_name} are not yet supported. This PR "
            f"implements recurrent PPO only; non-PPO on-policy algorithms must "
            f"use networks: {{actor: mlp, critic: mlp}} or the plain string form."
        )


def _reject_offpolicy_recurrent(algo_name: str, kwargs) -> None:
    """Guard for off-policy algos that do NOT yet have recurrent variants: accept
    the actor_network / critic_network kwargs (YAML dict form) but reject non-MLP.
    Only DQN and SAC have recurrent variants; DDPG and the offline algos
    (offline_dqn, cql, iql, bcq, *_continuous) still reject."""
    if kwargs.get("actor_network", "mlp") != "mlp" or (
        kwargs.get("critic_network", "mlp") != "mlp"
    ):
        raise ValueError(
            f"Recurrent networks for {algo_name} are not supported. Only DQN and "
            f"SAC have recurrent variants (LSTM/GRU/RNN); for {algo_name}, use the "
            f"plain string form or networks: {{actor: mlp, critic: mlp}}."
        )


def _offpolicy_recurrent_guard(name: str, builder):
    """Wrap an (imported) off-policy builder so it rejects non-MLP networks."""

    def _wrapped(**kwargs):
        _reject_offpolicy_recurrent(name, kwargs)
        return builder(**kwargs)

    return _wrapped


def register_default_algorithms() -> None:
    # On-policy builders
    def build_vanilla(**kwargs):
        _reject_recurrent("vanilla (and vanilla_ac)", kwargs)
        policy = _build_actor_critic(kwargs)
        agent = VanillaPolicyGradient(policy, device=kwargs["device"])
        return policy, agent

    def build_a2c(**kwargs):
        _reject_recurrent("a2c", kwargs)
        policy = _build_actor_critic(kwargs)
        agent = A2C(policy, device=kwargs["device"])
        return policy, agent

    def build_ppo(**kwargs):
        policy = _build_actor_critic(kwargs)
        agent = PPO(policy, device=kwargs["device"])
        return policy, agent

    def build_trpo(**kwargs):
        _reject_recurrent("trpo", kwargs)
        policy = _build_actor_critic(kwargs)
        agent = TRPO(policy, device=kwargs["device"])
        return policy, agent

    # Off-policy builders
    def build_dqn(**kwargs):
        obs_dim = kwargs["obs_dim"]
        action_dim = kwargs["action_dim"]
        device = kwargs["device"]
        obs_shape = kwargs.get("obs_shape", (obs_dim,))
        # DQN has no actor/critic split; the Q-network IS the critic, so it uses
        # critic_network. build_trunk("mlp") == select_backbone (bitwise); a
        # recurrent type returns an LSTM/GRU/RNN trunk. trunk_kwargs carries
        # hidden_dim/num_layers when recurrent.
        net = kwargs.get("critic_network", "mlp")
        tk = _ac_trunk_kwargs(kwargs)
        q_net = build_trunk(net, obs_shape, obs_dim, action_dim, **tk)
        target_net = build_trunk(net, obs_shape, obs_dim, action_dim, **tk)
        # Buffer: the runner passes a SequenceReplayBuffer for recurrent runs;
        # otherwise the flat ReplayBuffer is created here (capacity unchanged ->
        # golden bitwise).
        buffer = kwargs.get("buffer")
        if buffer is None:  # not `or`: an empty SequenceReplayBuffer is falsy
            buffer = ReplayBuffer(capacity=100_000, device=device)
        agent = DQN(q_net.to(device), target_net.to(device), buffer, device=device)
        return q_net.to(device), agent

    def build_ddpg(**kwargs):
        _reject_offpolicy_recurrent("ddpg", kwargs)
        obs_dim = kwargs["obs_dim"]
        action_dim = kwargs["action_dim"]
        device = kwargs["device"]
        action_space = kwargs["action_space"]
        obs_shape = kwargs.get("obs_shape", (obs_dim,))
        actor = select_backbone(
            obs_shape, obs_dim, action_dim, output_activation=nn.Tanh
        )
        critic = select_backbone((obs_dim + action_dim,), obs_dim + action_dim, 1)
        target_actor = select_backbone(
            obs_shape, obs_dim, action_dim, output_activation=nn.Tanh
        )
        target_critic = select_backbone(
            (obs_dim + action_dim,), obs_dim + action_dim, 1
        )
        buffer = ReplayBuffer(capacity=100_000, device=device)
        agent = DDPG(
            actor.to(device),
            critic.to(device),
            target_actor.to(device),
            target_critic.to(device),
            buffer,
            device=device,
        )
        # set action bounds if available
        try:
            low = torch.as_tensor(action_space.low, device=device)
            high = torch.as_tensor(action_space.high, device=device)
            agent.action_low = low
            agent.action_high = high
        except Exception:
            pass
        return actor.to(device), agent

    registry.register("vanilla", AlgorithmSpec(builder=build_vanilla, kind="on_policy"))
    # vanilla_ac: alias of vanilla (same builder/spec); keep the vanilla key.
    registry.register(
        "vanilla_ac", AlgorithmSpec(builder=build_vanilla, kind="on_policy")
    )
    registry.register("a2c", AlgorithmSpec(builder=build_a2c, kind="on_policy"))
    registry.register("ppo", AlgorithmSpec(builder=build_ppo, kind="on_policy"))
    registry.register("trpo", AlgorithmSpec(builder=build_trpo, kind="on_policy"))

    def build_sac(**kwargs):
        from src.rl.off_policy.sac import SAC, SquashedGaussianActor

        obs_dim = kwargs["obs_dim"]
        action_dim = kwargs["action_dim"]
        device = kwargs["device"]
        action_space = kwargs["action_space"]
        obs_shape = kwargs.get("obs_shape", (obs_dim,))
        # Actor uses actor_network; twin critics use critic_network. mlp keeps the
        # exact pre-recurrent construction (SAC golden bitwise).
        actor_net = kwargs.get("actor_network", "mlp")
        critic_net = kwargs.get("critic_network", "mlp")
        tk = _ac_trunk_kwargs(kwargs)
        actor = SquashedGaussianActor(
            obs_dim, action_dim, obs_shape=obs_shape, network=actor_net, **tk
        ).to(device)
        if critic_net == "mlp":
            mk_q = lambda: select_backbone(  # noqa: E731
                (obs_dim + action_dim,),
                obs_dim + action_dim,
                1,
                hidden_dims=(256, 256),
                activation=nn.ReLU,
            )
        else:
            mk_q = lambda: build_trunk(  # noqa: E731
                critic_net, (obs_dim + action_dim,), obs_dim + action_dim, 1, **tk
            )
        q1, q2, q1t, q2t = (mk_q().to(device) for _ in range(4))
        # SAC needs a large buffer for million-step reference training. Runner
        # passes a SequenceReplayBuffer for recurrent runs; else create the flat
        # one here (capacity unchanged -> golden bitwise).
        buffer = kwargs.get("buffer")
        if buffer is None:  # not `or`: an empty SequenceReplayBuffer is falsy
            buffer = ReplayBuffer(capacity=1_000_000, device=device)
        try:
            scale = float(abs(action_space.high[0]))
        except Exception:
            scale = 1.0
        agent = SAC(
            actor,
            q1,
            q2,
            q1t,
            q2t,
            buffer,
            device=device,
            action_dim=action_dim,
            action_scale=scale,
        )
        return actor, agent

    registry.register("dqn", AlgorithmSpec(builder=build_dqn, kind="off_policy"))
    registry.register("sac", AlgorithmSpec(builder=build_sac, kind="off_policy"))
    registry.register("ddpg", AlgorithmSpec(builder=build_ddpg, kind="off_policy"))

    # Offline (fixed-dataset) algorithms: data_regime="offline" routes run() to
    # _train_offline. Online dqn is left untouched.
    from src.rl.offline.bcq import build_bcq
    from src.rl.offline.bcq_continuous import build_bcq_continuous
    from src.rl.offline.cql import build_cql
    from src.rl.offline.cql_continuous import build_cql_continuous
    from src.rl.offline.dqn import build_offline_dqn
    from src.rl.offline.iql import build_iql
    from src.rl.offline.iql_continuous import build_iql_continuous

    for _name, _builder in (
        ("offline_dqn", build_offline_dqn),
        ("bcq", build_bcq),
        ("cql", build_cql),
        ("iql", build_iql),
        # Continuous offline (CQL-on-SAC, IQL-Gaussian, CVAE-BCQ).
        ("cql_continuous", build_cql_continuous),
        ("iql_continuous", build_iql_continuous),
        ("bcq_continuous", build_bcq_continuous),
    ):
        registry.register(
            _name,
            AlgorithmSpec(
                builder=_offpolicy_recurrent_guard(_name, _builder),
                kind="off_policy",
                data_regime="offline",
            ),
        )

    # Oracle-U deconfounding variants (discrete Cell-7 arm) as FIRST-CLASS algos,
    # selected via --algos like any other. They read the per-transition latent U
    # (requires_confounder_u=True) -> the ORACLE REFERENCE ceiling, not a reported
    # competitor. Only the discrete bases have an oracle builder; there is no
    # continuous variant, so e.g. "cql_continuous_oracle_u"/"sac_oracle_u" is a
    # clean registry KeyError.
    from src.rl.offline.oracle_u import (
        build_oracle_u_bcq,
        build_oracle_u_cql,
        build_oracle_u_dqn,
        build_oracle_u_iql,
    )

    for _vname, _vbuilder in (
        ("offline_dqn_oracle_u", build_oracle_u_dqn),
        ("bcq_oracle_u", build_oracle_u_bcq),
        ("cql_oracle_u", build_oracle_u_cql),
        ("iql_oracle_u", build_oracle_u_iql),
    ):
        registry.register(
            _vname,
            AlgorithmSpec(
                builder=_offpolicy_recurrent_guard(_vname, _vbuilder),
                kind="off_policy",
                data_regime="offline",
                requires_confounder_u=True,
            ),
        )

    # Proximal deconfounding variants (STUB, PR-1 scaffolding). base x Proximal()
    # strategy: it INFERS U (requires_confounder_u=False, never reads it) and
    # consumes episode-grouped sequences (needs_episode_grouping=True). The stub
    # degrades to the Observational floor; the estimator math is PR-2. Same
    # discrete bases; selecting a continuous *_proximal is a clean KeyError.
    from src.rl.off_policy.identification import Proximal

    def _proximal_builder(base_builder):
        def _wrapped(**kwargs):
            return base_builder(strategy=Proximal(), **kwargs)

        return _wrapped

    for _pname, _pbuilder in (
        ("offline_dqn_proximal", build_offline_dqn),
        ("bcq_proximal", build_bcq),
        ("cql_proximal", build_cql),
        ("iql_proximal", build_iql),
    ):
        registry.register(
            _pname,
            AlgorithmSpec(
                builder=_offpolicy_recurrent_guard(
                    _pname, _proximal_builder(_pbuilder)
                ),
                kind="off_policy",
                data_regime="offline",
                needs_episode_grouping=True,
            ),
        )

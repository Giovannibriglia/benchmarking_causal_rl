"""Registry builders for the model-based planners (online + offline).

Same shape as ``build_dqn`` / ``build_offline_dqn``: the builder returns
``(policy_module, agent)``; the agent owns its replay buffer; the offline
variant differs only in buffer capacity (the Minari fill needs the whole
dataset resident) and in the pessimism default.
"""

from __future__ import annotations

from src.rl.models.backbone import select_backbone
from src.rl.off_policy.replay_buffer import ReplayBuffer
from .mlp_vi import DynamicsEnsemble, MLPValueIteration
from .tabular_vi import TabularVI

ONLINE_CAPACITY = 100_000
OFFLINE_CAPACITY = 1_000_000


def _discrete_only(name: str, kwargs) -> None:
    if kwargs.get("action_type", "discrete") != "discrete":
        raise ValueError(
            f"{name} is discrete-action only: value iteration takes a max over "
            "actions. Continuous action spaces need an actor (sac/ddpg/ppo)."
        )


def _buffer(kwargs, capacity: int, device) -> ReplayBuffer:
    buf = kwargs.get("buffer")
    if buf is None:  # not `or`: an empty SequenceReplayBuffer is falsy
        buf = ReplayBuffer(capacity=capacity, device=device)
    return buf


def _build_tabular(name: str, capacity: int, optimistic: bool, **kwargs):
    _discrete_only(name, kwargs)
    obs_shape = tuple(kwargs.get("obs_shape", (kwargs["obs_dim"],)))
    if len(obs_shape) != 1:
        raise ValueError(
            f"{name} needs a flat (one-hot) observation, got shape {obs_shape}"
        )
    device = kwargs["device"]
    agent = TabularVI(
        n_states=int(kwargs["obs_dim"]),
        n_actions=int(kwargs["action_dim"]),
        buffer=_buffer(kwargs, capacity, device),
        device=device,
        optimistic=bool(kwargs.get("optimistic", optimistic)),
        **{
            k: kwargs[k]
            for k in ("gamma", "epsilon", "plan_every", "unvisited_value")
            if k in kwargs
        },
    )
    return agent.q_table, agent


def _build_mlp(name: str, capacity: int, penalty_coef: float, **kwargs):
    _discrete_only(name, kwargs)
    obs_dim, action_dim, device = (
        kwargs["obs_dim"],
        kwargs["action_dim"],
        kwargs["device"],
    )
    obs_shape = tuple(kwargs.get("obs_shape", (obs_dim,)))
    if len(obs_shape) != 1:
        raise ValueError(
            f"{name} needs a flat vector observation, got shape {obs_shape}"
        )
    q_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    target_net = select_backbone(obs_shape, obs_dim, action_dim).to(device)
    model = DynamicsEnsemble(
        obs_dim,
        action_dim,
        n_members=int(kwargs.get("n_members", 3)),
        hidden_dims=tuple(kwargs.get("model_hidden_dims", (200, 200))),
    ).to(device)
    agent = MLPValueIteration(
        model,
        q_net,
        target_net,
        _buffer(kwargs, capacity, device),
        device=device,
        penalty_coef=float(kwargs.get("penalty_coef", penalty_coef)),
        **{
            k: kwargs[k]
            for k in ("gamma", "epsilon", "tau", "model_warmup", "lr_model", "lr_q")
            if k in kwargs
        },
    )
    return q_net, agent


def build_tabular_vi(**kwargs):
    # Online: R-max optimism for unvisited (s,a) -- the count-based model's
    # exploration mechanism (see TabularVI). Offline: 0 (nothing to explore).
    # The count model is rebuilt from the WHOLE buffer, so the buffer must never
    # evict history: OFFLINE_CAPACITY (1M rows) online too -- tabular rows are tiny.
    return _build_tabular("tabular_vi", OFFLINE_CAPACITY, optimistic=True, **kwargs)


def build_offline_tabular_vi(**kwargs):
    return _build_tabular(
        "offline_tabular_vi", OFFLINE_CAPACITY, optimistic=False, **kwargs
    )


def build_mlp_vi(**kwargs):
    return _build_mlp("mlp_vi", ONLINE_CAPACITY, penalty_coef=0.0, **kwargs)


def build_offline_mlp_vi(**kwargs):
    # Offline default: a small disagreement penalty (MOPO's idea) so the
    # planner does not chase model error off the data's support. 0 online.
    return _build_mlp("offline_mlp_vi", OFFLINE_CAPACITY, penalty_coef=1.0, **kwargs)

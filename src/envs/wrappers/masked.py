from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box


class MaskedObservationWrapper:
    """Observation wrapper that DROPS a fixed set of components from the flat
    observation vector — the agent never sees the masked positions, but the
    underlying env (state, reward, action space, termination) is untouched.

    This realizes the ``Z hidden`` axis of the experimental matrix
    (``docs/experimental_design.md`` §5): a per-env set of velocity indices is
    removed so the learner faces a partially observed context.

    Composition with confounding (``docs/experimental_design.md`` §8, Cell 8).
    The wrapper stack is::

        base env -> ConfoundedCollectionWrapper -> MaskedObservationWrapper

    i.e. masking is applied on the OUTSIDE. ``ConfoundedCollectionWrapper`` reads
    the env's internal latent ``U`` to perturb the reward (the confounding
    mechanism is reward-side, not observation-side), so it must sit UNDER the
    mask: masking on the outside hides observation components from the agent
    while leaving the confounding mechanism fully intact. ``current_u`` and every
    other attribute of the inner stack remain reachable through this wrapper via
    attribute delegation.

    Scope. ``Box`` (continuous, flat-vector) observation spaces only. The masked
    indices address the LAST axis, so the same wrapper handles both unbatched
    ``(obs_dim,)`` and vectorized ``(n_envs, obs_dim)`` observations. The action
    space, reward, ``info`` dict, and termination flags are never touched.
    """

    def __init__(self, env, indices) -> None:
        self.env = env
        indices = tuple(int(i) for i in indices)

        base_space = self._base_obs_space(env)
        if not isinstance(base_space, Box):
            raise TypeError(
                "MaskedObservationWrapper supports only Box observation spaces; "
                f"got {type(base_space).__name__}. Masking is Box-only by design "
                "(image/Dict/Tuple observations are out of scope)."
            )

        obs_dim = int(base_space.shape[-1])
        for idx in indices:
            if idx < 0 or idx >= obs_dim:
                raise ValueError(
                    f"mask index {idx} is out of range for observation dim "
                    f"{obs_dim} (valid indices are 0..{obs_dim - 1})."
                )
        seen: set[int] = set()
        for idx in indices:
            if idx in seen:
                raise ValueError(
                    f"duplicate mask index {idx} in {indices}; indices must be unique."
                )
            seen.add(idx)

        self.indices = indices
        self._keep = [i for i in range(obs_dim) if i not in seen]
        # The most recent PRE-mask observation (full obs vector), updated on each
        # reset/step. PR2's eval_per_context writer reads the masked indices off
        # this to bin returns by the hidden Z component — the value the agent
        # never sees. None until the first reset.
        self.last_unmasked_obs = None

        low = np.delete(np.asarray(base_space.low), indices, axis=-1)
        high = np.delete(np.asarray(base_space.high), indices, axis=-1)
        projected = Box(low=low, high=high, dtype=base_space.dtype)
        # Expose under both names: ``observation_space`` for the raw-gymnasium
        # contract, ``obs_space`` for this project's BaseEnv contract.
        self.observation_space = projected
        self.obs_space = projected

    @staticmethod
    def _base_obs_space(env):
        """The inner env's observation space, under whichever attribute it
        exposes (``observation_space`` for raw gymnasium envs, ``obs_space`` for
        this project's ``BaseEnv`` stack)."""
        space = getattr(env, "observation_space", None)
        if space is None:
            space = getattr(env, "obs_space", None)
        if space is None:
            raise TypeError(
                "MaskedObservationWrapper requires the wrapped env to expose an "
                "'observation_space' or 'obs_space' attribute."
            )
        return space

    def __getattr__(self, name):
        # Delegate everything not set on this wrapper (act_space/action_space,
        # current_u, n_envs, device, close, render, start_video, ...) to the
        # inner env, so the confounded latent and the env API stay reachable
        # through the mask.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def observation(self, obs):
        """Drop the masked indices from the last axis of ``obs``.

        Keeps a ``torch.Tensor`` a tensor (the project pipeline stays on-device);
        a NumPy array stays a NumPy array (raw-gymnasium path).
        """
        if isinstance(obs, torch.Tensor):
            keep = torch.as_tensor(self._keep, dtype=torch.long, device=obs.device)
            return obs.index_select(-1, keep)
        return np.delete(np.asarray(obs), self.indices, axis=-1)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.last_unmasked_obs = obs
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_unmasked_obs = obs
        return self.observation(obs), reward, terminated, truncated, info

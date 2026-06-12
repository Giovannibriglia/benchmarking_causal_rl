from __future__ import annotations

import torch


class ConfoundedCollectionWrapper:
    """Collection-only env wrapper that injects a per-episode latent ``U``.

    ``U`` drives BOTH the behavior policy's action (read via ``current_u``) and
    the env's reward (``r += c_r * U``), while it is NEVER added to the returned
    observation — genuine *unobserved* confounding: the learner's stored
    ``(obs, action, reward, next_obs)`` carries an action<->reward association
    with no ``U`` column.

    Wraps the vectorized env, so ``U`` is per-sub-env and per-EPISODE: it is
    resampled at each sub-env's ``done`` (NEXT_STEP autoreset), AFTER perturbing
    that terminal step's reward — so within any step the action and the reward
    share the same ``U`` (a stable spurious signal, not within-step noise).

    Train-env ONLY: the runner never wraps ``eval_env`` (eval is always clean).
    ``U`` sampling stays in the main RNG stream (``torch.bernoulli``/``randn``,
    no explicit per-instance RNG seeding) — it's the behavior+env dynamics,
    reproducible via the run-start global seed (so it adds no new seeding site).
    """

    def __init__(
        self, env, c_a: float = 1.0, c_r: float = 1.0, u_dist: str = "bernoulli"
    ):
        self.env = env
        self.c_a = float(c_a)
        self.c_r = float(c_r)
        self.u_dist = u_dist
        self.n_envs = env.n_envs
        self.device = env.device
        self.current_u = self._sample_u()

    def __getattr__(self, name):
        # Delegate everything not overridden (obs_space, act_space, close,
        # start_video, render, ...) to the inner env. ``obs_space`` is the inner
        # (clean) space -> the learner builds for U-free observations.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def _sample_u(self) -> torch.Tensor:
        if self.u_dist == "normal":
            return torch.randn(self.n_envs, device=self.device)
        return torch.bernoulli(torch.full((self.n_envs,), 0.5, device=self.device))

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.current_u = self._sample_u()
        info = {**info, "confounder_u": self.current_u.clone()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Perturb the reward with THIS episode's U (shared with the action that
        # produced it), then expose U; obs is passed through untouched.
        reward = reward + self.c_r * self.current_u
        info = {**info, "confounder_u": self.current_u.clone()}
        # Resample U for sub-envs whose episode just ended (after perturbing the
        # terminal reward) so the next episode draws a fresh latent.
        done = torch.logical_or(terminated, truncated)
        if bool(done.any()):
            self.current_u = torch.where(done, self._sample_u(), self.current_u)
        return obs, reward, terminated, truncated, info

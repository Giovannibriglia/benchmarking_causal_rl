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

    ``U`` RNG is dual-mode (issue #36), selected by the ``seed`` arg:
      * ``seed=None`` (the runner's online A1 collection path): ``U`` is drawn
        from the GLOBAL torch stream (``torch.bernoulli``/``randn`` with
        ``generator=None``) — byte-identical to the pre-#36 behavior, so the
        off-policy golden and the run-seed reproducibility of online confounded
        collection are untouched (it adds no new seeding site there).
      * ``seed=int`` (the offline GENERATE path threads
        ``generate_offline_dataset``'s seed): ``U`` is drawn from an isolated
        per-instance ``torch.Generator``, so a freshly generated dataset's
        gate-test outcome is reproducible regardless of cumulative process RNG
        state. Mirrors ``CuriosityBehaviorPolicy``'s per-instance generator.
    """

    def __init__(
        self,
        env,
        c_a: float = 1.0,
        c_r: float = 1.0,
        u_dist: str = "bernoulli",
        seed: int | None = None,
        confounder_kind: str = "additive",
        a_bad: int = 1,
    ):
        self.env = env
        self.c_a = float(c_a)
        self.c_r = float(c_r)
        self.u_dist = u_dist
        self.n_envs = env.n_envs
        self.device = env.device
        # Confounder kind (issue: action-dependent confounding, new cell). "additive"
        # (default) is the BYTE-FROZEN cells-7/8 path: r += c_r*U on every step.
        # "action_gated" gates the shift on the taken action == a_bad, so U inflates
        # ONE action's apparent reward -> U changes the confounded-optimal policy (a
        # stronger regime). a_bad defaults to 1, the U=1-preferred action that
        # ConfoundedBehaviorPolicy already biases toward (pref = round(U)); both edges
        # then land on the same action (U -> A and the gated U -> R).
        if confounder_kind not in ("additive", "action_gated"):
            raise ValueError(
                f"confounder_kind must be 'additive' or 'action_gated', got "
                f"{confounder_kind!r}."
            )
        self.confounder_kind = confounder_kind
        self.a_bad = int(a_bad)
        # Isolated U RNG only when a seed is given; None keeps the global stream
        # (see class doc). The generator lives on the sampling device so the
        # generator/tensor devices match for both CPU and CUDA generation.
        if seed is None:
            self._gen = None
        else:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(int(seed))
        self.current_u = self._sample_u()
        # Continuous action_gated cell indicator h(a), set per-step by the paired
        # partition-swap policy; None keeps the discrete a_bad gate (byte-frozen).
        self.current_h = None

    def __getattr__(self, name):
        # Delegate everything not overridden (obs_space, act_space, close,
        # start_video, render, ...) to the inner env. ``obs_space`` is the inner
        # (clean) space -> the learner builds for U-free observations.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def _sample_u(self) -> torch.Tensor:
        if self.u_dist == "normal":
            return torch.randn(self.n_envs, device=self.device, generator=self._gen)
        return torch.bernoulli(
            torch.full((self.n_envs,), 0.5, device=self.device), generator=self._gen
        )

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.current_u = self._sample_u()
        info = {**info, "confounder_u": self.current_u.clone()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Perturb the reward with THIS episode's U (shared with the action that
        # produced it), then expose U; obs is passed through untouched.
        if self.confounder_kind == "additive":
            reward = reward + self.c_r * self.current_u  # cells 7/8 — BYTE-FROZEN
        else:  # action_gated: shift ONLY on the "upper cell" (h==1) transitions
            # Continuous: the paired policy exposes the executed action's cell
            # h(a)=1[a>=median] via current_h (the discrete a_bad indicator has no
            # median). Discrete falls back to action==a_bad -> byte-frozen (cell 9).
            h = getattr(self, "current_h", None)
            if h is not None:
                gate = h.to(reward.dtype)
            else:
                gate = (action == self.a_bad).to(reward.dtype)
            reward = reward + self.c_r * self.current_u * gate
        info = {**info, "confounder_u": self.current_u.clone()}
        # Resample U for sub-envs whose episode just ended (after perturbing the
        # terminal reward) so the next episode draws a fresh latent.
        done = torch.logical_or(terminated, truncated)
        if bool(done.any()):
            self.current_u = torch.where(done, self._sample_u(), self.current_u)
        return obs, reward, terminated, truncated, info

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
        proxy_strength: float | None = None,
        instrument_strength: float | None = None,
        u_drift: float = 0.0,
        gate_probs: tuple[float, float] | None = None,
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
        # --- GRACE v2 arms. All default OFF, and every one draws from its OWN
        # generator, so an arm that does not enable them consumes exactly the
        # RNG the byte-frozen paths always consumed.
        #
        # D-D proxies: two COVARIATE-FREE noisy measurements of U. "Covariate
        # free" is the load-bearing property, not a nicety -- because
        # P(Z|U), P(W|U) do not depend on (s,a), the proxy measurement matrices
        # are GLOBAL and pin the latent's labelling globally. Letting the noise
        # scale with the state (an easy, natural-looking choice) would silently
        # turn them covariate-conditional and collapse D-D into D-B's situation.
        # Hence: the noise below is drawn from a dedicated generator and NEVER
        # reads obs.
        self.proxy_strength = None if proxy_strength is None else float(proxy_strength)
        # D-E instrument: drawn per episode INDEPENDENTLY of U, read by the
        # behaviour policy, with no path to the reward.
        self.instrument_strength = (
            None if instrument_strength is None else float(instrument_strength)
        )
        # D-B' persistence: per-step flip probability. 0.0 = episode-static
        # (D-B) and consumes NO draws, so D-B is reproduced byte-for-byte.
        # D-E (R2 route (a)): make U shift the PROBABILITY of the gated bonus
        # rather than its magnitude --
        #     r += c_r * 1[a = a_bad] * Bernoulli(q_U),   q_1 > q_0
        # -- instead of the deterministic r += c_r * U * 1[a = a_bad].
        #
        # This resolves BOTH halves of R2 at once, which a continuous noise term
        # could not:
        #   * R stays BINARY-valued in {r_base, r_base + c_r}, so L4's
        #     Balke-Pearl anchor keeps the binary closed form. Continuous noise
        #     would have forced either a discretisation (which v2 forbids) or an
        #     LP solved numerically in place of the formula.
        #   * R is now genuinely STOCHASTIC given (A, U), so residualising on
        #     them leaves variance and the exclusion check acquires real power
        #     instead of measuring nothing.
        # The declared diagram is unchanged: R's parents are still (S, A, U) and
        # the Bernoulli draw is R's exogenous term, which every diagram already
        # has. The U->R edge magnitude becomes c_r * (q_1 - q_0).
        self.gate_probs = None if gate_probs is None else tuple(map(float, gate_probs))
        if self.gate_probs is not None:
            q0, q1 = self.gate_probs
            if not (0.0 <= q0 <= 1.0 and 0.0 <= q1 <= 1.0):
                raise ValueError(f"gate_probs must be in [0,1], got {self.gate_probs}.")
            if q1 <= q0:
                raise ValueError(
                    f"gate_probs = (q0, q1) needs q1 > q0 or there is no U->R edge "
                    f"left to identify; got {self.gate_probs}."
                )
            if confounder_kind != "action_gated":
                raise ValueError(
                    "gate_probs is meaningful only for the action_gated confounder."
                )
        self.u_drift = float(u_drift)
        if not 0.0 <= self.u_drift <= 0.5:
            raise ValueError(
                f"u_drift is a per-step flip probability in [0, 0.5]; 0.5 is full "
                f"refresh (U_t independent of U_{{t-1}}). Got {self.u_drift}."
            )
        # Proxies measure the latent as it stood when they were drawn (episode
        # reset). Under drift that is the episode-INITIAL U, not the U each
        # transition actually shares -- an unannounced measurement error that
        # would look like weak proxies rather than a bug. D-D declares static U
        # and D-B' declares no explicit proxies, so nothing needs the
        # combination; refuse it rather than let it pass quietly.
        if self.proxy_strength is not None and self.u_drift > 0.0:
            raise NotImplementedError(
                "proxy_strength with u_drift > 0 is not supported: the proxies are "
                "drawn once per episode and would measure the episode-INITIAL U "
                "while the reward uses the drifted U. D-D declares a static latent "
                "(u_drift = 0); D-B' uses lagged views, not declared proxies. A "
                "drifting-latent proxy arm needs per-step proxy draws and a "
                "catalogue entry to declare them."
            )
        self._aux_gen = None
        if seed is not None and (
            self.proxy_strength is not None
            or self.instrument_strength is not None
            or self.u_drift > 0.0
            or self.gate_probs is not None
        ):
            self._aux_gen = torch.Generator(device=self.device)
            self._aux_gen.manual_seed(int(seed) + 8_675_309)
        self.current_u = self._sample_u()
        self.current_z, self.current_w = self._sample_proxies()
        self.current_i = self._sample_instrument()
        # action_gated is DISCRETE-ONLY: the a_bad reward gate has no meaning on a
        # continuous action space (a float action never equals a_bad -> the gate would
        # never fire -> a silently UNCONFOUNDED dataset). The continuous confounder is
        # deferred (see MarginallyMatchedConfoundedBehaviorPolicy + docs). Raise here
        # rather than fail silently; test fakes without ``act_space`` are left alone.
        if self.confounder_kind == "action_gated":
            act_space = getattr(env, "act_space", None)
            if act_space is not None and not hasattr(act_space, "n"):
                raise NotImplementedError(
                    "action_gated (action-dependent) confounding is DISCRETE-ONLY; the "
                    "continuous confounded arm is not runnable (see "
                    "docs/rl_regimes_restructure.md)."
                )

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

    def _sample_proxies(self):
        """Two conditionally-independent measurements of U, COVARIATE-FREE.

        ``Z = s*(2U-1) + eps_z``, ``W = s*(2U-1) + eps_w`` with independent
        standard-normal noise and ``s = proxy_strength`` the signal knob (R4
        sweeps it). Neither reads the observation, so ``parents(Z) =
        parents(W) = {U}`` exactly, and ``Z indep W | U`` holds by independent
        draws. Neither is a function of the action, so neither is affected by A.
        """
        if self.proxy_strength is None:
            return None, None
        centred = 2.0 * self.current_u - 1.0
        z = self.proxy_strength * centred + torch.randn(
            self.n_envs, device=self.device, generator=self._aux_gen
        )
        w = self.proxy_strength * centred + torch.randn(
            self.n_envs, device=self.device, generator=self._aux_gen
        )
        return z, w

    def _sample_instrument(self):
        """A per-episode instrument, drawn INDEPENDENTLY of U and never
        entering the reward. The behaviour policy reads ``current_i``; the
        wrapper's reward perturbation ignores it entirely, which is what makes
        the exclusion restriction hold by construction."""
        if self.instrument_strength is None:
            return None
        return torch.bernoulli(
            torch.full((self.n_envs,), 0.5, device=self.device),
            generator=self._aux_gen,
        )

    def _drift_u(self, u: torch.Tensor) -> torch.Tensor:
        """D-B': flip each sub-env's U with probability ``u_drift``. At 0.0 this
        consumes no randomness, so the static path stays byte-identical."""
        if self.u_drift <= 0.0:
            return u
        flip = (
            torch.rand(self.n_envs, device=self.device, generator=self._aux_gen)
            < self.u_drift
        )
        return torch.where(flip, 1.0 - u, u)

    def _aux_info(self) -> dict:
        out = {}
        if self.current_z is not None:
            out["proxy_z"] = self.current_z.clone()
            out["proxy_w"] = self.current_w.clone()
        if self.current_i is not None:
            out["instrument_i"] = self.current_i.clone()
        return out

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.current_u = self._sample_u()
        self.current_z, self.current_w = self._sample_proxies()
        self.current_i = self._sample_instrument()
        info = {**info, "confounder_u": self.current_u.clone(), **self._aux_info()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Perturb the reward with THIS episode's U (shared with the action that
        # produced it), then expose U; obs is passed through untouched.
        if self.confounder_kind == "additive":
            reward = reward + self.c_r * self.current_u  # cells 7/8 — BYTE-FROZEN
        else:  # action_gated (discrete-only): shift ONLY on the a_bad transitions
            gate = (action == self.a_bad).to(reward.dtype)
            if self.gate_probs is None:
                reward = reward + self.c_r * self.current_u * gate  # BYTE-FROZEN
            else:
                q0, q1 = self.gate_probs
                q = torch.where(
                    self.current_u > 0.5,
                    torch.full_like(reward, q1),
                    torch.full_like(reward, q0),
                )
                bonus = torch.bernoulli(q, generator=self._aux_gen)
                reward = reward + self.c_r * bonus * gate
        info = {**info, "confounder_u": self.current_u.clone(), **self._aux_info()}
        # D-B' drift: U evolves WITHIN the episode, after the reward that this
        # transition's U produced. At u_drift = 0 this is a no-op consuming no
        # randomness (D-B).
        self.current_u = self._drift_u(self.current_u)
        # Resample U for sub-envs whose episode just ended (after perturbing the
        # terminal reward) so the next episode draws a fresh latent.
        done = torch.logical_or(terminated, truncated)
        if bool(done.any()):
            self.current_u = torch.where(done, self._sample_u(), self.current_u)
            fresh_z, fresh_w = self._sample_proxies()
            if fresh_z is not None:
                self.current_z = torch.where(done, fresh_z, self.current_z)
                self.current_w = torch.where(done, fresh_w, self.current_w)
            fresh_i = self._sample_instrument()
            if fresh_i is not None:
                self.current_i = torch.where(done, fresh_i, self.current_i)
        return obs, reward, terminated, truncated, info

"""GRACE template CBN — tabular two-slice causal Bayesian network built from a
DECLARED ``CellGraph`` (amendment A9), with two likelihood channels sharing
every non-A CPT.

Structure comes ONLY from the ``CellGraph`` (``cell_graph.py``): the
observational channel uses the declared edge list; the mutilated channel is
``graph.mutilated_edges()`` — the same list minus edges into A. No edge is
introduced anywhere else.

Channels:
  * OBSERVATIONAL — full-graph likelihood incl. ``P_b(A | ., U)``. Offline
    trajectories fit this channel with latents marginalized: exact enumeration
    EM over the episode-static U (MDP cells), Baum-Welch over the latent
    S chain x U (POMDP cells).
  * MUTILATED — derived by graph surgery at construction, never via nbn's
    ``intervene()`` (the Phase-1 audit: ``intervene`` breaks exact VE, and VE
    silently ignores ``do=``). All non-A CPTs are the same tensors/modules, so
    one fit updates both channels (asserted in tests, A7).

Five-keys invariant: the fit consumes ONLY (obs, actions, rewards, next_obs,
dones) — never the dataset's realized ``confounder_u`` (leakage rule R5).

Phase-3 seams (A10 — obligations, not implementations):
  * the DISCRETE LATENT BLOCK (U enumeration: E-step responsibilities, the
    U-belief, the mixture inside the backup) is separated from the OBSERVABLE
    EVIDENCE evaluation — ``_step_loglik`` / ``_local_evidence`` are the only
    places observable likelihoods are computed, so continuous mechanisms swap
    there without touching the algorithms;
  * ``q_do`` is the SINGLE target-computation boundary (the continuous slice
    replaces its interior with mutilated-net sampling, same signature);
  * nothing tabular leaks into the ``Grace`` / ``RegimeRouter`` public
    signatures (they exchange stats dicts and verdicts, not bin tables).

Determinism: all randomness (EM inits, bootstrap resampling) flows through a
dedicated ``torch.Generator`` — the global torch stream is never touched, so a
grace arm never shifts the base learner's RNG relative to the observational
floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.rl.offline.grace.cell_graph import CellGraph

_EPS = 1e-12
_NEG = -1e30
# Guarded label-swap canonicalization threshold — the SAME convention (and
# value) as ``ProximalEM._canon_eps``: no-spread -> symmetric prior; posterior
# anti-correlated with the mean-residual observable -> flip.
_CANON_EPS = 0.05


@dataclass
class GraceOptions:
    """The approved config surface (ablation switches).

    ``n_bins`` None -> auto: 6 per dim for obs_dim <= 4, else 4 (CartPole
    joint 6^4=1296 stays exact-joint; Acrobot 4^6=4096 exceeds ``joint_cap``
    and takes the factored-transition path, amendment A8).
    """

    u_card: int = 2
    rho: float = 0.0  # U persistence switch — declared in the graph, default off
    ensemble_k: int = 5
    router: bool = True
    interval: bool = True
    n_bins: Optional[int] = None
    n_latent: int = 32  # POMDP latent-S cardinality
    em_iters: int = 20
    pomdp_em_iters: int = 30
    alpha: float = 0.5  # Dirichlet smoothing on all fitted CPT counts
    joint_cap: int = 2048  # joint-transition CPT above this -> factored (A8)
    reward_max_classes: int = 12
    holdout_frac: float = 0.2
    min_state_count: int = 10  # coverage: a state bin below this is unvisited
    seed: int = 0  # grace-internal generator seed (never the global stream)
    # Served-Q_do deploy: "marginal" = E_U[Q(s,.,u)] (the ablation's oracle-
    # comparable convention); "u0" = the U=0 reference stratum (the return-
    # correct deploy for the action-dependent confounder — cell-9 finding;
    # default for the {base}_grace classical algo variants).
    deploy: str = "marginal"

    def bins_for(self, obs_dim: int) -> int:
        if self.n_bins is not None:
            return int(self.n_bins)
        return 6 if obs_dim <= 4 else 4


@dataclass
class EpisodeData:
    """Padded per-episode index tensors (device-resident).

    MDP cells: ``s`` = joint state-bin index (``sdims`` per-dim indices ride
    along for the factored path). POMDP cells: ``s`` holds the OBSERVATION
    symbol (binned masked obs) — the latent S chain is fit by Baum-Welch.
    """

    s: torch.Tensor  # (N, T) long
    a: torch.Tensor  # (N, T) long
    rc: torch.Tensor  # (N, T) long reward class
    s_next: torch.Tensor  # (N, T) long
    done: torch.Tensor  # (N, T) float — 1 where the transition terminated
    mask: torch.Tensor  # (N, T) float
    lengths: torch.Tensor  # (N,) float
    rew: torch.Tensor  # (N, T) float raw rewards (residual observable)
    sdims_next: Optional[torch.Tensor] = None  # (N, T, D) per-dim next bins

    @property
    def n_episodes(self) -> int:
        return int(self.s.shape[0])

    def subset(self, idx: torch.Tensor) -> "EpisodeData":
        return EpisodeData(
            s=self.s[idx],
            a=self.a[idx],
            rc=self.rc[idx],
            s_next=self.s_next[idx],
            done=self.done[idx],
            mask=self.mask[idx],
            lengths=self.lengths[idx],
            rew=self.rew[idx],
            sdims_next=None if self.sdims_next is None else self.sdims_next[idx],
        )


@dataclass
class CBNParams:
    """Fitted probability tensors for one model variant.

    ``pb_a`` / ``p_r`` carry a leading U axis; a U-restricted variant (the
    router's M0/M_A/M_R fits, A3) has that axis of size 1 on the restricted
    channel. ``trans``/``p_done``/``p0``/``emis`` never carry U — the declared
    graphs have no U -> S' edge and the reduction guarantee (B6) leans on it.
    """

    p_u: torch.Tensor  # (n_u,)
    pb_a: torch.Tensor  # (n_u_a, n_s, n_a) observational behavior channel
    p_r: torch.Tensor  # (n_u_r, n_s, n_a, n_c)
    p0: torch.Tensor  # (n_s,)
    trans: Optional[torch.Tensor] = None  # (n_s, n_a, n_s) joint mode
    p_done: Optional[torch.Tensor] = None  # (n_s, n_a)
    trans_f: Optional[List[torch.Tensor]] = None  # factored: per-dim (n_s, n_a, b)
    emis: Optional[torch.Tensor] = None  # (n_s, n_o) POMDP emission
    q_u: Optional[torch.Tensor] = None  # (N, n_u) per-episode posterior
    visited_sa: Optional[torch.Tensor] = None  # (n_s, n_a) bool
    visited_s: Optional[torch.Tensor] = None  # (n_s,) bool


def _norm(counts: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return counts / counts.sum(dim=dim, keepdim=True).clamp_min(_EPS)


class TemplateCBN:
    """One template CBN (observational + mutilated channels) built from a
    declared ``CellGraph``.

    Lifecycle: ``fit(data)`` -> ``router_components(data)`` / ``q_do(gamma)``
    / ``build_nbn()``. ``fit`` is also the online-channel refresh entry (refit
    on the current rolling episode view; canonicalization re-runs every
    refresh — the Gate-B label-persistence convention).
    """

    def __init__(
        self,
        graph: CellGraph,
        n_states: int,
        n_actions: int,
        n_classes: int,
        class_values: torch.Tensor,
        opts: GraceOptions,
        device: torch.device,
        n_obs_symbols: int | None = None,
        factored_dims: int | None = None,
        n_bins: int | None = None,
    ) -> None:
        graph.validate()
        self.graph = graph
        self.pomdp = graph.pomdp
        self.has_u = graph.has_node("U")
        self.n_s = int(n_states)
        self.n_a = int(n_actions)
        self.n_c = int(n_classes)
        self.class_values = class_values.to(device)
        self.opts = opts
        self.device = device
        self.n_o = int(n_obs_symbols) if n_obs_symbols is not None else None
        self.factored_dims = factored_dims  # None -> joint transition mode
        self.n_bins = n_bins
        self.n_u = int(opts.u_card) if self.has_u else 1
        if graph.rho_persistence and float(opts.rho) > 0.0:
            raise NotImplementedError(
                "rho persistence (U -> U') is declared but its fit is out of this "
                "slice; run with rho=0.0 (static per-episode U, the wired design)."
            )
        self.params: CBNParams | None = None
        self._nbn_obs = None
        self._nbn_mut = None
        self.gen = torch.Generator().manual_seed(int(opts.seed))

    # ------------------------------------------------------------------ fit
    def fit(self, data: EpisodeData) -> CBNParams:
        if self.pomdp:
            self.params = self._fit_pomdp(data, self.n_u)
        else:
            self.params = self._fit_mdp(data, self.n_u)
        return self.params

    # ---------------------------------------------------- observable evidence
    # A10 seam: these two are the ONLY places observable likelihoods are
    # computed. The continuous slice replaces their interiors with neural
    # mechanism log-probs; every E-step / filter / router computation above
    # them treats the result as an opaque log-evidence tensor.
    def _step_loglik_mdp(
        self,
        log_pb: torch.Tensor,
        log_pr: torch.Tensor,
        ua: int,
        ur: int,
        s: torch.Tensor,
        a: torch.Tensor,
        rc: torch.Tensor,
    ) -> torch.Tensor:
        """Per-step log-evidence of the (A, R) observables given observed S and
        stratum u — flat ``(M,)`` over valid steps."""
        return log_pb[ua, s, a] + log_pr[ur, s, a, rc]

    def _local_evidence_pomdp(
        self,
        log_em: torch.Tensor,
        log_pb: torch.Tensor,
        log_pr: torch.Tensor,
        ua: int,
        ur: int,
        data: EpisodeData,
    ) -> torch.Tensor:
        """Per-step log-evidence over the latent state, ``(N, T, S)``:
        emission + behavior + reward channels at stratum u."""
        n_ep, t_len = data.s.shape
        s_lat = log_em.shape[0]
        loc = log_em[:, data.s.reshape(-1)].T.reshape(n_ep, t_len, s_lat).clone()
        loc += log_pb[ua][:, data.a.reshape(-1)].T.reshape(n_ep, t_len, s_lat)
        loc += (
            log_pr[ur]
            .reshape(s_lat, -1)[:, (data.a * self.n_c + data.rc).reshape(-1)]
            .T.reshape(n_ep, t_len, s_lat)
        )
        return loc

    # -- MDP: S observed, U = per-episode latent, exact enumeration EM --
    def _fit_mdp(
        self,
        data: EpisodeData,
        n_u: int,
        fit_ua: bool | None = None,
        fit_ur: bool | None = None,
    ) -> CBNParams:
        """Exact EM. ``fit_ua``/``fit_ur`` gate which channels carry the U
        axis. Defaults derive from the DECLARED graph (U->A and U->R edges);
        the router's restricted variants (A3) override them: M0 = (False,
        False), M_A = (True, False), M_R = (False, True)."""
        if fit_ua is None:
            fit_ua = self.graph.has_edge("U", "A")
        if fit_ur is None:
            fit_ur = self.graph.has_edge("U", "R")
        dev = self.device
        alpha = float(self.opts.alpha)
        mask = data.mask
        valid = mask > 0
        s_f = data.s[valid]
        a_f = data.a[valid]
        rc_f = data.rc[valid]
        n_ep = data.n_episodes

        # U-free pieces: dynamics, initial state, visit masks (hard counts).
        p0 = _norm(
            torch.bincount(data.s[:, 0], minlength=self.n_s).float().to(dev) + alpha
        )
        trans, p_done, trans_f = self._fit_transitions(data)
        sa_counts = torch.zeros(self.n_s * self.n_a, device=dev)
        sa_counts.index_add_(
            0, s_f * self.n_a + a_f, torch.ones_like(a_f, dtype=torch.float32)
        )
        sa_counts = sa_counts.view(self.n_s, self.n_a)
        visited_sa = sa_counts >= 1
        visited_s = sa_counts.sum(1) >= float(self.opts.min_state_count)

        # Residual observable for the canonical labeling: per-episode mean of
        # (r - rbar(s, a)) under the MARGINAL per-(s, a) reward baseline —
        # state-conditional by construction, the same observable ProximalEM
        # canonicalizes on.
        r_f = data.rew[valid]
        r_sum = torch.zeros(self.n_s * self.n_a, device=dev)
        r_sum.index_add_(0, s_f * self.n_a + a_f, r_f)
        rbar = (r_sum / sa_counts.view(-1).clamp_min(1.0)).view(self.n_s, self.n_a)
        resid_step = torch.zeros_like(data.rew)
        resid_step[valid] = r_f - rbar[s_f, a_f]
        residuals = (resid_step * mask).sum(1) / data.lengths  # (N,)

        eff_u = n_u if (fit_ua or fit_ur) else 1
        n_ua = eff_u if fit_ua else 1
        n_ur = eff_u if fit_ur else 1
        flat_sa = s_f * self.n_a + a_f
        flat_sac = flat_sa * self.n_c + rc_f
        ep_of = valid.nonzero(as_tuple=True)[0]  # episode id per valid step

        # Warm start (the proximal convention, generalized to n_u strata by a
        # residual-quantile split): no-spread -> symmetric prior.
        if eff_u == 1 or float(residuals.std()) < _CANON_EPS:
            q_u = torch.full((n_ep, eff_u), 1.0 / eff_u, device=dev)
        else:
            qs = torch.quantile(
                residuals, torch.linspace(0, 1, eff_u + 1, device=dev)[1:-1]
            )
            stratum = torch.bucketize(residuals, qs.contiguous())
            q_u = torch.zeros(n_ep, eff_u, device=dev)
            q_u[torch.arange(n_ep, device=dev), stratum] = 1.0

        def _m_step(q: torch.Tensor):
            w_a = q if fit_ua else torch.ones(n_ep, 1, device=dev)
            w_r = q if fit_ur else torch.ones(n_ep, 1, device=dev)
            pb_counts = torch.zeros(n_ua, self.n_s * self.n_a, device=dev)
            pr_counts = torch.zeros(n_ur, self.n_s * self.n_a * self.n_c, device=dev)
            for u in range(n_ua):
                pb_counts[u].index_add_(0, flat_sa, w_a[ep_of, u])
            for u in range(n_ur):
                pr_counts[u].index_add_(0, flat_sac, w_r[ep_of, u])
            pb = _norm(pb_counts.view(n_ua, self.n_s, self.n_a) + alpha)
            pr = _norm(pr_counts.view(n_ur, self.n_s, self.n_a, self.n_c) + alpha)
            pu = _norm(q.sum(0) + alpha)
            return pb, pr, pu

        pb_a, p_r, p_u = _m_step(q_u)
        if eff_u > 1:
            for _ in range(max(1, int(self.opts.em_iters))):
                # E-step: exact per-episode enumeration over the latent block.
                log_pb = torch.log(pb_a.clamp_min(_EPS))
                log_pr = torch.log(p_r.clamp_min(_EPS))
                ll = torch.zeros(n_ep, eff_u, device=dev)
                for u in range(eff_u):
                    step_ll = self._step_loglik_mdp(
                        log_pb,
                        log_pr,
                        min(u, n_ua - 1),
                        min(u, n_ur - 1),
                        s_f,
                        a_f,
                        rc_f,
                    )
                    ll[:, u].index_add_(0, ep_of, step_ll)
                ll = ll + torch.log(p_u.clamp_min(_EPS))
                q_u = torch.softmax(ll, dim=1)
                # Guarded label-swap canonicalization, every E-step.
                q_u = self._canonicalize(q_u, residuals)
                # M-step so the CPT labeling always matches the posterior.
                pb_a, p_r, p_u = _m_step(q_u)

        return CBNParams(
            p_u=p_u,
            pb_a=pb_a,
            p_r=p_r,
            p0=p0,
            trans=trans,
            p_done=p_done,
            trans_f=trans_f,
            q_u=q_u,
            visited_sa=visited_sa,
            visited_s=visited_s,
        )

    def _canonicalize(self, q_u: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """The label-swap convention, mirrored from ``ProximalEM.run_em``: no
        residual spread -> symmetric prior; degenerate responsibilities ->
        untouched; else enforce "higher stratum index = higher mean residual"
        (for n_u=2: flip when q(U=1) anti-correlates with the residual)."""
        n_u = q_u.shape[1]
        if float(residuals.std()) < _CANON_EPS:
            return torch.full_like(q_u, 1.0 / n_u)
        if float(q_u.std()) < _CANON_EPS:
            return q_u
        if n_u == 2:
            corr = torch.corrcoef(torch.stack([q_u[:, 1], residuals]))[0, 1]
            if bool(corr < 0.0):
                q_u = q_u.flip(1)
            return q_u
        # n_u > 2: sort strata ascending by responsibility-weighted mean
        # residual (the binary convention's natural generalization).
        w = q_u / q_u.sum(0, keepdim=True).clamp_min(_EPS)
        stratum_resid = (w * residuals.unsqueeze(1)).sum(0)
        return q_u[:, torch.argsort(stratum_resid)]

    def _fit_transitions(self, data: EpisodeData):
        """Hard-count dynamics (U-free per the declared graphs). Joint mode:
        dense ``(n_s, n_a, n_s)``. Factored mode (A8): per-dim
        ``(n_s, n_a, n_bins)`` next-bin CPTs whose product approximates the
        joint next-state law — an explicit, reduction-test-validated
        approximation."""
        dev = self.device
        alpha = float(self.opts.alpha)
        valid = data.mask > 0
        s_f, a_f, d_f = data.s[valid], data.a[valid], data.done[valid]
        cont = valid & (data.done < 0.5)
        sc, ac, sn = data.s[cont], data.a[cont], data.s_next[cont]

        done_counts = torch.zeros(self.n_s * self.n_a, device=dev)
        done_counts.index_add_(0, s_f * self.n_a + a_f, d_f)
        tot_counts = torch.zeros(self.n_s * self.n_a, device=dev)
        tot_counts.index_add_(0, s_f * self.n_a + a_f, torch.ones_like(d_f))
        p_done = (
            ((done_counts + alpha) / (tot_counts + 2 * alpha))
            .view(self.n_s, self.n_a)
            .clamp(0.0, 1.0)
        )

        flat = sc * self.n_a + ac
        if self.factored_dims is None:
            counts = torch.zeros(self.n_s * self.n_a, self.n_s, device=dev)
            counts.index_put_(
                (flat, sn), torch.ones_like(sn, dtype=torch.float32), accumulate=True
            )
            trans = _norm(counts + alpha).view(self.n_s, self.n_a, self.n_s)
            return trans, p_done, None

        b = int(self.n_bins)
        sd_next = data.sdims_next[cont]  # (M, D)
        trans_f: List[torch.Tensor] = []
        for d in range(self.factored_dims):
            counts = torch.zeros(self.n_s * self.n_a, b, device=dev)
            counts.index_put_(
                (flat, sd_next[:, d]),
                torch.ones_like(flat, dtype=torch.float32),
                accumulate=True,
            )
            trans_f.append(_norm(counts + alpha).view(self.n_s, self.n_a, b))
        return None, p_done, trans_f

    # -- POMDP: latent S chain (Baum-Welch) x per-episode latent U --
    def _fit_pomdp(
        self,
        data: EpisodeData,
        n_u: int,
        fit_ua: bool | None = None,
        fit_ur: bool | None = None,
    ) -> CBNParams:
        """Joint EM over the latent state chain (forward-backward) and the
        episode-latent U (exact enumeration — the discrete latent block).
        ``data.s`` holds OBSERVATION symbols; latent cardinality is
        ``opts.n_latent``. One alpha/beta pass per (EM iter, u) computes the
        episode loglik, gamma and xi counts together."""
        if fit_ua is None:
            fit_ua = self.graph.has_edge("U", "A")
        if fit_ur is None:
            fit_ur = self.graph.has_edge("U", "R")
        dev = self.device
        alpha = float(self.opts.alpha)
        s_lat = int(self.opts.n_latent)
        n_ep, t_len = data.s.shape
        mask = data.mask
        eff_u = n_u if (fit_ua or fit_ur) else 1
        n_ua = eff_u if fit_ua else 1
        n_ur = eff_u if fit_ur else 1

        # Residual observable (vs the marginal per-(obs-symbol, action) reward
        # mean) — the canonicalization anchor, exactly as the MDP path.
        valid = mask > 0
        o_f, a_f, r_f = data.s[valid], data.a[valid], data.rew[valid]
        oa_counts = torch.zeros(self.n_o * self.n_a, device=dev)
        oa_counts.index_add_(0, o_f * self.n_a + a_f, torch.ones_like(r_f))
        r_sum = torch.zeros(self.n_o * self.n_a, device=dev)
        r_sum.index_add_(0, o_f * self.n_a + a_f, r_f)
        rbar = (r_sum / oa_counts.clamp_min(1.0)).view(self.n_o, self.n_a)
        resid_step = torch.zeros_like(data.rew)
        resid_step[valid] = r_f - rbar[o_f, a_f]
        residuals = (resid_step * mask).sum(1) / data.lengths

        # Seeded near-uniform init (dedicated generator; global stream
        # untouched). Emission gets an identity-leaning tilt when the symbol
        # and latent spaces align (the MDP-limit prior of the declared S -> O).
        def _rand(*shape):
            return torch.rand(*shape, generator=self.gen).to(dev) * 0.5 + 0.75

        p0 = _norm(_rand(s_lat))
        trans = _norm(_rand(s_lat, self.n_a, s_lat))
        emis = _rand(s_lat, self.n_o)
        if s_lat == self.n_o:
            emis = emis + 2.0 * torch.eye(s_lat, device=dev)
        emis = _norm(emis)
        pb_a = _norm(_rand(n_ua, s_lat, self.n_a))
        p_r = _norm(_rand(n_ur, s_lat, self.n_a, self.n_c))
        p_u = torch.full((eff_u,), 1.0 / eff_u, device=dev)
        p_done_s = torch.full((s_lat, self.n_a), 0.01, device=dev)
        q_u = torch.full((n_ep, eff_u), 1.0 / eff_u, device=dev)

        arange_ep = torch.arange(n_ep, device=dev)
        last = (data.lengths.long() - 1).clamp_min(0)
        flat_ac = (data.a * self.n_c + data.rc).reshape(-1)
        dn_idx = (data.a.reshape(-1) * 2) + (data.done.reshape(-1) > 0.5).long()

        for _ in range(max(1, int(self.opts.pomdp_em_iters))):
            log_p0 = torch.log(p0.clamp_min(_EPS))
            log_tr = torch.log(trans.clamp_min(_EPS))
            log_em = torch.log(emis.clamp_min(_EPS))
            log_pb = torch.log(pb_a.clamp_min(_EPS))
            log_pr = torch.log(p_r.clamp_min(_EPS))

            ep_ll = torch.zeros(n_ep, eff_u, device=dev)
            # Per-u sufficient statistics from ONE alpha/beta pass each.
            stats = []
            for u in range(eff_u):
                ua, ur = min(u, n_ua - 1), min(u, n_ur - 1)
                loc = self._local_evidence_pomdp(log_em, log_pb, log_pr, ua, ur, data)
                log_alpha = torch.full((n_ep, t_len, s_lat), _NEG, device=dev)
                log_alpha[:, 0] = log_p0 + loc[:, 0]
                for t in range(1, t_len):
                    prev = log_alpha[:, t - 1].unsqueeze(2)  # (N, S, 1)
                    step = torch.logsumexp(
                        prev + log_tr[:, data.a[:, t - 1]].permute(1, 0, 2), dim=1
                    )
                    cur = step + loc[:, t]
                    live = (mask[:, t] > 0).unsqueeze(1)
                    log_alpha[:, t] = torch.where(live, cur, log_alpha[:, t - 1])
                log_beta = torch.zeros(n_ep, t_len, s_lat, device=dev)
                for t in range(t_len - 2, -1, -1):
                    nxt = (loc[:, t + 1] + log_beta[:, t + 1]).unsqueeze(1)
                    step = torch.logsumexp(
                        log_tr[:, data.a[:, t]].permute(1, 0, 2) + nxt, dim=2
                    )
                    live = (mask[:, t + 1] > 0).unsqueeze(1)
                    log_beta[:, t] = torch.where(live, step, torch.zeros_like(step))
                ll_u = torch.logsumexp(log_alpha[arange_ep, last], dim=1)
                ep_ll[:, u] = ll_u
                stats.append((loc, log_alpha, log_beta, ll_u))

            # Discrete latent block: episode posterior over U (exact), then
            # the guarded canonicalization — every iteration, like the E-step.
            if eff_u > 1:
                q_u = torch.softmax(ep_ll + torch.log(p_u.clamp_min(_EPS)), dim=1)
                q_u = self._canonicalize(q_u, residuals)
                p_u = _norm(q_u.sum(0) + alpha)

            # Accumulate expected counts (gamma / xi, q_u-weighted).
            c_p0 = torch.zeros(s_lat, device=dev)
            c_tr = torch.zeros(s_lat, self.n_a, s_lat, device=dev)
            c_em = torch.zeros(s_lat, self.n_o, device=dev)
            c_pb = torch.zeros(n_ua, s_lat, self.n_a, device=dev)
            c_pr = torch.zeros(n_ur, s_lat, self.n_a * self.n_c, device=dev)
            c_done = torch.zeros(s_lat, self.n_a * 2, device=dev)
            for u in range(eff_u):
                ua, ur = min(u, n_ua - 1), min(u, n_ur - 1)
                loc, log_alpha, log_beta, ll_u = stats[u]
                log_gamma = (log_alpha + log_beta - ll_u.reshape(n_ep, 1, 1)).clamp(
                    min=-60.0
                )
                gamma = torch.exp(log_gamma) * mask.unsqueeze(2)
                gamma = gamma * q_u[:, u].reshape(n_ep, 1, 1)
                c_p0 += gamma[:, 0].sum(0)
                g_flat = gamma.reshape(-1, s_lat).T  # (S, N*T)
                c_em.index_add_(1, data.s.reshape(-1), g_flat)
                c_pb[ua].index_add_(1, data.a.reshape(-1), g_flat)
                c_pr[ur].index_add_(1, flat_ac, g_flat)
                c_done.index_add_(1, dn_idx, g_flat)
                # Xi counts, streamed over t; per-action scatter (n_a is tiny).
                wq = q_u[:, u]
                for t in range(t_len - 1):
                    live = mask[:, t + 1] > 0
                    if not bool(live.any()):
                        continue
                    lx = (
                        log_alpha[:, t].unsqueeze(2)
                        + log_tr[:, data.a[:, t]].permute(1, 0, 2)
                        + (loc[:, t + 1] + log_beta[:, t + 1]).unsqueeze(1)
                        - ll_u.reshape(n_ep, 1, 1)
                    ).clamp(min=-60.0)
                    xi = torch.exp(lx) * (wq * live.float()).reshape(n_ep, 1, 1)
                    for a_val in torch.unique(data.a[:, t][live]):
                        sel = live & (data.a[:, t] == a_val)
                        c_tr[:, int(a_val)] += xi[sel].sum(0)

            # M-step (latent chain + channels), Dirichlet-smoothed.
            p0 = _norm(c_p0 + alpha)
            trans = _norm(c_tr + alpha)
            emis = _norm(c_em + alpha)
            pb_a = _norm(c_pb + alpha)
            p_r = _norm(c_pr.view(n_ur, s_lat, self.n_a, self.n_c) + alpha)
            c_done2 = c_done.view(s_lat, self.n_a, 2)
            tot = c_done2.sum(-1).clamp_min(_EPS)
            p_done_s = ((c_done2[..., 1] + alpha) / (tot + 2 * alpha)).clamp(0.0, 1.0)

        return CBNParams(
            p_u=p_u,
            pb_a=pb_a,
            p_r=p_r,
            p0=p0,
            trans=trans,
            p_done=p_done_s,
            emis=emis,
            q_u=q_u,
            visited_sa=torch.ones(s_lat, self.n_a, dtype=torch.bool, device=dev),
            visited_s=torch.ones(s_lat, dtype=torch.bool, device=dev),
        )

    # ------------------------------------------------- router components (A3)
    def router_components(self, data: EpisodeData) -> Dict[str, float]:
        """Channel-split latent-evidence statistics (amendment A3).

        Fits three RESTRICTED variants on a deterministic episode split and
        scores held-out per-transition log-likelihood improvements:

          * ``delta_a`` — latent-improves-P(A|.) (U -> A only): the
            CONFOUNDING-specific component.
          * ``delta_r`` — latent-improves-P(R|.) (U -> R only): the
            heterogeneity / partial-observability component. It legitimately
            fires on the basic arm too (c_r > 0 injects action-INDEPENDENT
            U reward noise there — ``c_r_for``), which is exactly why the
            split exists: a confounded verdict requires ``delta_a``.

        Never consumes the realized U (leakage rule R5)."""
        n_ep = data.n_episodes
        every = max(2, round(1.0 / max(self.opts.holdout_frac, 1e-6)))
        hold = torch.arange(n_ep) % every == 0
        train = data.subset((~hold).nonzero(as_tuple=True)[0])
        test = data.subset(hold.nonzero(as_tuple=True)[0])
        fitter = self._fit_pomdp if self.pomdp else self._fit_mdp
        m0 = fitter(train, self.n_u, fit_ua=False, fit_ur=False)
        m_a = fitter(train, self.n_u, fit_ua=True, fit_ur=False)
        m_r = fitter(train, self.n_u, fit_ua=False, fit_ur=True)
        ll0 = self._heldout_ll(m0, test)
        lla = self._heldout_ll(m_a, test)
        llr = self._heldout_ll(m_r, test)
        n_tr = float(test.mask.sum().item())
        return {
            "delta_a": (lla - ll0) / max(n_tr, 1.0),
            "delta_r": (llr - ll0) / max(n_tr, 1.0),
            "heldout_transitions": n_tr,
        }

    def _heldout_ll(self, params: CBNParams, data: EpisodeData) -> float:
        """Held-out marginal loglik of the observable stream. MDP: (A, R)
        conditioned on observed S, exact U enumeration. POMDP: (O, A, R) with
        the latent chain forward-marginalized per u."""
        dev = self.device
        n_u_a, n_u_r = params.pb_a.shape[0], params.p_r.shape[0]
        eff_u = max(n_u_a, n_u_r)
        n_ep = data.n_episodes
        lp_u = torch.log(params.p_u.clamp_min(_EPS))
        if params.p_u.numel() != eff_u:
            lp_u = torch.full((eff_u,), -math.log(eff_u), device=dev)
        if self.pomdp:
            log_p0 = torch.log(params.p0.clamp_min(_EPS))
            log_tr = torch.log(params.trans.clamp_min(_EPS))
            log_em = torch.log(params.emis.clamp_min(_EPS))
            log_pb = torch.log(params.pb_a.clamp_min(_EPS))
            log_pr = torch.log(params.p_r.clamp_min(_EPS))
            t_len = data.s.shape[1]
            ll = torch.zeros(n_ep, eff_u, device=dev)
            for u in range(eff_u):
                loc = self._local_evidence_pomdp(
                    log_em, log_pb, log_pr, min(u, n_u_a - 1), min(u, n_u_r - 1), data
                )
                log_alpha = log_p0 + loc[:, 0]
                for t in range(1, t_len):
                    prev = log_alpha.unsqueeze(2)
                    step = torch.logsumexp(
                        prev + log_tr[:, data.a[:, t - 1]].permute(1, 0, 2), dim=1
                    )
                    cur = step + loc[:, t]
                    live = (data.mask[:, t] > 0).unsqueeze(1)
                    log_alpha = torch.where(live, cur, log_alpha)
                ll[:, u] = torch.logsumexp(log_alpha, dim=1)
            return float(torch.logsumexp(ll + lp_u, dim=1).sum().item())
        valid = data.mask > 0
        s_f, a_f, rc_f = data.s[valid], data.a[valid], data.rc[valid]
        ep_of = valid.nonzero(as_tuple=True)[0]
        log_pb = torch.log(params.pb_a.clamp_min(_EPS))
        log_pr = torch.log(params.p_r.clamp_min(_EPS))
        ll = torch.zeros(n_ep, eff_u, device=dev)
        for u in range(eff_u):
            step_ll = self._step_loglik_mdp(
                log_pb, log_pr, min(u, n_u_a - 1), min(u, n_u_r - 1), s_f, a_f, rc_f
            )
            ll[:, u].index_add_(0, ep_of, step_ll)
        return float(torch.logsumexp(ll + lp_u, dim=1).sum().item())

    # ----------------------------------------------------------- Q_do (B4)
    def q_do(self, gamma: float, tol: float = 1e-6, max_iter: int = 500):
        """THE target-computation boundary (A10 seam): interventional Q by
        exact value iteration on the MUTILATED channel,

            Q(u, s, a) = E[R | s, a, u] + gamma * (1 - p_done(s, a))
                         * E_{s' ~ T(.|s, a)} [ max_a' Q(u, s', a') ]

        The behavior CPD never enters — the mutilated channel has no edges
        into A (``CellGraph.mutilated_edges``). U indexes strata (static per
        episode; the finite mixture over U is the discrete latent block); the
        marginal-vs-u0 DEPLOY is the head's concern, not this backup's.
        Returns ``(n_u_r, n_s, n_a)``."""
        p = self.params
        r_mean = (p.p_r * self.class_values.view(1, 1, 1, -1)).sum(-1)
        n_u, n_s, n_a = r_mean.shape
        cont = (1.0 - p.p_done).clamp(0.0, 1.0).view(1, n_s, n_a)
        q = torch.zeros_like(r_mean)
        for _ in range(max_iter):
            v = q.max(dim=-1).values  # (n_u, n_s)
            ev = self._expected_next_v(v)
            q_new = r_mean + float(gamma) * cont * ev
            delta = float((q_new - q).abs().max().item())
            q = q_new
            if delta < tol:
                break
        return q

    def _expected_next_v(self, v: torch.Tensor) -> torch.Tensor:
        """E_{s'|s,a}[v(u, s')] under the fitted dynamics — dense joint mode,
        or the A8 factored contraction (per-dim CPT product), ``(n_u, n_s,
        n_a)``."""
        if self.params.trans is not None:
            return torch.einsum("saj,uj->usa", self.params.trans, v)
        b = int(self.n_bins)
        d_dims = int(self.factored_dims)
        n_u = v.shape[0]
        x = v.reshape(n_u, *([b] * d_dims))
        # Successive contraction of each next-state dim with its per-dim CPT:
        # after step d, x has shape (n_u, n_s, n_a, b^(D-d-1)).
        x = torch.einsum("sab,ub...->usa...", self.params.trans_f[0], x)
        for d in range(1, d_dims):
            x = torch.einsum("sab,usab...->usa...", self.params.trans_f[d], x)
        return x

    # ------------------------------------------------------------- NBN mirror
    def build_nbn(self):
        """Mirror the two channels as ``nbn`` networks whose DAGs come from
        the DECLARED graph (A9.2: observational = ``graph.edges``, mutilated =
        ``graph.mutilated_edges()``), with SHARED ``CategoricalTableMechanism``
        modules for every non-A node (A7). Joint-transition MDP mode only (the
        factored path has no single S' CPT to mirror; documented in
        docs/grace.md)."""
        if self.pomdp or self.params is None or self.params.trans is None:
            return None, None
        from nbn import (
            CategoricalTableMechanism,
            DiscreteVariable,
            NeuralBayesianNetwork,
        )

        p = self.params
        n_u = max(p.pb_a.shape[0], p.p_r.shape[0], p.p_u.numel())
        card = {
            "U": n_u,
            "S": self.n_s,
            "O": self.n_s,  # identity emission in MDP cells
            "A": self.n_a,
            "R": self.n_c,
            "S_next": self.n_s,
        }
        node_names = [n.name for n in self.graph.nodes]
        variables = {name: DiscreteVariable(name, card[name]) for name in node_names}
        obs_edges = list(self.graph.edges)
        mut_edges = list(self.graph.mutilated_edges())
        net_obs = NeuralBayesianNetwork(obs_edges, variables, device=str(self.device))
        net_mut = NeuralBayesianNetwork(mut_edges, variables, device=str(self.device))

        def _mech(probs: torch.Tensor, parent_cards, n_classes: int):
            # The same private-field population pattern as
            # NeuralBayesianNetwork.from_bif (network.py) — incl. _n_classes,
            # which tabulate()/VE need.
            m = CategoricalTableMechanism()
            flat = probs.reshape(-1, n_classes).clamp_min(_EPS)
            m._logits = torch.log(flat).to(self.device)
            m._n_classes = int(n_classes)
            m._parent_cards = list(parent_cards)
            strides: list[int] = []
            acc = 1
            for c in reversed(list(parent_cards)):
                strides.append(acc)
                acc *= c
            m._parent_strides = list(reversed(strides))
            m._class_values = torch.arange(n_classes, device=self.device).float()
            return m

        def _parents(net, node: str) -> list[str]:
            return list(net.dag.parents(node))

        def _cpt_for(node: str, net, mutilated: bool) -> torch.Tensor:
            """The fitted CPT laid out in the net's own parent order."""
            parents = _parents(net, node)
            if node == "U":
                return p.p_u.view(1, -1)
            if node == "S":
                return p.p0.view(1, -1)
            if node == "O":  # identity emission (MDP mirror)
                return torch.eye(self.n_s, device=self.device)
            if node == "S_next":  # parents (S, A)
                t = p.trans  # (S, A, S')
                perm = [("S", 0), ("A", 1)]
                order = [dict(perm)[x] for x in parents]
                return t.permute(*order, 2)
            if node == "R":
                # p_r is (U, S, A, C); missing-U variants broadcast to n_u.
                pr = p.p_r
                if pr.shape[0] != n_u:
                    pr = pr.expand(n_u, -1, -1, -1)
                axis = {"U": 0, "S": 1, "A": 2}
                order = [axis[x] for x in parents]
                return pr.permute(*order, 3)
            if node == "A":
                if mutilated:
                    raise AssertionError("mutilated A is a root (surgery)")
                pb = p.pb_a  # (U, S|O, A)
                if pb.shape[0] != n_u:
                    pb = pb.expand(n_u, -1, -1)
                axis = {"U": 0, "S": 1, "O": 1}
                order = [axis[x] for x in parents]
                return pb.permute(*order, 2)
            raise KeyError(node)

        # Shared modules for every node whose CPT is channel-invariant (A7):
        # the SAME object registered in both nets.
        for node in node_names:
            if node == "A":
                continue
            parents = _parents(net_obs, node)
            probs = _cpt_for(node, net_obs, mutilated=False)
            mech = _mech(probs, [card[x] for x in parents], card[node])
            net_obs.set_mechanism(node, mech)
            net_mut.set_mechanism(node, mech)  # same module object
        # A: behavior CPD observationally; a parentless uniform prior in the
        # mutilated net (the do-surgery makes A a root).
        pa = _parents(net_obs, "A")
        net_obs.set_mechanism(
            "A", _mech(_cpt_for("A", net_obs, False), [card[x] for x in pa], self.n_a)
        )
        net_mut.set_mechanism(
            "A",
            _mech(
                torch.full((1, self.n_a), 1.0 / self.n_a, device=self.device),
                [],
                self.n_a,
            ),
        )
        self._nbn_obs, self._nbn_mut = net_obs, net_mut
        # A7: the VE plan/factor caches key on id(model) — every (re)build MUST
        # invalidate, or a stale mutilation answers the next query.
        self.invalidate_nbn_cache()
        return net_obs, net_mut

    def invalidate_nbn_cache(self) -> None:
        """A7: nbn's VE caches are INSTANCE-level dicts keyed by ``id(model)``
        (``tensor_ve.py``), and each model lazily holds its own engine. After
        any in-place CPT change or a rebuild, drop the caches on any LIVE
        engine (HybridRouter wraps its VE as ``_ve``) and null the lazy engine
        slot so the next query rebuilds clean — the same invalidation
        ``NeuralBayesianNetwork.to()`` performs."""
        for net in (self._nbn_obs, self._nbn_mut):
            if net is None:
                continue
            eng = getattr(net, "_engine", None)
            ve = getattr(eng, "_ve", None)
            if ve is None and hasattr(eng, "_factor_cache"):
                ve = eng
            if ve is not None:
                ve.invalidate_cache(net)
            net._engine = None

    # --------------------------------------------------------------- ensemble
    def bootstrap_q_do(
        self, data: EpisodeData, gamma: float, k: int
    ) -> List[torch.Tensor]:
        """K episode-level bootstrap refits -> K ``q_do`` tables (B4's minimal
        interval mode). Resampling uses the dedicated generator only."""
        out: List[torch.Tensor] = []
        n = data.n_episodes
        for member in range(int(k)):
            idx = torch.randint(0, n, (n,), generator=self.gen)
            member_cbn = TemplateCBN(
                self.graph,
                self.n_s,
                self.n_a,
                self.n_c,
                self.class_values,
                self.opts,
                self.device,
                n_obs_symbols=self.n_o,
                factored_dims=self.factored_dims,
                n_bins=self.n_bins,
            )
            member_cbn.gen = torch.Generator().manual_seed(
                int(self.opts.seed) * 1000 + 7 * member + 1
            )
            member_cbn.fit(data.subset(idx))
            out.append(member_cbn.q_do(gamma))
        return out

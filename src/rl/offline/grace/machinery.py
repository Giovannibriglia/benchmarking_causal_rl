"""GRACE per-agent machinery: dataset ingestion, CBN fit, router, serving.

One ``GraceMachinery`` per grace arm. ``fit_from_buffer`` runs ONCE at the
runner's ``set_sequence_buffer`` handoff (offline: the dataset is static, so
the CBN, router verdict, ensemble and Q_do tables are all deterministic
functions of it); the online arms re-enter it on a refresh cadence through the
wrapped ``learn`` (the Gate-B label-persistence convention re-runs the
canonicalization every refresh).

R5 (leakage): ingestion reads ONLY the five base keys (obs, actions, rewards,
next_obs, dones) from the buffer's transitions — never ``confounder_u`` and
never the generation-time gate metadata. Asserted in ``test_grace_router``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from src.rl.offline.grace.cbn import EpisodeData, GraceOptions, TemplateCBN
from src.rl.offline.grace.cell_graph import CellGraph, identification_report
from src.rl.offline.grace.discretizer import Discretizer, RewardCodec
from src.rl.offline.grace.filter import BeliefFilter
from src.rl.offline.grace.router import RegimeRouter, RouterVerdict

_FIVE_KEYS = ("obs", "actions", "rewards", "next_obs", "dones")


class GraceMachinery:
    """Everything grace-side that is not the trainable Q_obs head."""

    def __init__(
        self,
        graph: CellGraph,
        opts: GraceOptions,
        n_actions: int,
        device: torch.device,
        gamma: float,
        env_id: str | None = None,
    ) -> None:
        graph.validate()
        self.graph = graph
        self.opts = opts
        self.n_a = int(n_actions)
        self.device = device
        self.gamma = float(gamma)
        self.env_id = env_id
        self.id_report = identification_report(graph)
        self.ready = False
        self.online_refresh_interval: Optional[int] = None  # online arms set it
        self._updates = 0
        self._seq_buffer = None
        # Fit products.
        self.discretizer: Discretizer | None = None
        self.codec: RewardCodec | None = None
        self.cbn: TemplateCBN | None = None
        self.belief_filter: BeliefFilter | None = None
        self.verdict: RouterVerdict | None = None
        self.q_do_table: torch.Tensor | None = None  # (n_u, n_s, n_a)
        self.q_lo: torch.Tensor | None = None  # (n_s, n_a) deployed lower
        self.q_hi: torch.Tensor | None = None
        self.visited_s: torch.Tensor | None = None
        self._diag: Dict[str, float] = {}

    # ------------------------------------------------------------- ingestion
    def _episode_data(self, episodes) -> tuple[EpisodeData, torch.Tensor]:
        dev = self.device
        n = len(episodes)
        lengths = torch.tensor([len(e) for e in episodes], dtype=torch.float32)
        t_max = int(lengths.max().item())
        obs_dim = int(episodes[0][0]["obs"].reshape(-1).shape[0])
        obs = torch.zeros(n, t_max, obs_dim)
        nxt = torch.zeros(n, t_max, obs_dim)
        act = torch.zeros(n, t_max, dtype=torch.long)
        rew = torch.zeros(n, t_max)
        done = torch.zeros(n, t_max)
        mask = torch.zeros(n, t_max)
        for i, ep in enumerate(episodes):
            t = len(ep)
            # Five keys ONLY (R5) — anything else in the transition dicts
            # (confounder_u, r_tau, coverage_min, ...) is never read.
            obs[i, :t] = torch.stack([tr["obs"].reshape(-1) for tr in ep]).float()
            nxt[i, :t] = torch.stack([tr["next_obs"].reshape(-1) for tr in ep]).float()
            act[i, :t] = torch.stack([tr["actions"].reshape(()) for tr in ep]).long()
            rew[i, :t] = torch.stack([tr["rewards"].reshape(()) for tr in ep]).float()
            done[i, :t] = torch.stack([tr["dones"].reshape(()).float() for tr in ep])
            mask[i, :t] = 1.0

        n_bins = self.opts.bins_for(obs_dim)
        self.discretizer = Discretizer(n_bins).fit(
            obs.reshape(-1, obs_dim)[mask.reshape(-1) > 0]
        )
        self.codec = RewardCodec(self.opts.reward_max_classes).fit(
            rew.reshape(-1)[mask.reshape(-1) > 0]
        )
        flat_mask = mask.reshape(-1) > 0
        s = torch.zeros(n * t_max, dtype=torch.long)
        s[flat_mask] = self.discretizer.encode(obs.reshape(-1, obs_dim)[flat_mask])
        s_next = torch.zeros(n * t_max, dtype=torch.long)
        s_next[flat_mask] = self.discretizer.encode(nxt.reshape(-1, obs_dim)[flat_mask])
        rc = torch.zeros(n * t_max, dtype=torch.long)
        rc[flat_mask] = self.codec.encode(rew.reshape(-1)[flat_mask])
        sdims_next = None
        if self.discretizer.joint_size > int(self.opts.joint_cap):
            sd = torch.zeros(n * t_max, obs_dim, dtype=torch.long)
            sd[flat_mask] = self.discretizer.encode_dims(
                nxt.reshape(-1, obs_dim)[flat_mask]
            )
            sdims_next = sd.reshape(n, t_max, obs_dim).to(dev)
        data = EpisodeData(
            s=s.reshape(n, t_max).to(dev),
            a=act.to(dev),
            rc=rc.reshape(n, t_max).to(dev),
            s_next=s_next.reshape(n, t_max).to(dev),
            done=done.to(dev),
            mask=mask.to(dev),
            lengths=lengths.to(dev),
            rew=rew.to(dev),
            sdims_next=sdims_next,
        )
        return data, obs

    # ------------------------------------------------------------------- fit
    def fit_from_buffer(self, seq_buffer) -> None:
        self._seq_buffer = seq_buffer
        episodes = list(seq_buffer.iter_episodes())
        if not episodes:
            return
        data, _ = self._episode_data(episodes)
        opts = self.opts
        pomdp = self.graph.pomdp
        n_bins = self.discretizer.n_bins
        joint = self.discretizer.joint_size
        factored = None if joint <= int(opts.joint_cap) else self.discretizer.obs_dim
        # POMDP: the observed symbol space is the discretized MASKED obs; the
        # latent S cardinality is opts.n_latent. MDP: S == the joint bin.
        n_states = int(opts.n_latent) if pomdp else joint
        cbn = TemplateCBN(
            self.graph,
            n_states=n_states,
            n_actions=self.n_a,
            n_classes=self.codec.n_classes,
            class_values=self.codec.class_values,
            opts=opts,
            device=self.device,
            n_obs_symbols=joint if pomdp else None,
            factored_dims=factored,
            n_bins=n_bins,
        )
        cbn.fit(data)
        self.cbn = cbn
        self.belief_filter = BeliefFilter(
            cbn.params, self.codec.n_classes, pomdp, self.device
        )
        self.visited_s = cbn.params.visited_s
        # Q_do (the single target-computation boundary) + the ensemble.
        self.q_do_table = cbn.q_do(self.gamma)
        members = (
            cbn.bootstrap_q_do(data, self.gamma, opts.ensemble_k)
            if opts.interval
            else []
        )
        if members:
            deployed = torch.stack([self._deploy(m) for m in members])
            self.q_lo = deployed.min(dim=0).values
            self.q_hi = deployed.max(dim=0).values
        else:
            served = self._deploy(self.q_do_table)
            self.q_lo = served.clone()
            self.q_hi = served.clone()

        # Router statistics (data-derivable only) + verdict.
        stats: Dict[str, float] = {}
        stats["coverage"] = self._coverage_stat(data)
        stats["width"] = float(
            (self.q_hi - self.q_lo)[self.visited_s.to(self.q_hi.device)].mean().item()
            if bool(self.visited_s.any())
            else 0.0
        )
        q_u = cbn.params.q_u
        stats["separability"] = (
            float(q_u.max(dim=1).values.median().item()) if q_u is not None else 0.0
        )
        prior_belief = self.belief_filter.reset(1)
        stats["belief_entropy"] = float(self.belief_filter.entropy(prior_belief).item())
        if opts.router:
            stats.update(cbn.router_components(data))
        router = RegimeRouter.from_reference(
            self.env_id, graph_ok=self.id_report["confounded_serving_ok"]
        )
        if opts.router:
            self.verdict = router.verdict(stats)
        else:
            # grace_no_router: ALWAYS-CAUSAL (the approved ablation switch) —
            # serve Q_do unconditionally; label mirrors that choice.
            self.verdict = RouterVerdict(
                label="always_causal",
                serve="do",
                calibrated=False,
                reasons=("router disabled (grace_no_router arm)",),
                stats=stats,
            )
        self._diag = {
            "router_delta_a": stats.get("delta_a", float("nan")),
            "router_delta_r": stats.get("delta_r", float("nan")),
            "router_coverage": stats.get("coverage", float("nan")),
            "ensemble_width": stats.get("width", float("nan")),
            "grace_separability": stats.get("separability", float("nan")),
            "grace_belief_entropy": stats.get("belief_entropy", float("nan")),
        }
        self.ready = True

    def refresh(self) -> None:
        """Online cadence: rebuild from the current (rolling) buffer. The
        canonicalization re-runs inside the fit — labels re-anchor every
        refresh (the Gate-B convention)."""
        if self._seq_buffer is not None:
            self.fit_from_buffer(self._seq_buffer)

    def tick_update(self) -> None:
        """Called by the wrapped ``learn`` once per optimiser step; drives the
        online refresh cadence (offline arms have no cadence — the dataset is
        static and the fit already happened at the handoff)."""
        self._updates += 1
        if (
            self.online_refresh_interval
            and self._updates % int(self.online_refresh_interval) == 0
        ):
            self.refresh()

    # -------------------------------------------------------------- serving
    def _deploy(self, q_do: torch.Tensor) -> torch.Tensor:
        """(n_u, n_s, n_a) -> deployed (n_s, n_a). ``marginal`` = E_U under
        the fitted prior (the ablation's oracle-comparable convention);
        ``u0`` = the U=0 reference stratum — index 0 IS the low-residual
        (clean-world) stratum by the label-swap canonicalization convention
        (cell-9 deploy semantics)."""
        if self.opts.deploy == "u0":
            return q_do[0]
        p_u = self.cbn.params.p_u.view(-1, 1, 1).to(q_do.device)
        if p_u.shape[0] != q_do.shape[0]:
            p_u = torch.full(
                (q_do.shape[0], 1, 1), 1.0 / q_do.shape[0], device=q_do.device
            )
        return (p_u * q_do).sum(0)

    def serve(self, x: torch.Tensor, q_obs_fn: Callable) -> torch.Tensor:
        """The routed head (B5). Falls back to Q_obs on state bins the dataset
        never visited (tabular coverage holes must not serve stale zeros)."""
        with torch.no_grad():
            q_obs = q_obs_fn(x)
            mode = self.verdict.serve if self.verdict is not None else "obs"
            if not self.ready or mode == "obs":
                return q_obs
            if self.graph.pomdp:
                sym = self.discretizer.encode(x.reshape(x.shape[0], -1))
                belief = self.belief_filter.condition_obs(
                    self.belief_filter.reset(x.shape[0]), sym
                )
                b = belief.reshape(x.shape[0], self.cbn.n_s, self.cbn.n_u)
                table = self.q_do_table if mode == "do" else None
                if table is not None:
                    # QMDP-style belief average over (S, U).
                    q = torch.einsum("bsu,usa->ba", b, table.to(x.device))
                else:
                    bs = b.sum(dim=2)  # (B, S) marginal over U
                    q = torch.einsum("bs,sa->ba", bs, self.q_lo.to(x.device))
                return q
            bins = self.discretizer.encode(x.reshape(x.shape[0], -1)).to(
                self.q_lo.device
            )
            table = (self._deploy(self.q_do_table) if mode == "do" else self.q_lo).to(
                x.device
            )
            q_tab = table[bins]
            visited = self.visited_s.to(x.device)[bins]
            return torch.where(visited.unsqueeze(1), q_tab, q_obs)

    # ------------------------------------------------------------ statistics
    def _coverage_stat(self, data: EpisodeData) -> float:
        """State-conditional action coverage: the 10th percentile over
        sufficiently visited state bins of ``min_a pi_b_hat(a | s)``. Never a
        marginal action frequency (the documented recurring trap)."""
        valid = data.mask > 0
        s_f, a_f = data.s[valid], data.a[valid]
        n_s = int(self.cbn.n_s if not self.graph.pomdp else self.cbn.n_o)
        counts = torch.zeros(n_s * self.n_a, device=s_f.device)
        counts.index_add_(
            0, s_f * self.n_a + a_f, torch.ones_like(a_f, dtype=torch.float32)
        )
        counts = counts.view(n_s, self.n_a)
        row_tot = counts.sum(1)
        rows = row_tot >= float(self.opts.min_state_count)
        if not bool(rows.any()):
            return 0.0
        pi_hat = counts[rows] / row_tot[rows].unsqueeze(1)
        min_per_state = pi_hat.min(dim=1).values
        return float(torch.quantile(min_per_state, 0.1).item())

    def diagnostics(self) -> Dict[str, float | str]:
        out: Dict[str, float | str] = dict(self._diag)
        if self.verdict is not None:
            out["router_verdict"] = self.verdict.label
            out["router_serve"] = self.verdict.serve
        return out

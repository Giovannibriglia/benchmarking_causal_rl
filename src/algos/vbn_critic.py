from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import networkx as nx
import pandas as pd
import torch
from torch import nn

from vbn.core import CausalBayesNet

from src.algos.utils import set_learning_and_inference_objects


class VBNCritic:
    """
    Causal-critic mixin providing:
      • BN init / (re)fit hooks and auto-warmup
      • Q_c(s,a), V_c(s) computation
      • causal advantages + old_logp filling
      • optional causal prior π_c ∝ exp(β Q_c)
      • single-file checkpoint bundle for BN state (+ back-compat sidecars)

    Expected to be mixed into a BaseActorCritic-like class exposing:
      - self.device (torch.device)
      - self.env (optional, for auto-warmup)
      - self._encode(), self.actor / self.actor_mu, self.dist_fn, self.is_discrete
      - (optional) self.log_std for continuous policies
    """

    # ────────────────────────── init ──────────────────────────
    def __init__(
        self,
        *,
        causality_init: Optional[dict] = None,
        pi_samples: int = 32,
        kl_coeff: float = 0.0,
        kl_beta: float = 5.0,
    ):
        cfg = causality_init or {}

        # knobs
        self.pi_samples = pi_samples
        self.kl_coeff = kl_coeff
        self.kl_beta = kl_beta

        # BN config / metadata
        self.num_samples = cfg.get("num_samples", 32)
        self.dag: Optional[nx.DiGraph] = cfg.get("dag", None)
        self.types: Optional[dict] = cfg.get("types", None)
        self.cards: Optional[dict] = cfg.get("cards", None)
        self.discrete: bool = cfg.get("discrete", False)
        self.approximate: bool = cfg.get("approximate", True)
        chkpt_path = cfg.get("chkpt_path", None)

        # learning & inference backends
        self.fit_method, self.inf_method = set_learning_and_inference_objects(
            self.discrete, self.approximate
        )
        self.kwargs_inf = {"num_samples": self.num_samples}
        self.kwargs_fit = cfg.get("kwargs_fit", {"epochs": 100, "batch_size": 1024})

        # BN runtime holders
        self.bn: Optional[CausalBayesNet] = None
        self.lp = None  # learned parameters (tensor dict or similar)
        self.inf_obj = None

        # optional pre-instantiate from provided dag/types
        if self.dag is not None and self.types is not None:
            self.bn = CausalBayesNet(self.dag, self.types, self.cards or {})
            if chkpt_path is not None:
                self.lp = self.bn.load_params(str(chkpt_path))
            if self.inf_method is not None:
                self.inf_obj = self.bn.setup_inference(self.inf_method)

        # cache for discrete Q-table -> causal prior
        self._cached_q_table = None

    # ────────────────────────── readiness / warmup ──────────────────────────
    def vbn_is_ready(self) -> bool:
        return self.bn is not None and self.inf_obj is not None

    def _ensure_backends(self) -> None:
        if self.fit_method is None or self.inf_method is None:
            fm, im = set_learning_and_inference_objects(
                bool(getattr(self, "discrete", False)),
                bool(getattr(self, "approximate", True)),
            )
            self.fit_method, self.inf_method = fm, im

    def schema_from_spaces(
        self,
        obs_space: gym.Space,
        act_space: Optional[gym.Space],
        target: str = "return",
    ) -> Tuple[nx.DiGraph, dict, dict]:
        """
        Minimal schema: obs_* -> target; action_* -> target.
        Discrete spaces => type='discrete' (+cards); others => 'continuous'.
        """
        G = nx.DiGraph()
        types, cards = {}, {}

        # observation nodes
        if isinstance(obs_space, gym.spaces.Discrete):
            types["obs_0"] = "discrete"
            cards["obs_0"] = int(obs_space.n)
            obs_nodes = ["obs_0"]
        else:
            k = gym.spaces.utils.flatdim(obs_space)
            obs_nodes = [f"obs_{i}" for i in range(k)]
            for n in obs_nodes:
                types[n] = "continuous"

        # action nodes
        act_nodes = []
        if act_space is not None:
            if isinstance(act_space, gym.spaces.Discrete):
                types["action_0"] = "discrete"
                cards["action_0"] = int(act_space.n)
                act_nodes = ["action_0"]
            else:
                k = gym.spaces.utils.flatdim(act_space)
                act_nodes = [f"action_{i}" for i in range(k)]
                for n in act_nodes:
                    types[n] = "continuous"

        for n in obs_nodes + act_nodes:
            G.add_edge(n, target)

        types[target] = "continuous"
        return G, types, cards

    def ensure_bn(
        self, *, env=None, schema: Optional[Tuple[nx.DiGraph, dict, dict]] = None
    ) -> None:
        """
        Ensure BN object + inference object exist (graph + types/cards available).
        """
        self._ensure_backends()
        if self.bn is None:
            if schema is None:
                if env is None:
                    raise RuntimeError("ensure_bn requires either env or schema.")
                G, types, cards = self.schema_from_spaces(
                    env.observation_space, getattr(env, "action_space", None)
                )
            else:
                G, types, cards = schema
            self.dag, self.types, self.cards = G, types, cards or {}
            self.bn = CausalBayesNet(self.dag, self.types, self.cards)
        if self.inf_obj is None and self.inf_method is not None:
            self.inf_obj = self.bn.setup_inference(self.inf_method)

    def fit_bn_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        schema: Optional[Tuple[nx.DiGraph, dict, dict]] = None,
    ) -> None:
        """
        Fit (or update) BN from a DataFrame with columns obs_* , action_* , return.
        """
        self._ensure_backends()
        if self.bn is None:
            if schema is None:
                G = self._get_dag(df, target_feature="return", do_key="action")
                if self.types is None or self.cards is None:
                    self.types = {str(n): "continuous" for n in G.nodes()}
                    self.cards = {}
                self.dag = G
                self.bn = CausalBayesNet(self.dag, self.types, self.cards)
            else:
                G, types, cards = schema
                self.dag, self.types, self.cards = G, types, cards or {}
                self.bn = CausalBayesNet(self.dag, self.types, self.cards)

        if self.lp is None:
            self.lp = self.bn.fit(self.fit_method, df, **self.kwargs_fit)
        else:
            self.lp = self.bn.add_data(df, update_params=True, return_lp=True)

        if self.inf_obj is None and self.inf_method is not None:
            self.inf_obj = self.bn.setup_inference(self.inf_method)

    def fit_bn_from_rollout(
        self, mem: Optional[dict] = None, *, steps: int = 2048
    ) -> None:
        """
        Fit BN from a rollout buffer. If mem=None, it calls self._collect_rollout().
        """
        self._ensure_backends()
        if mem is None:
            if not hasattr(self, "_collect_rollout"):
                raise RuntimeError(
                    "fit_bn_from_rollout requires _collect_rollout or pass mem=..."
                )
            mem, _, _ = self._collect_rollout()
        self._causal_update(mem)

    def ensure_vbn_initialized(
        self, *, env=None, mem: Optional[dict] = None, steps: int = 2048
    ) -> None:
        """
        High-level: ensure BN+inference exist; if no params, warm-start via rollout.
        """
        if self.vbn_is_ready() and self.lp is not None:
            return
        self.ensure_bn(env=env or getattr(self, "env", None))
        if self.lp is None:
            self.fit_bn_from_rollout(mem=mem, steps=steps)

    # ────────────────────────── core helpers ──────────────────────────
    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, 1) if x.ndim == 1 else x

    def _build_evidence(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        states = self._ensure_2d(states)
        if states.shape[1] == 1:
            return {"obs_0": states.view(-1)}
        return {f"obs_{i}": states[:, i] for i in range(states.shape[1])}

    def _build_do(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        actions = self._ensure_2d(actions)
        if actions.shape[1] == 1:
            return {"action_0": actions.view(-1)}
        return {f"action_{i}": actions[:, i] for i in range(actions.shape[1])}

    @staticmethod
    def _replace_infs(t: torch.Tensor) -> torch.Tensor:
        is_pos_inf = torch.isposinf(t)
        is_neg_inf = torch.isneginf(t)
        finite = torch.isfinite(t)
        t = t.clone()
        if finite.any():
            t[is_pos_inf] = t[finite].max()
            t[is_neg_inf] = t[finite].min()
        else:
            t[is_pos_inf] = 1e6
            t[is_neg_inf] = -1e6
        return t

    def _posterior_expectation(self, posterior_pdfs, posterior_samples) -> torch.Tensor:
        if posterior_samples is not None and torch.isfinite(posterior_samples).any():
            return (
                posterior_samples
                if posterior_samples.ndim == 1
                else posterior_samples.mean(-1)
            )
        if posterior_pdfs is not None:
            return (
                posterior_pdfs
                if posterior_pdfs.ndim == 1
                else posterior_pdfs.max(-1).values
            )
        return torch.zeros(1, device=self.device)

    # ────────────────────────── value / policy ──────────────────────────
    def _q_causal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.vbn_is_ready():
            # try to self-init if we have an env
            if hasattr(self, "env") and self.env is not None:
                self.ensure_vbn_initialized(env=self.env, steps=1024)
        assert self.bn is not None and self.inf_obj is not None, "VBN not initialized."
        evidence = self._build_evidence(states)
        do = self._build_do(actions)
        posterior_pdfs, posterior_samples, _ = self.bn.infer(
            self.inf_obj,
            lp=self.lp,
            evidence=evidence,
            query=["return"],
            do=do,
            return_samples=True,
            **self.kwargs_inf,
        )
        if isinstance(posterior_pdfs, dict):
            posterior_pdfs = posterior_pdfs.get("return", None)
        if isinstance(posterior_samples, dict):
            posterior_samples = posterior_samples.get("return", None)
        q = (
            self._posterior_expectation(posterior_pdfs, posterior_samples)
            .to(self.device)
            .view(-1)
        )
        q = torch.nan_to_num(
            q, nan=(q[torch.isfinite(q)].mean() if torch.isfinite(q).any() else 0.0)
        )
        return self._replace_infs(q)

    def _enumerate_actions(self, dist, B: int) -> torch.Tensor:
        nA = dist.probs.shape[-1]
        a = torch.arange(nA, device=self.device).view(1, nA).repeat(B, 1)
        return a.view(-1, 1)

    def _v_causal(
        self, states: torch.Tensor, dist
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = states.shape[0]
        # exact for single Categorical
        if hasattr(dist, "probs") and dist.probs.ndim == 2:
            nA = dist.probs.shape[-1]
            actions_enum = self._enumerate_actions(dist, B)
            states_rep = states.repeat_interleave(nA, dim=0)
            q_enum = self._q_causal(states_rep, actions_enum).view(B, nA)
            v = (dist.probs * q_enum).sum(-1)
            return v, q_enum
        # MC for multi-discrete/continuous
        with torch.no_grad():
            a_samp = dist.sample((self.pi_samples,))  # [K,B,...]
            aBK = a_samp.permute(1, 0, *range(2, a_samp.ndim))  # [B,K,...]
        if aBK.ndim == 2:  # [B,K] -> [B,K,1]
            aBK = aBK.unsqueeze(-1)
        states_rep = (
            states.unsqueeze(1)
            .expand(-1, self.pi_samples, -1)
            .reshape(-1, states.shape[-1])
        )
        actions_rep = aBK.reshape(-1, aBK.shape[-1])
        q_mc = self._q_causal(states_rep, actions_rep).view(B, self.pi_samples)
        return q_mc.mean(-1), None

    def _policy_dist_from_latent(self, latent):
        """
        Return the current policy distribution from a latent.
        Works for TRPO (has _distribution) and for A2C/PPO (have dist_fn).
        """
        if hasattr(self, "_distribution"):
            return self._distribution(latent)
        if getattr(self, "is_discrete", False):
            logits = self.actor(latent)
            return self.dist_fn(logits)
        mu = self.actor_mu(latent)
        return self.dist_fn(mu)

    # ───────────────────── rollout-time hooks ─────────────────────
    def _get_dag(
        self, data: pd.DataFrame, target_feature: str = "return", do_key: str = "action"
    ):
        if do_key is not None:
            obs_feats = [
                s for s in data.columns if do_key not in s and s != target_feature
            ]
            int_feats = [s for s in data.columns if do_key in s and s != target_feature]
        else:
            obs_feats, int_feats = [s for s in data.columns], []
        edges = [(f, target_feature) for f in obs_feats] + [
            (a, target_feature) for a in int_feats
        ]
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G

    def _causal_update(self, mem):
        obs = mem["obs"].reshape(-1, *mem["obs"].shape[2:])
        acts = mem["actions"].reshape(-1, *mem["actions"].shape[2:])
        rets = mem["returns"].reshape(-1)
        df = pd.DataFrame(
            {
                **(
                    {f"obs_{i}": obs[:, i].cpu().numpy() for i in range(obs.shape[1])}
                    if obs.ndim == 2
                    else {"obs_0": obs.view(-1).cpu().numpy()}
                ),
                **(
                    {
                        f"action_{i}": acts[:, i].cpu().numpy()
                        for i in range(acts.shape[1])
                    }
                    if acts.ndim == 2
                    else {"action_0": acts.view(-1).cpu().numpy()}
                ),
                "return": rets.cpu().numpy(),
            }
        )
        if self.bn is not None:
            self.lp = self.bn.add_data(df, update_params=True, return_lp=True)
        else:
            G = self._get_dag(df)
            if self.types is None or self.cards is None:
                self.types = {str(n): "continuous" for n in G.nodes()}
                self.cards = {}
            self.bn = CausalBayesNet(G, self.types, self.cards)
            self.lp = self.bn.fit(self.fit_method, df, **self.kwargs_fit)
            self.inf_obj = self.bn.setup_inference(self.inf_method)

    def _post_update_fill_adv_and_logp(self, mem):
        """
        Fills:
          • mem["advantages"] = returns - V_causal(s)   [T,N]
          • mem["old_logp"]   = log πθ(a|s)            [T,N]
        Assumes mem has "obs", "actions", "returns".
        """
        device = self.device
        obs = mem["obs"].to(device)  # [T,N,flat] (or [T,N,1] if Discrete index)
        act = mem["actions"].to(device)  # [T,N] or [T,N,Ad]
        ret = mem["returns"].to(device)  # [T,N]

        T, N = obs.shape[:2]
        B = T * N

        # states for BN + latent for policy
        if isinstance(self.encoder, nn.Embedding):
            if obs.ndim == 3 and obs.shape[-1] == 1:
                obs_idx = obs.squeeze(-1)  # [T,N]
            else:
                obs_idx = obs[..., 0]
            states = obs_idx.reshape(B, 1).long()
            latent = self._encode(obs_idx.reshape(B))
        else:
            states = obs.view(B, -1).float()
            latent = self._encode(states)

        dist = self._policy_dist_from_latent(latent)

        # V_causal(s)
        Vc, _ = self._v_causal(states, dist)
        Vc = Vc.view(T, N).detach()

        # advantages
        mem["advantages"] = (ret - Vc).detach()

        # old_logp
        if self.is_discrete:
            act_vec = act.view(B).long()
            logp_vec = dist.log_prob(act_vec)
        else:
            act_dim = self.log_std.numel()
            act_vec = act.view(B, act_dim)
            logp_vec = dist.log_prob(act_vec).sum(-1)
        mem["old_logp"] = logp_vec.view(T, N).detach()

    # ────────────────────────── inference / plotting ──────────────────────────
    def infer_conditional(
        self,
        states: torch.Tensor,
        *,
        actions: Optional[torch.Tensor] = None,
        query: str = "return",
        num_samples: Optional[int] = None,
        return_samples: bool = True,
    ):
        # self-init if needed (eval-time call)
        if not self.vbn_is_ready() and hasattr(self, "env") and self.env is not None:
            self.ensure_vbn_initialized(env=self.env, steps=1024)
        assert self.bn is not None and self.inf_obj is not None, "VBN not initialized."

        def _squeeze_first(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[0] == 1:
                return x[0]
            return x

        num_samples = num_samples or self.num_samples
        evidence = self._build_evidence(self._ensure_2d(states))
        do = None if actions is None else self._build_do(self._ensure_2d(actions))

        posterior_pdfs, posterior_samples, support = self.bn.infer(
            self.inf_obj,
            lp=self.lp,
            evidence=evidence,
            query=[query],
            do=do,
            return_samples=return_samples,
            num_samples=num_samples,
        )

        if isinstance(posterior_pdfs, dict):
            posterior_pdfs = posterior_pdfs.get(query, None)
        if isinstance(posterior_samples, dict):
            posterior_samples = posterior_samples.get(query, None)
        if isinstance(support, dict):
            support = support.get(query, None)

        posterior_pdfs = _squeeze_first(posterior_pdfs)
        posterior_samples = _squeeze_first(posterior_samples)
        support = _squeeze_first(support)

        return {"pdf": posterior_pdfs, "samples": posterior_samples, "support": support}

    # (plot_* helpers unchanged from your version; keep if you use them)

    # ────────────────────────── single-file BN bundle ──────────────────────────
    def _bn_to_bundle(self) -> dict | None:
        """
        Serialize causal BN state into a dict suitable for embedding inside a .pt.
        """
        if self.bn is None or self.lp is None:
            return None

        dag = self.dag
        if dag is None and hasattr(self.bn, "G"):
            dag = self.bn.G
        dag_edges = list(dag.edges()) if isinstance(dag, nx.DiGraph) else []

        buf = io.BytesIO()
        torch.save(self.lp, buf)
        bn_blob = buf.getvalue()

        return {
            "dag_edges": [(str(u), str(v)) for (u, v) in dag_edges],
            "types": self.types or {},
            "cards": self.cards or {},
            "num_samples": int(getattr(self, "num_samples", 32)),
            "discrete": bool(getattr(self, "discrete", False)),
            "approximate": bool(getattr(self, "approximate", True)),
            "inf_method": self.inf_method,
            "fit_method": self.fit_method,
            "bn_blob": bn_blob,
        }

    def _bn_from_bundle(self, bundle: dict) -> None:
        """
        Restore BN from a bundle produced by _bn_to_bundle.
        """
        if not bundle:
            return

        G = nx.DiGraph()
        G.add_edges_from(bundle.get("dag_edges", []))
        self.dag = G
        self.types = bundle.get("types", {})
        self.cards = bundle.get("cards", {})
        self.num_samples = bundle.get("num_samples", self.num_samples)
        self.discrete = bundle.get("discrete", self.discrete)
        self.approximate = bundle.get("approximate", self.approximate)
        self.inf_method = bundle.get("inf_method", self.inf_method)
        self.fit_method = bundle.get("fit_method", self.fit_method)

        if self.bn is None:
            self.bn = CausalBayesNet(self.dag, self.types, self.cards or {})

        bn_blob = bundle.get("bn_blob", None)
        if bn_blob is not None:
            buf = io.BytesIO(bn_blob)
            self.lp = torch.load(buf, map_location=self.device, weights_only=False)
        else:
            self.lp = None

        if self.inf_method is not None and self.bn is not None:
            self.inf_obj = self.bn.setup_inference(self.inf_method)

        # ensure methods/objects exist for future use
        self._ensure_backends()
        if self.inf_obj is None and self.inf_method is not None:
            self.inf_obj = self.bn.setup_inference(self.inf_method)

    # ────────────────────────── legacy sidecars (back-compat) ─────────────────
    def _bn_params_path(self, base_path: Union[str, Path]) -> Path:
        p = Path(base_path)
        if p.suffix != ".pt":
            p = p.with_suffix(".pt")
        return p.with_suffix(".pt").with_name(p.stem + ".td")

    def _bn_meta_path(self, base_path: Union[str, Path]) -> Path:
        p = Path(base_path)
        if p.suffix != ".pt":
            p = p.with_suffix(".pt")
        return p.with_name(p.stem + ".bnmeta.json")

    def save_bn_params(self, base_path: Union[str, Path]) -> None:
        """Legacy: save learned BN params AND sidecar metadata (dag/types/cards)."""
        if self.bn is None or self.lp is None:
            return
        td_path = self._bn_params_path(base_path)
        td_path.parent.mkdir(parents=True, exist_ok=True)
        self.bn.save_params(self.lp, str(td_path))

        meta_path = self._bn_meta_path(base_path)
        dag = self.dag if self.dag is not None else getattr(self.bn, "G", None)
        dag_edges = list(dag.edges()) if isinstance(dag, nx.DiGraph) else []
        meta = {
            "dag_edges": [(str(u), str(v)) for (u, v) in dag_edges],
            "types": self.types or {},
            "cards": self.cards or {},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def load_bn_params(self, base_path: Union[str, Path]) -> None:
        """Legacy: load BN params; also restore dag/types/cards from sidecar if present."""
        td_path = self._bn_params_path(base_path)
        meta_path = self._bn_meta_path(base_path)
        if not td_path.exists():
            return

        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            G = nx.DiGraph()
            G.add_edges_from(meta.get("dag_edges", []))
            self.dag = G
            self.types = meta.get("types", None)
            self.cards = meta.get("cards", None)

        if self.bn is None:
            assert (
                self.dag is not None and self.types is not None
            ), "CBN load requires dag/types available either in memory or in sidecar metadata."
            self.bn = CausalBayesNet(self.dag, self.types, self.cards or {})

        self.lp = self.bn.load_params(str(td_path))
        if self.inf_method is not None:
            self.inf_obj = self.bn.setup_inference(self.inf_method)

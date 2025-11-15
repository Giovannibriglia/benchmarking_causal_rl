# vbn_critic.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import networkx as nx
import pandas as pd
import torch

from new_vbn import VBN
from new_vbn.io import load_vbn, save_vbn
from torch import nn


class VBNCritic:
    """
    Clean causal-critic mixin built on your VBN API.

    • Static mode (time_size=0):   obs_*, action_*  →  return
    • Temporal mode (time_size=K): obs_tk_*, action_tk_* → reward_tk; transitions to obs_t{k+1}_*

    Expects the host class (Actor-Critic) to provide:
      - self.device (torch.device)
      - self.env (optional; used to auto-build schema)
      - self._encode(), self.actor / self.actor_mu, self.dist_fn, self.is_discrete
      - optional self.log_std for continuous policies
    """

    # ────────────────────────── init / config ──────────────────────────
    def __init__(
        self,
        *,
        causality_init: Optional[dict] = None,
        time_size: int = 0,
        learning_method: str = "mdn",
        inference_method: str = "montecarlo.lw",
        num_samples: int = 64,
        pi_samples: int = 32,
        kl_coeff: float = 0.0,
        kl_beta: float = 5.0,
    ):
        cfg = causality_init or {}

        self.time_size = int(cfg.get("time_size", time_size))
        self.learning_method = cfg.get("learning_method", learning_method)
        self.inference_method = cfg.get("inference_method", inference_method)
        self.num_samples = int(cfg.get("num_samples", num_samples))
        self.pi_samples = int(cfg.get("pi_samples", pi_samples))
        self.kl_coeff = float(cfg.get("kl_coeff", kl_coeff))
        self.kl_beta = float(cfg.get("kl_beta", kl_beta))
        self.kwargs_fit = cfg.get("kwargs_fit", {"epochs": 50, "batch_size": 4096})
        self.kwargs_inf = cfg.get("kwargs_inf", {"num_samples": 64})

        # runtime holders
        self.vbn: Optional[VBN] = None
        self.dag: Optional[nx.DiGraph] = None

    # ────────────────────────── readiness ──────────────────────────
    def vbn_is_ready(self) -> bool:
        if self.vbn is None:
            return False
        nodes = getattr(self.vbn, "_nodes", None)
        return isinstance(nodes, dict) and len(nodes) > 0

    # ────────────────────────── schema builders ──────────────────────────
    def _expand_space(
        self, space: Optional[gym.Space], base_name: str
    ) -> Tuple[list[str], Dict[str, int]]:
        """Return node names and (for discrete) cardinalities."""
        if space is None:
            return [], {}
        if isinstance(space, gym.spaces.Discrete):
            return [f"{base_name}_0"], {f"{base_name}_0": int(space.n)}
        k = gym.spaces.utils.flatdim(space)
        return [f"{base_name}_{i}" for i in range(k)], {}

    def _build_static_schema(
        self, obs_space: gym.Space, act_space: Optional[gym.Space]
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        obs_nodes, _ = self._expand_space(obs_space, "obs")
        act_nodes, _ = self._expand_space(act_space, "action")
        for n in obs_nodes + act_nodes:
            G.add_edge(n, "return")
        return G

    def _build_temporal_schema(
        self, obs_space: gym.Space, act_space: Optional[gym.Space], K: int
    ) -> nx.DiGraph:
        assert K > 0
        G = nx.DiGraph()
        obs_slices = []
        act_slices = []
        for k in range(K):
            obs_k, _ = self._expand_space(obs_space, f"obs_t{k}")
            act_k, _ = self._expand_space(act_space, f"action_t{k}")
            r_k = f"reward_t{k}"
            obs_slices.append(obs_k)
            act_slices.append(act_k)
            # parents of reward_tk
            for n in obs_k + act_k:
                G.add_edge(n, r_k)
        # terminal obs_tK
        obs_K, _ = self._expand_space(obs_space, f"obs_t{K}")
        obs_slices.append(obs_K)
        # transitions
        for k in range(K):
            for a in act_slices[k]:
                for n1 in obs_slices[k + 1]:
                    G.add_edge(a, n1)
            for n0 in obs_slices[k]:
                for n1 in obs_slices[k + 1]:
                    G.add_edge(n0, n1)
        return G

    def schema_from_spaces(
        self, obs_space: gym.Space, act_space: Optional[gym.Space]
    ) -> nx.DiGraph:
        if self.time_size == 0:
            return self._build_static_schema(obs_space, act_space)
        return self._build_temporal_schema(obs_space, act_space, self.time_size)

    # ────────────────────────── ensure BN ──────────────────────────
    def ensure_bn(self, *, env=None, schema: Optional[nx.DiGraph] = None) -> None:
        """
        Ensure a VBN object exists and is configured with learning method.
        We intentionally defer binding the inferencer until after first fit/update.
        """
        if self.vbn is not None:
            return
        if schema is None:
            if env is None:
                raise RuntimeError("ensure_bn requires either env or schema.")
            schema = self.schema_from_spaces(
                env.observation_space, getattr(env, "action_space", None)
            )
        self.dag = schema
        self.vbn = VBN(
            dag=self.dag, device=self.device, seed=getattr(self, "seed", None)
        )
        self.vbn.set_learning_method(self.learning_method)
        # NOTE: do NOT call set_inference_method here. We re-bind it after each fit/update.

    # ────────────────────────── helpers ──────────────────────────
    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, 1) if x.ndim == 1 else x

    def _build_evidence(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        states = self._ensure_2d(states)
        if self.time_size == 0:
            if states.shape[1] == 1:
                return {"obs_0": states.view(-1)}
            return {f"obs_{i}": states[:, i] for i in range(states.shape[1])}
        if states.shape[1] == 1:
            return {"obs_t0_0": states.view(-1)}
        return {f"obs_t0_{i}": states[:, i] for i in range(states.shape[1])}

    def _build_do(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        actions = self._ensure_2d(actions)
        if self.time_size == 0:
            if actions.shape[1] == 1:
                return {"action_0": actions.view(-1)}
            return {f"action_{i}": actions[:, i] for i in range(actions.shape[1])}
        if actions.shape[1] == 1:
            return {"action_t0_0": actions.view(-1)}
        return {f"action_t0_{i}": actions[:, i] for i in range(actions.shape[1])}

    def _get_dag_from_df(self, df: pd.DataFrame) -> nx.DiGraph:
        """Static fallback when building from dataframe columns."""
        obs_feats = [c for c in df.columns if c.startswith("obs_")]
        act_feats = [c for c in df.columns if c.startswith("action_")]
        G = nx.DiGraph()
        for c in obs_feats + act_feats:
            G.add_edge(c, "return")
        return G

    def _refresh_inferencer(self) -> None:
        """Re-bind the inferencer to the *current* VBN nodes after fit/update."""
        assert self.vbn is not None
        self.vbn.set_inference_method(
            self.inference_method, num_samples=self.num_samples
        )

    # ────────────────────────── fit/update from rollout ──────────────────────────
    def _causal_update(self, mem: dict) -> None:
        """
        Update/fit the VBN from rollout memory.

        Static:   rows = {obs_*, action_*, return}
        Temporal: sliding windows with obs_tk_*, action_tk_*, reward_tk (k=0..K-1) + obs_tK_*
        """
        # ----- Static -----
        if self.time_size == 0:
            obs = mem["obs"].reshape(-1, *mem["obs"].shape[2:])
            acts = mem["actions"].reshape(-1, *mem["actions"].shape[2:])
            rets = mem["returns"].reshape(-1)
            df = pd.DataFrame(
                {
                    **(
                        {
                            f"obs_{i}": obs[:, i].cpu().numpy()
                            for i in range(obs.shape[1])
                        }
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

            if self.vbn is None:
                # initialize VBN if needed
                self.ensure_bn(schema=self._get_dag_from_df(df))
                self.vbn.fit(df, **(self.kwargs_fit or {}))
                self._refresh_inferencer()
            else:
                try:
                    self.vbn.update(df)
                except Exception:
                    self.vbn.fit(df, **(self.kwargs_fit or {}))
                finally:
                    self._refresh_inferencer()
            return

        # ----- Temporal (K-slice) -----
        K = self.time_size
        obs = mem["obs"]  # [T,N,D] or [T,N,1]
        acts = mem["actions"]  # [T,N,A] or [T,N,1]
        rew = (mem["rewards"] if "rewards" in mem else mem["returns"]).float()

        T, N = obs.shape[:2]
        W = T - K
        if W <= 0:
            return  # not enough steps yet

        rows = []
        for n in range(N):
            for t in range(W):
                row = {}
                for k in range(K):
                    o = obs[t + k, n]
                    a = acts[t + k, n]
                    r = rew[t + k, n]
                    if o.ndim == 1:
                        for i in range(o.shape[0]):
                            row[f"obs_t{k}_{i}"] = float(o[i].item())
                    else:
                        row[f"obs_t{k}_0"] = float(o.view(-1).item())
                    if a.ndim == 1:
                        for j in range(a.shape[0]):
                            row[f"action_t{k}_{j}"] = float(a[j].item())
                    else:
                        row[f"action_t{k}_0"] = float(a.view(-1).item())
                    row[f"reward_t{k}"] = float(r.item())
                oK = obs[t + K, n]
                if oK.ndim == 1:
                    for i in range(oK.shape[0]):
                        row[f"obs_t{K}_{i}"] = float(oK[i].item())
                else:
                    row[f"obs_t{K}_0"] = float(oK.view(-1).item())
                rows.append(row)

        df = pd.DataFrame(rows)

        if self.vbn is None:
            # Build schema from env when available, else dense fallback from headers
            if hasattr(self, "env") and self.env is not None:
                self.ensure_bn(env=self.env)
            else:
                G = nx.DiGraph()
                for k in range(K):
                    r_k = f"reward_t{k}"
                    obs_k = [c for c in df.columns if c.startswith(f"obs_t{k}_")]
                    act_k = [c for c in df.columns if c.startswith(f"action_t{k}_")]
                    for c in obs_k + act_k:
                        G.add_edge(c, r_k)
                    obs_k1 = [c for c in df.columns if c.startswith(f"obs_t{k+1}_")]
                    for c in obs_k + act_k:
                        for d in obs_k1:
                            G.add_edge(c, d)
                self.ensure_bn(schema=G)
            self.vbn.fit(df, **(self.kwargs_fit or {}))
            self._refresh_inferencer()
        else:
            try:
                self.vbn.update(df)
            except Exception:
                self.vbn.fit(df, **(self.kwargs_fit or {}))
            finally:
                self._refresh_inferencer()

    def ensure_vbn_initialized(
        self, *, env=None, mem: Optional[dict] = None, steps: int = 1024
    ) -> None:
        """Ensure a VBN exists and is fitted at least once."""
        if self.vbn_is_ready():
            return
        self.ensure_bn(env=env or getattr(self, "env", None))
        if mem is None and hasattr(self, "_collect_rollout"):
            mem, _, _ = self._collect_rollout()
        if mem is not None:
            self._causal_update(mem)

    # ────────────────────────── value / policy ──────────────────────────
    def _policy_dist_from_latent(self, latent):
        if hasattr(self, "_distribution"):
            return self._distribution(latent)
        if getattr(self, "is_discrete", False):
            return self.dist_fn(self.actor(latent))
        return self.dist_fn(self.actor_mu(latent))

    def _enumerate_actions(self, dist, B: int) -> torch.Tensor:
        nA = dist.probs.shape[-1]
        return torch.arange(nA, device=self.device).view(1, nA).repeat(B, 1).view(-1, 1)

    def _q_causal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Q_c(s,a) := E[ outcome | do(a), evidence(s) ].
        Static:   outcome='return'
        Temporal: outcome='reward_t0'
        """
        if not self.vbn_is_ready():
            if hasattr(self, "env") and self.env is not None:
                self.ensure_vbn_initialized(env=self.env, steps=1024)
            if not self.vbn_is_ready():
                raise RuntimeError(
                    "VBN not fitted yet; call _causal_update(mem) first."
                )

        states = self._ensure_2d(states).to(self.device)
        actions = self._ensure_2d(actions).to(self.device)

        evidence = self._build_evidence(states)
        do = self._build_do(actions)
        query = "return" if self.time_size == 0 else "reward_t0"

        pdf, samples = self.vbn.infer_posterior(
            query=query, evidence=evidence, do=do, **self.kwargs_inf
        )

        if isinstance(samples, dict):
            samples = samples.get(query, None)
        if isinstance(pdf, dict):
            pdf = pdf.get("weights", None)

        if samples is not None and torch.isfinite(samples).any():
            q = samples if samples.ndim == 1 else samples.mean(-1)
        else:
            q = torch.zeros(states.shape[0], device=self.device)

        q = q.to(self.device).view(-1)
        if not torch.isfinite(q).all():
            finite = q[torch.isfinite(q)]
            if finite.numel() > 0:
                q = torch.nan_to_num(q, nan=float(finite.mean()))
                q[q == float("inf")] = finite.max()
                q[q == float("-inf")] = finite.min()
            else:
                q = torch.zeros_like(q)
        return q

    def _v_causal(self, states: torch.Tensor, dist):
        B = states.shape[0]
        # exact for single categorical head
        if hasattr(dist, "probs") and dist.probs.ndim == 2:
            nA = dist.probs.shape[-1]
            actions_enum = self._enumerate_actions(dist, B)
            states_rep = states.repeat_interleave(nA, dim=0)
            q_enum = self._q_causal(states_rep, actions_enum).view(B, nA)
            v = (dist.probs * q_enum).sum(-1)
            return v, q_enum
        # Monte Carlo for others
        with torch.no_grad():
            a_samp = dist.sample((self.pi_samples,))  # [K,B,...]
            aBK = a_samp.permute(1, 0, *range(2, a_samp.ndim))  # [B,K,...]
        if aBK.ndim == 2:
            aBK = aBK.unsqueeze(-1)
        states_rep = (
            states.unsqueeze(1)
            .expand(-1, self.pi_samples, -1)
            .reshape(-1, states.shape[-1])
        )
        actions_rep = aBK.reshape(-1, aBK.shape[-1])
        q_mc = self._q_causal(states_rep, actions_rep).view(B, self.pi_samples)
        return q_mc.mean(-1), None

    # ────────────────────────── rollout-time: fill adv & logp ─────────────────
    def _post_update_fill_adv_and_logp(self, mem: dict) -> None:
        """
        Fills:
          mem["advantages"] = returns - V_causal(s)   [T,N]
          mem["old_logp"]   = log π(a|s)              [T,N]
        Ensures the VBN is fitted on this rollout before use.
        """
        if not self.vbn_is_ready():
            self.ensure_bn(env=getattr(self, "env", None))
            self._causal_update(mem)

        device = self.device
        obs = mem["obs"].to(device)  # [T,N,flat] or [T,N,1]
        act = mem["actions"].to(device)  # [T,N] or [T,N,Ad]
        ret = mem["returns"].to(device)  # [T,N]

        T, N = obs.shape[:2]
        B = T * N

        # states for VBN + latent for policy
        if isinstance(self.encoder, nn.Embedding):
            obs_idx = (
                obs.squeeze(-1)
                if (obs.ndim == 3 and obs.shape[-1] == 1)
                else obs[..., 0]
            )
            states = obs_idx.reshape(B, 1).long()
            latent = self._encode(obs_idx.reshape(B))
        else:
            states = obs.view(B, -1).float()
            latent = self._encode(states)

        dist = self._policy_dist_from_latent(latent)

        # V_causal and advantages
        Vc, _ = self._v_causal(states, dist)
        Vc = Vc.view(T, N).detach()
        mem["advantages"] = (ret - Vc).detach()

        # old_logp
        if self.is_discrete:
            logp_vec = dist.log_prob(act.view(B).long())
        else:
            act_dim = getattr(self, "log_std", torch.zeros(1, device=device)).numel()
            act_vec = act.view(B, act_dim) if act.ndim == 3 else act.view(B, -1)
            logp_vec = dist.log_prob(act_vec).sum(-1)
        mem["old_logp"] = logp_vec.view(T, N).detach()

    # ────────────────────────── high-level fit from DataFrame ─────────────────
    def fit_bn_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        schema: Optional[nx.DiGraph] = None,
    ) -> None:
        """Convenience function to fit/update from an external DataFrame."""
        if self.vbn is None:
            if schema is None:
                if self.time_size == 0:
                    schema = self._get_dag_from_df(df)
                else:
                    # Dense temporal fallback from headers
                    K = self.time_size
                    G = nx.DiGraph()
                    for k in range(K):
                        r_k = f"reward_t{k}"
                        obs_k = [c for c in df.columns if c.startswith(f"obs_t{k}_")]
                        act_k = [c for c in df.columns if c.startswith(f"action_t{k}_")]
                        for c in obs_k + act_k:
                            G.add_edge(c, r_k)
                        obs_k1 = [c for c in df.columns if c.startswith(f"obs_t{k+1}_")]
                        for c in obs_k + act_k:
                            for d in obs_k1:
                                G.add_edge(c, d)
                    schema = G
            self.ensure_bn(schema=schema)
            self.vbn.fit(df, **(self.kwargs_fit or {}))
            self._refresh_inferencer()
        else:
            try:
                self.vbn.update(df)
            except Exception:
                self.vbn.fit(df, **(self.kwargs_fit or {}))
            finally:
                self._refresh_inferencer()

    def fit_bn_from_rollout(self, mem: Optional[dict] = None, *, steps: int = 2048):
        if mem is None:
            if not hasattr(self, "_collect_rollout"):
                raise RuntimeError(
                    "fit_bn_from_rollout requires _collect_rollout or mem=..."
                )
            mem, _, _ = self._collect_rollout()
        self._causal_update(mem)

    # ────────────────────────── persistence (VBN snapshot) ────────────────────
    def save_causal_model(self, path: Union[str, Path]) -> None:
        if self.vbn is None:
            return
        save_vbn(self.vbn, str(path))

    def load_causal_model(
        self, path: Union[str, Path], map_location: Optional[torch.device] = None
    ) -> None:
        self.vbn = load_vbn(str(path), map_location=map_location or self.device)
        self.dag = self.vbn.dag
        # After loading, re-bind inferencer to ensure it sees current nodes
        self._refresh_inferencer()

    # ── inside class VBNCritic ──────────────────────────────────────────────────
    def _bn_to_bundle(self):
        """
        Snapshot VBN via new_vbn.io.save_vbn into bytes, safe for torch.save.
        Returns None if VBN is absent or untrained.
        """
        if self.vbn is None:
            return None
        # write to a secure temp path, then read bytes
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, "vbn_snapshot.pt")
            try:
                save_vbn(self.vbn, tmp_path)
                with open(tmp_path, "rb") as f:
                    blob = f.read()
            except Exception as e:
                # as a last resort, don't block policy saving
                print(
                    f"[VBNCritic] Warning: save_vbn failed ({e}); skipping causal bundle."
                )
                return None

        return {
            "format": "vbn_bytes_v1",
            "blob": blob,
            "time_size": int(getattr(self, "time_size", 0)),
            "learning_method": getattr(self, "learning_method", None),
            "inference_method": getattr(self, "inference_method", None),
            "num_samples": int(getattr(self, "num_samples", 32)),
        }

    def _bn_from_bundle(self, bundle):
        """
        Restore VBN from the bytes produced by _bn_to_bundle. Safe against lambdas
        because we delegate deserialization to new_vbn.io.load_vbn.
        """
        if not bundle:
            return
        if bundle.get("format") != "vbn_bytes_v1":
            print("[VBNCritic] Unknown causal bundle format; skipping.")
            return

        blob = bundle["blob"]
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, "vbn_snapshot.pt")
            try:
                with open(tmp_path, "wb") as f:
                    f.write(blob)
                self.vbn = load_vbn(tmp_path, map_location=self.device)
                self.dag = self.vbn.dag
            except Exception as e:
                print(
                    f"[VBNCritic] Warning: load_vbn failed ({e}); causal model not restored."
                )
                return

        # restore knobs (non-critical)
        self.time_size = int(bundle.get("time_size", getattr(self, "time_size", 0)))
        self.learning_method = bundle.get(
            "learning_method", getattr(self, "learning_method", "mdn")
        )
        self.inference_method = bundle.get(
            "inference_method", getattr(self, "inference_method", "montecarlo.lw")
        )
        self.num_samples = int(
            bundle.get("num_samples", getattr(self, "num_samples", 32))
        )

        # re-bind inferencer to current nodes
        try:
            self._refresh_inferencer()
        except Exception:
            pass

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn

from vbn.core import CausalBayesNet

from src.algos.utils import set_learning_and_inference_objects


class VBNCritic:
    """
    Mixin that provides:
      - BN init / (re)fit hooks
      - Q_c(x,a), V_c(x) computation
      - causal advantages + old_logp fill-in for on-policy algos
      - optional causal prior pi_c ∝ exp(β Q_c)

    Call CBNCriticMixin.__init__(..., causality_init=..., pi_samples=..., kl_coeff=..., kl_beta=...)
    from your algo subclass *after* calling the algo's super().__init__ (so self.device exists).
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
        causality_init = causality_init or {}

        # knobs
        self.pi_samples = pi_samples
        self.kl_coeff = kl_coeff
        self.kl_beta = kl_beta

        # BN config
        self.num_samples = causality_init.get("num_samples", 32)
        self.dag = causality_init.get("dag", None)
        self.types = causality_init.get("types", None)
        self.cards = causality_init.get("cards", None)
        self.discrete = causality_init.get("discrete", False)
        self.approximate = causality_init.get("approximate", True)
        chkpt_path = causality_init.get("chkpt_path", None)

        # learning & inference backends
        self.fit_method, self.inf_method = set_learning_and_inference_objects(
            self.discrete, self.approximate
        )
        self.kwargs_inf = {"num_samples": self.num_samples}
        self.kwargs_fit = causality_init.get(
            "kwargs_fit", {"epochs": 100, "batch_size": 1024}
        )

        # (optional) pre-instantiate BN
        self.bn: Optional[CausalBayesNet] = None
        self.lp = None
        self.inf_obj = None
        if self.dag is not None and self.types is not None:
            self.bn = CausalBayesNet(self.dag, self.types, self.cards or {})
            if chkpt_path is not None:
                self.lp = self.bn.load_params(str(chkpt_path))
            if self.inf_method is not None:
                self.inf_obj = self.bn.setup_inference(self.inf_method)

        # cache for discrete Q-table -> causal prior
        self._cached_q_table = None

    # ────────────────────────── helpers ──────────────────────────
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

    def _q_causal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert self.bn is not None and self.inf_obj is not None, "CBN not initialized."
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

    def _pi_causal_prior(self, q_table: torch.Tensor):
        return torch.distributions.Categorical(logits=self.kl_beta * q_table)

    # Default DAG if none provided by subclass
    def _get_dag(
        self, data: pd.DataFrame, target_feature: str = "return", do_key: str = "action"
    ):
        import networkx as nx

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

    # ───────────────────── rollout-time hooks ─────────────────────
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
        self._causal_update(mem)
        states = mem["obs"].reshape(-1, *mem["obs"].shape[2:])
        actions = mem["actions"].reshape(-1, *mem["actions"].shape[2:])
        returns = mem["returns"].reshape(-1).detach()

        latent = self.encoder(
            states.long() if isinstance(self.encoder, nn.Embedding) else states.float()
        )
        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)
            old_logp = dist.log_prob(actions.long().view(-1))
        else:
            mu = self.actor_mu(latent)
            dist = self.dist_fn(mu)
            old_logp = dist.log_prob(actions).sum(-1)

        Vc, Qtab = self._v_causal(states, dist)
        adv = returns - Vc.detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        mem["advantages"] = adv
        mem["values"] = Vc.detach()
        mem["old_logp"] = old_logp.detach()
        self._cached_q_table = Qtab

    def _bn_params_path(self, base_path: Union[str, Path]) -> Path:
        p = Path(base_path)
        if p.suffix != ".pt":
            p = p.with_suffix(".pt")
        return p.with_suffix(".pt").with_name(p.stem + ".td")

    def save_bn_params(self, base_path: Union[str, Path]) -> None:
        """Save learned BN params tensor next to the policy checkpoint."""
        if getattr(self, "bn", None) is None or getattr(self, "lp", None) is None:
            return
        td_path = self._bn_params_path(base_path)
        td_path.parent.mkdir(parents=True, exist_ok=True)
        self.bn.save_params(self.lp, str(td_path))

    def load_bn_params(self, base_path: Union[str, Path]) -> None:
        """Load BN params tensor; rebuild BN if needed; re-init inference."""
        td_path = self._bn_params_path(base_path)
        if not td_path.exists():
            return  # nothing to load

        # If BN not yet built, we must have dag/types/cards from init
        if getattr(self, "bn", None) is None:
            assert (
                self.dag is not None and self.types is not None
            ), "CBN load requires dag/types available in cbn_kwargs['causality_init']"
            from vbn.core import CausalBayesNet

            self.bn = CausalBayesNet(self.dag, self.types, self.cards or {})

        # Load params + set inference object
        self.lp = self.bn.load_params(str(td_path))
        if getattr(self, "inf_method", None) is not None:
            self.inf_obj = self.bn.setup_inference(self.inf_method)

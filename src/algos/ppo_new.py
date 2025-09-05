from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import networkx as nx
import pandas as pd
import torch
from torch import nn
from vbn.core import CausalBayesNet

from src.algos.ppo import PPO

from src.algos.utils import set_learning_and_inference_objects


class CBNOnlyPPO(PPO):
    """
    PPO variant that **replaces** the neural critic with a Causal Bayesian Network (CBN).
    - Advantages are computed as: adv = returns - V_causal(x)
    - No critic MSE term (vf_coeff should be 0.0)
    - Optional KL(pi || pi_causal_prior) regularization toward a Boltzmannized causal prior
    Assumptions:
      * Discrete action spaces are handled exactly via enumeration (single Categorical).
      * Multi-discrete / continuous actions are approximated by MC sampling from pi.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pi_samples = kwargs.get("pi_samples", 32)
        self.kl_coeff = kwargs.get(
            "kl_coeff", 0.0
        )  # KL(pi || pi_c) weight; set >0 to enable
        self.kl_beta = kwargs.get(
            "kl_beta", 5.0
        )  # temperature for causal prior pi_c ~ softmax(beta * Q_c)

        causality_init = kwargs.pop("causality_init", {})

        self.num_samples = causality_init.get("num_samples", 32)

        self.dag = causality_init.get("dag", None)
        self.types = causality_init.get("types", None)
        self.cards = causality_init.get("cards", None)

        self.discrete = causality_init.get("discrete", False)
        self.approximate = causality_init.get("approximate", False)

        chkpt_path = causality_init.get("chkpt_path", None)

        if self.dag is not None and self.cards is not None and self.types is not None:
            self.bn: CausalBayesNet = CausalBayesNet(self.dag, self.types, self.cards)
        else:
            self.bn = None

        if chkpt_path is not None and self.bn is not None:
            self.lp = self.bn.load_params(str(chkpt_path))
        else:
            self.lp = None

        self.fit_method, self.inf_method = set_learning_and_inference_objects(
            self.discrete, self.approximate
        )

        if self.bn is not None and self.inf_method is not None:
            self.inf_obj = self.bn.setup_inference(self.inf_method)

        self.kwargs_inf = {"num_samples": self.num_samples}
        self.kwargs_fit = {
            "epochs": 100,
            "batch_size": 1024,
        }

    # ---------- small helpers ----------

    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, 1) if x.ndim == 1 else x

    @staticmethod
    def _replace_infs(t: torch.Tensor) -> torch.Tensor:
        is_pos_inf = torch.isposinf(t)
        is_neg_inf = torch.isneginf(t)
        finite_mask = torch.isfinite(t)
        if finite_mask.any():
            finite_min = t[finite_mask].min()
            finite_max = t[finite_mask].max()
        else:
            finite_min = torch.tensor(-1e6, device=t.device, dtype=t.dtype)
            finite_max = torch.tensor(1e6, device=t.device, dtype=t.dtype)
        t = t.clone()
        t[is_pos_inf] = finite_max
        t[is_neg_inf] = finite_min
        return t

    # ---------- CBN core: Q_c(x,a) and V_c(x) ----------

    def _build_evidence(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        states = self._ensure_2d(states)
        B, F = states.shape
        if F == 1:
            return {"obs_0": states.view(-1)}
        return {f"obs_{i}": states[:, i] for i in range(F)}

    def _build_do(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        actions = self._ensure_2d(actions)
        B, A = actions.shape
        if A == 1:
            return {"action_0": actions.view(-1)}
        return {f"action_{i}": actions[:, i] for i in range(A)}

    def _posterior_expectation(
        self,
        posterior_pdfs: torch.Tensor | None,
        posterior_samples: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Best-effort expectation of the 'return' node:
          - if samples available: mean over samples
          - else: fallback to mode probability * (assume reward normalized) as in your code
        """
        if posterior_samples is not None and torch.isfinite(posterior_samples).any():
            # posterior_samples: [B, S] or [S,] -> mean over sample dim
            if posterior_samples.ndim == 1:
                return posterior_samples
            return posterior_samples.mean(dim=-1)
        # Fallback: max-prob (mode) proxy
        if posterior_pdfs is not None:
            if posterior_pdfs.ndim == 1:
                return posterior_pdfs
            return posterior_pdfs.max(dim=-1).values
        # ultimate fallback
        return torch.zeros(1, device=self.device)

    def _q_causal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Q_c(x,a) = E[R | do(A=a), X=state] via BN.
        Returns: [B] tensor of expected returns.
        """
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

        # Harmonize device & shapes; assume the query returns tensors keyed by 'return'
        # If your bn.infer returns plain tensors, adapt accordingly.
        if isinstance(posterior_pdfs, dict):
            posterior_pdfs = posterior_pdfs.get("return", None)
        if isinstance(posterior_samples, dict):
            posterior_samples = posterior_samples.get("return", None)

        q = self._posterior_expectation(posterior_pdfs, posterior_samples).to(
            self.device
        )
        q = q.view(-1)
        q = torch.nan_to_num(
            q, nan=q[torch.isfinite(q)].mean() if torch.isfinite(q).any() else 0.0
        )
        return self._replace_infs(q)

    def _enumerate_actions(self, dist, B: int) -> torch.Tensor:
        """
        Enumerate all discrete actions for a single Categorical action head.
        Returns a [B * nA, 1] tensor of actions.
        """
        assert hasattr(
            dist, "probs"
        ), "Enumeration requires discrete Categorical policy."
        nA = dist.probs.shape[-1]
        a = torch.arange(nA, device=self.device).view(1, nA).repeat(B, 1)  # [B, nA]
        return a.view(-1, 1)  # [B*nA, 1]

    def _v_causal(
        self,
        states: torch.Tensor,
        dist,
        actions_current: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute V_c(x) = E_{a~pi}[Q_c(x,a)]
        - Discrete single-head: exact via enumeration.
        - Otherwise: MC with `self.pi_samples`.
        Also returns optional Q_c(x,·) for KL prior when discrete.
        """
        B = states.shape[0]

        # DISCRETE (single Categorical head)
        if hasattr(dist, "probs") and dist.probs.ndim == 2:
            nA = dist.probs.shape[-1]
            # enumerate actions
            actions_enum = self._enumerate_actions(dist, B)  # [B*nA, 1]
            states_rep = states.repeat_interleave(nA, dim=0)  # [B*nA, F]
            q_enum = self._q_causal(states_rep, actions_enum).view(B, nA)  # [B, nA]
            pi_probs = dist.probs  # [B, nA]
            v = (pi_probs * q_enum).sum(dim=-1)  # [B]
            return v, q_enum  # return Q-table for KL prior

        # MULTI-DISCRETE or CONTINUOUS: MC under pi
        with torch.no_grad():
            # ask for (K, B, A?) samples in one call
            a_samples = dist.sample((self.pi_samples,))  # [K, B, A] or [K, B]
            a_cat = a_samples.permute(
                1, 0, *range(2, a_samples.ndim)
            )  # [B, K, A] or [B, K]

        if a_cat.ndim == 2:  # [B, K] -> add action dim
            a_cat = a_cat.unsqueeze(-1)

        states_rep = (
            states.unsqueeze(1).repeat(1, self.pi_samples, 1).view(-1, states.shape[-1])
        )  # [B*K, F]
        actions_rep = a_cat.reshape(-1, a_cat.shape[-1])  # [B*K, A]
        q_mc = self._q_causal(states_rep, actions_rep).view(
            B, self.pi_samples
        )  # [B, K]
        v = q_mc.mean(dim=-1)  # [B]
        return v, None

    def _pi_causal_prior(
        self, q_table: torch.Tensor
    ) -> torch.distributions.Categorical:
        """
        Build pi_c(a|x) ∝ exp(beta * Q_c(x,a)) from a [B, nA] Q-table.
        """
        logits = self.kl_beta * q_table  # [B, nA]
        return torch.distributions.Categorical(logits=logits)

    # ---------- Training-time plumbing ----------

    def _causal_update(self, mem):
        """
        Same as your `_causal_update`: update BN with fresh rollout data.
        """
        # Flatten
        n_obs = 1 if mem["obs"].dim() == 2 else mem["obs"].shape[-1]
        n_actions = 1 if mem["actions"].dim() == 2 else mem["actions"].shape[-1]

        if n_obs == 1:
            obs_flat = mem["obs"].reshape(-1)
        else:
            obs_flat = mem["obs"].reshape(-1, mem["obs"].shape[-1])

        if n_actions == 1:
            acts_flat = mem["actions"].reshape(-1)
        else:
            acts_flat = mem["actions"].reshape(-1, mem["actions"].shape[-1])

        rets_flat = mem["returns"].reshape(-1)

        data_dict = {}
        if n_obs == 1:
            data_dict["obs_0"] = obs_flat.cpu().numpy()
        else:
            for i in range(obs_flat.shape[1]):
                data_dict[f"obs_{i}"] = obs_flat[:, i].cpu().numpy()

        if n_actions == 1:
            data_dict["action_0"] = acts_flat.cpu().numpy()
        else:
            for i in range(acts_flat.shape[1]):
                data_dict[f"action_{i}"] = acts_flat[:, i].cpu().numpy()

        data_dict["return"] = rets_flat.cpu().numpy()
        df = pd.DataFrame(data_dict)

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

    def _get_dag(
        self, data: pd.DataFrame, target_feature: str = "return", do_key: str = "action"
    ) -> nx.DiGraph:
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

    # ---------- Hook called after each rollout to prep losses ----------

    def _post_update(self, mem):
        """
        Compute causal advantages and (optionally) a causal prior for KL regularization.
        Expects mem['obs'], mem['actions'], mem['returns'] shaped [T, N, ...].
        """
        # 1) keep BN up-to-date
        self._causal_update(mem)

        # 2) flatten rollout tensors
        states = mem["obs"].reshape(-1, *mem["obs"].shape[2:])
        actions = mem["actions"].reshape(-1, *mem["actions"].shape[2:])
        returns = mem["returns"].reshape(-1).detach()

        # 3) build current policy dist πθ(·|x)
        if isinstance(self.encoder, nn.Embedding):
            enc_in = states.long()
        else:
            enc_in = states.float()
        latent = self.encoder(enc_in)
        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)  # Categorical
        else:
            mu = self.actor_mu(latent)
            dist = self.dist_fn(mu)  # e.g., Normal/TanhNormal

        # 4) V_c(x) from BN (and Q-table if discrete)
        Vc, Q_table = self._v_causal(states, dist)

        # 5) causal advantages
        adv = returns - Vc.detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        mem["advantages"] = adv
        mem["values"] = Vc.detach()  # useful if you still compute diagnostics
        # PPO-style bookkeeping if needed:
        mem["old_logp"] = (
            dist.log_prob(actions).sum(-1)
            if not self.is_discrete
            else dist.log_prob(actions.long().view(-1))
        )

        # Store a causal prior for KL if discrete
        self._cached_q_table = Q_table  # [B, nA] or None
        self._cached_states_latent = latent  # optional reuse

    # ---------- The main optimization step (no neural critic MSE) ----------

    def _algo_update(self, mem):
        obs = self.flat(mem["obs"])
        actions = (
            self.flat(mem["actions"]).long()
            if self.is_discrete
            else self.flat(mem["actions"])
        )
        # returns = self.flat(mem["returns"]).detach()
        adv = self.flat(mem["advantages"]).detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        latent = self.encoder(
            obs.float() if not isinstance(self.encoder, nn.Embedding) else obs.long()
        )

        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)
            logp = dist.log_prob(actions.view(-1))
            entropy = dist.entropy().mean()
            extra_a = self.extra_actor_loss(latent, dist)
        else:
            mu = self.actor_mu(latent)
            dist = self.dist_fn(mu)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            extra_a = self.extra_actor_loss(latent, dist)

        # Actor loss only (no critic MSE)
        base_actor_loss = -(logp * adv).mean()

        # Optional KL(pi || pi_causal) when discrete and Q-table available
        kl_loss = torch.tensor(0.0, device=self.device)
        if (
            self.kl_coeff > 0.0
            and self.is_discrete
            and hasattr(self, "_cached_q_table")
            and self._cached_q_table is not None
        ):
            # Build causal prior pi_c
            pi_c = self._pi_causal_prior(
                self._cached_q_table.detach()
            )  # Categorical with logits = beta*Q_c
            # KL(pi || pi_c) = E_pi[log pi - log pi_c]
            probs = dist.probs.clamp_min(1e-8)
            log_probs = probs.log()
            log_probs_c = (pi_c.probs.clamp_min(1e-8)).log()
            kl = (probs * (log_probs - log_probs_c)).sum(dim=-1).mean()
            kl_loss = self.kl_coeff * kl

        # No critic loss:
        extra_c = self.extra_critic_loss(latent, torch.zeros_like(logp))
        critic_loss = torch.tensor(0.0, device=self.device)
        loss = base_actor_loss + extra_a + kl_loss - self.ent_coeff * entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        # metrics
        self.train_metrics.add(
            total_loss=float(loss.item()),
            actor_loss=float(base_actor_loss),
            extra_actor_loss=float(extra_a),
            critic_loss=float(critic_loss),
            extra_critic_loss=float(extra_c),
            kl_loss=float(kl_loss.item()),
        )
        self._log_ac_metrics(
            mse=0.0, adv_var=adv.var(unbiased=False).item(), entropy=entropy.item()
        )

    # ---------- persistence ----------

    def save_policy(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "vf_coeff": self.vf_coeff,
                "ent_coeff": self.ent_coeff,
                "is_discrete": self.is_discrete,
                "kl_coeff": self.kl_coeff,
                "kl_beta": self.kl_beta,
                "pi_samples": self.pi_samples,
            },
            self._ensure_pt_path(path),
        )
        if self.bn is not None:
            self.bn.save_params(self.lp, str(path) + ".td")

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        self.vf_coeff = ckpt.get("vf_coeff", 0.0)
        self.ent_coeff = ckpt.get("ent_coeff", 0.01)
        self.kl_coeff = ckpt.get("kl_coeff", 0.0)
        self.kl_beta = ckpt.get("kl_beta", 5.0)
        self.pi_samples = ckpt.get("pi_samples", 8)

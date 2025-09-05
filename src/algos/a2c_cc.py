from pathlib import Path
from typing import Union

import networkx as nx
import pandas as pd
import torch

from torch import nn
from vbn.core import CausalBayesNet

# from vbn.utils import infer_types_and_cards

from src.algos import A2C
from src.algos.utils import set_learning_and_inference_objects


class CausalCriticA2C(A2C):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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

    # --------  -- persistence ----------
    def save_policy(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "vf_coeff": self.vf_coeff,
                "ent_coeff": self.ent_coeff,
                "is_discrete": self.is_discrete,
            },
            self._ensure_pt_path(path),
        )

        self.bn.save_params(self.lp, str(path) + ".td")

    @staticmethod
    def _replace_infs(t: torch.Tensor) -> torch.Tensor:
        is_pos_inf = torch.isposinf(t)
        is_neg_inf = torch.isneginf(t)
        finite_mask = torch.isfinite(t)

        if finite_mask.any():
            finite_min = t[finite_mask].min()
            finite_max = t[finite_mask].max()
        else:
            # fallback values if everything is inf or nan
            finite_min = torch.tensor(-1e6, device=t.device, dtype=t.dtype)
            finite_max = torch.tensor(1e6, device=t.device, dtype=t.dtype)

        t = t.clone()
        t[is_pos_inf] = finite_max
        t[is_neg_inf] = finite_min
        return t

    def causal_prior_probs(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Dummy prior – replace with your own CPD.
        Returns categorical probabilities over actions for each state.
          states : [B, obs_dim]
          actions: [B, act_dim]
          out    : [B, n_actions]  (rows sum to 1)
        """
        if states.ndim == 1:
            # batch_size = states.shape
            # n_features = 1
            evidence = {"obs_0": states}
        else:
            batch_size, n_features = states.shape
            evidence = {f"obs_{f}": states[:, f] for f in range(n_features)}

        do = {}
        if actions.ndim == 1:
            do["action_0"] = actions
        else:
            for i in range(actions.shape[1]):
                do[f"action_{i}"] = actions[:, i]

        posterior_pdfs, posterior_samples, _ = self.bn.infer(
            self.inf_obj,
            lp=self.lp,
            evidence=evidence,
            query=["return"],
            do=do,
            return_samples=True,
            **self.kwargs_inf,
        )

        nan_mask = torch.isnan(posterior_pdfs)
        if nan_mask.any():
            mean_pdfs = posterior_pdfs[~nan_mask].mean()
            posterior_pdfs[nan_mask] = mean_pdfs

        most_probable_reward = posterior_pdfs.max(dim=-1).values.view(-1)

        log_p = torch.log(most_probable_reward)
        log_p = self._replace_infs(log_p)

        return log_p

    def _post_update(self, mem):
        """
        Compute causal-baseline advantages for PPO.
        Expects mem['obs'], ['actions'], ['returns'] shaped [T, N, ...].
        """

        self._causal_update(mem)

        # ---------- flatten rollout tensors ----------
        states = mem["obs"].reshape(-1, *mem["obs"].shape[2:])  # [B, obs_dim]
        actions = mem["actions"].reshape(-1, *mem["actions"].shape[2:])  # [B, act_dim]
        returns = mem["returns"].reshape(-1).detach()  # [B]

        # ---------- prepare encoder input ----------
        # if encoder is an Embedding (discrete obs), keep long indices
        if isinstance(self.encoder, nn.Embedding):
            enc_in = states.long()
        else:
            enc_in = states.float()
        latent = self.encoder(enc_in)  # [B, hidden]

        # ---------- get V(s) (collapse Q-vector if needed) ----------
        crit_out = self.critic(latent)  # [B,1] or [B,nA]
        if crit_out.dim() == 2 and crit_out.size(1) > 1:  # Q-vector → scalar
            state_values = crit_out.mean(dim=1)
        else:
            state_values = crit_out.squeeze(-1)  # [B]

        # ---------- causal prior log-probs ----------
        old_logp = self.causal_prior_probs(states, actions)  # [B, nA]

        # ---------- causal advantages ----------
        causal_adv = returns - state_values.detach()
        causal_adv = (causal_adv - causal_adv.mean()) / (
            causal_adv.std(unbiased=False) + 1e-8
        )

        # ---------- stash for algo update ----------
        mem["advantages"] = causal_adv
        mem["old_logp"] = old_logp

    def _causal_update(self, mem):
        # Flatten tensors: [rollout_len, n_train_envs, ...] -> [rollout_len * n_train_envs, ...]

        n_obs = 1 if mem["obs"].dim() == 2 else mem["obs"].shape[-1]

        n_actions = 1 if mem["actions"].dim() == 2 else mem["actions"].shape[-1]

        if n_obs == 1:
            obs_flat = mem["obs"].reshape(-1)
        else:
            obs_flat = mem["obs"].reshape(
                -1, mem["obs"].shape[-1]
            )  # shape: [T*N, obs_dim]

        if n_actions == 1:
            actions_flat = mem["actions"].reshape(-1)
        else:
            actions_flat = mem["actions"].reshape(
                -1, mem["actions"].shape[-1]
            )  # shape: [T*N, qct_dim

        returns_flat = mem["returns"].reshape(-1)  # shape: [T*N]

        if n_obs == 1:
            data_dict = {
                "obs_0": obs_flat.cpu().numpy(),
            }
        else:
            data_dict = {}
            for i in range(obs_flat.shape[1]):
                data_dict[f"obs_{i}"] = obs_flat[:, i].cpu().numpy()

        if n_actions == 1:
            data_dict["action_0"] = actions_flat.cpu().numpy()
        else:
            for i in range(actions_flat.shape[1]):
                data_dict[f"action_{i}"] = actions_flat[:, i].cpu().numpy()

        data_dict["return"] = returns_flat.cpu().numpy()

        data = pd.DataFrame(data_dict)

        if self.bn is not None:
            # Use the updated DataFrame to update your Bayesian Network or knowledge representation
            self.lp = self.bn.add_data(data, update_params=True, return_lp=True)
        else:
            dag = self._get_dag(data)

            if self.types is None or self.cards is None:
                self.types = {str(n): "continuous" for n in dag.nodes()}
                self.cards = {}
                # self.types, self.cards = infer_types_and_cards(data)

            self.bn = CausalBayesNet(dag, self.types, self.cards)
            self.lp = self.bn.fit(self.fit_method, data, **self.kwargs_fit)
            self.inf_obj = self.bn.setup_inference(self.inf_method)

    def _get_dag(
        self, data: pd.DataFrame, target_feature: str = "return", do_key: str = "action"
    ) -> nx.DiGraph:
        if do_key is not None:
            observation_features = [
                s for s in data.columns if do_key not in s and s != target_feature
            ]
            intervention_features = [
                s for s in data.columns if do_key in s and s != target_feature
            ]
        else:
            observation_features = [s for s in data.columns]
            intervention_features = []

        dag = [(feature, target_feature) for feature in observation_features]

        for int_feature in intervention_features:
            dag.append((int_feature, target_feature))

        G = nx.DiGraph()
        G.add_edges_from(dag)

        return G

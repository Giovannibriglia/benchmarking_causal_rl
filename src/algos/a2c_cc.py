from pathlib import Path
from typing import Union

import networkx as nx
import pandas as pd
import torch
import yaml

from cbn.base.bayesian_network import BayesianNetwork
from torch import nn

from src.algos import A2C


class CausalCriticA2C(A2C):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        causality_init = kwargs.pop("causality_init", {})

        self.N_max = causality_init.get("N_max", 16)

        self.dag = causality_init.get("dag", None)
        data = causality_init.get("data", None)

        parameter_learning_algo = causality_init.get(
            "parameter_learning_algo", "logistic_regression"
        )
        inference_mechanism = causality_init.get("inference_mechanism", "exact")

        with open(
            f"cbn/conf/parameter_learning/{parameter_learning_algo}.yaml", "r"
        ) as file:
            self.parameters_learning_config = yaml.safe_load(file)

        with open(f"cbn/conf/inference/{inference_mechanism}.yaml", "r") as file:
            self.inference_config = yaml.safe_load(file)

        self.kwargs = {"log": False, "plot_prob": False}

        if self.dag is not None and data is not None:
            self.bn = BayesianNetwork(
                dag=self.dag,
                data=data,
                parameters_learning_config=self.parameters_learning_config,
                inference_config=self.inference_config,
                **self.kwargs,
            )
        else:
            self.bn = None

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

        self.bn.save_model(path)

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        self.vf_coeff = ckpt.get("vf_coeff", 0.5)
        self.ent_coeff = ckpt.get("ent_coeff", 0.01)

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
          states : [B, feat]
          out    : [B, n_actions]  (rows sum to 1)
        """
        if states.ndim == 1:
            batch_size = states.shape
            n_features = 1
            evidence = {"obs_0": states.unsqueeze(-1)}
        else:
            batch_size, n_features = states.shape
            evidence = {
                f"obs_{f}": states[:, f].unsqueeze(-1) for f in range(n_features)
            }

        evidence["action"] = actions.unsqueeze(-1)

        pdfs, target_node_domains, parents_domains = self.bn.get_pdf(
            "return", evidence, N_max=self.N_max
        )

        nan_mask = torch.isnan(pdfs)
        if nan_mask.any():
            mean_pdfs = pdfs[~nan_mask].mean()
            pdfs[nan_mask] = mean_pdfs

        most_probable_reward = pdfs.max(dim=-1).values.view(-1)

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
        actions = mem["actions"].reshape(-1).long()  # [B]
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

        # ---------- stash for PPO update ----------
        mem["advantages"] = causal_adv
        mem["old_logp"] = old_logp

    def _causal_update(self, mem):
        # Flatten tensors: [rollout_len, n_train_envs, ...] -> [rollout_len * n_train_envs, ...]

        n_obs = 1 if mem["obs"].dim() == 2 else mem["obs"].shape[-1]

        if n_obs == 1:
            obs_flat = mem["obs"].reshape(-1)
        else:
            obs_flat = mem["obs"].reshape(
                -1, mem["obs"].shape[-1]
            )  # shape: [T*N, obs_dim]
        actions_flat = mem["actions"].reshape(-1)  # shape: [T*N]
        returns_flat = mem["returns"].reshape(-1)  # shape: [T*N]

        # Use a dict comprehension for obs columns efficiently
        if n_obs == 1:
            data_dict = {
                "obs_0": obs_flat.cpu().numpy(),
                "action": actions_flat.cpu().numpy(),
                "return": returns_flat.cpu().numpy(),
            }
        else:
            data_dict = {
                **{
                    f"obs_{i}": obs_flat[:, i].cpu().numpy()
                    for i in range(obs_flat.shape[1])
                },
                "action": actions_flat.cpu().numpy(),
                "return": returns_flat.cpu().numpy(),
            }

        data = pd.DataFrame(data_dict)

        if self.bn is not None:
            # Use the updated DataFrame to update your Bayesian Network or knowledge representation
            self.bn.update_knowledge(data)
        else:
            dag = self._get_dag(data)

            self.bn = BayesianNetwork(
                dag=dag,
                data=data,
                parameters_learning_config=self.parameters_learning_config,
                inference_config=self.inference_config,
                **self.kwargs,
            )

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

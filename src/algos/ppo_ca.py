from pathlib import Path
from typing import Union

import networkx as nx
import pandas as pd
import torch
import yaml

from cbn.base.bayesian_network import BayesianNetwork
from torch.distributions import Categorical, kl_divergence, Normal

from src.algos import PPO


class CausalActorPPO(PPO):
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
                "clip_eps": self.clip_eps,
                "vf_coeff": self.vf_coeff,
                "ent_coeff": self.ent_coeff,
                "is_discrete": self.is_discrete,
            },
            self._ensure_pt_path(path),
        )

        self.bn.save_model(path)

    def load_policy(self, path: Union[str, Path]) -> None:
        # TODO: load bn
        print("YOU NEED TO LOG THE BN")

        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        # hyper‑params useful if you want to inspect them later
        self.clip_eps = ckpt.get("clip_eps", 0.2)
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
            # batch_size = states.shape
            # n_features = 1
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
        self._causal_update(mem)

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
            self.dag = self._get_dag(data)

            self.bn = BayesianNetwork(
                dag=self.dag,
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

    def extra_actor_loss(
        self,
        states: torch.Tensor | None = None,  # [B, feat]
        dist_rl: torch.Tensor | None = None,  # [B, n_actions]
    ) -> torch.Tensor:

        if states.ndim == 1:
            # batch_size = states.shape
            # n_features = 1
            evidence = {"obs_0": states.unsqueeze(-1)}
        else:
            batch_size, n_features = states.shape
            evidence = {
                f"obs_{f}": states[:, f].unsqueeze(-1) for f in range(n_features)
            }

        pdfs, target_node_domains, parents_domains = self.bn.get_pdf(
            "return", evidence, N_max=self.N_max
        )
        # pdfs shape  : [B, N_t, N1, …, Nk]

        # ------------------------------------------------ 1) collapse all parent axes
        reduce_dims = tuple(range(2, pdfs.ndim))  # axes 2 … end
        marginal = pdfs.sum(dim=reduce_dims)  # [B, N_t]

        weights = marginal / (marginal.sum(dim=-1, keepdim=True) + 1e-12)  # [B, N_t]

        # ------------------------------------------------ 2) build P_causal
        if isinstance(dist_rl, Categorical):
            causal_dist = Categorical(probs=weights)  # <-- discrete
        else:
            # -- continuous -------------------------------------------------------
            # ❶ obtain the grid  (size N_t)  *independent* of the batch dimension
            if target_node_domains is not None:
                # target_node_domains can be [N_t] or [B, N_t]; get the 1‑D view
                grid = target_node_domains
                if grid.ndim > 1:
                    grid = grid[0]  # take the first sample – the grid is the same
                grid = grid.to(pdfs.device)  # [N_t]
            else:
                # fallback: a simple 0,1,…,N_t‑1 grid
                grid = torch.arange(pdfs.shape[1], device=pdfs.device, dtype=pdfs.dtype)

            # ❷ broadcast so the shapes line up:   [B, N_t]  ⊙  [N_t] → [B, N_t]
            mean_prior = (weights * grid).sum(dim=-1, keepdim=True)  # [B, 1]
            var_prior = (weights * (grid - mean_prior) ** 2).sum(dim=-1, keepdim=True)
            std_prior = var_prior.clamp_min(1e-12).sqrt()

            causal_dist = Normal(mean_prior, std_prior)  # 1‑D

        # ------------------------------------------------ 3) KL and loss
        kl_batch = kl_divergence(causal_dist, dist_rl)  # [B]
        return kl_batch.mean()  # scalar

    @staticmethod
    def _extract_return_grid(parents_domains: torch.Tensor) -> torch.Tensor:
        """
        parents_domains shape  : [B, N, N, …, N]   (same #dims as `pdfs`)
        return value for bin i : parents_domains[..., i, 0, 0, …, 0]

        The grid is identical across the batch, so we slice the *first* sample
        and squeeze all but the return dimension.
        """
        # ❶ drop the batch dim
        grid_nd = parents_domains[0]  # [N, N, …, N]

        # ❷ keep the first axis (return), fix every parent index at 0
        #    e.g. (:) for axis 0, and 0 for the rest
        idx = (slice(None),) + (0,) * (grid_nd.ndim - 1)
        grid_1d = grid_nd[idx]  # [N]

        return grid_1d

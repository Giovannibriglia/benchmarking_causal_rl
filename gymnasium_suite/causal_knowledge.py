import networkx as nx

import pandas as pd
import torch
import yaml

from cbn.base.bayesian_network import BayesianNetwork


class CausalKnowledge:
    def __init__(
        self,
        dag: nx.DiGraph = None,
        data: pd.DataFrame = None,
        parameter_learning_algo: str = "logistic_regression",
        inference_mechanism: str = "exact",
        causal_update: int = 20000,
    ):
        self.causal_update = causal_update

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(
            f"../cbn/conf/parameter_learning/{parameter_learning_algo}.yaml", "r"
        ) as file:
            self.parameters_learning_config = yaml.safe_load(file)

        with open(f"../cbn/conf/inference/{inference_mechanism}.yaml", "r") as file:
            self.inference_config = yaml.safe_load(file)

        self.kwargs = {"log": False, "plot_prob": False}

        if dag is not None and data is not None:
            self.bn = BayesianNetwork(
                dag=dag,
                data=data,
                parameters_learning_config=self.parameters_learning_config,
                inference_config=self.inference_config,
                **self.kwargs,
            )
        else:
            self.bn = None

        self.storing = []
        self.n_obs = None
        self.n_actions = None
        self.n_rewards = None

    def store_data(
        self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        if obs.dim() == 1:
            obs = obs.unsqueeze(-1)
        n_envs, self.n_obs = obs.shape
        self.n_actions = actions.shape[0]
        self.n_rewards = rewards.shape[0]

        # Validate input dimensions
        if actions.shape[0] != n_envs or rewards.shape[0] != n_envs:
            raise ValueError(
                "Mismatch in number of environments between obs, actions, and rewards"
            )

        # Append data for each environment
        for env_idx in range(n_envs):
            row = {}

            row["action"] = actions[env_idx].cpu().numpy()

            row["reward"] = rewards[env_idx].cpu().numpy()

            for o in range(self.n_obs):
                row[f"obs_{o}"] = obs[env_idx][o].cpu().numpy()

            self.storing.append(row)

        if len(self.storing) >= self.causal_update:
            self._update_knowledge()
            self.storing = []

    def _update_knowledge(self):
        if not self.storing:
            return

        # Convert stored data into a DataFrame
        data = pd.DataFrame(self.storing)

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

    def get_reward_action_values(self, obs: torch.Tensor, N_max: int = 16):
        n_envs = obs.shape[0]
        if self.bn is not None:
            target_node = "reward"

            if obs.dim() == 1:
                obs = obs.unsqueeze(-1)

            evidence = {f"obs_{o}": obs[:, o].unsqueeze(-1) for o in range(self.n_obs)}

            pdfs, target_node_domains, parents_domains = self.bn.get_pdf(
                target_node, evidence, N_max=N_max
            )
            if pdfs.shape[-1] == 1:
                print("mario")
            rav = self._setup_rav(pdfs)
            assert rav.shape == (n_envs, N_max, N_max), ValueError(
                f"rav.shape: {rav.shape} instead ({n_envs, N_max, N_max})"
            )
            return rav
        else:
            obs = torch.ones(
                (n_envs, N_max, N_max), dtype=torch.float32, device=self.device
            )
            # print("2) ", obs.shape)
            return obs

    def _setup_rav(
        self,
        pdfs: torch.Tensor,
    ):
        parents_reward = self.bn.get_parents(self.bn.initial_dag, "reward")
        action_idx = parents_reward.index("action")

        # We compute which axes to average over (parent axes except action_idx)
        parent_dims = pdfs.shape[1:-1]  # exclude query and target_dim
        parent_axes = list(range(1, 1 + len(parent_dims)))
        axes_to_avg = tuple(i for i in parent_axes if i != action_idx + 1)

        # Sum across axes_to_avg
        pdf_plot = torch.sum(pdfs, dim=axes_to_avg, keepdim=False)

        # Normalize
        pdf_plot = pdf_plot / pdf_plot.sum(
            dim=-1, keepdim=True
        )  # normalize across target dim

        return pdf_plot  # shape: [n_queries, action_domain_size, target_domain_size]

    def _get_dag(
        self, data: pd.DataFrame, target_feature: str = "reward", do_key: str = "action"
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

    def save(self, path: str):
        self.bn.save_model(path)

    def load(self, path: str):
        raise NotImplementedError

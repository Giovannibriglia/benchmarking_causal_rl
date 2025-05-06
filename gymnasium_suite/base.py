import os
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Union

import gymnasium as gym

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from cbn.base.bayesian_network import BayesianNetwork
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical, Normal


class MetricBuffer(dict):
    def add(self, **kv):
        for k, v in kv.items():
            self[k] = self.get(k, 0.0) + float(v)

    def dump(self, div: int = 1):  # average & reset
        out = {k: v / max(div, 1) for k, v in self.items()}
        self.clear()
        return out


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(indices.long().view(-1), num_classes=num_classes).float()


def safe_clone(x, dtype=None, device: torch.device = "cuda"):
    """clone‑detach tensors, wrap numpy → tensor"""
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


class BasePolicy(ABC):
    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces,
        observation_space: gym.spaces,
        n_envs: int,
        **kwargs,
    ):
        self.algo_name = algo_name
        self.action_space = action_space
        self.observation_space = observation_space

        self.n_envs = n_envs

        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.metrics = MetricBuffer()

        # holder for the current extra state (set each env step)
        self._extra_state: torch.Tensor | None = None

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        time_to_update = time.time()
        self._update(observations, actions, rewards, next_observations, dones)
        time_to_update = time.time() - time_to_update
        self.metrics.add(
            time_to_update=time_to_update,
        )

    @abstractmethod
    def _update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        raise NotImplementedError

    def get_actions(
        self, observations: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        time_to_get_actions = time.time()

        n_envs = observations.shape[0]
        res = self._get_actions(observations, mask=mask)
        assert res.shape[0] == n_envs, ValueError(
            f"actions has wrong dimension: {res.shape} first dimension should be {n_envs}"
        )

        time_to_get_actions = time.time() - time_to_get_actions
        self.metrics.add(
            time_to_get_actions=time_to_get_actions,
        )

        return res

    @abstractmethod
    def _get_actions(
        self, observations: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def setup_actions(
        self,
        actions_tensor: torch.Tensor,  # (n_envs, ...) – raw policy output
        done_mask: torch.Tensor,  # (n_envs,)     – True where episode finished
        out_type: str = "numpy",  # "numpy" | "torch"
    ):
        """
        Convert `actions_tensor` to the exact datatype/shape expected by
        `SyncVectorEnv.step`.  Always returns a 1‑D container of length n_envs.
        """
        if out_type not in {"numpy", "torch"}:
            raise ValueError("out_type must be 'numpy' or 'torch'")

        # ---------------------------------------------------------------------
        # DISCRETE ACTION SPACE  (CartPole, MountainCar, etc.)
        # ---------------------------------------------------------------------
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions_tensor = (actions_tensor > 0).to(torch.long)

            # Case A ─ logits / Q‑values  [n_envs, n]  → choose arg‑max
            if actions_tensor.ndim > 1:
                actions_tensor = torch.argmax(actions_tensor, dim=1)

            # Case B ─ single float per env  [n_envs]  → threshold at 0
            elif actions_tensor.dtype.is_floating_point:
                actions_tensor = (actions_tensor > 0).to(torch.long)

            actions_tensor.to(device=self.device)

            # zero‑out finished envs
            actions_tensor = torch.where(
                done_mask,
                torch.zeros_like(actions_tensor, device=self.device),
                actions_tensor,
            )

            if out_type == "torch":
                return actions_tensor  # shape (n_envs,)  torch.long
            else:
                return (
                    actions_tensor.detach().cpu().numpy().astype(np.int32)
                )  # shape (n_envs,)

        # ---------------------------------------------------------------------
        # CONTINUOUS (BOX) ACTION SPACE
        # ---------------------------------------------------------------------
        elif isinstance(self.action_space, gym.spaces.Box):
            # If the Box has ≥1 dimension but you only need the first scalar,
            # keep actions_tensor[:, 0]; otherwise keep the full vector.
            if self.action_space.shape == ():
                # scalar Box → squeeze to (n_envs,)
                actions_tensor = actions_tensor.squeeze(-1)

            # zero‑out finished envs
            actions_tensor = torch.where(
                done_mask.view(-1, *([1] * (actions_tensor.ndim - 1))),
                torch.zeros_like(actions_tensor),
                actions_tensor,
            )

            if out_type == "torch":
                return actions_tensor  # dtype matches policy output
            else:
                return (
                    actions_tensor.detach().cpu().numpy()
                )  # np.ndarray, shape (n_envs, …)

        # ---------------------------------------------------------------------
        else:
            raise NotImplementedError(
                f"Unsupported action space type: {type(self.action_space)}"
            )

    @abstractmethod
    def pop_metrics(self) -> dict:  # <── call at episode end
        raise NotImplementedError

    @staticmethod
    def _ensure_pt_path(path: Union[str, os.PathLike]) -> Path:
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def save_policy(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load_policy(self, path: str):
        raise NotImplementedError

    def set_extra_state(self, x: torch.Tensor | None):
        """
        Store a tensor of shape (n_envs, *extra_dims).
        Pass None to disable augmentation for the next step.
        The tensor is **not copied**; keep it alive until after `update`.
        """
        self._extra_state = x.to(self.device) if x is not None else None


class BaseCausalStateAugmentationPolicy(BasePolicy, ABC):
    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces,
        observation_space: gym.spaces,
        n_envs: int,
        **kwargs,
    ):
        super().__init__(algo_name, action_space, observation_space, n_envs, **kwargs)

        self.action_space_dim = (
            self.action_space.shape[0]
            if isinstance(self.action_space, gym.spaces.Box)
            else 1
        )
        causality_init = kwargs.pop("causality_init", {})

        self.N_max = causality_init.get("N_max", 16)
        # Define the expected shape
        self.ok_shape = (n_envs,) + (self.N_max,) * self.action_space_dim

        self.dag = causality_init.get("dag", None)
        data = causality_init.get("data", None)
        parameter_learning_algo = causality_init.get(
            "parameter_learning_algo", "logistic_regression"
        )
        inference_mechanism = causality_init.get("inference_mechanism", "exact")
        self.samples_causal_update = int(
            causality_init.get("samples_causal_update", 2e4)
        )
        with open(
            f"../cbn/conf/parameter_learning/{parameter_learning_algo}.yaml", "r"
        ) as file:
            self.parameters_learning_config = yaml.safe_load(file)

        with open(f"../cbn/conf/inference/{inference_mechanism}.yaml", "r") as file:
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

        self.storing = []

    def save_policy(self, path: str):
        if self.bn is not None:
            path2 = path.replace("policy", "bn")
            self.bn.save_model(path2)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        time_to_update = time.time()

        augmented_observations = self._augment_observation(observations)
        augmented_next_observations = self._augment_observation(next_observations)
        self.store_data(augmented_observations, actions, rewards)

        self._update(
            augmented_observations, actions, rewards, augmented_next_observations, dones
        )

        time_to_update = time.time() - time_to_update
        self.metrics.add(
            time_to_update=time_to_update,
        )

    def get_actions(
        self, observations: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        time_to_get_actions = time.time()
        augmented_obs = self._augment_observation(observations)
        # causal_mask = mask  # TODO: make better

        n_envs = observations.shape[0]
        res = self._get_actions(augmented_obs, mask=mask)
        assert res.shape[0] == n_envs, ValueError(
            f"actions has wrong dimension: {res.shape} first dimension should be {n_envs}"
        )

        time_to_get_actions = time.time() - time_to_get_actions
        self.metrics.add(
            time_to_get_actions=time_to_get_actions,
        )

        return res

    def _augment_observation(self, observations: torch.Tensor) -> torch.Tensor:
        rav = self.get_reward_action_values(observations)
        rav_flat = rav.view(rav.size(0), -1)  # -> [n_envs, N*N]
        # (optional) cast dtype if needed
        if rav_flat.dtype != rav_flat.dtype:
            rav_flat = rav_flat.to(observations.dtype)

        assert (
            rav_flat.shape[0] == self.ok_shape[0]
        ), f"wrong first dimension augmented dimensions: rav_flat.shape should be [{self.ok_shape[0]}], instead is [{rav_flat.shape[0]}]"

        assert (
            rav_flat.shape[1] == torch.prod(torch.tensor(self.ok_shape[1:])).item()
        ), f"wrong other dimensions: rav_flat.shape should be [{self.ok_shape}], instead is [{rav_flat.shape}]"

        # concatenate along the feature axis
        res = torch.cat([observations, rav_flat], dim=1)  # -> [n_envs, n_obs + N*N]

        mean_res = torch.nanmean(res)
        res = torch.where(torch.isnan(res), mean_res, res)

        return res

    def store_data(
        self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        if obs.dim() == 1:
            obs = obs.unsqueeze(-1)
        n_envs, n_obs = obs.shape
        n_obs -= self.N_max**self.action_space_dim
        n_actions = actions.dim()

        # Validate input dimensions
        if actions.shape[0] != n_envs or rewards.shape[0] != n_envs:
            raise ValueError(
                "Mismatch in number of environments between obs, actions, and rewards"
            )

        # Append data for each environment
        for env_idx in range(n_envs):
            row = {}

            if n_actions == 1:
                row["action_0"] = actions[env_idx].cpu().numpy().item()
            else:
                for a in range(actions[env_idx].shape[0]):
                    row[f"action_{a}"] = actions[env_idx][a].cpu().numpy().item()

            row["reward"] = rewards[env_idx].cpu().numpy().item()

            for o in range(n_obs):
                row[f"obs_{o}"] = obs[env_idx][o].cpu().numpy().item()

            self.storing.append(row)

        if len(self.storing) >= self.samples_causal_update:
            self._update_knowledge()
            self.storing = []
            # print("update_done")

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

    def get_reward_action_values(self, obs: torch.Tensor):
        if self.bn is not None:
            target_node = "reward"

            n_envs, n_obs = obs.shape

            if obs.dim() == 1:
                obs = obs.unsqueeze(-1)

            evidence = {f"obs_{o}": obs[:, o].unsqueeze(-1) for o in range(n_obs)}

            pdfs, target_node_domains, parents_domains = self.bn.get_pdf(
                target_node, evidence, N_max=self.N_max
            )
            nan_mask = torch.isnan(pdfs)
            if nan_mask.any():
                mean_pdfs = pdfs[~nan_mask].mean()
                pdfs[nan_mask] = mean_pdfs

            rav = self._setup_rav(pdfs, target_node_domains, parents_domains)

            return rav
        else:
            obs = torch.rand(
                self.ok_shape,
                dtype=torch.float32,
                device=self.device,
            )
            # print("2) ", obs.shape)
            return obs

    def _setup_rav(
        self,
        pdfs: torch.Tensor,
        reward_domain: torch.Tensor,
        parents_domain: torch.Tensor,
    ):
        """
        Set up the reward-action-value (RAV) distribution by averaging over
        parent axes and normalizing.

        Args:
            pdfs (torch.Tensor): The input tensor of probability density functions.

        Returns:
            torch.Tensor: The final normalized PDF.
        """
        """# Get parents of the reward node from the Bayesian network
        parents_reward = self.bn.get_parents(self.bn.initial_dag, "reward")

        # Number of environments
        n_envs = pdfs.shape[0]

        # Prepare an empty tensor for the final normalized PDF
        final_pdf = torch.empty(self.ok_shape, device=self.device)

        # For each action dimension, compute the averaged and normalized PDF
        for action_i in range(self.action_space_dim):
            # Find the index of the current action among the parents
            action_idx = parents_reward.index(f"action_{action_i}")

            # Determine which axes to average over (all parent axes except the action index)
            parent_dims = pdfs.shape[1:-1]  # exclude query and target_dim
            parent_axes = list(range(1, 1 + len(parent_dims)))
            axes_to_avg = tuple(i for i in parent_axes if i != action_idx + 1)

            # Sum across the axes to average
            pdf_plot = torch.sum(pdfs, dim=axes_to_avg, keepdim=False)

            normalized_pdf = pdf_plot / (pdf_plot.sum(dim=-1, keepdim=True) + 1e-8)

            # Prepare indexing: final_pdf[env_i, :, :, ..., action_i, action_i]
            idx = [slice(None)] * (2 + action_i) + [action_i]
            idx = tuple(idx + [action_i])  # repeat for both dims

            final_pdf[idx] = normalized_pdf"""

        parents_reward = self.bn.get_parents(self.bn.initial_dag, "reward")
        axes_to_avg = tuple(
            i + 1 for i, p in enumerate(parents_reward) if "action" not in p
        )

        pdf_plot = torch.sum(pdfs, dim=axes_to_avg, keepdim=False)
        normalized_pdf = pdf_plot / (pdf_plot.sum(dim=-1, keepdim=True) + 1e-8)

        idx = normalized_pdf.argmax(dim=-1)  # (n_envs,)*(N,)*n

        while reward_domain.ndim < idx.ndim:
            reward_domain = reward_domain.unsqueeze(-1)

        reward_domain = reward_domain.expand_as(idx)
        most_probable_reward = torch.gather(reward_domain, 1, idx)

        # Verify the final shape matches the expectation
        assert most_probable_reward.shape == self.ok_shape, ValueError(
            f"final_pdf.shape: {most_probable_reward.shape} instead should be ({self.ok_shape})"
        )

        # Return the final normalized PDF
        return most_probable_reward

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


def build_base_acnet(is_causal: bool = False):
    base_policy = BaseCausalStateAugmentationPolicy if is_causal else BasePolicy

    class ACNet(nn.Module):
        """Actor–critic head for Box *or* Discrete action spaces."""

        def __init__(self, in_dim: int, act_space: gym.spaces.Space):
            super().__init__()
            self.discrete = isinstance(act_space, gym.spaces.Discrete)
            act_dim = act_space.n if self.discrete else int(np.prod(act_space.shape))

            self.torso = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())
            if self.discrete:
                self.logits = nn.Linear(128, act_dim)
            else:
                self.mu = nn.Linear(128, act_dim)
                self.std = nn.Linear(128, act_dim)
                self.bd = float(act_space.high[0])

            self.v_head = nn.Linear(128, 1)

        def dist(self, x):
            h = self.torso(x)
            if self.discrete:
                return Categorical(logits=self.logits(h))
            else:
                mu = self.bd * torch.tanh(self.mu(h))
                std = F.softplus(self.std(h)) + 1e-5
                return Normal(mu, std)

        def value(self, x: torch.Tensor) -> torch.Tensor:
            return self.v_head(self.torso(x)).squeeze(-1)

    class BaseACPolicy(base_policy, ABC):
        def __init__(
            self,
            algo_name: str,
            act_space: gym.spaces.Space,
            obs_space: gym.spaces.Space,
            n_envs: int,
            *,
            rollout_len: int = 128,
            gamma: float = 0.98,
            lr: float = 3e-4,
            extra_input_dim: int = 0,  # N*M if you have extra tensor (flattened)
            **kwargs,
        ):
            super().__init__(algo_name, act_space, obs_space, n_envs, **kwargs)

            # ---------- obs encoder ------------------------------------------
            # Encoder setup
            if isinstance(obs_space, Box):
                self.obs_dim = int(np.prod(obs_space.shape))
                # Option A: explicit view
                # self._base_enc = lambda x: safe_clone(x, torch.float32, self.device).view(x.size(0), self.obs_dim)

                # Option B: flatten
                self._base_enc = lambda x: safe_clone(
                    x, torch.float32, self.device
                ).flatten(start_dim=1)

            elif isinstance(obs_space, Discrete):
                self.obs_dim = obs_space.n
                self._base_enc = lambda x: one_hot(
                    safe_clone(x, torch.long, self.device), self.obs_dim
                )
            else:
                raise NotImplementedError

            in_dim = self.obs_dim + extra_input_dim
            self.net = ACNet(in_dim, act_space).to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.gamma, self.rollout_len = gamma, rollout_len

            # buffer & metrics
            self._reset_buf()

            # placeholder for extra tensor (set each step)
            self._extra_state: torch.Tensor | None = None

            # wrap encoder with augmentation
            def _enc(x):
                out = self._base_enc(x)
                if self._extra_state is not None:
                    aug = self._extra_state.reshape(
                        self._extra_state.size(0), -1
                    ).float()
                    out = torch.cat((out, aug.to(self.device)), dim=1)
                return out

            self._enc = _enc

        # ---------------------------------------------------------------- buffer
        def _reset_buf(self):
            self.buf: Dict[str, List] = {
                k: [] for k in ("s", "a", "r", "d", "logp", "v")
            }

        # ---------------------------------------------------------------- setter
        def set_extra_state(self, x: torch.Tensor | None):
            self._extra_state = x.to(self.device) if x is not None else None

        # ---------------------------------------------------------------- common act
        def _get_actions(
            self, obs: torch.Tensor, mask: torch.Tensor = None
        ) -> torch.Tensor:
            enc = self._enc(obs)
            dist = self.net.dist(enc)
            acts = dist.sample()
            logp = dist.log_prob(acts)
            ent = dist.entropy()
            if logp.ndim > 1:
                logp, ent = logp.sum(-1), ent.sum(-1)

            val = self.net.value(enc)
            self.metrics.add(entropy=ent.mean().item())
            self._store(enc, acts.detach(), 0.0, 0.0, logp.detach(), val.detach())
            return acts

        def _store(self, s, a, r, d, lp, v):
            self.buf["s"].append(s)
            self.buf["a"].append(a)
            self.buf["r"].append(safe_clone(r, torch.float32, self.device))
            self.buf["d"].append(safe_clone(d, torch.float32, self.device))
            self.buf["logp"].append(lp)
            self.buf["v"].append(v)

        # ─── abstract: subclasses implement algorithm‑specific update ───────────
        @abstractmethod
        def _update(self, obs, acts, rews, next_obs, dones): ...

        # pop averaged metrics once per episode
        def pop_metrics(self) -> Dict[str, float]:
            return self.metrics.dump(div=self.rollout_len)

        def _extra_to_save(self) -> dict:
            """Sub‑classes may override to add algorithm‑specific scalars."""
            return {}

        def _load_extra(self, d: dict):
            """Sub‑classes may override to read what they saved."""
            pass

        def save_policy(self, path: str):
            if is_causal:
                super().save_policy(path)

            p = self._ensure_pt_path(path)
            torch.save(
                {
                    "net": self.net.state_dict(),
                    "opt": self.opt.state_dict(),
                    "extra": self._extra_to_save(),
                },
                p,
            )
            # print(f"[{self.algo_name}] saved → {p}")

        def load_policy(self, path: str):
            p = self._ensure_pt_path(path)
            d = torch.load(p, map_location=self.device)
            self.net.load_state_dict(d["net"])
            self.opt.load_state_dict(d["opt"])
            self._load_extra(d.get("extra", {}))
            print(f"[{self.algo_name}] loaded ← {p}")

    return BaseACPolicy


def build_base_q_policy(is_causal: bool = False):
    base_policy = BaseCausalStateAugmentationPolicy if is_causal else BasePolicy

    # Replay buffer shared by all off-policy methods
    class ReplayBuffer:
        def __init__(self, capacity: int = 20000):
            self.buf: deque = deque(maxlen=capacity)

        def __len__(self):
            return len(self.buf)

        def put(self, obs, act, rew, next_obs, non_terminal: float):
            self.buf.append((obs, act, rew, next_obs, non_terminal))

        def sample(
            self, batch: int, device
        ) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            batch_items = random.sample(self.buf, batch)
            o, a, r, o2, m = zip(*batch_items)
            obs = torch.as_tensor(np.array(o), dtype=torch.float32, device=device)
            actions = torch.as_tensor(
                a,
                dtype=torch.float32 if isinstance(a[0], float) else torch.long,
                device=device,
            )
            rewards = torch.as_tensor(r, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(np.array(o2), dtype=torch.float32, device=device)
            mask = torch.as_tensor(m, dtype=torch.float32, device=device)
            return obs, actions, rewards, next_obs, mask

    # Generic Q-network: input_dim -> hidden -> ... -> output_dim
    class QNetwork(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128)):
            super().__init__()
            layers = []
            dims = [input_dim] + list(hidden_dims)
            for i in range(len(hidden_dims)):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            layers.append(nn.Linear(dims[-1], output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # Continuous-action policy network for SAC
    class ActorNetwork(nn.Module):
        def __init__(self, input_dim: int, action_dim: int, hidden_dims=(256, 256)):
            super().__init__()
            layers = []
            dims = [input_dim] + list(hidden_dims)
            for i in range(len(hidden_dims)):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            self.net = nn.Sequential(*layers)
            self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        def forward(self, x):
            x = self.net(x)
            mu = self.mu_head(x)
            log_std = self.log_std_head(x).clamp(-20, 2)
            return mu, log_std

    # Unified BaseQPolicy handling both discrete and continuous actions
    class BaseQPolicy(base_policy, ABC):
        def __init__(
            self,
            algo_name: str,
            act_space: gym.spaces.Space,
            obs_space: gym.spaces.Space,
            n_envs: int,
            n_episodes: int,
            *,
            buffer_size=50_000,
            batch=32,
            gamma=0.99,
            lr=5e-4,
            pi_lr=5e-4,
            tgt_sync=100,
            entropy_coef=0.01,
            device=None,
            extra_input_dim: int = 0,
        ):
            super().__init__(algo_name, act_space, obs_space, n_envs)
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.algo_name = algo_name
            self.n_envs = n_envs
            self.cur_ep = 0
            self.update_ct = 0

            # Encoder setup
            if isinstance(obs_space, gym.spaces.Box):
                self.obs_dim = int(np.prod(obs_space.shape))
                self._base_enc = lambda x: safe_clone(
                    x, torch.float32, self.device
                ).view(-1, self.obs_dim)
            elif isinstance(obs_space, gym.spaces.Discrete):
                self.obs_dim = obs_space.n
                self._base_enc = lambda x: one_hot(
                    safe_clone(x, torch.long, self.device), self.obs_dim
                )
            else:
                raise NotImplementedError

            def _enc(x):
                out = self._base_enc(x)
                if hasattr(self, "_extra_state") and self._extra_state is not None:
                    aug = self._extra_state.reshape(
                        self._extra_state.size(0), -1
                    ).float()
                    out = torch.cat((out, aug.to(self.device)), dim=1)
                return out

            self._enc = _enc
            self.mem = ReplayBuffer(buffer_size)
            self.batch, self.gamma, self.tgt_sync, self.entropy_coef = (
                batch,
                gamma,
                tgt_sync,
                entropy_coef,
            )
            self.extra_input_dim = extra_input_dim

            # Action-space specific networks
            self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
            input_dim = self.obs_dim + extra_input_dim

            if self.is_discrete:
                action_dim = act_space.n
                # Q-network outputs one Q-value per action
                self.q_net = QNetwork(input_dim, action_dim).to(self.device)
                self.target_q = QNetwork(input_dim, action_dim).to(self.device)
                self.target_q.load_state_dict(self.q_net.state_dict())
                self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=lr)
            else:
                action_dim = act_space.shape[0]
                # Actor network for continuous actions
                self.actor = ActorNetwork(input_dim, action_dim).to(self.device)
                # Twin critics: input (obs + act) -> scalar
                critic_in = input_dim + action_dim
                self.q_net1 = QNetwork(critic_in, 1).to(self.device)
                self.q_net2 = QNetwork(critic_in, 1).to(self.device)
                self.target_q1 = QNetwork(critic_in, 1).to(self.device)
                self.target_q2 = QNetwork(critic_in, 1).to(self.device)
                self.target_q1.load_state_dict(self.q_net1.state_dict())
                self.target_q2.load_state_dict(self.q_net2.state_dict())
                self.opt_q = torch.optim.Adam(
                    list(self.q_net1.parameters()) + list(self.q_net2.parameters()),
                    lr=lr,
                )
                self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)

        def set_extra_state(self, x):
            self._extra_state = x.to(self.device) if x is not None else None

        def update_episode(self, ep):
            self.cur_ep = ep

        def pop_metrics(self):
            return {}

        @abstractmethod
        def _get_actions(self, observations: torch.Tensor, mask: torch.Tensor = None):
            raise NotImplementedError

        @abstractmethod
        def _update(self, obs, acts, rews, next_obs, dones):
            raise NotImplementedError

        def save_policy(self, path: str):
            if is_causal:
                super().save_policy(path)

            p = self._ensure_pt_path(path)
            data = {
                "cur_ep": self.cur_ep,
                "update_ct": self.update_ct,
            }
            if self.is_discrete:
                data.update(
                    {
                        "q_net": self.q_net.state_dict(),
                        "target_q": self.target_q.state_dict(),
                        "opt_q": self.opt_q.state_dict(),
                    }
                )
            else:
                data.update(
                    {
                        "actor": self.actor.state_dict(),
                        "q_net1": self.q_net1.state_dict(),
                        "q_net2": self.q_net2.state_dict(),
                        "target_q1": self.target_q1.state_dict(),
                        "target_q2": self.target_q2.state_dict(),
                        "opt_q": self.opt_q.state_dict(),
                        "opt_pi": self.opt_pi.state_dict(),
                    }
                )
            torch.save(data, p)

        def load_policy(self, path: str):
            p = self._ensure_pt_path(path)
            data = torch.load(p, map_location=self.device)
            self.cur_ep = data["cur_ep"]
            self.update_ct = data["update_ct"]
            if self.is_discrete:
                self.q_net.load_state_dict(data["q_net"])
                self.target_q.load_state_dict(data["target_q"])
                self.opt_q.load_state_dict(data["opt_q"])
            else:
                self.actor.load_state_dict(data["actor"])
                self.q_net1.load_state_dict(data["q_net1"])
                self.q_net2.load_state_dict(data["q_net2"])
                self.target_q1.load_state_dict(data["target_q1"])
                self.target_q2.load_state_dict(data["target_q2"])
                self.opt_q.load_state_dict(data["opt_q"])
                self.opt_pi.load_state_dict(data["opt_pi"])

    return BaseQPolicy

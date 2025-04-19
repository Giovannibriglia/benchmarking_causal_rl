from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch


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

    @abstractmethod
    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
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
                return actions_tensor.cpu().numpy().astype(np.int32)  # shape (n_envs,)

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
                return actions_tensor.cpu().numpy()  # np.ndarray, shape (n_envs, …)

        # ---------------------------------------------------------------------
        else:
            raise NotImplementedError(
                f"Unsupported action space type: {type(self.action_space)}"
            )


class RandomPolicy(BasePolicy):
    def __init__(
        self,
        algo_name: str,
        action_space: gym.spaces,
        observation_space: gym.spaces,
        n_envs: int,
    ):
        super().__init__(algo_name, action_space, observation_space, n_envs)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        pass

    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:

        if isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (self.n_envs,)

            actions_tensor = torch.randint(
                0, self.action_space.n, size=action_shape, device=self.device
            )
        elif isinstance(self.action_space, gym.spaces.Box):
            action_shape = (self.n_envs, self.action_space.shape[0])

            # Original low/high from the action space
            low = torch.tensor(self.action_space.low, device=self.device)
            high = torch.tensor(self.action_space.high, device=self.device)

            # Replace infs with large finite values
            finite_low = torch.where(
                torch.isinf(low), torch.full_like(low, -1e6, device=self.device), low
            )
            finite_high = torch.where(
                torch.isinf(high), torch.full_like(high, 1e6, device=self.device), high
            )

            actions_tensor = finite_low + (finite_high - finite_low) * torch.rand(
                size=action_shape, device=self.device
            )
        else:
            raise ValueError(
                f"Unsupported action space type: {type(self.action_space)}"
            )

        return actions_tensor

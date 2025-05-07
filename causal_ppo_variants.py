# -*- coding: utf-8 -*-
"""ppo_causal_variants.py – *benchmark suite*
==============================================
Trains **all four agents** (baseline, prior, critic, joint) in one go,
collects per‑episode metrics, dumps them to a single JSON file, and offers
handy helpers to **compare metrics across agents** with line‑plots and a
summary table.

Quick start
-----------
Train and save metrics::

    python ppo_causal_variants.py --episodes 300 --outfile bench.json

Plot returns + variance for every agent::

    python - <<'PY'
    from ppo_causal_variants import plot_benchmark, summary_table
    plot_benchmark('bench.json', keys=['ret', 'adv_var'])
    print(summary_table('bench.json'))
    PY
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium_suite.run import convert_ndarray
from torch.distributions import Categorical

from tqdm import tqdm


# ───────────────────────── utilities ──────────────────────────


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_prior_cartpole(obs: torch.Tensor) -> torch.Tensor:
    angle = obs[:, 2]
    p_left = torch.where(angle < 0, 0.8, 0.2).to(obs)
    return torch.stack((p_left, 1 - p_left), dim=-1)


def random_prior_cartpole(obs: torch.Tensor) -> torch.Tensor:
    batch_size = obs.shape[0]
    probs = torch.rand(batch_size, 1)
    return torch.cat([probs, 1 - probs], dim=1)


# ───────────────────────── networks ───────────────────────────
class CausalQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):  # returns Q(s,·)  shape (B, A)
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


class CPDNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class RandomCPDNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_actions = self.net[
            -1
        ].out_features  # assuming self.net ends with a Linear layer
        logits = torch.rand(
            batch_size, num_actions, device=x.device, requires_grad=True
        )
        return F.softmax(logits, dim=-1)


# ───────── rollout buffer (GAE) ─────────
@dc.dataclass
@dc.dataclass
class T:
    s: np.ndarray
    a: int
    lp: float
    r: float
    d: bool
    v: float


class Buffer:
    def __init__(self, gamma: float, lam: float):
        self.g, self.lam = gamma, lam
        self.data: List[T] = []

    def add(self, *a):
        self.data.append(T(*a))

    def reset(self):
        self.data.clear()

    def compute(self, device):
        S = torch.as_tensor(
            [t.s for t in self.data], dtype=torch.float32, device=device
        )
        A = torch.as_tensor([t.a for t in self.data], dtype=torch.long, device=device)
        LP = torch.as_tensor(
            [t.lp for t in self.data], dtype=torch.float32, device=device
        )
        V = torch.as_tensor(
            [t.v for t in self.data], dtype=torch.float32, device=device
        )
        R = [t.r for t in self.data]
        advs, gae, nxt = [], 0.0, 0.0
        for i in reversed(range(len(R))):
            delta = R[i] + self.g * nxt * (1 - self.data[i].d) - V[i]
            gae = delta + self.g * self.lam * (1 - self.data[i].d) * gae
            advs.insert(0, gae)
            nxt = V[i]
        ADV = torch.as_tensor(advs, dtype=torch.float32, device=device)
        RET = ADV + V
        raw_var = ADV.var(unbiased=False).item()
        if ADV.numel() > 1:
            ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
        return S, A, LP, RET, ADV, raw_var


# ───────── PPO backbone ─────────
class PPOBase:
    def __init__(
        self,
        env: gym.Env,
        ac: ActorCritic,
        *,
        clip_eps=0.2,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        k_epochs=4,
        entropy_coef=0.0,
        device="cpu",
    ):
        self.env, self.ac, self.device = env, ac.to(device), torch.device(device)
        self.clip, self.k_epochs, self.entropy = clip_eps, k_epochs, entropy_coef
        self.buf = Buffer(gamma, lam)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.metrics = {}

    def _select(self, s):
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        lg, v = self.ac(st)
        d = Categorical(logits=lg)
        a = d.sample()
        return a.item(), d.log_prob(a).item(), v.item()

    def rollout(self, n):
        s, _ = self.env.reset()
        for _ in range(n):
            a, lp, v = self._select(s)
            s2, r, t, tr, _ = self.env.step(a)
            self.buf.add(s, a, lp, r, t or tr, v)
            s = s2
            if t or tr:
                s, _ = self.env.reset()

    def _extra_actor_loss(self, *_):
        return 0.0

    def _extra_critic_loss(self, *_):
        return 0.0

    def _post_update(self, *_):
        pass

    def update(self):
        S, A, oldLP, RET, ADV, var = self.buf.compute(self.device)
        self.metrics = {"adv_var": var}
        for _ in range(self.k_epochs):
            lg, vals = self.ac(S)
            dist = Categorical(logits=lg)
            lp = dist.log_prob(A)
            ratio = torch.exp(lp - oldLP)
            surr1 = ratio * ADV
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV
            act_loss = -torch.min(surr1, surr2).mean() + self._extra_actor_loss(
                lg, S, ADV
            )
            val_loss = 0.5 * F.mse_loss(vals, RET) + self._extra_critic_loss(vals, RET)
            loss = act_loss + val_loss - self.entropy * dist.entropy().mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.metrics["value_mse"] = F.mse_loss(vals.detach(), RET).item()
        self._post_update(S, A, ADV, RET)
        self.buf.reset()


# ───────── causal variants ─────────
class PPOBaseline(PPOBase):
    pass


class PPOCausalPrior(PPOBase):
    def __init__(self, *a, beta=1.0, **kw):
        super().__init__(*a, **kw)
        self.beta = beta

    def _extra_actor_loss(self, lg, S, ADV):
        with torch.no_grad():
            p = causal_prior_cartpole(S)
        kl = (
            (F.softmax(lg, -1) * (F.log_softmax(lg, -1) - torch.log(p + 1e-8)))
            .sum(-1)
            .mean()
        )
        self.metrics["prior_kl"] = kl.item()
        return self.beta * kl


class PPORandomPrior(PPOBase):
    def __init__(self, *a, beta=1.0, **kw):
        super().__init__(*a, **kw)
        self.beta = beta

    def _extra_actor_loss(self, lg, S, ADV):
        with torch.no_grad():
            p = random_prior_cartpole(S)
        kl = (
            (F.softmax(lg, -1) * (F.log_softmax(lg, -1) - torch.log(p + 1e-8)))
            .sum(-1)
            .mean()
        )
        self.metrics["prior_kl"] = kl.item()
        return self.beta * kl


class PPOCausalCritic(PPOBase):
    """
    • q_net predicts Q(s,·)
    • causal baseline  bᶜ(s)=Σ_a P_C(a|s)·Q(s,a)
    • actor still uses ordinary advantages, but the critic is trained on Q–targets
    """

    def __init__(self, env, *args, **kw):
        super().__init__(env, *args, **kw)
        self.q_net = CausalQCritic(
            env.observation_space.shape[0], env.action_space.n
        ).to(self.device)
        self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=3e-4)

    # ---------- actor loss gets an **extra** causal‑KL term if you like ----------
    def _extra_actor_loss(self, logits, states, advs):
        # causal baseline (variance reduction only)
        with torch.no_grad():
            p_causal = causal_prior_cartpole(states)  # (B,A)
            q_all = self.q_net(states)  # (B,A)
            baseline = (p_causal * q_all).sum(-1)  # (B,)
        # save for later (see _post_update) and return 0 – we don't bias the loss here
        self._cached_baseline = baseline
        return 0.0

    # ---------- critic loss is done after PPO update so we have the targets ----------
    def _post_update(self, states, acts, advs, returns):
        # train Q(s,a) on TD/MC target (here: full‑return)
        q_pred = self.q_net(states)[torch.arange(len(acts)), acts]
        loss_q = F.mse_loss(q_pred, returns)
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        self.metrics["q_loss"] = loss_q.item()


class PPORandomCausalCritic(PPOBase):
    """
    • q_net predicts Q(s,·)
    • causal baseline  bᶜ(s)=Σ_a P_C(a|s)·Q(s,a)
    • actor still uses ordinary advantages, but the critic is trained on Q–targets
    """

    def __init__(self, env, *args, **kw):
        super().__init__(env, *args, **kw)
        self.q_net = CausalQCritic(
            env.observation_space.shape[0], env.action_space.n
        ).to(self.device)
        self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=3e-4)

    # ---------- actor loss gets an **extra** causal‑KL term if you like ----------
    def _extra_actor_loss(self, logits, states, advs):
        # causal baseline (variance reduction only)
        with torch.no_grad():
            p_causal = random_prior_cartpole(states)  # (B,A)
            q_all = self.q_net(states)  # (B,A)
            baseline = (p_causal * q_all).sum(-1)  # (B,)
        # save for later (see _post_update) and return 0 – we don't bias the loss here
        self._cached_baseline = baseline
        return 0.0

    # ---------- critic loss is done after PPO update so we have the targets ----------
    def _post_update(self, states, acts, advs, returns):
        # train Q(s,a) on TD/MC target (here: full‑return)
        q_pred = self.q_net(states)[torch.arange(len(acts)), acts]
        loss_q = F.mse_loss(q_pred, returns)
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        self.metrics["q_loss"] = loss_q.item()


class PPOJointCPD(PPOCausalPrior):
    def __init__(self, env, ac, *, beta=1.0, cpd_lr=3e-4, **kw):
        super().__init__(env, ac, beta=beta, **kw)
        self.cpd = CPDNet(env.observation_space.shape[0], env.action_space.n).to(
            self.device
        )
        self.opt_cpd = torch.optim.Adam(self.cpd.parameters(), lr=cpd_lr)

    def _extra_actor_loss(self, lg, S, ADV):
        with torch.no_grad():
            p = self.cpd(S)
        kl = (
            (F.softmax(lg, -1) * (F.log_softmax(lg, -1) - torch.log(p + 1e-8)))
            .sum(-1)
            .mean()
        )
        self.metrics["prior_kl"] = kl.item()
        return self.beta * kl

    def _post_update(self, S, A, ADV, *_):
        logp = torch.log(self.cpd(S) + 1e-8)
        loss = -(F.relu(ADV).detach() * logp[torch.arange(len(A)), A]).mean()
        self.opt_cpd.zero_grad()
        loss.backward()
        self.opt_cpd.step()
        self.metrics["cpd_update"] = loss.item()


class PPORandomJointCPD(PPOCausalPrior):
    def __init__(self, env, ac, *, beta=1.0, cpd_lr=3e-4, **kw):
        super().__init__(env, ac, beta=beta, **kw)
        self.cpd = RandomCPDNet(env.observation_space.shape[0], env.action_space.n).to(
            self.device
        )
        self.opt_cpd = torch.optim.Adam(self.cpd.parameters(), lr=cpd_lr)

    def _extra_actor_loss(self, lg, S, ADV):
        with torch.no_grad():
            p = self.cpd(S)
        kl = (
            (F.softmax(lg, -1) * (F.log_softmax(lg, -1) - torch.log(p + 1e-8)))
            .sum(-1)
            .mean()
        )
        self.metrics["prior_kl"] = kl.item()
        return self.beta * kl

    def _post_update(self, S, A, ADV, *_):
        logp = torch.log(self.cpd(S) + 1e-8)
        loss = -(F.relu(ADV).detach() * logp[torch.arange(len(A)), A]).mean()
        self.opt_cpd.zero_grad()
        loss.backward()
        self.opt_cpd.step()
        self.metrics["cpd_update"] = loss.item()


AGENT_CLS = dict(
    baseline=PPOBaseline,
    prior=PPOCausalPrior,
    random_prior=PPORandomPrior,
    critic=PPOCausalCritic,
    random_critic=PPORandomCausalCritic,
    joint=PPOJointCPD,
    random_joint=PPORandomJointCPD,
)

METRICS = ["ret", "adv_var", "prior_kl", "cpd_update", "value_mse", "q_loss"]


# ───────── benchmark runner ─────────
def run_benchmark(
    env_id: str, episodes: int, rollout: int, seeds: int, checkpoints: int
) -> Dict[str, Dict[str, List[List[float]]]]:

    ck_idx = np.linspace(1, episodes, checkpoints, dtype=int)

    logs: Dict[str, Dict[str, List[List[float]]]] = {v: {} for v in AGENT_CLS}

    pbar = tqdm(range(seeds), desc="training seeds...")

    for seed_i in pbar:
        set_seed(seed_i)
        envs = {v: gym.make(env_id) for v in AGENT_CLS}
        agents = {
            v: AGENT_CLS[v](
                envs[v],
                ActorCritic(envs[v].observation_space.shape[0], envs[v].action_space.n),
            )
            for v in AGENT_CLS
        }
        for ep in range(episodes):
            for v, a in agents.items():
                a.rollout(rollout)
                a.update()
            if ep in ck_idx:
                for v, a in agents.items():
                    a.metrics["ret"] = greedy_return(envs[v], a)
                    for k, val in a.metrics.items():
                        logs[v].setdefault(k, [[] for _ in range(seeds)])[
                            seed_i
                        ].append(val)
            pbar.set_postfix(episodes=f"{ep}/{episodes}")
        for e in envs.values():
            e.close()

    save_logs(logs, "benchmark.json")
    return logs


def greedy_return(env, agent):
    s, _ = env.reset()
    ret = 0
    done = False
    while not done:
        st = torch.as_tensor(s, dtype=torch.float32)
        a = torch.argmax(torch.softmax(agent.ac(st)[0], -1)).item()
        s, r, t, tr, _ = env.step(a)
        ret += r
        done = t or tr
    return ret


# ───────── I/O helpers ─────────
def save_logs(logs: Dict[str, List[Dict[str, float]]], path: str):
    with open(path, "w") as fp:
        json.dump(convert_ndarray(logs), fp, indent=2)
    print(f"saved metrics 👉 {path}")


# ───────── plotting & tables ─────────


def plot_metric(json_path: str, key: str):
    logs = json.loads(Path(json_path).read_text())
    xs = np.arange(len(next(iter(next(iter(logs.values())).values()))[0]))
    plt.figure()
    for v in logs:
        if key in logs[v].keys():
            arr = np.array(logs[v][key])  # shape (seeds, checkpoints)
            mean = arr.mean(0)
            std = arr.std(0)
            plt.plot(xs, mean, label=v)
            plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.xticks(xs)
    plt.xlabel("checkpoint")
    plt.ylabel(key)
    plt.title(f"{key} across agents (mean±std over seeds)")
    plt.legend()
    plt.show()


# summary table of last checkpoint
def table_summary(json_path: str) -> pd.DataFrame:
    """Return a *variant × metric* table (mean over seeds, last checkpoint).
    Missing metrics are filled with NaN so every variant has the same columns."""
    logs = json.loads(Path(json_path).read_text())
    # gather the union of all metric names
    metrics = sorted({m for d in logs.values() for m in d})
    rows = {}
    for variant, var_dict in logs.items():
        row = {}
        for m in metrics:
            if m in var_dict:
                # take the last checkpoint value for every seed, then average
                arr = np.array(var_dict[m])  # shape (seeds, checkpoints)
                row[m] = (
                    f"{np.mean(arr):.3f} ± {np.std(arr):.3f}" if arr.size else np.nan
                )
            else:
                row[m] = np.nan
        rows[variant] = row
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.to_csv("res.csv")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--rollout", type=int, default=1024)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--checkpoints", type=int, default=50)
    args = p.parse_args()

    logs = run_benchmark(
        args.env, args.episodes, args.rollout, args.seeds, args.checkpoints
    )

    # run_benchmark("CartPole-v1", 10, 512, 2, 10)

    for m in METRICS:
        plot_metric("bench.json", m)

    table_summary("bench.json")

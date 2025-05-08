from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from torch.distributions import Categorical
from tqdm import tqdm

# ------------- util --------------------------------------------------------
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_prior_cartpole(obs: torch.Tensor) -> torch.Tensor:
    ang = obs[:, 2]
    p_left = torch.where(ang < 0, 0.8, 0.2).to(obs)
    return torch.stack((p_left, 1 - p_left), dim=-1)


def random_prior_cartpole(obs: torch.Tensor) -> torch.Tensor:
    batch_size = obs.shape[0]
    probs = torch.rand(batch_size, 1)
    return torch.cat([probs, 1 - probs], dim=1)


# ------------- nets --------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs: int, act: int, hid: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs, hid),
            nn.Tanh(),
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, act),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs, hid),
            nn.Tanh(),
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1),
        )

    def forward(self, x):
        lg = self.actor(x)
        v = self.critic(x).squeeze(-1)
        return lg, v


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


class RandomQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):  # returns Q(s,·), shape (B, A)
        B = x.shape[0]  # Batch size
        A = self.net[
            -1
        ].out_features  # Number of actions from the last layer of the net
        return torch.rand(B, A, device=x.device)


class CPDNet(nn.Module):
    def __init__(self, obs: int, act: int, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, act),
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


# ------------- buffer (works with vector env) ------------------------------
class T(dict):
    pass


class Buffer:
    def __init__(self, gamma: float, lam: float):
        self.g, self.lam = gamma, lam
        self.data: List[T] = []

    def add(self, **kw):
        self.data.append(T(kw))

    def reset(self):
        self.data.clear()

    def _cat(self, key):
        return np.concatenate([d[key] for d in self.data])

    def compute(self, dev):
        S = torch.as_tensor(self._cat("s"), dtype=torch.float32, device=dev)
        A = torch.as_tensor(self._cat("a"), dtype=torch.long, device=dev)
        LP = torch.as_tensor(self._cat("lp"), dtype=torch.float32, device=dev)
        V = torch.as_tensor(self._cat("v"), dtype=torch.float32, device=dev)
        R = self._cat("r")
        D = self._cat("d")
        advs, gae, nxt = [], 0.0, 0.0
        for i in range(len(R) - 1, -1, -1):
            delta = R[i] + self.g * nxt * (1 - D[i]) - V[i]
            gae = delta + self.g * self.lam * (1 - D[i]) * gae
            advs.insert(0, gae)
            nxt = V[i]
        ADV = torch.as_tensor(advs, dtype=torch.float32, device=dev)
        RET = ADV + V
        if ADV.numel() > 1:
            ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
        return S, A, LP, RET, ADV, ADV.var(unbiased=False).item()


# ------------- PPO backbone -----------------------------------------------
class PPOBase:
    hp = dict(clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95, k_epochs=4, entropy_coef=0.0)

    def __init__(self, env: gym.Env, *, device=DEFAULT_DEVICE, **override):
        self.env = env
        self.device = torch.device(device)
        p = {**self.hp, **override}
        self.clip, self.k_epochs, self.ent = (
            p["clip_eps"],
            p["k_epochs"],
            p["entropy_coef"],
        )
        self.buf = Buffer(p["gamma"], p["lam"])
        self.ac = ActorCritic(
            env.single_observation_space.shape[0], env.single_action_space.n
        ).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=p["lr"])
        self.metrics: Dict[str, float] = {}
        self.last_train_len = self.last_train_ret = 0.0

    # -- rollout --
    def _select(self, s):
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        lg, v = self.ac(st)
        dist = Categorical(logits=lg)
        a = dist.sample()
        return (
            a.cpu().numpy(),
            dist.log_prob(a).detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            dist.entropy().detach().cpu().numpy(),
        )

    def rollout(self, n_steps):
        s, _ = self.env.reset()
        steps = 0
        ep_ret = np.zeros(len(s))
        while steps < n_steps:
            a, lp, v, ent = self._select(s)
            s2, r, term, trunc, _ = self.env.step(a)
            self.buf.add(s=s, a=a, lp=lp, r=r, d=np.logical_or(term, trunc), v=v)
            ep_ret += r
            steps += len(r)
            s = s2
            if term.any() or trunc.any():
                self.last_train_len = int((term | trunc).sum())
                self.last_train_ret = float(ep_ret.sum())
                ep_ret = np.zeros(len(r))
        self.metrics["entropy"] = float(np.mean(ent))

    # -- hooks --
    def _extra_actor(self, *_) -> float:
        return 0.0

    def _extra_critic(self, *_) -> float:
        return 0.0

    def _post(self):
        pass

    # -- update --
    def update(self):
        S, A, oldLP, RET, ADV, var = self.buf.compute(self.device)
        self.metrics["adv_var"] = var
        for _ in range(self.k_epochs):
            lg, vals = self.ac(S)
            dist = Categorical(logits=lg)
            lp = dist.log_prob(A)
            ratio = torch.exp(lp - oldLP)
            act_loss = -torch.min(
                ratio * ADV, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV
            ).mean() + self._extra_actor(lg, S, ADV)
            val_loss = 0.5 * F.mse_loss(vals, RET) + self._extra_critic(vals, RET)
            loss = act_loss + val_loss - self.ent * dist.entropy().mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.metrics["value_mse"] = val_loss.item()
        self.buf.reset()
        self._post()


# ------------- Variants ----------------------------------------------------
class PPOBaseline(PPOBase):
    pass


class PPOCausalPrior(PPOBase):
    def __init__(self, env, *, beta=1.0, **kw):
        super().__init__(env, **kw)
        self.beta = beta

    def _extra_actor(self, lg, S, _):
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
    * Adds a Q‑network so we can form a causal baseline
      bᶜ(s)=Σ_a P_C(a|s)·Q(s,a).
    * Actor loss unchanged (baseline affects variance only).
    * `_post` does a single TD/M‑C regression step on Q.
    """

    def __init__(self, env, *args, **kw):
        super().__init__(env, *args, **kw)
        self.q_net = CausalQCritic(
            env.single_observation_space.shape[0], env.single_action_space.n
        ).to(self.device)
        self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=3e-4)

    # ------- override actor baseline inside the extra‑actor hook ----------
    def _extra_actor(self, logits, states, _):
        with torch.no_grad():
            pc = causal_prior_cartpole(states)  # (B, A)
            q_all = self.q_net(states)  # (B, A)
            b_c = (pc * q_all).sum(-1)  # (B,)
        self.metrics["causal_baseline_var"] = float(torch.var(b_c).item())
        return 0.0  # no extra actor loss term

    # ------- train Q(s,a) after each PPO update ---------------------------
    def _post(self):
        # Re‑use the last minibatch stored in self.buf for TD‑MC target.
        if not self.buf.data:  # safety: nothing to do first iteration
            return
        S, A, _, RET, _, _ = self.buf.compute(self.device)
        q_pred = self.q_net(S)[torch.arange(len(A)), A]  # pick Q(s,a)
        loss_q = F.mse_loss(q_pred, RET)  # MC return target
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        self.metrics["q_loss"] = loss_q.item()


class PPORandomCritic(PPOBase):
    """
    * Adds a Q‑network so we can form a causal baseline
      bᶜ(s)=Σ_a P_C(a|s)·Q(s,a).
    * Actor loss unchanged (baseline affects variance only).
    * `_post` does a single TD/M‑C regression step on Q.
    """

    def __init__(self, env, *args, **kw):
        super().__init__(env, *args, **kw)
        self.q_net = RandomQCritic(
            env.single_observation_space.shape[0], env.single_action_space.n
        ).to(self.device)
        self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=3e-4)

    # ------- override actor baseline inside the extra‑actor hook ----------
    def _extra_actor(self, logits, states, _):
        with torch.no_grad():
            pc = random_prior_cartpole(states)  # (B, A)
            q_all = self.q_net(states)  # (B, A)
            b_c = (pc * q_all).sum(-1)  # (B,)
        # Replace default entropy metric with variance after baseline?
        self.metrics["causal_baseline_var"] = float(torch.var(b_c).item())
        return 0.0  # no extra actor loss term

    # ------- train Q(s,a) after each PPO update ---------------------------
    def _post(self):
        # Re‑use the last minibatch stored in self.buf for TD‑MC target.
        if not self.buf.data:  # safety: nothing to do first iteration
            return
        S, A, _, RET, _, _ = self.buf.compute(self.device)
        q_pred = self.q_net(S)[torch.arange(len(A)), A]  # pick Q(s,a)
        loss_q = F.mse_loss(q_pred, RET)  # MC return target
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        self.metrics["q_loss"] = loss_q.item()


class PPOJointCPD(PPOCausalPrior):
    def __init__(self, env, *, beta=1.0, cpd_lr=3e-4, **kw):
        super().__init__(env, beta=beta, **kw)
        self.cpd = CPDNet(
            env.single_observation_space.shape[0], env.single_action_space.n
        ).to(self.device)
        self.opt_cpd = torch.optim.Adam(self.cpd.parameters(), lr=cpd_lr)

    def _extra_actor(self, lg, S, _):
        with torch.no_grad():
            p = self.cpd(S)
        kl = (
            (F.softmax(lg, -1) * (F.log_softmax(lg, -1) - torch.log(p + 1e-8)))
            .sum(-1)
            .mean()
        )
        self.metrics["prior_kl"] = kl.item()
        return self.beta * kl

    def _post(self):
        pass  # could train CPD here


class PPORandomJoint(PPOCausalPrior):
    def __init__(self, env, *, beta=1.0, cpd_lr=3e-4, **kw):
        super().__init__(env, beta=beta, **kw)
        self.cpd = RandomCPDNet(
            env.single_observation_space.shape[0], env.single_action_space.n
        ).to(self.device)
        self.opt_cpd = torch.optim.Adam(self.cpd.parameters(), lr=cpd_lr)

    def _extra_actor(self, lg, S, _):
        with torch.no_grad():
            p = self.cpd(S)
        kl = (
            (F.softmax(lg, -1) * (F.log_softmax(lg, -1) - torch.log(p + 1e-8)))
            .sum(-1)
            .mean()
        )
        self.metrics["prior_kl"] = kl.item()
        return self.beta * kl

    def _post(self):
        pass  # could train CPD here


AGENTS = dict(
    baseline=PPOBaseline,
    prior=PPOCausalPrior,
    random_prior=PPORandomPrior,
    critic=PPOCausalCritic,
    random_critic=PPORandomCritic,
    joint=PPOJointCPD,
    random_joint=PPORandomJoint,
)


# ------------- evaluation --------------------------------------------------
def greedy_eval(env_id: str, agent: PPOBase) -> Tuple[float, int]:
    env = gym.make(env_id)
    s, _ = env.reset()
    ret = steps = 0
    while True:
        st = torch.as_tensor(s, dtype=torch.float32, device=agent.device)
        a = torch.argmax(torch.softmax(agent.ac(st)[0], -1)).item()
        s, r, term, trunc, _ = env.step(a)
        ret += r
        steps += 1
        if term or trunc:
            break
    env.close()
    return ret, steps


# ------------- helpers -----------------------------------------------------
def make_vec_env(env_id: str, n: int, seed: int):
    def thunk(i):
        def _init():
            env = gym.make(env_id)
            env.reset(seed=seed + i)
            return env

        return _init

    return SyncVectorEnv([thunk(i) for i in range(n)])


def evaluate_vec(env_id: str, agent: PPOBase, n_envs: int, seed_offset: int):
    env = make_vec_env(env_id, n_envs, seed_offset)
    s, _ = env.reset()
    done = np.zeros(n_envs, bool)
    rets = np.zeros(n_envs)
    lens = np.zeros(n_envs, int)
    while not done.all():
        st = torch.as_tensor(s, dtype=torch.float32, device=agent.device)
        a = torch.argmax(torch.softmax(agent.ac(st)[0], -1), -1).cpu().numpy()
        s, r, term, trunc, _ = env.step(a)
        rets += (~done) * r
        lens += (~done) * 1
        done |= term | trunc
    env.close()
    return rets.tolist(), lens.tolist()


def append_metrics(file: Path, seed: int, metrics: Dict[str, float | list]):
    data: Dict = json.loads(file.read_text())
    for k, v in metrics.items():
        data.setdefault(k, {}).setdefault(str(seed), [])
        data[k][str(seed)].append(v)  # v can be float *or* list
    file.write_text(json.dumps(data, indent=2))


# ------------- training per variant ---------------------------------------
def train_variant(
    variant: str,
    env_id: str,
    episodes: int,
    rollout: int,
    seeds: int,
    checkpoints: int,
    n_train_envs: int,
    n_eval_envs: int,
    root: Path,
    device: str,
):

    ck_every = episodes // checkpoints
    ck_root = root / "checkpoints" / variant
    ck_root.mkdir(parents=True, exist_ok=True)
    metric_file = root / f"metrics_{variant}.json"
    if not metric_file.exists():
        metric_file.write_text("{}")

    for seed in range(seeds):
        set_seed(seed)
        env = make_vec_env(env_id, n_train_envs, seed)
        agent = AGENTS[variant](env, device=device)
        for ep in tqdm(
            range(1, episodes + 1), desc=f"seed: {seed+1}/{seeds} - {variant}"
        ):
            agent.rollout(rollout)
            agent.update()

            if ep % ck_every == 0:
                rets, lens = evaluate_vec(env_id, agent, n_eval_envs, seed + seeds + 1)
                agent.metrics.update(
                    {
                        "eval_ret": rets,  # list
                        "eval_len": lens,  # list
                        "train_len": agent.last_train_len,
                        "train_ret": agent.last_train_ret,
                    }
                )
                # save nets
                torch.save(
                    dict(
                        actor_critic=agent.ac.state_dict(),
                        cpd=(
                            getattr(agent, "cpd", None).state_dict()
                            if hasattr(agent, "cpd")
                            else None
                        ),
                    ),
                    ck_root / f"seed{seed}_ck{ep//ck_every}.pt",
                )
                # append metrics
                append_metrics(metric_file, seed, agent.metrics)
        env.close()


# ------------- plot / table -----------------------------------------------
def _iqr_stats(arr: np.ndarray, q: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust mean/std along axis‑0 using the inter‑quartile range.
    Guaranteed to return finite numbers without numpy RuntimeWarnings.
    Falls back to plain mean/std for columns where the IQR mask removes
    every sample (or when there is only one sample row).
    """
    if arr.size == 0:  # completely empty array
        shape = arr.shape[1:] or (1,)
        return np.zeros(shape), np.zeros(shape)

    # single seed → raw stats
    if arr.shape[0] < 2:
        return arr.astype(float), np.zeros_like(arr, dtype=float)

    q1 = np.percentile(arr, q, axis=0)
    q3 = np.percentile(arr, 100 - q, axis=0)
    mask = (arr >= q1) & (arr <= q3)
    trimmed = np.where(mask, arr, np.nan)

    with np.errstate(all="ignore"):  # silence empty‑slice warnings
        mean_iqr = np.nanmean(trimmed, axis=0)
        std_iqr = np.nanstd(trimmed, axis=0)

    # columns where IQR wiped everything (all NaN) → fall back to full range
    fallback = np.isnan(mean_iqr)
    if fallback.any():
        mean_full = arr[:, fallback].mean(axis=0)
        std_full = arr[:, fallback].std(axis=0)
        mean_iqr[fallback] = mean_full
        std_iqr[fallback] = std_full

    return mean_iqr, std_iqr


def plot_metric(outdir: str, key: str, n_episodes: int):
    """Plot mean curve with *IQR‑based* ribbon for eval_* metrics.

    * eval_ret / eval_len → first flatten seeds×envs, keep only values in the
      25‑75 % percentile band per checkpoint; ribbon = μ ± σ of those inliers.
    * all other metrics → ribbon = μ ± σ across seeds (legacy behaviour).
    """
    fontsize = 25

    dirp = Path(outdir)
    min_T = math.inf
    series = []

    def to_vec(step):
        return (
            np.asarray(step, float)
            if isinstance(step, (list, tuple, np.ndarray))
            else np.asarray([float(step)])
        )

    for mf in dirp.glob("metrics_*.json"):
        var = mf.stem.split("_", 1)[1]
        data = json.loads(mf.read_text()).get(key, {})
        if not data:
            continue
        # build (seeds, T, n_eval)
        seqs = []
        for seq in data.values():
            vec_steps = [to_vec(st) for st in seq]
            width = max(len(v) for v in vec_steps)
            seqs.append(
                np.vstack(
                    [
                        np.pad(v, (0, width - len(v)), constant_values=np.nan)
                        for v in vec_steps
                    ]
                )
            )
        arr = np.stack(seqs)  # (seeds, T, n_eval)
        min_T = min(min_T, arr.shape[1])
        series.append((var, arr))
    if not series:
        print(f"no '{key}' logged")
        return

    plt.figure(dpi=500, figsize=(16, 9))
    xs = np.linspace(0, n_episodes - 1, min_T, dtype=int)
    for var, arr in series:
        arr = arr[:, :min_T]
        if key.startswith("eval_"):
            flat = arr.reshape(arr.shape[0], arr.shape[1], -1)  # seeds,T,S
            mean, lo, hi = [], [], []
            for t in range(min_T):
                samples = flat[:, t, :].ravel()
                q25, q75 = np.nanpercentile(samples, [25, 75])
                inl = samples[(samples >= q25) & (samples <= q75)]
                mean.append(np.nanmean(inl))
                lo.append(q25)
                hi.append(q75)
            mu = np.array(mean)
            lo = np.array(lo)
            hi = np.array(hi)
        else:
            mu = np.nanmean(arr, axis=(0, 2))
            sd = np.nanstd(arr, axis=(0, 2))
            lo, hi = mu - sd, mu + sd
        linewidth = 4 if var == "baseline" else 2
        plt.plot(xs, mu, label=var, linewidth=linewidth)
        plt.fill_between(xs, lo, hi, alpha=0.1)

    xtic = np.linspace(0, n_episodes - 1, 9, dtype=int)
    plt.xticks(xtic, fontsize=fontsize)
    plt.xlabel("episodes", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel(key, fontsize=fontsize)

    plt.grid(True)
    plt.legend(loc="best", fontsize=20)
    plt.tight_layout()
    plt.show()


def table_summary(outdir: str):
    print("TABLE IS GENERATED WRONG, YOU SHOULD TAKE IQR MEAN AND STD")
    """Variant × metric table with *IQR mean* for eval_ret/len."""
    dirp = Path(outdir)
    union = set()
    rows = {}
    for mf in dirp.glob("metrics_*.json"):
        union |= set(json.loads(mf.read_text()).keys())
    metrics = sorted(union)

    def last_iqr_mean(seq):
        last = seq[-1]
        if isinstance(last, (list, tuple)):
            q25, q75 = np.percentile(last, [25, 75])
            inl = [x for x in last if q25 <= x <= q75]
            return float(np.mean(inl)) if inl else float("nan")
        return float(last)

    for mf in dirp.glob("metrics_*.json"):
        var = mf.stem.split("_", 1)[1]
        data = json.loads(mf.read_text())
        row = {}
        for m in metrics:
            seeds_dict = data.get(m, {})
            if seeds_dict:
                vals = [last_iqr_mean(seq) for seq in seeds_dict.values() if seq]
                row[m] = float(np.mean(vals)) if vals else math.nan
            else:
                row[m] = math.nan
        rows[var] = row
    df = (
        pd.DataFrame.from_dict(rows, orient="index")
        .sort_index()
        .from_dict(rows, orient="index")
    )
    df.to_csv(outdir / "results.csv")


# ------------- CLI ---------------------------------------------------------
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--variant", choices=list(AGENTS) + ["all"], default="all")
    cli.add_argument("--env", default="CartPole-v1")
    cli.add_argument("--episodes", type=int, default=50000)
    cli.add_argument("--rollout", type=int, default=1024)
    cli.add_argument("--seeds", type=int, default=1)
    cli.add_argument("--checkpoints", type=int, default=250)
    cli.add_argument("--n_train_envs", type=int, default=32)
    cli.add_argument("--n_eval_envs", type=int, default=32)
    cli.add_argument("--device", default=DEFAULT_DEVICE)
    cli.add_argument("--outdir", default="runs_ok2")

    args = cli.parse_args()

    root = Path(args.outdir)
    root.mkdir(exist_ok=True)
    variants = AGENTS if args.variant == "all" else {args.variant: AGENTS[args.variant]}
    for v in variants:
        train_variant(
            v,
            args.env,
            args.episodes,
            args.rollout,
            args.seeds,
            args.checkpoints,
            args.n_train_envs,
            args.n_eval_envs,
            root,
            args.device,
        )

    for k in [
        "train_ret",
        "eval_ret",
        "train_len",
        "eval_len",
        "adv_var",
        "entropy",
        "value_mse",
        "prior_kl",
        "causal_baseline_var",
    ]:
        plot_metric(root, k, args.episodes)

    table_summary(root)

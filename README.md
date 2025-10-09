# 🧠 Causal Critic — Vectorized Bayesian Networks for Actor–Critic RL

This repository accompanies the paper
**“Causal Value Function Estimation for Actor–Critic Architectures”** (submitted to AAMAS 2026).

---

## 📖 Overview

We introduce a **Causal Critic** for RL, implemented through **Vectorized Bayesian Networks (VBNs)** — a differentiable,
GPU-accelerated representation of Structural Causal Models (SCMs).
The Causal Critic explicitly models the causal effect of actions on rewards, removing spurious correlations and
improving **sample efficiency**, **stability**, and **policy convergence**.

Our implementation extends the **Vanilla Actor–Critic** and **Advantage Actor–Critic (A2C)** frameworks with a
drop-in causal replacement of the critic.

---

## ✨ Highlights

* 🔍 **Causal Value Estimation** — learns ($Q_{\mathrm{do}}(s,a) = \mathbb{E}[R \mid \mathrm{do}(A=a),S=s]$)
* ⚙️ **Vectorized Bayesian Networks (VBNs)** — continuous causal reasoning on GPU
* 🧩 **Modular Integration** — plug-and-play with any Actor–Critic variant
* 📈 **Performance Gains** — up to **+32 %** on Vanilla AC and **+15 %** on A2C across Gymnasium benchmarks
* 🎥 **Smooth and Stable Learning Curves** — reduced variance and policy-independent critics

---


## ⚡ Quick Start

```bash
# build python environment
python3 -m venv .venv
#
# Install dependencies
pip install -r requirements.txt

# run main experiments
python3 __main__.py
```

---

## 🎬 Visual Results


| Environment        | Video                             |
|--------------------|-----------------------------------|
| **Acrobot-v1**     | ![](figures/gifs/acrobot.mp4)     |
| **CartPole-v1**    | ![](figures/gifs/cart_pole.mp4)   |
| **LunarLander-v3** | ![](figures/gifs/lunar_lander.mp4) |
| **Pendulum-v1**    | ![](figures/gifs/pendululum.mp4)  |

---

## 📈 Evaluation Plots

| Environemnt         | Plot                                                    |
|---------------------|---------------------------------------------------------|
| **BipedalWalker-v3** | ![](figures/plots/BipedalWalker-v3_evaluation_return.pdf) |
| **FrozenLake-v1**   | ![](figures/plots/FrozenLake-v1_evaluation_return.pdf)  |
| **LunarLander-v3** | ![](figures/plots/LunarLander-v3_evaluation_return.pdf) |
| **Walker2d-v5** | ![](figures/plots/Walker2d-v5_evaluation_return.pdf)    |

## 📈 Computational Complexity
| Inference only                                    | Train and Inference                           |
|---------------------------------------------------|-----------------------------------------------|
| ![](figures/plots/cost_vs_nobs_ALL_inference.pdf) | ![](figures/plots/cost_vs_nobs_ALL_refit.pdf) |

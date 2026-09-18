# Model-based value iteration vs DQN / PPO — first trial

Online budget: 8 envs x 256 steps x 50 rounds = 102,400 env steps per run. Offline budget: 10,000 gradient steps per run. Final-checkpoint eval return, mean ± sd over seeds (0, 1) (n shown). PPO has no offline form; CQL is the second offline comparator. THE METRIC: the runner's eval sums rewards over a FIXED window of steps per env (256 online, 500 offline) with autoreset, averaged over eval envs -- for episodic tasks that is reward per window, not per episode (CliffWalking: -1 per step, so the optimal 13-step loop reads about -(window - window/14); Taxi: +20 per delivery; FrozenLake: goals reached per window).

## online

| env | tier | mlp_vi | tabular_vi | dqn | ppo |
|---|---|---|---|---|---|
| FrozenLake-v1 | — | — | 3.8 ± 0.1 (n=2) | 4.3 ± 0.2 (n=2) | 4.0 ± 0.4 (n=2) |
| Taxi-v3 | — | — | 126.2 ± 0.0 (n=2) | -255.0 ± 0.0 (n=2) | -255.0 ± 0.0 (n=2) |
| CliffWalking200-v1 | — | — | -238.0 ± 0.0 (n=2) | -255.0 ± 0.0 (n=2) | -255.0 ± 0.0 (n=2) |
| CartPole-v1 | — | 254.1 ± 0.1 (n=2) | — | 252.5 ± 2.5 (n=2) | 256.0 ± 0.0 (n=2) |
| Acrobot-v1 | — | -254.4 ± 1.6 (n=2) | — | -256.0 ± 0.0 (n=2) | -250.9 ± 0.3 (n=2) |
| MountainCar-v0 | — | -255.0 ± 0.0 (n=2) | — | -255.0 ± 0.0 (n=2) | -255.0 ± 0.0 (n=2) |
| LunarLander-v3 | — | -3.2 ± 58.2 (n=2) | — | -16.0 ± 9.0 (n=2) | -46.5 ± 27.9 (n=2) |

## offline

| env | tier | offline_mlp_vi | offline_tabular_vi | cql | offline_dqn |
|---|---|---|---|---|---|
| FrozenLake-v1 | uniform | — | 6.4 ± 0.1 (n=2) | 1.6 ± 1.6 (n=2) | 5.2 ± 0.8 (n=2) |
| FrozenLake-v1 | medium | — | 3.2 ± 0.1 (n=2) | 3.8 ± 0.1 (n=2) | 0.9 ± 0.9 (n=2) |
| Taxi-v3 | uniform | — | 277.5 ± 0.7 (n=2) | -497.3 ± 0.7 (n=2) | -1323.9 ± 45.2 (n=2) |
| Taxi-v3 | medium | — | -505.6 ± 4.8 (n=2) | -498.0 ± 0.0 (n=2) | -692.1 ± 84.4 (n=2) |
| CliffWalking200-v1 | uniform | — | -465.0 ± 0.0 (n=2) | -498.0 ± 0.0 (n=2) | -498.0 ± 0.0 (n=2) |
| CliffWalking200-v1 | medium | — | -498.0 ± 0.0 (n=2) | -498.0 ± 0.0 (n=2) | -498.0 ± 0.0 (n=2) |
| CartPole-v1 | medium | 498.2 ± 1.1 (n=2) | — | 469.8 ± 0.2 (n=2) | 462.3 ± 1.5 (n=2) |
| Acrobot-v1 | medium | -499.5 ± 0.5 (n=2) | — | -500.0 ± 0.0 (n=2) | -500.0 ± 0.0 (n=2) |
| LunarLander-v3 | medium | -3250.0 ± 305.3 (n=2) | — | -1243.2 ± 202.1 (n=2) | -2936.1 ± 4.9 (n=2) |

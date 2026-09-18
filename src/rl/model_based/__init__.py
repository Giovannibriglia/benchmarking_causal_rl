"""Model-based value iteration: learn a model of the environment from the
replay buffer, then plan on it.

Two members, one contract:

* ``TabularVI`` -- discrete observation + discrete action spaces. Counts
  ``N(s, a, s')`` give the maximum-likelihood transition/reward/termination
  tables; exact value iteration on those tables gives ``Q``.
* ``MLPValueIteration`` -- continuous (vector) observations + discrete
  actions. An ensemble of MLP dynamics models predicts ``(delta_s, r, done)``;
  a Q-network is trained by FULL Bellman backups through the model at every
  action, on states drawn from the buffer (fitted value iteration).

Both are ``BaseOffPolicy`` agents that own a ``buffer``, so they ride the
existing online loop (collect -> buffer -> ``update(batch)``) and the existing
offline loop (Minari fill -> ``update(sample)``) unchanged: ``tabular_vi`` /
``mlp_vi`` are online, ``offline_tabular_vi`` / ``offline_mlp_vi`` train on a
fixed dataset, exactly like ``dqn`` / ``offline_dqn``.
"""

from .mlp_vi import DynamicsEnsemble, MLPValueIteration
from .tabular_vi import TabularVI

__all__ = ["DynamicsEnsemble", "MLPValueIteration", "TabularVI"]

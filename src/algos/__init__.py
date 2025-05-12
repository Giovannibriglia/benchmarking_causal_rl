from .a2c import A2C
from .a2c_cc import CausalCriticA2C
from .ppo import PPO
from .ppo_ca import CausalActorPPO
from .ppo_cc import CausalCriticPPO

AGENTS = {
    # "ppo_ca": CausalActorPPO,
    "ppo_cc": CausalCriticPPO,
    "ppo_base": PPO,
    # "ppo_cc": CausalCriticPPO,
    "a2c_base": A2C,
    "a2c_cc": CausalCriticA2C,
}

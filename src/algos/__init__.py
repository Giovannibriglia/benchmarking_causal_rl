from .a2c import A2C
from .ppo import PPO
from .ppo_cc import CausalCriticPPO

AGENTS = {
    "ppo_cc": CausalCriticPPO,
    "ppo": PPO,
    "a2c": A2C,
}

from .a2c import A2C
from .a2c_new import CBNOnlyA2C
from .ppo import PPO
from .ppo_new import CBNOnlyPPO

AGENTS = {
    "a2c_cc": CBNOnlyA2C,
    "a2c": A2C,
    "ppo_cc": CBNOnlyPPO,
    "ppo": PPO,
}

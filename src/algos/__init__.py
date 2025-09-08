from .a2c import A2C
from .a2c_cc import A2C_CC
from .ppo import PPO
from .ppo_cc import PPO_CC
from .trpo import TRPO
from .trpo_cc import TRPO_CC

AGENTS = {
    "trpo_cc": TRPO_CC,
    "trpo": TRPO,
    "a2c_cc": A2C_CC,
    "a2c": A2C,
    "ppo_cc": PPO_CC,
    "ppo": PPO,
}

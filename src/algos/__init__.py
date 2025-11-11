from .a2c import A2C
from .a2c_cc import A2C_CC
from .a2c_empiricalcheck import A2C_EmpiricalCheck
from .ppo import PPO
from .ppo_cc import PPO_CC
from .ppo_empiricalcheck import PPO_EmpiricalCheck
from .trpo import TRPO
from .trpo_cc import TRPO_CC
from .trpo_empiricalcheck import TRPO_EmpiricalCheck
from .vanilla_ac import VanillaAC
from .vanilla_ac_cc import VanillaAC_CC
from .vanilla_empiricalcheck import VanillaAC_EmpiricalCheck

AGENTS = {
    "vanilla": VanillaAC,
    "vanilla_cc": VanillaAC_CC,
    "a2c_cc": A2C_CC,
    "a2c": A2C,
    # "ppo_cc": PPO_CC,
    # "ppo": PPO,
    # "trpo_cc": TRPO_CC,
    # "trpo": TRPO,
}


EMPIRICAL_CHECKS = {
    "a2c": A2C_EmpiricalCheck,
    "vanilla": VanillaAC_EmpiricalCheck,
    # "ppo": PPO_EmpiricalCheck,
    # "trpo": TRPO_EmpiricalCheck,
}

from .a2c import A2C
from .a2c_ablation import A2C_Ablation
from .a2c_cc import A2C_CC
from .ppo import PPO
from .ppo_ablation import PPO_Ablation
from .ppo_cc import PPO_CC
from .trpo import TRPO
from .trpo_ablation import TRPO_Ablation
from .trpo_cc import TRPO_CC
from .vanilla_ablation import VanillaAC_Ablation
from .vanilla_ac import VanillaAC
from .vanilla_ac_cc import VanillaAC_CC

AGENTS = {
    "vanilla_cc": VanillaAC_CC,
    "vanilla": VanillaAC,
    "a2c_cc": A2C_CC,
    "a2c": A2C,
    "ppo_cc": PPO_CC,
    "ppo": PPO,
    "trpo_cc": TRPO_CC,
    "trpo": TRPO,
}

ABLATIONS = {
    "a2c": A2C_Ablation,
    "vanilla": VanillaAC_Ablation,
    "ppo": PPO_Ablation,
    "trpo": TRPO_Ablation,
}

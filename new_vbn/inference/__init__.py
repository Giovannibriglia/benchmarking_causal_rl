from .exact.clg import ExactCLGInferencer
from .exact.gaussian import ExactLinearGaussianInferencer
from .exact.ve import VariableEliminationInferencer
from .montecarlo.gibbs import GibbsSamplingInferencer
from .montecarlo.likelihood_weighting import LikelihoodWeightingInferencer

INFERENCE_METHODS = {
    "exact.ve": lambda **kw: VariableEliminationInferencer(**kw),
    "exact.gaussian": lambda **kw: ExactLinearGaussianInferencer(**kw),
    "montecarlo.lw": lambda **kw: LikelihoodWeightingInferencer(**kw),
    "montecarlo.gibbs": lambda **kw: GibbsSamplingInferencer(**kw),
    "exact.clg": lambda **kw: ExactCLGInferencer(**kw),
}

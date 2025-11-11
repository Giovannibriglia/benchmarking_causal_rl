from .exact.clg import ExactCLGInferencer
from .exact.gaussian import ExactLinearGaussianInferencer
from .exact.ve import VariableEliminationInferencer
from .likelyhood_free.abc import ABCInferencer
from .montecarlo.gibbs import GibbsSamplingInferencer
from .montecarlo.likelihood_weighting import LikelihoodWeightingInferencer
from .smc.rao_blackwellized import RaoBlackwellizedSMCInferencer
from .variational.meanfield import MeanFieldFullVIInferencer

INFERENCE_METHODS = {
    "exact.ve": lambda **kw: VariableEliminationInferencer(**kw),
    "exact.gaussian": lambda **kw: ExactLinearGaussianInferencer(**kw),
    "montecarlo.lw": lambda **kw: LikelihoodWeightingInferencer(**kw),
    "montecarlo.gibbs": lambda **kw: GibbsSamplingInferencer(**kw),
    "variational.mf_full": lambda **kw: MeanFieldFullVIInferencer(**kw),
    "lfi.abc": lambda **kw: ABCInferencer(**kw),
    "exact.clg": lambda **kw: ExactCLGInferencer(**kw),
    "smc.rb": lambda **kw: RaoBlackwellizedSMCInferencer(**kw),
}

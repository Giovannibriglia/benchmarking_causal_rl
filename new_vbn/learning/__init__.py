from .differentiable_cpds.flow_rnvp import CondRealNVPFlowCPD
from .differentiable_cpds.gaussian_nn import GaussianNNCPD
from .differentiable_cpds.kde import KDEGaussianLearnBW
from .differentiable_cpds.mdn import MDNCPD
from .differentiable_cpds.softmax_nn import SoftmaxNNCPD
from .implicit_generators.c_mmd import CondMMDGenerator
from .implicit_generators.c_wgan import CondWGANGenerator
from .parametric_cpds.kde import KDEGaussianCPD
from .parametric_cpds.linear_gaussian import ParametricLinearGaussianCPD
from .parametric_cpds.mle import MLESoftmaxCPD

LEARNING_METHODS = {
    "linear_gaussian": lambda name, in_dim, out_dim, **kw: ParametricLinearGaussianCPD(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
    "mle_softmax": lambda name, in_dim, out_dim, **kw: MLESoftmaxCPD(
        name=name,
        in_dim=in_dim,
        num_classes=kw.pop("num_classes", out_dim),  # <-- consume num_classes here
        **kw,
    ),
    "kde_gaussian": lambda name, in_dim, out_dim, **kw: KDEGaussianCPD(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
    "kde_gaussian_diff": lambda name, in_dim, out_dim, **kw: KDEGaussianLearnBW(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
    "gaussian_nn": lambda name, in_dim, out_dim, **kw: GaussianNNCPD(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
    "mdn": lambda name, in_dim, out_dim, **kw: MDNCPD(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
    "softmax_nn": lambda name, in_dim, out_dim, **kw: SoftmaxNNCPD(
        name=name,
        in_dim=in_dim,
        out_dim=kw.pop("num_classes", out_dim),
        **kw,
    ),
    "flow_rnvp": lambda name, in_dim, out_dim, **kw: CondRealNVPFlowCPD(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
    "implicit_c_mmd": lambda name, in_dim, out_dim, **kw: CondMMDGenerator(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
    "implicit_c_wgan": lambda name, in_dim, out_dim, **kw: CondWGANGenerator(
        name=name, in_dim=in_dim, out_dim=out_dim, **kw
    ),
}

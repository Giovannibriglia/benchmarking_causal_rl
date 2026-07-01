"""Step 3 — vectorized ProximalEM parity gate.

The vectorized run_em (padded masked-batched E/M) must be the SAME estimator as the
frozen scalar run_em, only faster. From an IDENTICAL init (same seeded params, same
warm-start r_tau, same episodes), one run_em is compared:

  * delta trajectory: BITWISE (torch.equal) — the hard bar. If delta is not bitwise
    the masking logic is wrong.
  * r_tau: allclose(atol=1e-6) — masked single-reduction reorders the FP sum vs the
    scalar per-episode sum; the only effect is <=1 float32 ulp on the final sigmoid
    (reordering noise, not logic divergence).

CPU parity is the hard gate (the estimator was validated on CPU). GPU parity runs
when CUDA is present (reduction kernels may need a slightly looser but still tight
r_tau tol; delta stays bitwise).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from tests._proximal_scalar_reference import scalar_run_em, scalar_warm_start

warnings.filterwarnings("ignore")

pytest.importorskip("minari")
pytest.importorskip("h5py")

_CP = "generated/cartpole/random-bias_confounded-sigma050-v0"
_AC = "generated/acrobot/random-bias_confounded-sigma050-v0"


def _make(device, obs_dim, action_dim):
    from src.envs.registry import register_default_env_wrappers

    torch.manual_seed(0)
    register_default_algorithms()
    register_default_env_wrappers()
    _, ag = registry.get("cql_proximal").builder(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type="discrete",
        device=device,
        action_space=None,
        obs_shape=(obs_dim,),
    )
    return ag._proximal_em


def _parity(device, did, obs_dim, action_dim):
    from src.envs.offline.minari_loader import fill_sequence_buffer_from_minari
    from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

    def fresh():
        em = _make(device, obs_dim, action_dim)
        buf = SequenceReplayBuffer(capacity=200_000, device=device)
        fill_sequence_buffer_from_minari(did, buf, device, load_u=False)
        return em, buf

    # SCALAR reference: warm-start (scalar) then scalar run_em.
    em_r, buf_r = fresh()
    em_r.seq_buffer = buf_r
    scalar_warm_start(em_r)
    init_rtau = [float(ep[0]["r_tau"]) for ep in buf_r.iter_episodes()]

    # VECTORIZED: identical params + identical init r_tau (copied from the scalar
    # warm-start so the ONLY difference measured is scalar-vs-vectorized run_em).
    em_v, buf_v = fresh()
    em_v.seq_buffer = buf_v
    em_v._episodes = list(buf_v.iter_episodes())
    em_v._batch = em_v._build_batch(em_v._episodes)
    em_v.rm.load_state_dict(em_r.rm.state_dict())
    rt0 = torch.tensor(init_rtau, device=device)
    em_v._batch["r_tau"] = rt0
    em_v._scatter_r_tau(rt0)

    scalar_run_em(em_r)
    em_v.run_em()

    delta_r, delta_v = float(em_r.rm.delta), float(em_v.rm.delta)
    rtau_r = torch.tensor([float(ep[0]["r_tau"]) for ep in buf_r.iter_episodes()])
    rtau_v = torch.tensor([float(ep[0]["r_tau"]) for ep in buf_v.iter_episodes()])
    return delta_r, delta_v, rtau_r, rtau_v


@pytest.mark.parametrize(
    "did,od,ad,label", [(_CP, 4, 2, "cartpole"), (_AC, 6, 3, "acrobot")]
)
def test_vectorized_parity_cpu(did, od, ad, label):
    dr, dv, rr, rv = _parity(torch.device("cpu"), did, od, ad)
    max_rtau = float((rr - rv).abs().max())
    print(
        f"\n[CPU {label}] delta scalar={dr:.10f} vec={dv:.10f} |Δ|={abs(dr-dv):.2e} "
        f"| r_tau max|Δ|={max_rtau:.2e}"
    )
    assert dr == dv, f"delta not bitwise ({label}): {dr} vs {dv}"  # hard bar
    assert max_rtau <= 1e-6, f"r_tau drift too large ({label}): {max_rtau}"  # <=1 ulp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "did,od,ad,label", [(_CP, 4, 2, "cartpole"), (_AC, 6, 3, "acrobot")]
)
def test_vectorized_parity_gpu(did, od, ad, label):
    dr, dv, rr, rv = _parity(torch.device("cuda"), did, od, ad)
    max_rtau = float((rr - rv).abs().max())
    print(
        f"\n[GPU {label}] delta scalar={dr:.10f} vec={dv:.10f} |Δ|={abs(dr-dv):.2e} "
        f"| r_tau max|Δ|={max_rtau:.2e}"
    )
    # delta bitwise within-device; r_tau tol slightly looser for GPU reduction kernels.
    assert dr == dv, f"delta not bitwise on GPU ({label}): {dr} vs {dv}"
    assert max_rtau <= 1e-5, f"r_tau GPU drift too large ({label}): {max_rtau}"

"""The POMDP branch's wiring — selection, collapse, augmentation, exhaustion.

The heavy fit is stubbed (it has its own tests); what these pin is the
BRANCH: dr2_cut required, k=0 collapsing to the MDP branch, k>=1 fitting the
augmented view and substituting into the REAL buffer with full coverage, and
exhaustion abstaining as BUDGET-BOUND with the evidence attached.
"""

from __future__ import annotations

import numpy as np
import pytest
import src.rl.offline.grace.pomdp_branch as pb
import torch
from src.rl.offline.grace.serving import GraceServing, SERVE_PESSIMISTIC


def _dict_buffer(n_ep=8, t_len=12, d=2, hidden_velocity=False, seed=0):
    """A dict-layout buffer (the extractor's third path) with known content."""
    rng = np.random.default_rng(seed)
    obs, act, rew, eps = [], [], [], []
    for e in range(n_ep):
        v = rng.standard_normal()
        o = np.zeros((t_len + 1, d))
        o[0] = rng.standard_normal(d)
        a = rng.integers(0, 2, size=t_len)
        for t in range(t_len):
            if hidden_velocity:
                v = 0.9 * v + 0.4 * (a[t] - 0.5)
                o[t + 1, 0] = o[t, 0] + v
                o[t + 1, 1:] = 0.8 * o[t, 1:]
            else:
                o[t + 1] = 0.8 * o[t] + 0.3 * a[t] + 0.1 * rng.standard_normal(d)
        obs.append(o[:-1])
        act.append(a)
        rew.append(rng.rand(t_len) if hasattr(rng, "rand") else rng.random(t_len))
        eps.append(np.full(t_len, e))
    return dict(
        obs=torch.tensor(np.concatenate(obs), dtype=torch.float32),
        actions=torch.tensor(np.concatenate(act)),
        rewards=torch.tensor(np.concatenate(rew), dtype=torch.float32),
        episode_ids=torch.tensor(np.concatenate(eps)),
        dones=torch.zeros(n_ep * t_len),
    )


def _stub_select(k, n_verdicts=1):
    class _V:
        p_value, statistic = 0.5, 1e-9

        def label(self, a):
            return "stub"

    def fake(episodes, **kw):
        return k, [_V() for _ in range(max(n_verdicts, 1))]

    return fake


def _stub_transform(monkeypatch, record):
    def fake(buffer, **options):
        record["buffer"] = buffer
        record["options"] = options
        n = int(buffer["rewards"].shape[0]) if isinstance(buffer, dict) else 96
        return GraceServing(
            mode=SERVE_PESSIMISTIC,
            fit_label="fit",
            l4_kind="interval",
            lo=0.4,
            hi=0.6,
            rewards=torch.full((n,), 7.0),
        )

    monkeypatch.setattr(pb, "transform_offline_rewards", fake)


def test_dr2_cut_is_required():
    with pytest.raises(ValueError, match="calibration"):
        pb.transform_offline_rewards_pomdp(_dict_buffer(), dr2_cut=None)


def test_k0_collapses_to_the_mdp_branch(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec)
    monkeypatch.setattr(pb.l5, "select_window", _stub_select(0))
    buf = _dict_buffer()
    s = pb.transform_offline_rewards_pomdp(buf, dr2_cut=1e-3)
    assert s.meta["window_k"] == 0 and not s.abstained
    assert rec["buffer"] is buf  # the REAL buffer, no augmentation


def test_k1_fits_augmented_view_and_substitutes_full_coverage(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec)
    monkeypatch.setattr(pb.l5, "select_window", _stub_select(1, 2))
    # production buffers expose a write target (episodes or _data); the dict
    # fixture gets the same via the shim
    buf = pb._DictBuffer(_dict_buffer(hidden_velocity=True))
    n = int(buf["rewards"].shape[0])
    s = pb.transform_offline_rewards_pomdp(buf, dr2_cut=1e-3, dataset_id="ds")
    # the inner fit saw the AUGMENTED view: base d=2 obs + (action + d=2 obs) lag
    assert rec["buffer"]["obs"].shape == (n, 2 + 1 + 2)
    assert rec["options"]["dataset_id"] == "ds#k=1"
    # and the REAL buffer's rewards were substituted, every row
    assert torch.all(buf["rewards"] == 7.0)
    assert s.meta["window_k"] == 1 and s.fit_label.startswith("pomdp[window=1]")


def test_augmentation_edge_pads_early_rows():
    from src.rl.offline.grace.serving import _episode_data_from_buffer

    buf = _dict_buffer(n_ep=2, t_len=5)
    data, _, _ = _episode_data_from_buffer(buf)
    cols = pb._augmented_cols(data, k=2)
    n, d = data.state.shape
    assert cols["obs"].shape == (n, d + 2 * (1 + d))
    # row 0 of each episode: lag features equal its OWN first row (edge pad)
    first = (data.episode_ids == 0).nonzero(as_tuple=True)[0][0]
    row = cols["obs"][first]
    assert torch.equal(row[d + 1 : d + 1 + d], data.state[first])  # lag-1 state
    assert torch.equal(row[2 * d + 2 :], data.state[first])  # lag-2 state


def test_exhaustion_abstains_budget_bound(monkeypatch):
    monkeypatch.setattr(pb.l5, "select_window", _stub_select(None, 3))
    s = pb.transform_offline_rewards_pomdp(_dict_buffer(), dr2_cut=1e-3, k_max=2)
    assert s.abstained and "BUDGET-BOUND" in s.reason
    assert s.meta["window_k"] is None and s.meta["transform_applied"] is False
    assert "l5_stage_p" in s.meta

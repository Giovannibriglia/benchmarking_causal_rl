"""The declared-observability path — one code path for (observability, k).

The heavy fit is stubbed (it has its own tests); what these pin is the
BRANCH: declared MDP is k=0 through the same path; a supplied k is used as
given (not subject to k_max) with the two report-only diagnostics; a
delegated selection picks the smallest k whose next lag does NOT move the
served contrast beyond L4's half-width (materiality-by-refit, no constant);
exhaustion abstains BUDGET-BOUND; an abstaining fit inside the selection
abstains with its reason; the REAL buffer is written exactly once, by the
served fit, with full coverage; the k=0 fit is requested on the REAL buffer
unwritten (the cache collapse); the augmented view carries exact next_obs
and dones.
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
        rew.append(rng.random(t_len))
        eps.append(np.full(t_len, e))
    return dict(
        obs=torch.tensor(np.concatenate(obs), dtype=torch.float32),
        actions=torch.tensor(np.concatenate(act)),
        rewards=torch.tensor(np.concatenate(rew), dtype=torch.float32),
        episode_ids=torch.tensor(np.concatenate(eps)),
        dones=torch.zeros(n_ep * t_len),
    )


D = 2  # obs dim of the fixture


def _k_of(buffer) -> int:
    """Recover the window from the view's width: d + k (1 + d)."""
    return (int(buffer["obs"].shape[1]) - D) // (1 + D)


def _stub_transform(monkeypatch, record, contrasts, widths=None, abstain=()):
    """Per-k stub fits: ``contrasts[k]`` is the served contrast, ``widths[k]``
    L4's half-width (default 0.05); ``abstain`` lists k that abstain."""
    widths = widths or {}
    record.setdefault("calls", [])

    def fake(buffer, *, cache_dir=None, dataset_id="", apply=True, **options):
        k = _k_of(buffer)
        record["calls"].append((k, dataset_id, apply, buffer))
        if k in abstain:
            return GraceServing(reason=f"stub abstained at k={k}")
        n = int(buffer["rewards"].shape[0])
        w = widths.get(k, 0.05)
        c = contrasts[k]
        return GraceServing(
            mode=SERVE_PESSIMISTIC,
            fit_label=f"fit{k}",
            l4_kind="interval",
            lo=c - w,
            hi=c + w,
            rewards=torch.full((n,), 7.0 + k),
            meta={"contrast_point": c},
        )

    monkeypatch.setattr(pb, "transform_offline_rewards", fake)


def test_unknown_observability_raises():
    with pytest.raises(ValueError, match="mdp|pomdp"):
        pb.transform_offline_rewards_declared(_dict_buffer(), observability="hmm")


def test_declared_mdp_with_k_raises():
    with pytest.raises(ValueError, match="k = 0"):
        pb.transform_offline_rewards_declared(_dict_buffer(), observability="mdp", k=1)


def test_declared_mdp_is_k0_through_the_same_path(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {0: 0.50, 1: 0.51})
    buf = pb._DictBuffer(_dict_buffer())
    s = pb.transform_offline_rewards_declared(buf, observability="mdp", l5_report=False)
    assert s.meta["window_k"] == 0 and s.meta["window_source"] == "declared-mdp"
    # the k=0 fit is requested on the REAL buffer, UNWRITTEN (apply=False):
    # its content address is the MDP-declared arm's -- the cache collapse
    k0 = [c for c in rec["calls"] if c[0] == 0]
    assert len(k0) == 1 and k0[0][3] is buf and k0[0][2] is False
    # the ONE write, by the branch, with the served fit's rewards
    assert torch.all(buf["rewards"] == 7.0)
    # sufficient? diagnostic ran (fit at k=1) and passed: |0.51-0.50| <= 0.05
    assert s.meta["window_sufficient"] is True
    assert s.meta["window_necessary"] is None  # k=0 has no shorter window
    assert not s.abstained and s.fit_label.startswith("window[k=0|declared-mdp]")


def test_declared_mdp_too_short_warns_and_serves_anyway(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {0: 0.50, 1: 0.80})
    buf = pb._DictBuffer(_dict_buffer())
    s = pb.transform_offline_rewards_declared(buf, observability="mdp", l5_report=False)
    assert s.meta["window_k"] == 0 and s.meta["window_sufficient"] is False
    assert "WINDOW-TOO-SHORT" in s.fit_label
    assert torch.all(buf["rewards"] == 7.0)  # served AS DECLARED


def test_supplied_k_is_used_not_subject_to_k_max(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {1: 0.60, 2: 0.61, 3: 0.61})
    buf = pb._DictBuffer(_dict_buffer(hidden_velocity=True))
    n = int(buf["rewards"].shape[0])
    s = pb.transform_offline_rewards_declared(
        buf, observability="pomdp", k=2, k_max=1, dataset_id="ds", l5_report=False
    )
    assert s.meta["window_k"] == 2 and s.meta["window_source"] == "declared"
    # the served fit saw the lag-2 view under the '#k=2' audit id
    served = [c for c in rec["calls"] if c[0] == 2][0]
    assert served[1] == "ds#k=2" and served[3]["obs"].shape == (n, D + 2 * (1 + D))
    assert torch.all(buf["rewards"] == 9.0)  # rewards of the k=2 fit
    # diagnostics: sufficient (k=3 does not move it), NOT necessary (k=1 already suffices)
    assert s.meta["window_sufficient"] is True
    assert s.meta["window_necessary"] is False
    assert "WINDOW-LONGER-THAN-NEEDED" in s.fit_label


def test_diagnostics_are_a_budget_switch(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {1: 0.60})
    buf = pb._DictBuffer(_dict_buffer(hidden_velocity=True))
    s = pb.transform_offline_rewards_declared(
        buf, observability="pomdp", k=1, k_diagnostics=False, l5_report=False
    )
    assert [c[0] for c in rec["calls"]] == [1]  # exactly one fit
    assert s.meta["window_k_diagnostics"] is False
    assert "window_sufficient" not in s.meta


def test_delegated_selection_picks_smallest_immaterial_k(monkeypatch):
    rec = {}
    # k=0 -> k=1 moves the contrast by 0.30 (> w=0.05): material;
    # k=1 -> k=2 moves it by 0.01 (<= w): NOT material -> k* = 1
    _stub_transform(monkeypatch, rec, {0: 0.50, 1: 0.80, 2: 0.81})
    buf = pb._DictBuffer(_dict_buffer(hidden_velocity=True))
    s = pb.transform_offline_rewards_declared(
        buf, observability="pomdp", k_max=2, l5_report=False
    )
    assert s.meta["window_k"] == 1 and s.meta["window_source"] == "selected"
    assert sorted({c[0] for c in rec["calls"]}) == [0, 1, 2]
    assert torch.all(buf["rewards"] == 8.0)  # the k=1 fit's rewards, once
    assert s.meta["window_stage0_delta"] == pytest.approx(0.30)
    assert s.meta["window_stage1_delta"] == pytest.approx(0.01)
    assert "k=0:" in s.meta["window_stages"] and "k=2:" in s.meta["window_stages"]


def test_delegated_selection_stops_at_k0_on_markov_data(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {0: 0.50, 1: 0.52})
    buf = pb._DictBuffer(_dict_buffer())
    s = pb.transform_offline_rewards_declared(
        buf, observability="pomdp", k_max=2, l5_report=False
    )
    assert s.meta["window_k"] == 0
    assert sorted({c[0] for c in rec["calls"]}) == [0, 1]  # two fits, not three
    assert torch.all(buf["rewards"] == 7.0)


def test_exhaustion_abstains_budget_bound(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9})
    buf = pb._DictBuffer(_dict_buffer(hidden_velocity=True))
    before = buf["rewards"].clone()
    s = pb.transform_offline_rewards_declared(
        buf, observability="pomdp", k_max=2, l5_report=False
    )
    assert s.abstained and "BUDGET-BOUND" in s.reason
    assert s.meta["window_k"] is None and s.meta["transform_applied"] is False
    assert torch.equal(buf["rewards"], before)  # nothing written


def test_abstaining_fit_inside_selection_abstains_with_reason(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {0: 0.5, 1: 0.5}, abstain=(1,))
    buf = pb._DictBuffer(_dict_buffer())
    s = pb.transform_offline_rewards_declared(
        buf, observability="pomdp", k_max=2, l5_report=False
    )
    assert s.abstained and "k=1 abstained" in s.reason
    assert s.meta["window_k"] is None


def test_l5_record_travels_on_the_served_value(monkeypatch):
    rec = {}
    _stub_transform(monkeypatch, rec, {0: 0.50, 1: 0.51})
    buf = pb._DictBuffer(_dict_buffer(n_ep=10, t_len=15))
    s = pb.transform_offline_rewards_declared(
        buf, observability="mdp", l5_b=9, l5_report=True
    )
    for key in ("l5_p", "l5_dr2", "l5_shrink", "l5_base_r2", "l5_label"):
        assert key in s.meta, key
    assert s.meta["l5_lag"] == 0
    assert "l5_material_material" in s.meta  # serving_material's record, prefixed
    # a record, never a gate: served regardless of what it says
    assert not s.abstained and torch.all(buf["rewards"] == 7.0)


def test_augmentation_edge_pads_and_carries_exact_next_obs_and_dones():
    from src.rl.offline.grace.serving import _episode_data_from_buffer

    buf = _dict_buffer(n_ep=2, t_len=5)
    data, nxt, dn = _episode_data_from_buffer(buf)
    cols = pb._augmented_cols(data, 2, nxt, dn)
    n, d = data.state.shape
    assert cols["obs"].shape == (n, d + 2 * (1 + d))
    assert cols["next_obs"].shape == (n, d + 2 * (1 + d))
    assert torch.equal(cols["dones"], dn)
    # row 0 of each episode: lag features equal its OWN first row (edge pad)
    first = (data.episode_ids == 0).nonzero(as_tuple=True)[0][0]
    row = cols["obs"][first]
    assert torch.equal(row[d + 1 : d + 1 + d], data.state[first])  # lag-1 state
    assert torch.equal(row[2 * d + 2 :], data.state[first])  # lag-2 state
    # next_obs is EXACT: for a non-terminal row t, next_obs[t] == obs[t+1]
    t = int(first) + 1
    assert torch.allclose(cols["next_obs"][t], cols["obs"][t + 1])


def test_l5_record_is_budgeted_and_content_cached(monkeypatch, tmp_path):
    """The record reads the first ``l5_n_ep`` episodes (disclosed), and a
    second call on the same content is a cache HIT keyed by content, never by
    dataset id (the fingerprint lesson)."""
    rec = {}
    _stub_transform(monkeypatch, rec, {0: 0.50, 1: 0.51})
    buf = pb._DictBuffer(_dict_buffer(n_ep=10, t_len=15))
    s1 = pb.transform_offline_rewards_declared(
        buf,
        observability="mdp",
        l5_b=9,
        l5_n_ep=6,
        cache_dir=str(tmp_path),
        dataset_id="A",
    )
    assert s1.meta["l5_n_ep"] == 6 and s1.meta["l5_n_ep_used"] == 6
    assert s1.meta["l5_record_cache"] == "stored" and s1.meta["l5_n_episodes"] == 6
    buf2 = pb._DictBuffer(_dict_buffer(n_ep=10, t_len=15))  # same content, other id
    s2 = pb.transform_offline_rewards_declared(
        buf2,
        observability="mdp",
        l5_b=9,
        l5_n_ep=6,
        cache_dir=str(tmp_path),
        dataset_id="B",
    )
    assert (
        s2.meta["l5_record_cache"] == "hit"
        and s2.meta["l5_record_key"] == s1.meta["l5_record_key"]
    )
    assert s2.meta["l5_dr2"] == s1.meta["l5_dr2"]
    # serving_material grades the cached record exactly like the fresh one
    assert s2.meta["l5_material_material"] == s1.meta["l5_material_material"]

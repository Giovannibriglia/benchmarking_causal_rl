"""The transform cache — key discipline and roundtrip, pinned.

The cache's licence is bitwise (10/10 production pairs, equal perturbed-RNG
hashes), so what these tests pin is the DISCIPLINE: content-addressing, the
full-dict equality that makes collision impossible rather than unlikely, the
abstention roundtrip, and the verify-after-write (S12).
"""

from __future__ import annotations

import torch
from src.rl.offline.grace import transform_cache as tc
from src.rl.offline.grace.estimator import EpisodeData
from src.rl.offline.grace.serving import GraceServing, SERVE_PESSIMISTIC


def _data(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        EpisodeData(
            state=torch.randn(n, 3, generator=g),
            action=torch.randint(0, 2, (n,), generator=g),
            reward=torch.rand(n, generator=g),
            episode_ids=torch.arange(n) // 10,
            proxy={},
        ),
        torch.randn(n, 3, generator=g),
        torch.zeros(n),
    )


def _key(data_sha, dataset_id="ds-1"):
    return tc.build_key(
        dataset_id=dataset_id,
        data_sha256=data_sha,
        proxy_names=("Z",),
        alpha=0.1,
        b=19,
        fit_seed=0,
        init_seeds=(1, 2),
        fit_kwargs=dict(max_iter=30),
        device_kind="cpu",
    )


def test_data_fingerprint_is_content_sensitive_and_stable():
    d, nxt, dn = _data()
    a = tc.data_fingerprint(d, nxt, dn)
    assert a == tc.data_fingerprint(d, nxt, dn)  # deterministic
    d.reward[7] += 1e-6  # one changed reward => a different dataset
    assert a != tc.data_fingerprint(d, nxt, dn)


def test_roundtrip_served_and_abstained(tmp_path):
    d, nxt, dn = _data()
    key = _key(tc.data_fingerprint(d, nxt, dn))
    served = GraceServing(
        mode=SERVE_PESSIMISTIC,
        fit_label="fit",
        l4_kind="interval",
        lo=0.4,
        hi=0.6,
        rewards=torch.rand(40),
        meta=dict(contrast_point=0.5, n_transitions=40),
    )
    tc.store(tmp_path, key, served)
    back = tc.load(tmp_path, key)
    assert back is not None and not back.abstained
    assert torch.equal(back.rewards, served.rewards)  # bitwise, not close
    assert back.meta["transform_cache_hit"] is True
    assert back.lo == served.lo and back.hi == served.hi

    key2 = _key(tc.data_fingerprint(d, nxt, dn), dataset_id="ds-2")
    abst = GraceServing(reason="fit stuck")
    tc.store(tmp_path, key2, abst)
    back2 = tc.load(tmp_path, key2)
    assert back2 is not None and back2.abstained and back2.rewards is None
    assert back2.reason == "fit stuck"


def test_any_key_field_change_is_a_miss(tmp_path):
    d, nxt, dn = _data()
    sha = tc.data_fingerprint(d, nxt, dn)
    key = _key(sha)
    tc.store(tmp_path, key, GraceServing(reason="x"))
    for mutate in (
        dict(alpha=0.2),
        dict(b=39),
        dict(fit_seed=1),
        dict(init_seeds=(1, 3)),
        dict(fit_kwargs=dict(max_iter=31)),
        dict(proxy_names=("Z", "W")),
        dict(device_kind="cuda"),
        dict(data_sha256="0" * 64),
    ):
        q = tc.build_key(
            **{
                **dict(
                    dataset_id="ds-1",
                    data_sha256=sha,
                    proxy_names=("Z",),
                    alpha=0.1,
                    b=19,
                    fit_seed=0,
                    init_seeds=(1, 2),
                    fit_kwargs=dict(max_iter=30),
                    device_kind="cpu",
                ),
                **mutate,
            }
        )
        assert tc.load(tmp_path, q) is None, mutate


def test_full_dict_equality_decides_not_the_hash(tmp_path):
    """A forged entry at the right path with a wrong stored key must MISS."""
    d, nxt, dn = _data()
    key = _key(tc.data_fingerprint(d, nxt, dn))
    entry = tc.store(tmp_path, key, GraceServing(reason="x"))
    forged = dict(key, alpha=0.999)
    (entry / "key.json").write_text(__import__("json").dumps(forged))
    assert tc.load(tmp_path, key) is None

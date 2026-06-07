"""Dataset overwrite guard + collection-config metadata (Phase-6A gate 2)."""

from __future__ import annotations

import pytest
import torch
from src.data.behavior_policies import UniformExplorer
from src.data.minari_io import (
    assert_collection_config,
    collect_dataset,
    read_collection_config,
    to_offline_source,
)

DEVICE = torch.device("cpu")
DSID = "causal/cartpole/_guard_test-v0"


@pytest.fixture()
def tiny_dataset():
    collect_dataset(
        "CartPole-v1",
        UniformExplorer(2, DEVICE),
        DSID,
        n_episodes=2,
        seed=0,
        device=DEVICE,
        collection_config={"tier": "guard_test", "epsilon": 1.0},
        force=True,  # test fixture may legitimately overwrite its own id
    )
    yield DSID
    import minari

    if DSID in minari.list_local_datasets():
        minari.delete_dataset(DSID)


def test_existing_id_is_hard_error(tiny_dataset):
    with pytest.raises(FileExistsError, match="content-versioned"):
        collect_dataset(
            "CartPole-v1",
            UniformExplorer(2, DEVICE),
            tiny_dataset,
            n_episodes=1,
            seed=1,
            device=DEVICE,
        )


def test_collection_config_embedded_and_readable(tiny_dataset):
    cfg = read_collection_config(tiny_dataset)
    assert cfg is not None
    assert cfg["tier"] == "guard_test"
    assert cfg["epsilon"] == 1.0
    assert cfg["n_episodes"] == 2
    assert cfg["policy"] == "UniformExplorer"


def test_assert_collection_config(tiny_dataset):
    assert_collection_config(tiny_dataset, {"tier": "guard_test"})
    with pytest.raises(AssertionError, match="mismatch on 'epsilon'"):
        assert_collection_config(tiny_dataset, {"epsilon": 0.5})


def test_load_time_expectation(tiny_dataset):
    src = to_offline_source(
        tiny_dataset, DEVICE, expect={"tier": "guard_test", "n_episodes": 2}
    )
    assert src.n_episodes == 2
    with pytest.raises(AssertionError, match="mismatch"):
        to_offline_source(tiny_dataset, DEVICE, expect={"n_episodes": 999})

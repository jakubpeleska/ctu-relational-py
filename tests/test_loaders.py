from collections import Counter

import pytest
import torch_geometric.typing
from torch_geometric.data import HeteroData

from redelex.loaders import ComposedLoader, HGTMultiInputLoader

LOADERS = {
    "a": [f"a{i}" for i in range(5)],
    "b": [f"b{i}" for i in range(8)],
}


def _epoch(loader):
    return list(iter(loader))


def test_composed_loader_minimum():
    loader = ComposedLoader(dict(LOADERS), mode="minimum")
    batches = _epoch(loader)
    assert len(loader) == 10
    assert len(batches) == 10
    counts = Counter(b[0] for b in batches)
    assert counts == {"a": 5, "b": 5}


def test_composed_loader_full():
    loader = ComposedLoader(dict(LOADERS), mode="full")
    batches = _epoch(loader)
    assert len(loader) == 13
    assert sorted(batches) == sorted(LOADERS["a"] + LOADERS["b"])


def test_composed_loader_rnd_uni():
    loader = ComposedLoader(dict(LOADERS), mode="rnd_uni")
    batches = _epoch(loader)
    assert len(batches) == 10
    counts = Counter(b[0] for b in batches)
    assert counts["a"] <= 5 and counts["b"] <= 8


def test_composed_loader_validation():
    with pytest.raises(ValueError, match="Unknown mode"):
        ComposedLoader(dict(LOADERS), mode="bogus")


@pytest.mark.skipif(
    not torch_geometric.typing.WITH_TORCH_SPARSE,
    reason="HGTLoader requires a working torch-sparse installation",
)
def test_hgt_multi_input_loader(hetero_graph):
    data, _ = hetero_graph
    loader = HGTMultiInputLoader(
        data,
        num_samples=[4],
        input_nodes=["users", "visits"],
        batch_size=4,
    )
    batches = _epoch(loader)
    assert len(batches) == len(loader) == sum(loader.loaders_len)

    counts = Counter(node_type for node_type, _ in batches)
    assert counts["users"] == loader.loaders_len[0]
    assert counts["visits"] == loader.loaders_len[1]
    for node_type, batch in batches:
        assert isinstance(batch, HeteroData)
        assert batch[node_type].batch_size <= 4

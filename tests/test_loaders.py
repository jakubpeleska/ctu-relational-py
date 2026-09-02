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


class _RatioLoader:
    """Length-only stand-in; ComposedLoader needs just __len__ and __iter__."""

    def __init__(self, n: int, tag: str):
        self.n, self.tag = n, tag

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter([(self.tag, i) for i in range(self.n)])


def _ratio(mode, n_new, n_old, weights=None, reps=20):
    counts = Counter()
    for _ in range(reps):
        loader = ComposedLoader(
            {"new": _RatioLoader(n_new, "new"), "old": _RatioLoader(n_old, "old")},
            mode=mode,
            **({"weights": weights} if weights is not None else {}),
        )
        counts.update(b[0] for b in loader)
    return counts["new"] / sum(counts.values())


def test_rnd_uni_ratio_is_proportional_not_uniform():
    # Regression guard: `rnd_uni` truncates the epoch to min(len)*n_loaders but
    # builds the draw order from the FULL loader lengths, so each loader's share
    # tracks its own size. With history 9x the increment the new-data share is
    # ~10%, not the 50% one might expect. See ComposedLoader docstring.
    assert _ratio("rnd_uni", 100, 100) == pytest.approx(0.5, abs=0.05)
    assert _ratio("rnd_uni", 100, 900) == pytest.approx(0.10, abs=0.03)
    assert _ratio("rnd_uni", 50, 1000) == pytest.approx(0.05, abs=0.02)


def test_minimum_ratio_is_uniform_regardless_of_sizes():
    for n_new, n_old in [(100, 100), (100, 900), (50, 1000)]:
        assert _ratio("minimum", n_new, n_old) == pytest.approx(0.5, abs=0.02)


def test_weighted_ratio_is_honoured():
    for frac in (0.25, 0.5, 0.75):
        got = _ratio("weighted", 500, 5000, weights={"new": frac, "old": 1 - frac})
        assert got == pytest.approx(frac, abs=0.02)


def test_weighted_never_exhausts_a_loader():
    loader = ComposedLoader(
        {"new": _RatioLoader(10, "new"), "old": _RatioLoader(1000, "old")},
        mode="weighted",
        weights={"new": 0.5, "old": 0.5},
    )
    batches = _epoch(loader)
    assert len(batches) == len(loader)
    counts = Counter(b[0] for b in batches)
    assert counts["new"] <= 10 and counts["old"] <= 1000


def test_weighted_requires_weights():
    with pytest.raises(ValueError, match="requires `weights`"):
        ComposedLoader(dict(LOADERS), mode="weighted")

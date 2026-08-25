import torch

from redelex.nn.models import DBFormer

COL_STATS = {"a": {"c1": {}, "c2": {}}, "b": {"c1": {}}}
EDGE_TYPES = [("a", "e", "b")]
C = 8


def _inputs():
    x_dict = {"a": torch.randn(3, 2, C), "b": torch.randn(4, 1, C)}
    edge_index_dict = {("a", "e", "b"): torch.tensor([[0, 1, 2], [0, 1, 2]])}
    return x_dict, edge_index_dict


def test_dbformer_preserves_node_types_without_incoming_edges():
    """Regression: node types missing from HeteroConv's output (no incoming
    edge type) raised KeyError with norm enabled or silently vanished."""
    model = DBFormer(["a", "b"], EDGE_TYPES, COL_STATS, channels=C, num_layers=2)
    x_dict, edge_index_dict = _inputs()

    out = model(x_dict, edge_index_dict)
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (3, 2, C)
    assert out["b"].shape == (4, 1, C)


def test_dbformer_residuals_without_norm():
    """Regression: residuals were nested inside the normalization branch, so
    with_norm=False silently disabled them."""
    x_dict, edge_index_dict = _inputs()

    kwargs = dict(channels=C, num_layers=1, dropout=0.0)
    torch.manual_seed(1)
    with_res = DBFormer(
        ["a", "b"], EDGE_TYPES, COL_STATS, with_norm=False, with_residuals=True, **kwargs
    )
    torch.manual_seed(1)
    without_res = DBFormer(
        ["a", "b"], EDGE_TYPES, COL_STATS, with_norm=False, with_residuals=False, **kwargs
    )

    with_res.eval()
    without_res.eval()
    out_res = with_res(x_dict, edge_index_dict)
    out_plain = without_res(x_dict, edge_index_dict)

    # Identical weights, so the difference must come from the residual paths.
    assert not torch.allclose(out_res["b"], out_plain["b"])


def test_dbformer_output_transform():
    model = DBFormer(
        ["a", "b"],
        EDGE_TYPES,
        COL_STATS,
        channels=C,
        num_layers=1,
        with_output_transform=True,
    )
    x_dict, edge_index_dict = _inputs()
    out = model(x_dict, edge_index_dict)
    assert out["a"].shape == (3, C)
    assert out["b"].shape == (4, C)

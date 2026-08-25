import torch
from torch_geometric.data import HeteroData

from redelex.nn.loss import (
    ContextContrastiveLoss,
    EdgeContrastiveLoss,
    TableContrastiveLoss,
)

C = 16


def _x_dict(data):
    return {nt: torch.randn(data[nt].tf.num_rows, C) for nt in data.node_types}


def test_table_contrastive_loss_finite():
    loss_fn = TableContrastiveLoss(C, node_types=["users", "visits"])
    x = {"users": torch.randn(10, C), "visits": torch.randn(12, C)}
    cor = {k: v + 0.1 * torch.randn_like(v) for k, v in x.items()}

    loss = loss_fn(x, cor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_table_contrastive_loss_subsampling():
    loss_fn = TableContrastiveLoss(C, node_types=["users"], max_negatives=3)
    x = {"users": torch.randn(10, C)}
    loss = loss_fn(x, {"users": x["users"].clone()})
    assert torch.isfinite(loss)


def test_table_contrastive_loss_empty():
    loss_fn = TableContrastiveLoss(C, node_types=["users"])
    x = {"users": torch.randn(1, C)}
    loss = loss_fn(x, x)
    assert loss.item() == 0.0
    loss.backward()  # the fallback must be usable in a training step


def test_context_contrastive_loss(hetero_graph):
    data, _ = hetero_graph
    loss_fn = ContextContrastiveLoss(
        C, node_types=list(data.node_types), edge_types=list(data.edge_types)
    )
    loss = loss_fn(data, _x_dict(data))
    assert torch.isfinite(loss)

    # Subsampling path.
    loss_fn = ContextContrastiveLoss(
        C,
        node_types=list(data.node_types),
        edge_types=list(data.edge_types),
        max_negatives=3,
    )
    assert torch.isfinite(loss_fn(data, _x_dict(data)))


def _mini_edge_data():
    data = HeteroData()
    data["a"].num_nodes = 6
    data["b"].num_nodes = 8
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    # Base edge name starts with characters from the set {r, e, v, _} on
    # purpose: lstrip("rev_") used to mangle it into a missing weights key.
    data["a", "ref_edge", "b"].edge_index = edge_index
    data["b", "rev_ref_edge", "a"].edge_index = edge_index.flip(0)
    return data


def test_edge_contrastive_loss_rev_prefix():
    """Regression: name.lstrip('rev_') strips a character set, not the prefix,
    so reversed edges whose base name starts with r/e/v/_ raised KeyError."""
    data = _mini_edge_data()
    loss_fn = EdgeContrastiveLoss(C, edge_types=list(data.edge_types))
    x_dict = {"a": torch.randn(6, C), "b": torch.randn(8, C)}

    loss = loss_fn(data, x_dict)
    assert torch.isfinite(loss)
    loss.backward()


def test_edge_contrastive_loss_numerically_stable():
    """Regression: exp(sim / 0.1) on unnormalized embeddings overflowed to
    inf/NaN. The log-space formulation must stay finite."""
    data = _mini_edge_data()
    loss_fn = EdgeContrastiveLoss(C, edge_types=list(data.edge_types))
    x_dict = {"a": 100 * torch.randn(6, C), "b": 100 * torch.randn(8, C)}

    loss = loss_fn(data, x_dict)
    assert torch.isfinite(loss)


def test_edge_contrastive_loss_subsampling():
    data = _mini_edge_data()
    loss_fn = EdgeContrastiveLoss(C, edge_types=list(data.edge_types), max_negatives=1)
    x_dict = {"a": torch.randn(6, C), "b": torch.randn(8, C)}
    assert torch.isfinite(loss_fn(data, x_dict))


def test_edge_contrastive_negatives_are_sampled_per_row():
    """Regression: the negatives were subsampled from one pool shared by every
    destination row (a budget of max_negatives * num_dst), so how many negatives
    a row got depended on the other rows and some rows were left with none.
    """
    loss_fn = EdgeContrastiveLoss(C, edge_types=[("a", "e", "b")], max_negatives=3)

    adj = torch.zeros(5, 20, dtype=torch.bool)
    adj[0, :5] = True  # row 0 is partly linked; rows 1-4 are not linked at all

    neg_idx = loss_fn._sample_negatives(adj)
    counts = torch.bincount(neg_idx[:, 0], minlength=5)

    assert (counts <= 3).all()
    # A row with no edges can never lose a sampled column, so it always keeps
    # the full quota. Under the pooled scheme this held only by chance.
    assert (counts[1:] == 3).all()
    # Negatives must never be a linked pair.
    assert not adj[neg_idx[:, 0], neg_idx[:, 1]].any()


def test_edge_contrastive_negatives_use_all_pairs_when_small():
    """Below the cap every unlinked pair of the batch is a negative."""
    loss_fn = EdgeContrastiveLoss(C, edge_types=[("a", "e", "b")], max_negatives=99)

    adj = torch.zeros(3, 4, dtype=torch.bool)
    adj[0, 0] = True
    neg_idx = loss_fn._sample_negatives(adj)

    assert neg_idx.size(0) == 3 * 4 - 1


def test_edge_contrastive_loss_reduced_precision():
    """Regression: the logsumexp accumulators were hardcoded to float32, so
    scatter raised "Expected self.dtype to be equal to src.dtype" as soon as the
    logits came out in reduced precision, i.e. under torch.autocast."""
    data = _mini_edge_data()
    loss_fn = EdgeContrastiveLoss(C, edge_types=list(data.edge_types))
    x_dict = {"a": torch.randn(6, C), "b": torch.randn(8, C)}

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = loss_fn(data, x_dict)
    assert torch.isfinite(loss)
    loss.float().backward()


def test_edge_contrastive_loss_degenerate_inputs():
    data = HeteroData()
    data["a"].num_nodes = 1
    data["b"].num_nodes = 1
    data["a", "e", "b"].edge_index = torch.tensor([[0], [0]])
    loss_fn = EdgeContrastiveLoss(C, edge_types=[("a", "e", "b")])
    x_dict = {"a": torch.randn(1, C), "b": torch.randn(1, C)}

    loss = loss_fn(data, x_dict)
    assert loss.item() == 0.0
    loss.backward()

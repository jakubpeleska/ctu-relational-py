import numpy as np
import pandas as pd
import pytest
import torch
from torch_frame import stype
from torch_frame.data import Dataset as TFDataset
from torch_frame.data import MultiNestedTensor

from redelex.data import TensorStatType, make_tensor_stats_dict
from redelex.nn.encoders import UniversalRowEncoder
from redelex.nn.encoders.universal_row_encoder import UniversalCategoricalEncoder


def test_multicategorical_value_embedding_alignment():
    """Regression: value-embedding shifts were assigned to contiguous blocks of
    the row-major value stream, so with more than one multicategorical column
    most elements looked up value embeddings of the wrong column."""
    torch.manual_seed(0)

    encoder = UniversalCategoricalEncoder(
        data_channels=8, stats_channels=8, embedding_dim=4, num_categories=10
    )

    # 2 rows x 2 columns, row-major bags:
    # row0: col0=[0, 1], col1=[2]; row1: col0=[1], col1=[0, 2]
    bags = [
        [torch.tensor([0, 1]), torch.tensor([2])],
        [torch.tensor([1]), torch.tensor([0, 2])],
    ]
    feat = MultiNestedTensor.from_tensor_mat(bags)

    max_card = 3
    value_emb = torch.randn(2, max_card, 4)
    stats = {
        TensorStatType.CARDINALITY: torch.tensor([3, 3]),
        TensorStatType.VALUE_EMBEDDINGS: value_emb,
    }

    out = encoder.encode_features(feat, stats)
    assert out.shape == (2, 2, 8)

    # Expected: per-bag sum of feature embeddings plus transformed per-column
    # value embeddings (padded with a zero row for the NA index).
    padded = torch.cat([torch.zeros(2, 1, 4), value_emb], dim=1)  # [col, card+1, dim]
    expected = torch.empty(2, 2, 8)
    for row in range(2):
        for col in range(2):
            values = bags[row][col]
            feat_emb = encoder.feat_embedding(values + 1).sum(dim=0)
            val_emb = padded[col][values + 1].sum(dim=0)
            expected[row, col] = feat_emb + encoder.text_transform(val_emb)

    torch.testing.assert_close(out, expected)


EMB_DIM = 16  # matches FakeTextEmbedder.embedding_dim


@pytest.fixture()
def row_encoder():
    return UniversalRowEncoder(out_channels=32, embedding_dim=EMB_DIM, col_channels=64)


def _stats_and_names(dataset, tf, text_embedder, tname):
    stats = make_tensor_stats_dict(dataset.col_stats, tf.col_names_dict, text_embedder)
    all_cols = [c for cols in tf.col_names_dict.values() for c in cols]
    name_embeddings = {c: text_embedder([c])[0] for c in all_cols}
    name_embeddings[tname] = text_embedder([tname])[0]
    return stats, name_embeddings


def test_universal_row_encoder_forward(all_stype_dataset, text_embedder, row_encoder):
    tf = all_stype_dataset.tensor_frame
    stats, name_embeddings = _stats_and_names(all_stype_dataset, tf, text_embedder, "users")

    out = row_encoder(tf, "users", stype_stats=stats, name_embeddings=name_embeddings)
    assert out.shape == (tf.num_rows, 32)
    assert torch.isfinite(out).all()


def test_universal_row_encoder_without_stats_warns(all_stype_tf, row_encoder):
    with pytest.warns(UserWarning):
        out = row_encoder(all_stype_tf, "users")
    assert out.shape == (all_stype_tf.num_rows, 32)
    assert torch.isfinite(out).all()


def test_universal_row_encoder_empty_frame(all_stype_tf, row_encoder):
    with pytest.warns(UserWarning):
        out = row_encoder(all_stype_tf[0:0], "users")
    assert out.shape == (0, 32)


def test_universal_row_encoder_embedding_stype(text_embedder, row_encoder):
    # Embedding columns must match the encoder's embedding_dim.
    df = pd.DataFrame(
        {
            "emb": [np.random.rand(EMB_DIM).tolist() for _ in range(12)],
            "num": np.random.rand(12),
        }
    )
    dataset = TFDataset(df, {"emb": stype.embedding, "num": stype.numerical}).materialize()
    tf = dataset.tensor_frame
    stats, name_embeddings = _stats_and_names(dataset, tf, text_embedder, "t")

    out = row_encoder(tf, "t", stype_stats=stats, name_embeddings=name_embeddings)
    assert out.shape == (12, 32)
    assert torch.isfinite(out).all()

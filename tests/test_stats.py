import torch
from torch_frame import stype

from redelex.data import TensorStatType, make_tensor_stats_dict


def test_make_tensor_stats_dict(all_stype_dataset, text_embedder):
    tf = all_stype_dataset.tensor_frame
    stats = make_tensor_stats_dict(
        all_stype_dataset.col_stats, tf.col_names_dict, text_embedder
    )

    # Numerical: one entry per stat, one value per column.
    num_cols = len(tf.col_names_dict[stype.numerical])
    for stat in [
        TensorStatType.MEAN,
        TensorStatType.STD,
        TensorStatType.MIN,
        TensorStatType.MAX,
        TensorStatType.MEDIAN,
        TensorStatType.Q1,
        TensorStatType.Q3,
    ]:
        assert stats[stype.numerical][stat].shape == (num_cols,)
        assert torch.isfinite(stats[stype.numerical][stat]).all()

    assert (
        stats[stype.numerical][TensorStatType.MIN]
        <= stats[stype.numerical][TensorStatType.MAX]
    ).all()

    # Categorical: cardinality + padded value embeddings.
    for st in [stype.categorical, stype.multicategorical]:
        n_cols = len(tf.col_names_dict[st])
        card = stats[st][TensorStatType.CARDINALITY]
        emb = stats[st][TensorStatType.VALUE_EMBEDDINGS]
        assert card.shape == (n_cols,)
        assert (card >= 1).all()
        assert emb.shape == (n_cols, card.max().item(), text_embedder.embedding_dim)

    # Timestamp: stacked 7-component date tensors and year range.
    ts_cols = len(tf.col_names_dict[stype.timestamp])
    assert stats[stype.timestamp][TensorStatType.EARLIEST_DATE].shape == (ts_cols, 7)
    assert stats[stype.timestamp][TensorStatType.LATEST_DATE].shape == (ts_cols, 7)
    assert stats[stype.timestamp][TensorStatType.MIN_YEAR].shape == (ts_cols,)
    assert (
        stats[stype.timestamp][TensorStatType.MIN_YEAR]
        <= stats[stype.timestamp][TensorStatType.MAX_YEAR]
    ).all()


def test_make_tensor_stats_dict_skips_unsupported(num_cat_dataset, text_embedder):
    tf = num_cat_dataset.tensor_frame
    stats = make_tensor_stats_dict(
        num_cat_dataset.col_stats, tf.col_names_dict, text_embedder
    )
    assert set(stats) == {stype.numerical, stype.categorical}

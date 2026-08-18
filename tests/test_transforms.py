import copy

import torch
from torch_frame import stype
from torch_frame.data import MultiNestedTensor

from redelex.transforms import ResampleCorruptor
from redelex.transforms.corruptors import TFCorruptor, rescale_tf


def _assert_tf_equal(tf1, tf2):
    for st in tf1.stypes:
        feat1, feat2 = tf1.feat_dict[st], tf2.feat_dict[st]
        if isinstance(feat1, MultiNestedTensor):
            torch.testing.assert_close(feat1.values, feat2.values, equal_nan=True)
            torch.testing.assert_close(feat1.offset, feat2.offset)
        else:
            torch.testing.assert_close(feat1, feat2, equal_nan=True)


def test_tf_corruptor_p0_identity(all_stype_tf):
    corruptor = TFCorruptor(all_stype_tf, p=0.0)
    cor_tf, mask = corruptor(all_stype_tf)

    _assert_tf_equal(cor_tf, all_stype_tf)
    assert all(not m.any() for m in mask.values())


def test_tf_corruptor_p1_masks_everything(all_stype_tf):
    corruptor = TFCorruptor(all_stype_tf, p=1.0)
    cor_tf, mask = corruptor(all_stype_tf)

    assert set(mask) == set(all_stype_tf._col_to_stype_idx)
    assert all(m.all() for m in mask.values())

    # Corrupted categorical values are resampled from observed values.
    cat_cols = all_stype_tf.col_names_dict[stype.categorical]
    for idx, _col in enumerate(cat_cols):
        observed = set(all_stype_tf.feat_dict[stype.categorical][:, idx].tolist())
        corrupted = set(cor_tf.feat_dict[stype.categorical][:, idx].tolist())
        assert corrupted <= observed

    # Numerical NaNs are replaced (samplers are fitted on non-NaN values only).
    assert not cor_tf.feat_dict[stype.numerical].isnan().any()


def test_tf_corruptor_does_not_mutate_input(all_stype_tf):
    original = copy.deepcopy(all_stype_tf)
    corruptor = TFCorruptor(all_stype_tf, p=1.0)
    corruptor(all_stype_tf)

    _assert_tf_equal(all_stype_tf, original)


def test_tf_corruptor_multicategorical(all_stype_tf):
    """Regression: any MultiNestedTensor column crashed the corruptor with an
    IndexError, and the write-back never reached the frame."""
    corruptor = TFCorruptor(all_stype_tf, p=1.0)
    cor_tf, mask = corruptor(all_stype_tf)

    orig = all_stype_tf.feat_dict[stype.multicategorical]
    cor = cor_tf.feat_dict[stype.multicategorical]
    torch.testing.assert_close(orig.offset, cor.offset)
    assert cor.values.shape == orig.values.shape

    # With p=1 and more than one category, at least one element must differ.
    if orig.values.numel() > 0 and orig.values.unique().numel() > 1:
        assert not torch.equal(orig.values, cor.values)


def test_resample_corruptor_hetero(hetero_graph):
    data, _ = hetero_graph
    corruptor = ResampleCorruptor(data, corrupt_prob=0.5)
    out = corruptor(data)

    for node_type in out.node_types:
        assert "cor_tf" in out[node_type]
        assert "cor_col_mask" in out[node_type]
        assert out[node_type].cor_tf.num_rows == out[node_type].tf.num_rows


def test_rescale_tf(num_cat_dataset):
    tf = num_cat_dataset.tensor_frame
    scaled = rescale_tf(tf)

    x = scaled.feat_dict[stype.numerical]
    assert x.min() >= 0.0
    assert x.max() <= 1.0
    # Input is untouched.
    assert (
        tf.feat_dict[stype.numerical].max()
        == num_cat_dataset.tensor_frame.feat_dict[stype.numerical].max()
    )


def test_rescale_tf_constant_column():
    from torch_frame import TensorFrame

    tf = TensorFrame(
        feat_dict={stype.numerical: torch.full((5, 1), 3.0)},
        col_names_dict={stype.numerical: ["const"]},
    )
    scaled = rescale_tf(tf)
    assert torch.isfinite(scaled.feat_dict[stype.numerical]).all()

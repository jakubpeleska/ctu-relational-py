import copy
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
from torch_frame import TensorFrame, stype
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor, StatType
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class ResampleCorruptor(BaseTransform):
    def __init__(
        self,
        data: HeteroData,
        corrupt_prob: float = 0.5,
        distribution: Literal["empirical", "uniform"] = "uniform",
    ):
        self.corrupt_prob = corrupt_prob
        self.distribution = distribution
        self.corruptors = {
            tname: TFCorruptor(tf, p=corrupt_prob, distribution=distribution)
            for tname, tf in data.collect("tf").items()
        }

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Corrupt the data by resampling features with a given probability.
        Args:
            data (HeteroData): The input heterogeneous data.
        Returns:
            HeteroData: The corrupted data with additional 'cor_tf' and
                'cor_col_mask' attributes per node type.
        """
        return self.corrupt_data(data)

    def corrupt_data(self, data: HeteroData) -> HeteroData:
        for node_type, tf in data.collect("tf").items():
            cor_tf, mask = self.corruptors[node_type](tf)
            data[node_type]["cor_tf"] = cor_tf
            data[node_type]["cor_col_mask"] = mask

        return data


class TFCorruptor:
    """
    A class to resample and corrupt features of a TensorFrame.
    """

    def __init__(
        self,
        tf: TensorFrame,
        p: float = 0.5,
        distribution: Literal["empirical", "uniform"] = "empirical",
    ):
        """
        Initialize the TFCorruptor with a TensorFrame and corruption probability.
        Args:
            tf (TensorFrame): The TensorFrame to corrupt.
            p (float): The probability of corruption.
            distribution (Literal["empirical", "uniform"]): The type of distribution to use for resampling.
        """
        self.p = p
        self.col_samplers: Dict[str, Callable] = {}

        # Create samplers for all features
        for col in tf._col_to_stype_idx:
            x = self.get_tf_col(tf, col)
            if x.ndim == 1:
                x = x[~x.isnan()]
            self.col_samplers[col] = self.get_categorical_sampler(
                x,
                empirical=(distribution == "empirical"),
            )

    def __call__(self, tf: TensorFrame) -> Tuple[TensorFrame, Dict[str, torch.Tensor]]:
        return self.corrupt_tf(tf, self.col_samplers, p=self.p)

    @classmethod
    def get_tf_col(cls, tf: TensorFrame, col: str) -> torch.Tensor:
        """Return the values of a column as a plain tensor.

        Dense columns come out as [num_rows, ...], multi-embedding columns as
        [num_rows, dim] and multi-nested columns as their flat value stream.
        """
        x = tf.get_col_feat(col)
        if isinstance(x, MultiNestedTensor):
            return x.values
        if isinstance(x, MultiEmbeddingTensor):
            return x.values
        return x.squeeze(1)

    @classmethod
    def get_categorical_sampler(
        cls, x: torch.Tensor, empirical=True, max_values: int = 10000
    ) -> Callable[[torch.Size], torch.Tensor]:
        """
        Build a sampler over the observed values of a feature.
        Args:
            x (torch.Tensor): The feature values (rows are treated as values).
            empirical (bool): If True, sample from the empirical distribution;
                otherwise sample uniformly over the distinct values.
        Returns:
            Callable[[torch.Size], torch.Tensor]: A function mapping a sample
                shape to sampled values.
        """
        u_values: torch.Tensor
        u_values, counts = x.unique(sorted=True, return_counts=True, dim=0)

        top_idx = torch.argsort(counts, descending=True)
        u_values = u_values[top_idx[:max_values]]
        counts = counts[top_idx[:max_values]]

        if empirical:
            # Get the empirical marginal distribution
            marginal_prob = counts.float() / counts.sum()
        else:
            # Use a uniform distribution
            marginal_prob = torch.ones(len(u_values), dtype=torch.float) / len(u_values)
        distribution = torch.distributions.Categorical(marginal_prob)

        def sample(size: torch.Size) -> torch.Tensor:
            return u_values[distribution.sample(size)]

        return sample

    @staticmethod
    def _mnt_column_element_indices(feat: MultiNestedTensor, col_idx: int) -> torch.Tensor:
        """Indices into ``feat.values`` of all elements belonging to a column.

        ``feat.values`` is laid out row-major over (row, column) bags.
        """
        device = feat.offset.device
        bags = torch.arange(feat.num_rows, device=device) * feat.num_cols + col_idx
        starts = feat.offset[bags]
        lengths = feat.offset[bags + 1] - starts

        total = int(lengths.sum())
        block_starts = torch.repeat_interleave(starts, lengths)
        offsets_within = torch.arange(total, device=device) - torch.repeat_interleave(
            torch.cumsum(lengths, dim=0) - lengths, lengths
        )
        return block_starts + offsets_within

    @classmethod
    def corrupt_tf(
        cls,
        tf: TensorFrame,
        col_samplers: Dict[str, Callable],
        p: float = 0.5,
    ) -> Tuple[TensorFrame, Dict[str, torch.Tensor]]:
        """
        Corrupts the features of a TensorFrame with probability p by resampling
        from the fitted per-column distributions.
        Args:
            tf (TensorFrame): The TensorFrame to corrupt.
            col_samplers (Dict[str, Callable]): Per-column samplers.
            p (float): The probability of corruption.
        Returns:
            Tuple[TensorFrame, Dict[str, torch.Tensor]]: The corrupted
                TensorFrame and per-column masks of the corrupted positions
                (per element for multi-nested columns, per row otherwise).
        """

        _tf = copy.deepcopy(tf)

        mask: dict[str, torch.Tensor] = {}
        for col, sampler in col_samplers.items():
            s, idx = _tf._col_to_stype_idx[col]
            feat = _tf.feat_dict[s]

            if isinstance(feat, MultiNestedTensor):
                element_idx = cls._mnt_column_element_indices(feat, idx)
                col_mask = torch.rand(element_idx.numel(), device=feat.values.device) < p
                mask[col] = col_mask
                if col_mask.any():
                    samples = sampler((int(col_mask.sum()),))
                    feat.values[element_idx[col_mask]] = samples
            elif isinstance(feat, MultiEmbeddingTensor):
                col_mask = torch.rand(feat.num_rows, device=feat.values.device) < p
                mask[col] = col_mask
                if col_mask.any():
                    samples = sampler((int(col_mask.sum()),))
                    start, end = feat.offset[idx], feat.offset[idx + 1]
                    feat.values[col_mask, start:end] = samples
            else:
                col_mask = torch.rand(feat.size(0), device=feat.device) < p
                mask[col] = col_mask
                if col_mask.any():
                    samples = sampler((int(col_mask.sum()),))
                    feat[col_mask, idx] = samples

        return _tf, mask


def rescale_tf(tf: TensorFrame, stats: Optional[Dict[str, Dict[StatType, Any]]] = None):
    """
    Rescale the numerical features of a TensorFrame to the range [0, 1].

    Args:
        tf (TensorFrame): The TensorFrame to rescale.
        stats (Dict[str, Dict[StatType, Any]], optional): Column statistics to
            rescale with; computed from the data when not given.
    Returns:
        TensorFrame: The rescaled TensorFrame.
    """
    if stype.numerical not in tf.stypes or len(tf) == 0:
        return tf

    _tf = copy.deepcopy(tf)

    for col in _tf.col_names_dict[stype.numerical]:
        x: torch.Tensor = _tf.get_col_feat(col).squeeze(1)
        if stats is not None and col in stats:
            # Use the provided statistics to rescale
            x_min = stats[col][StatType.QUANTILES][0]
            x_max = stats[col][StatType.QUANTILES][-1]
        else:
            # Compute the min and max values of the feature
            x_min, x_max = x.nanquantile(
                q=torch.tensor([0.0, 1.0], device=x.device)
            ).tolist()
        # Re-scale the numerical feature to [0, 1]; a constant column maps to 0.
        x = (x - x_min) / (x_max - x_min) if x_max != x_min else torch.zeros_like(x)
        _tf.feat_dict[stype.numerical][:, _tf._col_to_stype_idx[col][1]] = x

    return _tf

import math
from typing import Dict, List

import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType


class EdgeContrastiveLoss(torch.nn.Module):
    r"""InfoNCE-style loss over the edges of a heterogeneous graph.

    For every edge type, the (dst, src) pairs linked by the batch's
    own edges are the positives, and the batch's remaining pairs are
    the negatives. The loss is computed in log space, so it is numerically
    stable for unnormalized embeddings.

    Args:
        channels: Embedding dimension.
        edge_types: Edge types to score. Reversed types share the transposed
            weight matrix of their forward counterpart.
        temperature: Softmax temperature.
        max_negatives: Number of negatives sampled per destination row, when the
            batch offers more than that.
    """

    def __init__(
        self,
        channels: int,
        edge_types: List[EdgeType],
        temperature: float = 0.1,
        max_negatives: int = 255,
    ):
        super().__init__()
        self.channels = channels

        self.weights_dict = torch.nn.ParameterDict(
            {
                f"{src_node}_{name}_{dst_node}": torch.nn.Parameter(
                    torch.nn.init.kaiming_uniform_(
                        torch.empty((channels, channels)), mode="fan_in", a=math.sqrt(5)
                    )
                )
                for (src_node, name, dst_node) in edge_types
                if not name.startswith("rev_")
            }
        )

        self.temp = temperature
        self.max_negatives = max_negatives

    def _sample_negatives(self, adj_M: torch.Tensor) -> torch.Tensor:
        r"""Pick the unlinked (dst, src) pairs to use as negatives.

        Args:
            adj_M: Boolean [num_dst, num_src] adjacency of the batch.

        Returns:
            A [num_negatives, 2] tensor of (dst row, src column) pairs, holding
            at most ``max_negatives`` entries per destination row.
        """
        num_dst, num_src = adj_M.shape
        if num_src <= self.max_negatives:
            return (~adj_M).nonzero()

        # Draw `max_negatives` source columns for every dst row without
        # replacement, then drop the ones that turn out to be linked. Sampling
        # per row keeps a row's negatives independent of how densely the other
        # rows happen to be connected, and matches the table and context losses.
        cols = torch.rand(num_dst, num_src, device=adj_M.device).argsort(dim=1)[
            :, : self.max_negatives
        ]
        rows = torch.arange(num_dst, device=adj_M.device).unsqueeze(1).expand_as(cols)
        keep = ~adj_M[rows, cols]
        return torch.stack([rows[keep], cols[keep]], dim=1)

    def forward(
        self, data: HeteroData, x_dict: Dict[NodeType, torch.Tensor]
    ) -> torch.Tensor:
        edge_index_dict = data.collect("edge_index")

        loss = 0.0
        count = 0
        for edge_type, edge_index in edge_index_dict.items():
            src_node, name, dst_node = edge_type
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            src_x = x_dict[src_node]
            dst_x = x_dict[dst_node]

            if src_x.size(0) <= 1 or dst_x.size(0) <= 1:
                continue

            total_src = src_x.size(0)
            total_dst = dst_x.size(0)

            if name.startswith("rev_"):
                W = self.weights_dict[
                    f"{dst_node}_{name.removeprefix('rev_')}_{src_node}"
                ].T
            else:
                W = self.weights_dict[f"{src_node}_{name}_{dst_node}"]

            logits = (dst_x @ W @ src_x.T) / self.temp
            device = logits.device

            adj_M = torch.zeros((total_dst, total_src), dtype=torch.bool, device=device)
            adj_M[dst_idx, src_idx] = True
            pos_idx = adj_M.nonzero()

            if pos_idx.size(0) <= 1:
                continue

            neg_idx = self._sample_negatives(adj_M)
            if neg_idx.size(0) == 0:
                continue

            neg_rows = neg_idx[:, 0]
            neg_logits = logits[neg_rows, neg_idx[:, 1]]

            num_negatives = torch.zeros(total_dst, dtype=torch.long, device=device)
            num_negatives.scatter_add_(0, neg_rows, torch.ones_like(neg_rows))

            # Per-dst-row logsumexp of the negative logits (empty rows -> -inf).
            # The accumulators follow the logits' dtype, which is reduced
            # precision under `torch.autocast`; scatter requires an exact match.
            row_max = torch.full(
                (total_dst,), float("-inf"), device=device, dtype=logits.dtype
            )
            row_max.scatter_reduce_(0, neg_rows, neg_logits, reduce="amax")
            sum_exp = torch.zeros(total_dst, device=device, dtype=logits.dtype)
            sum_exp.scatter_add_(0, neg_rows, torch.exp(neg_logits - row_max[neg_rows]))
            neg_logsumexp = row_max + torch.log(sum_exp)

            pos_rows = pos_idx[:, 0]
            pos_logits = logits[pos_rows, pos_idx[:, 1]]

            pair_loss = -(pos_logits - torch.logaddexp(pos_logits, neg_logsumexp[pos_rows]))

            # A dst row keeps no negatives only when every column sampled for it was linked.
            valid = num_negatives[pos_rows] > 0
            if not valid.any():
                continue

            norm_factor = torch.log(num_negatives[pos_rows][valid].float() + 1)

            loss += (pair_loss[valid] / norm_factor).sum()
            count += int(valid.sum())

        if count > 0:
            return loss / count

        device = next(iter(x_dict.values())).device if x_dict else None
        return torch.zeros((), device=device, requires_grad=True)

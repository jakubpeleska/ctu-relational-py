import math
from typing import Dict, List

import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType


def _empty_loss(x_dict: Dict[NodeType, torch.Tensor]) -> torch.Tensor:
    device = next(iter(x_dict.values())).device if x_dict else None
    return torch.zeros((), device=device, requires_grad=True)


class EdgeContrastiveLoss(torch.nn.Module):
    r"""InfoNCE-style loss over the edges of a heterogeneous graph.

    For every edge type, linked (dst, src) pairs are positives and all
    unlinked pairs are negatives. The loss is computed in log space, so it is
    numerically stable for unnormalized embeddings.
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
            neg_idx = (~adj_M).nonzero()

            if pos_idx.size(0) <= 1 or neg_idx.size(0) == 0:
                continue

            # Subsample negatives globally to bound the computation.
            max_total_negatives = self.max_negatives * total_dst
            if neg_idx.size(0) > max_total_negatives:
                perm = torch.randperm(neg_idx.size(0), device=device)[:max_total_negatives]
                neg_idx = neg_idx[perm]

            neg_rows = neg_idx[:, 0]
            neg_logits = logits[neg_rows, neg_idx[:, 1]]

            num_negatives = torch.zeros(total_dst, dtype=torch.long, device=device)
            num_negatives.scatter_add_(0, neg_rows, torch.ones_like(neg_rows))

            # Per-dst-row logsumexp of the negative logits (empty rows -> -inf).
            row_max = torch.full((total_dst,), float("-inf"), device=device)
            row_max.scatter_reduce_(0, neg_rows, neg_logits, reduce="amax")
            sum_exp = torch.zeros(total_dst, device=device)
            sum_exp.scatter_add_(0, neg_rows, torch.exp(neg_logits - row_max[neg_rows]))
            neg_logsumexp = row_max + torch.log(sum_exp)

            pos_rows = pos_idx[:, 0]
            pos_logits = logits[pos_rows, pos_idx[:, 1]]

            pair_loss = -(pos_logits - torch.logaddexp(pos_logits, neg_logsumexp[pos_rows]))

            # Pairs whose dst row lost all its negatives in the subsample carry
            # no signal (and their norm factor would be zero).
            valid = num_negatives[pos_rows] > 0
            if not valid.any():
                continue

            norm_factor = torch.log(num_negatives[pos_rows][valid].float() + 1)

            loss += (pair_loss[valid] / norm_factor).sum()
            count += int(valid.sum())

        return loss / count if count > 0 else _empty_loss(x_dict)

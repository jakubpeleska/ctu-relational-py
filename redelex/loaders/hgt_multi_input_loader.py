from typing import Callable, List, Optional, Tuple, Union

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from torch_geometric.typing import NodeType


class HGTMultiInputLoader:
    r"""Round-robins one :class:`HGTLoader` per input node type.

    Iteration yields ``(node_type, batch)`` pairs in a shuffled order that is
    proportional to the loaders' lengths.
    """

    def __init__(
        self,
        data: HeteroData,
        num_samples: List[int],
        input_nodes: Union[List[NodeType], List[Tuple[NodeType, torch.Tensor]]],
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        self.data = data
        self.input_nodes = (
            [node[0] for node in input_nodes]
            if isinstance(input_nodes[0], tuple)
            else input_nodes
        )
        self.loaders = {
            node_name: HGTLoader(
                data,
                num_samples=num_samples,
                input_nodes=input_node,
                transform=transform,
                **kwargs,
            )
            for node_name, input_node in zip(self.input_nodes, input_nodes, strict=True)
        }
        self.loaders_len = [len(loader) for loader in self.loaders.values()]
        self.total_len = sum(self.loaders_len)

    def __iter__(self):
        self.idx = 0
        loader_indices = torch.arange(len(self.loaders_len))
        _rnd_cat = torch.repeat_interleave(
            loader_indices, repeats=torch.tensor(self.loaders_len)
        )
        _rnd_idx = torch.randperm(_rnd_cat.shape[0])
        self.rnd_loader_idx = _rnd_cat[_rnd_idx].long().tolist()
        self.loader_iter = [iter(self.loaders[e]) for e in self.input_nodes]
        return self

    def __next__(self) -> Tuple[NodeType, HeteroData]:
        if self.idx >= len(self):
            raise StopIteration
        _loader_idx = self.rnd_loader_idx[self.idx]
        self.idx += 1
        return self.input_nodes[_loader_idx], next(self.loader_iter[_loader_idx])

    def __len__(self):
        return self.total_len

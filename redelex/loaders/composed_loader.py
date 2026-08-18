from typing import Literal, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NodeLoader


class ComposedLoader:
    def __init__(
        self,
        loaders: dict[str, NodeLoader],
        mode: Literal["minimum", "full", "rnd_uni", "rnd_weighted"] = "minimum",
        weights: Optional[list[float]] = None,
    ):
        self.loaders = loaders
        self.names = list(self.loaders.keys())
        self.loaders_len = [len(self.loaders[tn]) for tn in self.names]
        self.mode = mode

        if self.mode in ["minimum", "rnd_uni", "rnd_weighted"]:
            self.total_len = min(self.loaders_len) * len(self.loaders)
        elif self.mode == "full":
            self.total_len = sum(self.loaders_len)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.mode == "rnd_weighted":
            if weights is None:
                raise ValueError("Weights must be provided for rnd_weighted mode")
            if len(weights) != len(self.loaders):
                raise ValueError("Weights length must match number of loaders")
            self.weights = torch.as_tensor(weights, dtype=torch.float)

    def __iter__(self):
        self.idx = 0
        if self.mode == "minimum":
            N = min(self.loaders_len)
            L = len(self.loaders)
            self.rnd_loader_idx = (
                torch.stack([torch.randperm(L) for _ in range(N)]).flatten().long().tolist()
            )
        elif self.mode in ["full", "rnd_uni"]:
            loader_indices = torch.arange(len(self.loaders_len))
            rnd_cat = torch.repeat_interleave(
                loader_indices, repeats=torch.tensor(self.loaders_len)
            )
            rnd_idx = torch.randperm(rnd_cat.shape[0])
            self.rnd_loader_idx = rnd_cat[rnd_idx].long().tolist()
        elif self.mode == "rnd_weighted":
            # Weighted sampling over per-batch slots without replacement, so no
            # loader can be drawn more often than its actual length.
            slot_loader = torch.repeat_interleave(
                torch.arange(len(self.loaders_len)), torch.tensor(self.loaders_len)
            )
            slot_weights = self.weights[slot_loader]
            picks = torch.multinomial(
                slot_weights, num_samples=self.total_len, replacement=False
            )
            self.rnd_loader_idx = slot_loader[picks].long().tolist()
        self.loader_iter = [iter(self.loaders[tn]) for tn in self.names]
        return self

    def __next__(self) -> HeteroData:
        if self.idx >= len(self):
            raise StopIteration
        _loader_idx = self.rnd_loader_idx[self.idx]
        self.idx += 1
        return next(self.loader_iter[_loader_idx])

    def __len__(self):
        return self.total_len

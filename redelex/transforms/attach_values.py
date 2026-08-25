from typing import Any, Union

from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class AttachValuesTransform(BaseTransform):
    def __init__(
        self,
        attach_data: Union[tuple[str, Any], list[tuple[str, Any]]],
    ):
        super().__init__()
        self.attach_data = attach_data
        if not isinstance(attach_data, list):
            self.attach_data = [attach_data]
        for t in self.attach_data:
            if not (isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str)):
                raise TypeError(
                    f"attach_data must be (name, value) tuples or a list of them, got {t!r}"
                )

    def forward(self, batch: HeteroData) -> HeteroData:
        for name, data in self.attach_data:
            batch[name] = data

        return batch

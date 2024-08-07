from typing import Optional

import pandas as pd

from .ctu_dataset import CTUDataset


class TPCD(CTUDataset):
    val_timestamp = pd.Timestamp(year=1996, month=8, day=2)
    test_timestamp = pd.Timestamp(year=1997, month=8, day=2)

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__("tpcd", cache_dir=cache_dir)

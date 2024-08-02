from typing import get_args

from relbench.datasets import register_dataset

from .db_dataset import *
from .ctu_dataset import *

for db in get_args(CTUDatabaseName):
    register_dataset(f"ctu-{db}", CTUDataset, database=db)

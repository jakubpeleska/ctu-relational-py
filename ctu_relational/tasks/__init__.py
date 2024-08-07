from typing import get_args

from relbench.tasks import register_task

from .ctu_task import CTUDatasetEntityTask, CTUDatasetRecommendationTask

for db in get_args(CTUDatabaseName):
    register_dataset(f"ctu-{db}", CTUDataset, database=db)

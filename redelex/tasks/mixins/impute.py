from typing import Callable, Optional

import pandas as pd
from relbench.base import Database, Table, TaskType

from .db_modify import ModifyDBTaskMixin
from .entity import EntityTaskMixin


class ImputeEntityTaskMixin(ModifyDBTaskMixin, EntityTaskMixin):
    r"""Mixin class for allowing to modify underlying database for a task.

    Classification targets are label-encoded over the sorted unique values of
    the target column. Rows with a missing target value are encoded with the
    sentinel label ``-1`` and are not dropped; downstream consumers must mask
    or filter them. The sentinel is not a class, so it is not counted by
    :attr:`num_classes`.

    Attributes:
        removed_entity_cols: list of entity columns to be removed from the
            entity table.
        num_classes: Number of target classes of a classification task.
        Other attributes are inherited from ModifyDBTaskMixin and EntityTaskMixin.
    """

    removed_entity_cols: list[str] = []
    entity_col: str
    entity_table: str
    target_col: str
    task_type: TaskType

    _target_mapping: Callable[[pd.Series], pd.Series] = None
    _target_dtype: type = None
    _target_values: Optional[pd.Index] = None

    @property
    def num_classes(self) -> Optional[int]:
        r"""Number of target classes, or None for non-classification tasks.

        The classes are the sorted unique non-missing values of the target
        column, i.e. the same encoding the task tables use.
        """
        if self.task_type not in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ]:
            return None

        if self._target_values is None:
            # Use the same database as `_get_table` does by default, so the
            # classes match the ones the task tables are built with.
            db = self.dataset.get_db(upto_test_timestamp=False)
            self._init_target_mapping(db.table_dict[self.entity_table].df)

        return len(self._target_values)

    def _init_target_mapping(self, df: pd.DataFrame) -> None:
        if self.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ]:
            _, target_values = df[self.target_col].factorize(
                sort=True, use_na_sentinel=True
            )

            if self.task_type == TaskType.BINARY_CLASSIFICATION and len(target_values) != 2:
                raise ValueError(
                    f"Binary classification target '{self.target_col}' must have "
                    f"exactly 2 categories, found {len(target_values)}: "
                    f"{list(target_values[:10])}"
                )

            def target_map(x):
                if pd.isna(x):
                    return -1
                else:
                    return target_values.get_loc(x)

            self._target_mapping = target_map
            self._target_values = target_values
        else:
            self._target_mapping = lambda x: x
            self._target_values = None

        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]:
            self._target_dtype = float
        elif self.task_type in [TaskType.MULTICLASS_CLASSIFICATION]:
            self._target_dtype = int
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _make_modified_db(self, db: Database) -> Database:
        r"""
        Modify the database for the task.
        Args:
            db: The database to make modifications on.
        Returns:
            A modified database.
        """

        remove_cols = set([self.target_col, *self.removed_entity_cols])
        # Tolerate repeated invocation on the same database object.
        remove_cols &= set(db.table_dict[self.entity_table].df.columns)

        db.table_dict[self.entity_table].df.drop(columns=remove_cols, inplace=True)

        return db

    def filter_dangling_entities(self, table: Table) -> Table:
        return table


__all__ = ["ImputeEntityTaskMixin"]

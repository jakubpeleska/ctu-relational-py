import pandas as pd

from relbench.base import EntityTask, Table

class ContinuousWrapper:
    def __init__(self, task: EntityTask):
        self.task = task

        self.train_table = self.task.get_table("train", mask_input_cols=False)
        self.val_table = self.task.get_table("val", mask_input_cols=False)
        self.test_table = self.task.get_table("test", mask_input_cols=False)

        df = (
            pd.concat([self.train_table.df, self.val_table.df, self.test_table.df], ignore_index=True)
            .sort_values(self.train_table.time_col)
            .reset_index(drop=True)
        )

        self.full_table = Table(
            df=df,
            fkey_col_to_pkey_table=self.train_table.fkey_col_to_pkey_table,
            pkey_col=self.train_table.pkey_col,
            time_col=self.train_table.time_col,
        )

    def get_table(self, start: pd.Timestamp, end: pd.Timestamp = None):
        if end is None:
            end = self.full_table.max_timestamp + pd.Timedelta(days=1)
        mask = (self.full_table.df[self.full_table.time_col] >= start) & (
            self.full_table.df[self.full_table.time_col] < end
        )
        return Table(
            df=self.full_table.df[mask].reset_index(drop=True),
            fkey_col_to_pkey_table=self.full_table.fkey_col_to_pkey_table,
            pkey_col=self.full_table.pkey_col,
            time_col=self.full_table.time_col,
        )

    def get_splits(
        self,
        val_delta: pd.Timedelta = None,
        align: str = "data",
        min_rows_frac: float = 0.1,
    ):
        r"""Episode boundaries for the incremental protocol.

        Args:
            val_delta: Episode width. Defaults to the dataset's own validation
                window (``test_timestamp - val_timestamp``), which makes the
                increment size an accident of how the benchmark happened to split
                the data rather than a controlled variable. Cannot usefully go
                below the task's ``timedelta``.
            align: ``"data"`` walks the task's own observed timestamps, which is
                what every published run did. ``"calendar"`` steps back from
                ``val_timestamp`` in fixed ``val_delta`` strides regardless of
                where rows happen to fall, so **two tasks on the same database
                given the same width get the same interior boundaries** and their
                episodes can be compared row for row.
            min_rows_frac: Drop an episode holding less than this fraction of the
                validation window's row count. Set to 0 to keep every boundary --
                necessary if alignment must hold exactly, since the filter is
                per-task and can otherwise remove different boundaries for
                different tasks.

        Returns:
            Increasing list of boundaries. ``splits[0]`` is the first data
            timestamp, ``splits[-2]`` is ``val_timestamp`` and ``splits[-1]`` is
            ``test_timestamp``; episode ``i`` covers ``[splits[i], splits[i+1])``.
        """
        if val_delta is None:
            val_delta = (
                self.task.dataset.test_timestamp - self.task.dataset.val_timestamp
            )
        if align not in ("data", "calendar"):
            raise ValueError(f"`align` must be 'data' or 'calendar', got {align!r}")

        timestamps = self.full_table.df[self.full_table.time_col].unique()
        first_timestamp = timestamps[0]

        splits = [self.task.dataset.test_timestamp, self.task.dataset.val_timestamp]

        if align == "calendar":
            # A fixed stride from val_timestamp. Independent of where rows land, so
            # the grid is a property of the database and the chosen width, not of
            # the task -- which is what lets tasks be aligned to each other.
            boundary = self.task.dataset.val_timestamp - val_delta
            while boundary > first_timestamp:
                splits.append(boundary)
                boundary = boundary - val_delta
        else:
            previous_timestamp = self.task.dataset.val_timestamp
            for timestamp in reversed(timestamps):
                if timestamp + val_delta <= previous_timestamp:
                    splits.append(timestamp)
                    previous_timestamp = timestamp

        splits.append(first_timestamp)
        splits.reverse()

        if min_rows_frac <= 0:
            return splits

        min_split_len = int(
            min_rows_frac
            * len(
                self.get_table(
                    start=self.task.dataset.val_timestamp,
                    end=self.task.dataset.test_timestamp,
                )
            )
        )
        filtered_splits = [splits[0]]
        for split in splits[1:]:
            split_len = len(self.get_table(start=filtered_splits[-1], end=split))
            if split_len >= min_split_len:
                filtered_splits.append(split)

        return filtered_splits

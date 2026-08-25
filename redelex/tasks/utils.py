from relbench.base import BaseTask as RelBenchTask

from redelex.tasks.mixins import TemporalTaskMixin


def is_temporal_task(task: object) -> bool:
    """Check if the given task is a temporal task.

    RelBench tasks are temporal by design, so any relbench ``BaseTask``
    instance counts as temporal.
    """
    return isinstance(task, (TemporalTaskMixin, RelBenchTask))

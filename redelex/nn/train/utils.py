from typing import Optional

import torch
from relbench.base import TaskType
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


def get_metrics(
    task_type: TaskType, num_classes: Optional[int] = None, **metrics_kwargs
) -> tuple[dict[str, Metric], str, bool]:
    """Return (metrics, tune_metric_name, higher_is_better) for a task type.

    Args:
        task_type: The relbench task type.
        num_classes: Number of classes; required for multiclass classification.
        **metrics_kwargs: Extra keyword arguments passed to every metric.
    """
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return (
            {
                "accuracy": BinaryAccuracy(**metrics_kwargs),
                "precision": BinaryPrecision(**metrics_kwargs),
                "f1": BinaryF1Score(**metrics_kwargs),
                "roc_auc": BinaryAUROC(**metrics_kwargs),
            },
            "roc_auc",
            True,
        )

    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if num_classes is None:
            raise ValueError(
                "num_classes is required for multiclass classification metrics"
            )
        return (
            {
                "macro_accuracy": MulticlassAccuracy(
                    num_classes=num_classes, average="macro", **metrics_kwargs
                ),
                "micro_accuracy": MulticlassAccuracy(
                    num_classes=num_classes, average="micro", **metrics_kwargs
                ),
                "macro_f1": MulticlassF1Score(
                    num_classes=num_classes, average="macro", **metrics_kwargs
                ),
                "micro_f1": MulticlassF1Score(
                    num_classes=num_classes, average="micro", **metrics_kwargs
                ),
                # MulticlassAUROC does not support average="micro".
                "macro_roc_auc": MulticlassAUROC(
                    num_classes=num_classes, average="macro", **metrics_kwargs
                ),
                "weighted_roc_auc": MulticlassAUROC(
                    num_classes=num_classes, average="weighted", **metrics_kwargs
                ),
            },
            "macro_roc_auc",
            True,
        )

    elif task_type == TaskType.REGRESSION:
        return (
            {
                "mae": MeanAbsoluteError(**metrics_kwargs),
                "mse": MeanSquaredError(**metrics_kwargs),
                "r2": R2Score(**metrics_kwargs),
            },
            "mae",
            False,
        )
    else:
        raise ValueError(f"Task type {task_type} is unsupported")


def get_loss(task_type: TaskType, **loss_kwargs) -> torch.nn.Module:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return torch.nn.BCEWithLogitsLoss(**loss_kwargs)

    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return torch.nn.CrossEntropyLoss(**loss_kwargs)

    elif task_type == TaskType.REGRESSION:
        return torch.nn.L1Loss(**loss_kwargs)

    else:
        raise ValueError(f"Task type {task_type} is unsupported")

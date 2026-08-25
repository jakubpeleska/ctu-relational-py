import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from .entity_wrapper import LightningEntityTaskWrapper


class SaveModelCallback(L.Callback):
    r"""Saves the wrapped model's state dict on tune-metric improvement.

    Args:
        save_dir: Directory to save checkpoints to.
        monitor: The logged metric to monitor. If None, ``val_<tune_metric>``
            of the :class:`LightningEntityTaskWrapper` is used together with
            its improvement direction.
        mode: "min" or "max". Only used with an explicit ``monitor``
            (defaults to "min").
        save_every_epoch: If True, additionally save a checkpoint every epoch.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: Optional[str] = None,
        mode: str = "min",
        save_every_epoch: bool = False,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_every_epoch = save_every_epoch
        self.best_score: Optional[float] = None
        self._warned_missing_monitor = False
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def _resolve_monitor(self, pl_module: LightningEntityTaskWrapper) -> tuple[str, str]:
        if self.monitor is not None:
            return self.monitor, self.mode
        monitor = f"val_{pl_module.tune_metric}"
        mode = "max" if pl_module.higher_is_better else "min"
        return monitor, mode

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: LightningEntityTaskWrapper
    ):
        if trainer.sanity_checking:
            return

        monitor, mode = self._resolve_monitor(pl_module)

        current_score = trainer.callback_metrics.get(monitor)
        if current_score is None:
            if not self._warned_missing_monitor:
                warnings.warn(
                    f"SaveModelCallback: monitored metric '{monitor}' has not "
                    "been logged; no checkpoint is saved.",
                    stacklevel=2,
                )
                self._warned_missing_monitor = True
            return

        current_score = (
            current_score.item()
            if isinstance(current_score, torch.Tensor)
            else current_score
        )

        if self.save_every_epoch:
            torch.save(
                pl_module.model.state_dict(),
                f"{self.save_dir}/epoch_{trainer.current_epoch}_{monitor}_{current_score:.3f}.pt",
            )

        if (
            self.best_score is None
            or (mode == "min" and current_score < self.best_score)
            or (mode == "max" and current_score > self.best_score)
        ):
            self.best_score = current_score
            torch.save(pl_module.model.state_dict(), f"{self.save_dir}/best_model.pt")


__all__ = ["SaveModelCallback"]

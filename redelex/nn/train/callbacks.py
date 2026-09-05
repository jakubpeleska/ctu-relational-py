import time
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

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: LightningEntityTaskWrapper
    ):
        # `on_validation_end`, not `on_validation_epoch_end`. Lightning calls the
        # callback hook BEFORE the LightningModule's, so at epoch-end the score in
        # `trainer.callback_metrics` is still validation k-1's while the weights in
        # hand are validation k's: the saved checkpoint was one validation stale,
        # and the final validation's score was never seen at all. Measured on the
        # production grid, the checkpoint saved was the pass immediately after the
        # best one in 121 of 132 runs.
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


class PhaseTimerCallback(L.Callback):
    r"""Accumulates wall-clock time spent training versus validating.

    Training cost here is bounded by a fixed step budget, but validation cost is
    not: it scales with the size of the validation window and with how often
    validation runs. On large datasets that can dominate a run, so the split is
    worth measuring rather than assuming.

    Logs ``time_train_s``, ``time_val_s``, ``n_val_passes`` and
    ``val_time_fraction`` at the end of training.
    """

    def __init__(self):
        super().__init__()
        self.train_seconds = 0.0
        self.val_seconds = 0.0
        self.n_val_passes = 0
        self.n_train_batches = 0
        self.n_val_batches = 0
        self._train_start: Optional[float] = None
        self._val_start: Optional[float] = None

    # -- training ---------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._train_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._train_start is not None:
            self.train_seconds += time.perf_counter() - self._train_start
            self._train_start = None
        self.n_train_batches += 1

    # -- validation -------------------------------------------------------
    def on_validation_start(self, trainer, pl_module):
        self._val_start = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if self._val_start is not None:
            self.val_seconds += time.perf_counter() - self._val_start
            self._val_start = None
        if not trainer.sanity_checking:
            self.n_val_passes += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.n_val_batches += 1

    # -- summary ----------------------------------------------------------
    def summary(self) -> dict:
        total = self.train_seconds + self.val_seconds
        return {
            "time_train_s": round(self.train_seconds, 2),
            "time_val_s": round(self.val_seconds, 2),
            "n_val_passes": self.n_val_passes,
            "n_train_batches": self.n_train_batches,
            "n_val_batches": self.n_val_batches,
            "val_time_fraction": round(self.val_seconds / total, 4) if total > 0 else 0.0,
        }

    def on_fit_end(self, trainer, pl_module):
        summary = self.summary()
        print(f"[PhaseTimer] {summary}", flush=True)
        if trainer.logger is not None:
            try:
                trainer.logger.log_metrics(summary)
            except Exception as exc:  # logging must never fail a run
                warnings.warn(f"PhaseTimerCallback could not log metrics: {exc}")


__all__ = ["SaveModelCallback", "PhaseTimerCallback"]

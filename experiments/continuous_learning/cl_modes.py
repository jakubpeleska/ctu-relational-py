r"""Continual-learning modes for the incremental RDL experiment.

The submitted paper compared four regimes that were all variants of one question --
how much old data to mix in. This module replaces them with the standard CL
taxonomy, keeping the reference points every CL paper reports.

Modes
-----
Reference frame:

* ``from_scratch`` -- retrain from random init on all history. The control the
  paper is about: is incremental updating worth it versus retraining?
* ``joint`` -- warm start, train on all history. **Joint/Cumulative**, the upper
  bound a method with unlimited memory would reach.
* ``naive`` -- warm start, train only on the new increment. **Naive fine-tuning**,
  the lower bound and the canonical catastrophic-forgetting case.

The four CL families:

* ``er`` -- Experience Replay over a *bounded* reservoir buffer at a controlled
  new:old ratio.
* ``der_pp`` -- DER++: replay plus distillation against the logits recorded when
  each exemplar was stored.
* ``ewc`` -- online Elastic Weight Consolidation. Stores no data at all, which is
  what makes it viable where a retention policy forbids keeping raw rows.
* ``lwf`` -- Learning without Forgetting: distil from the previous episode's model
  on current data. No buffer.
* ``freeze_extend`` -- freeze everything learned so far and add a small adapter per
  episode. Zero forgetting by construction.

``ft_full`` and ``ft_newonly`` are accepted as aliases of ``joint`` and ``naive``:
those two regimes were never ad-hoc, they were the standard bounds under
non-standard names. ``ft_upsample`` is retained only to reproduce the submitted
paper -- its mixing ratio was never the claimed 50%, decaying to roughly 5% as
history grew, and ``er`` replaces it with a ratio that is actually honoured.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from relbench.base import Table
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from redelex.continual import (
    ParameterAnchor,
    ReservoirBuffer,
    logit_distillation_loss,
)

__all__ = [
    "ModeSpec",
    "MODES",
    "MODE_ALIASES",
    "DEFAULT_ROSTER",
    "resolve_mode",
    "AttachAuxTransform",
    "buffer_to_table",
    "CLState",
    "make_ewc_penalty",
    "make_lwf_penalty",
    "make_der_penalty",
]


@dataclass(frozen=True)
class ModeSpec:
    """What a learning mode needs, declared once instead of scattered in ifs."""

    name: str
    warm_start: bool
    train_window: str  # "full" | "increment"
    uses_buffer: bool = False
    uses_anchor: bool = False
    uses_teacher: bool = False
    freeze_backbone: bool = False
    distil_stored_logits: bool = False
    legacy: bool = False

    @property
    def needs_prev_timestamp(self) -> bool:
        """Modes training on the increment need to know where it starts."""
        return self.train_window == "increment"

    @property
    def needs_chain_state(self) -> bool:
        """Modes carrying more than weights across an episode boundary."""
        return self.uses_buffer or self.uses_anchor or self.freeze_backbone


MODES: Dict[str, ModeSpec] = {
    "from_scratch": ModeSpec("from_scratch", warm_start=False, train_window="full"),
    "joint": ModeSpec("joint", warm_start=True, train_window="full"),
    "naive": ModeSpec("naive", warm_start=True, train_window="increment"),
    "er": ModeSpec("er", warm_start=True, train_window="increment", uses_buffer=True),
    "der_pp": ModeSpec(
        "der_pp",
        warm_start=True,
        train_window="increment",
        uses_buffer=True,
        distil_stored_logits=True,
    ),
    "ewc": ModeSpec("ewc", warm_start=True, train_window="increment", uses_anchor=True),
    "lwf": ModeSpec("lwf", warm_start=True, train_window="increment", uses_teacher=True),
    "freeze_extend": ModeSpec(
        "freeze_extend", warm_start=True, train_window="increment", freeze_backbone=True
    ),
    # Reproduction only. Not part of the roster.
    "ft_upsample": ModeSpec(
        "ft_upsample", warm_start=True, train_window="increment", legacy=True
    ),
}

MODE_ALIASES: Dict[str, str] = {"ft_full": "joint", "ft_newonly": "naive"}

DEFAULT_ROSTER = (
    "from_scratch",
    "joint",
    "naive",
    "er",
    "der_pp",
    "ewc",
    "lwf",
    "freeze_extend",
)


def resolve_mode(name: str) -> ModeSpec:
    """Look up a mode, accepting the paper's original names as aliases.

    Raises:
        ValueError: if the name is neither a mode nor an alias.
    """
    canonical = MODE_ALIASES.get(name, name)
    if canonical not in MODES:
        raise ValueError(
            f"unknown learning mode {name!r}; "
            f"expected one of {sorted(MODES)} or an alias {sorted(MODE_ALIASES)}"
        )
    return MODES[canonical]


class AttachAuxTransform(BaseTransform):
    r"""Attach a per-example tensor to the batch, indexed like the target.

    Mirrors :class:`~redelex.transforms.AttachTargetTransform`: the same input node
    can appear with different timestamps, so per-example values cannot live on the
    graph and must be indexed by ``input_id`` after the batch is built. Used to
    carry the stored logits that DER++ distils against.
    """

    def __init__(self, entity: str, name: str, values: torch.Tensor):
        self.entity = entity
        self.name = name
        self.values = values

    def forward(self, batch: HeteroData) -> HeteroData:
        batch[self.entity][self.name] = self.values[batch[self.entity].input_id]
        return batch


def buffer_to_table(
    buffer: ReservoirBuffer,
    entity_col: str,
    time_col: str,
    target_col: str,
    template: Table,
) -> Table:
    """Rebuild a task table from buffered exemplars.

    The buffer stores entity id, timestamp and target per exemplar, which is
    exactly what ``get_table_input`` needs. Timestamps are held as unix seconds
    (see :func:`redelex.utils.datetime.to_unix_time`) and converted back here.

    Args:
        buffer: Populated replay buffer.
        entity_col: Task's entity column name.
        time_col: Task's time column name.
        target_col: Task's target column name.
        template: Any table from the same task, used for the foreign-key metadata.

    Returns:
        A ``Table`` of the buffered rows, time-ordered.
    """
    if len(buffer) == 0:
        raise ValueError("cannot build a table from an empty buffer")

    df = pd.DataFrame(
        {
            entity_col: buffer.node_ids.astype("int64"),
            time_col: pd.to_datetime(buffer.timestamps, unit="s"),
            target_col: buffer.targets,
        }
    ).sort_values(time_col).reset_index(drop=True)

    return Table(
        df=df,
        fkey_col_to_pkey_table=template.fkey_col_to_pkey_table,
        pkey_col=None,
        time_col=time_col,
    )


class CLState:
    """Everything a mode carries across an episode boundary besides weights.

    Weights already travel via ``best_model.pt``; the replay buffer, the EWC
    anchor and the adapter stack have to travel the same way or every method
    silently restarts each episode.
    """

    FILENAME = "cl_state.pt"

    def __init__(
        self,
        buffer: Optional[ReservoirBuffer] = None,
        anchor: Optional[ParameterAnchor] = None,
        adapters: Optional[Dict[str, Any]] = None,
    ):
        self.buffer = buffer
        self.anchor = anchor
        self.adapters = adapters

    def state_dict(self) -> dict:
        return {
            "buffer": self.buffer.state_dict() if self.buffer is not None else None,
            "anchor": self.anchor.state_dict() if self.anchor is not None else None,
            "adapters": self.adapters,
        }

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path) -> "CLState":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            buffer=(
                ReservoirBuffer.from_state_dict(raw["buffer"])
                if raw.get("buffer") is not None
                else None
            ),
            anchor=(
                ParameterAnchor.from_state_dict(raw["anchor"])
                if raw.get("anchor") is not None
                else None
            ),
            adapters=raw.get("adapters"),
        )


# --- penalty builders -------------------------------------------------------
#
# Each returns a callable with the signature the wrapper's `cl_penalty` hook
# expects: fn(pl_module, batch, pred, target) -> Tensor | None.


def make_ewc_penalty(anchor: ParameterAnchor) -> Callable:
    """Quadratic pull toward consolidated weights, weighted by Fisher importance."""

    def penalty(pl_module, batch, pred, target):
        return anchor.penalty(pl_module.model)

    return penalty


def make_lwf_penalty(
    teacher: torch.nn.Module, entity_table: str, weight: float = 1.0
) -> Callable:
    """Distil the previous episode's model on the *current* batch. No buffer."""

    def penalty(pl_module, batch, pred, target):
        with torch.no_grad():
            teacher_out = teacher(batch, entity_table)
        teacher_out = teacher_out.view(-1) if teacher_out.size(-1) == 1 else teacher_out
        teacher_out = teacher_out[: pred.size(0)]
        return logit_distillation_loss(pred, teacher_out, weight=weight)

    return penalty


def make_der_penalty(entity_table: str, alpha: float = 0.5) -> Callable:
    """Distil against logits recorded when each exemplar was stored.

    Only fires on replayed batches: batches drawn from the new increment carry no
    stored logit, so the term is skipped rather than being computed against a
    placeholder.
    """

    def penalty(pl_module, batch, pred, target):
        store = batch[entity_table]
        stored = getattr(store, "teacher_logit", None)
        if stored is None:
            return None
        stored = stored[: pred.size(0)]
        return logit_distillation_loss(pred, stored, weight=alpha)

    return penalty

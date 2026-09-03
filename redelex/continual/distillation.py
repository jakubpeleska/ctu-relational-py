r"""Distillation terms for continual learning.

Two methods in this benchmark distil, and they differ only in *where the teacher
signal comes from* -- the loss itself is shared:

* **LwF** (Learning without Forgetting) runs the previous episode's model on the
  *current* batch. It needs no memory buffer, but it does need a frozen copy of
  the previous model, which is already on disk between episodes.
* **DER++** distils against logits recorded when each exemplar was *stored*. It
  needs no teacher model at training time, because the buffer carries the
  targets with it.

Every task in this benchmark is binary or regression with a single output, so
squared error on raw logits is the natural distillation term for both -- the
softmax-and-temperature formulation only applies once there are class
probabilities to soften. :func:`soft_target_kl` is provided for that case.
"""

from typing import Optional

import copy

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["logit_distillation_loss", "soft_target_kl", "frozen_teacher"]


def logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    r"""Squared error between student and teacher logits.

    This is the DER/DER++ term, and the right form of LwF for single-output
    heads. The teacher signal is detached, so gradients never flow into a stored
    target or a frozen teacher.

    Args:
        student_logits: Current model outputs.
        teacher_logits: Stored logits, or a frozen teacher's outputs.
        weight: Scalar multiplier (DER++'s ``alpha``).

    Returns:
        Scalar loss, zero when either side is empty.
    """
    if student_logits.numel() == 0 or teacher_logits.numel() == 0:
        return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
    student = student_logits.reshape(student_logits.shape[0], -1).float()
    teacher = teacher_logits.reshape(teacher_logits.shape[0], -1).to(student).detach()
    if student.shape != teacher.shape:
        raise ValueError(
            f"student logits {tuple(student.shape)} and teacher logits "
            f"{tuple(teacher.shape)} must match"
        )
    return weight * F.mse_loss(student, teacher)


def soft_target_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    weight: float = 1.0,
) -> torch.Tensor:
    r"""Temperature-softened KL divergence, for multi-class heads.

    Scaled by ``temperature ** 2`` so the gradient magnitude stays comparable as
    the temperature changes (Hinton et al., 2015). Unused by the current task
    grid, which is entirely binary and regression, but correct when a multiclass
    task enters.
    """
    if temperature <= 0:
        raise ValueError(f"`temperature` must be positive, got {temperature}")
    if student_logits.numel() == 0 or teacher_logits.numel() == 0:
        return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits.float().detach() / temperature, dim=-1)
    kl = F.kl_div(student, teacher, reduction="batchmean")
    return weight * (temperature**2) * kl


def frozen_teacher(model: nn.Module) -> nn.Module:
    r"""A detached, eval-mode copy of ``model`` for use as an LwF teacher.

    The copy has ``requires_grad`` cleared throughout and is put in eval mode, so
    batch-norm statistics stay fixed and the teacher cannot drift while the
    student trains.
    """
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher

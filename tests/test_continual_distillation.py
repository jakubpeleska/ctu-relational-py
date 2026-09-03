import pytest
import torch
from torch import nn

from redelex.continual.distillation import (
    frozen_teacher,
    logit_distillation_loss,
    soft_target_kl,
)


# --- logit_distillation_loss ------------------------------------------------


def test_zero_when_student_matches_teacher():
    x = torch.randn(8, 1)
    assert logit_distillation_loss(x, x.clone()).item() == pytest.approx(0.0)


def test_grows_with_disagreement():
    student = torch.zeros(8, 1)
    near = logit_distillation_loss(student, torch.full((8, 1), 0.5))
    far = logit_distillation_loss(student, torch.full((8, 1), 2.0))
    assert 0 < near.item() < far.item()


def test_scales_linearly_with_weight():
    s, t = torch.zeros(4, 1), torch.ones(4, 1)
    base = logit_distillation_loss(s, t).item()
    assert logit_distillation_loss(s, t, weight=3.0).item() == pytest.approx(3 * base)


def test_teacher_signal_is_detached():
    # gradients must never flow into a stored logit or a frozen teacher
    student = torch.zeros(4, 1, requires_grad=True)
    teacher = torch.ones(4, 1, requires_grad=True)
    logit_distillation_loss(student, teacher).backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_gradient_pushes_student_toward_teacher():
    student = torch.zeros(4, 1, requires_grad=True)
    logit_distillation_loss(student, torch.ones(4, 1)).backward()
    # teacher is above the student, so descending the gradient raises the student
    assert torch.all(student.grad < 0)


def test_accepts_flat_and_column_shapes():
    flat = logit_distillation_loss(torch.zeros(4), torch.ones(4))
    column = logit_distillation_loss(torch.zeros(4, 1), torch.ones(4, 1))
    assert flat.item() == pytest.approx(column.item())


def test_empty_input_is_zero_not_nan():
    out = logit_distillation_loss(torch.zeros(0, 1), torch.zeros(0, 1))
    assert out.item() == 0.0 and torch.isfinite(out)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must match"):
        logit_distillation_loss(torch.zeros(4, 2), torch.zeros(4, 3))


# --- soft_target_kl ---------------------------------------------------------


def test_kl_zero_for_identical_distributions():
    logits = torch.randn(6, 5)
    assert soft_target_kl(logits, logits.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_positive_for_different_distributions():
    a = torch.tensor([[10.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 10.0]])
    assert soft_target_kl(a, b).item() > 0


def test_kl_temperature_scaling_keeps_it_finite():
    a, b = torch.randn(6, 4), torch.randn(6, 4)
    for temp in (0.5, 1.0, 2.0, 8.0):
        assert torch.isfinite(soft_target_kl(a, b, temperature=temp))


def test_kl_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        soft_target_kl(torch.randn(2, 3), torch.randn(2, 3), temperature=0.0)


# --- frozen_teacher ---------------------------------------------------------


def test_teacher_is_a_copy_not_an_alias():
    model = nn.Linear(3, 1)
    teacher = frozen_teacher(model)
    with torch.no_grad():
        model.weight += 1.0
    assert not torch.allclose(teacher.weight, model.weight)


def test_teacher_parameters_require_no_grad():
    teacher = frozen_teacher(nn.Linear(3, 1))
    assert all(not p.requires_grad for p in teacher.parameters())


def test_teacher_is_in_eval_mode():
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    model.train()
    assert not frozen_teacher(model).training


def test_teacher_does_not_drift_while_student_trains():
    model = nn.Linear(3, 1)
    teacher = frozen_teacher(model)
    before = teacher.weight.detach().clone()

    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(model(torch.randn(8, 3)), torch.randn(8, 1)).backward()
        opt.step()

    torch.testing.assert_close(teacher.weight, before)

from redelex.continual.distillation import (
    frozen_teacher,
    logit_distillation_loss,
    soft_target_kl,
)
from redelex.continual.regularization import ParameterAnchor, fisher_diagonal
from redelex.continual.replay import ReservoirBuffer, herding_select
from redelex.continual.metrics import (
    average_accuracy,
    backward_transfer,
    evaluation_matrix_from_predictions,
    exp_decay_avg,
    forward_transfer,
    per_episode_forgetting,
)

__all__ = [
    "ParameterAnchor",
    "frozen_teacher",
    "logit_distillation_loss",
    "soft_target_kl",
    "fisher_diagonal",
    "ReservoirBuffer",
    "herding_select",
    "average_accuracy",
    "backward_transfer",
    "evaluation_matrix_from_predictions",
    "exp_decay_avg",
    "forward_transfer",
    "per_episode_forgetting",
]

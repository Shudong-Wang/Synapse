"""Event classification using archived teacher probabilities."""

import torch
from torch.nn import functional as F


class DistilledEventCrossEntropy(torch.nn.Module):
    """Mix truth with teacher probabilities during training and score truth in evaluation."""

    def __init__(self, teacher_fraction: float):
        super().__init__()
        if not 0 <= teacher_fraction <= 1:
            raise ValueError("teacher_fraction must be between 0 and 1")
        self.teacher_fraction = teacher_fraction

    def forward(self, pred, target, weight, mean_train_weight, extra_event_info=None, **kwargs):
        logits = pred[0]
        if self.training and self.teacher_fraction > 0:
            expected_shape = (len(target), logits.shape[1] + 1)
            if extra_event_info is None or extra_event_info.shape != expected_shape:
                raise ValueError("Expected one teacher probability per class followed by a validity flag")
            truth = F.one_hot(target, num_classes=logits.shape[1]).to(logits.dtype)
            teacher = extra_event_info[:, :-1].detach().to(logits.dtype)
            valid = extra_event_info[:, -1:].bool()
            teacher = torch.where(valid, teacher, truth)
            target = torch.lerp(truth, teacher, self.teacher_fraction)

        per_event = F.cross_entropy(logits, target, reduction="none")
        loss = (weight * per_event).mean() / mean_train_weight
        return loss, logits.softmax(dim=1)

import unittest

import torch
from torch.nn import functional as F

from synapse.loss_functions.distillation import DistilledEventCrossEntropy


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[2., -1., 0.], [.5, 1., 2.], [1., 0., 4.]], requires_grad=True)
        self.labels = torch.tensor([0, 2, 1])
        self.weights = torch.tensor([2., 0.5, 4.])
        self.mean_train_weight = 1.75
        self.teacher = torch.tensor([[.2, .3, .5, 1.], [-1., -1., -1., 0.],
                                     [float("nan"), 0., 0., 0.]], requires_grad=True)

    def test_weighted_mixture_and_student_gradient(self):
        objective = DistilledEventCrossEntropy(0.75)
        loss, score = objective((self.logits, None, None), self.labels, self.weights,
                                self.mean_train_weight, self.teacher)
        expected_target = torch.tensor([[.4, .225, .375], [0., 0., 1.], [0., 1., 0.]])
        per_event = -(expected_target * self.logits.log_softmax(dim=1)).sum(dim=1)
        torch.testing.assert_close(loss, (self.weights * per_event).mean() / self.mean_train_weight)
        torch.testing.assert_close(score, self.logits.softmax(dim=1))
        loss.backward()
        scale = self.weights[:, None] / (len(self.labels) * self.mean_train_weight)
        torch.testing.assert_close(self.logits.grad, scale * (score.detach() - expected_target))
        self.assertIsNone(self.teacher.grad)

    def test_baseline_and_evaluation_use_truth(self):
        truth_loss = F.cross_entropy(self.logits, self.labels, reduction="none")
        expected = (self.weights * truth_loss).mean() / self.mean_train_weight
        for objective in (DistilledEventCrossEntropy(0), DistilledEventCrossEntropy(.75).eval()):
            for info in (None, self.teacher):
                loss, _ = objective((self.logits, None, None), self.labels, self.weights,
                                    self.mean_train_weight, info)
                torch.testing.assert_close(loss, expected)

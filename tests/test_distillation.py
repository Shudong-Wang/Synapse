import unittest

import torch
from torch.nn import functional as F

from synapse.loss_functions.distillation import (
    DistilledEventCrossEntropy, DistilledEventWithLeptonOrigins, TruthEventWithAuxiliaryHeads,
)


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

    def test_independent_truth_teacher_and_balanced_process_gradients(self):
        ids = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 7])
        labels = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 1])
        n = len(labels)
        absolute = torch.arange(1, n + 1, dtype=torch.float32)
        info = torch.stack((ids, absolute), dim=1)
        event_weights = absolute.flip(0)
        mean_weight = event_weights.mean()
        objective = TruthEventWithAuxiliaryHeads(.25, .1, 0, 0)
        objective.fit_process_class_weights(info, labels)
        process_weights = absolute * objective.process_class_weights[ids]
        torch.testing.assert_close(torch.bincount(ids, weights=process_weights), torch.full((8,), n/8))

        main = torch.zeros(n, 3, requires_grad=True)
        teacher_logits = torch.zeros_like(main, requires_grad=True)
        process_logits = torch.zeros(n, 8, requires_grad=True)
        pred = (main, {"teacher": teacher_logits, "process": process_logits}, torch.zeros(n, 4))
        objects = (None, torch.zeros(n, 3, 1), torch.ones(n, 1, 1))
        teacher = torch.tensor([[.8, .15, .05, 1.]]).repeat(n, 1)
        teacher[-1] = torch.tensor([float("nan"), -1., -1., 0.])
        loss, _ = objective(pred, labels, event_weights, mean_weight, teacher, objects, info)
        loss.backward()
        event_scale = event_weights[:, None] / (n * mean_weight)
        torch.testing.assert_close(main.grad, event_scale * (1/3 - F.one_hot(labels, 3)))
        expected_teacher = .25 * event_scale[:-1] * (1/3 - teacher[:-1, :3])
        torch.testing.assert_close(teacher_logits.grad[:-1], expected_teacher)
        torch.testing.assert_close(teacher_logits.grad[-1], torch.zeros(3))
        torch.testing.assert_close(process_logits.grad, .1 * process_weights[:, None] / n * (1/8 - F.one_hot(ids, 8)))
        val_loss, score = objective.eval()((main, None, None), labels, event_weights, mean_weight)
        torch.testing.assert_close(val_loss, torch.log(torch.tensor(3.)))
        torch.testing.assert_close(score, main.softmax(dim=1))
        with self.assertRaisesRegex(ValueError, "Missing training weight"):
            objective.fit_process_class_weights(info[:-1], labels[:-1])

    def test_lepton_origin_gradients_include_real_jets_and_met(self):
        # Slots: three leptons, three jets, MET. The second event has two padded jets.
        origin = torch.tensor([[1, 2, 3, 1, 2, -1, 0], [4, 5, 0, 3, 0, -1, 0]])
        valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 1]])
        lepton = torch.tensor([[1, 1, 1, 0, 0, 0, 0]]).expand(2, -1)
        conversion = torch.zeros_like(origin)
        conversion[1, 2] = 1
        metadata = (None, torch.stack((origin, lepton, conversion), dim=1), valid.unsqueeze(1))
        objects = torch.zeros((2, 7, 4), requires_grad=True)
        prediction = (self.logits[:2], None, objects.flatten(0, 1))
        weights = self.weights[:2]
        objective = DistilledEventWithLeptonOrigins(.75, origin_loss_weight=.3, conversion_loss_weight=.2)
        loss, _ = objective(prediction, self.labels[:2], weights, self.mean_train_weight, self.teacher[:2], metadata)
        loss.backward()

        targets = torch.zeros_like(objects)
        targets[0, 0, 0] = targets[0, 1, 1] = targets[0, 2, 2] = 1  # Higgs, top, Z
        targets[1, :2, 2] = 1  # W and diboson
        targets[1, 2, 3] = 1  # origin 0, conversion
        scale = torch.zeros_like(objects)
        scale[:, :, :3] = (.3 * valid / (3 * valid.sum(1, keepdim=True))).unsqueeze(-1)
        scale[:, :, 3] = .2 * lepton / lepton.sum(1, keepdim=True)
        scale *= weights[:, None, None] / (len(weights) * self.mean_train_weight)
        torch.testing.assert_close(objects.grad, (.5 - targets) * scale)

        # Best-epoch selection remains based on event truth and needs no auxiliary labels.
        val_loss, scores = objective.eval()((self.logits[:2], None, None), self.labels[:2], weights,
                                            self.mean_train_weight)
        truth_loss = F.cross_entropy(self.logits[:2], self.labels[:2], reduction="none")
        torch.testing.assert_close(val_loss, (weights * truth_loss).mean() / self.mean_train_weight)
        torch.testing.assert_close(scores, self.logits[:2].softmax(dim=1))

    def test_balanced_origin_heads_equalize_weight_after_object_and_event_reduction(self):
        origin = torch.tensor([[1, 2, 3, 1, 0], [3, 0, 2, -1, 0]])
        lepton = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        conversion = torch.zeros_like(origin)
        conversion[1, 0] = 1
        labels = torch.stack((origin, lepton, conversion), dim=1)
        mask = valid.unsqueeze(1)
        weights = self.weights[:2]
        plain = DistilledEventWithLeptonOrigins(0, .3, .2)
        balanced = DistilledEventWithLeptonOrigins(0, .3, .2, balance_origin_classes=True)
        balanced.fit_origin_class_weights(labels, mask, weights)
        # The jet with a top parent in event two is a negative for the lepton-top head.
        positive_weight = torch.tensor([2/5, 2/5, 2/5 + .5/3])
        negative, positive = balanced.origin_class_weights
        half = weights.sum().expand(3) / 2
        torch.testing.assert_close(positive * positive_weight, half)
        torch.testing.assert_close(negative * (weights.sum() - positive_weight), half)

        def loss_and_gradient(objective):
            logits = torch.zeros(2, 5, 4, requires_grad=True)
            loss, _ = objective((self.logits[:2], None, logits.flatten(0, 1)), self.labels[:2],
                                weights, self.mean_train_weight, self.teacher[:2], (None, labels, mask))
            return loss, torch.autograd.grad(loss, logits)[0]

        plain_loss, plain_grad = loss_and_gradient(plain)
        balanced_loss, grad = loss_and_gradient(balanced)
        torch.testing.assert_close(plain_loss, balanced_loss)  # All logits are zero.
        torch.testing.assert_close(grad[:, :, :3].sum(dim=(0, 1)), torch.zeros(3), atol=1e-8, rtol=0)
        torch.testing.assert_close(grad[:, :, 3], plain_grad[:, :, 3])
        self.assertTrue((grad[~valid] == 0).all())
        self.assertTrue((grad[1, 2, :3] > 0).all())
        restored = DistilledEventWithLeptonOrigins(0, .3, .2, balance_origin_classes=True)
        restored.load_state_dict(balanced.state_dict())
        torch.testing.assert_close(restored.origin_class_weights, balanced.origin_class_weights)
        with self.assertRaisesRegex(ValueError, "top"):
            balanced.fit_origin_class_weights(labels[:, :, [0, 2]], mask[:, :, [0, 2]], weights)

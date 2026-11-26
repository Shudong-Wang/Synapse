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


class DistilledEventWithLeptonOrigins(DistilledEventCrossEntropy):
    """Three all-object lepton-origin logits and one lepton-only conversion logit.

    The event truth loss alone is used for validation and checkpoint selection.
    """

    requires_aligned_objects = True

    def __init__(self, teacher_fraction: float, origin_loss_weight: float, conversion_loss_weight: float,
                 balance_origin_classes: bool = False):
        super().__init__(teacher_fraction)
        if not (origin_loss_weight >= 0 and conversion_loss_weight >= 0):
            raise ValueError("Auxiliary loss weights must be nonnegative")
        self.origin_loss_weight = origin_loss_weight
        self.conversion_loss_weight = conversion_loss_weight
        self.register_buffer("origin_class_weights", torch.ones(2, 3) if balance_origin_classes else None)

    @staticmethod
    def _targets(labels, valid):
        origin, lepton_mask, conversion = labels.unbind(dim=1)
        lepton = lepton_mask.bool() & valid
        return torch.stack((
            lepton & (origin == 1),                  # Higgs lepton
            lepton & (origin == 2),                  # top lepton
            lepton & (origin >= 3) & (origin <= 5),  # Z, W or diboson lepton
            lepton & conversion.bool(),
        ), dim=-1), lepton

    @torch.no_grad()
    def fit_origin_class_weights(self, object_labels, object_mask, event_weights):
        """Equalize positive and negative weight per origin head over the prepared training fold."""
        totals = torch.zeros(2, 3, dtype=torch.float64, device="cpu")
        for start in range(0, len(event_weights), 65536):
            chunk = slice(start, start + 65536)
            weight = torch.as_tensor(event_weights[chunk], dtype=torch.float64, device="cpu")
            if not torch.all(torch.isfinite(weight) & (weight >= 0)):
                raise ValueError("Balancing origin classes requires finite, nonnegative training weights")
            valid = torch.as_tensor(object_mask[chunk], device="cpu").squeeze(1).bool()
            targets, _ = self._targets(torch.as_tensor(object_labels[chunk], device="cpu"), valid)
            n_valid = valid.sum(dim=1, keepdim=True)
            positive = targets[:, :, :3].sum(dim=1)
            per_object = weight[:, None] / n_valid.clamp_min(1)
            totals[0] += ((n_valid - positive) * per_object).sum(dim=0)
            totals[1] += (positive * per_object).sum(dim=0)
        for name, counts in zip(("Higgs", "top", "W/Z/diboson"), totals.T):
            if not (counts > 0).all():
                raise ValueError(f"The {name} origin head needs positive and negative training weight to balance")
        # Target half of the original total weight for each class in each head.
        self.origin_class_weights.copy_(totals.sum(dim=0) / (2 * totals))

    def forward(self, pred, target, weight, mean_train_weight, extra_event_info=None, extra_obj_info=None):
        event_loss, scores = super().forward(pred, target, weight, mean_train_weight, extra_event_info)
        if not self.training:
            return event_loss, scores

        # object_label is loss-only. object_mask is the same validity mask used by attention.
        if extra_obj_info is None or extra_obj_info[1] is None:
            raise ValueError("Training requires object_label fields [partOrigin, lepton_truth_mask, isConv]")
        _, object_labels, object_mask = extra_obj_info
        valid = object_mask.squeeze(1).bool()
        if object_labels[:, 0].shape != valid.shape or pred[2].shape != (valid.numel(), 4):
            raise ValueError("Expected four object logits aligned with each slot in the validity mask")
        targets, lepton = self._targets(object_labels, valid)
        logits = pred[2].reshape_as(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")

        # Real jets and MET are negative examples for every origin task.
        origin_bce = bce[:, :, :3]
        if self.origin_class_weights is not None:
            negative, positive = self.origin_class_weights.unbind()
            origin_bce = origin_bce * torch.where(targets[:, :, :3], positive, negative)
        origins = origin_bce.mean(dim=-1).masked_fill(~valid, 0)
        per_event_origin = origins.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        conversions = bce[:, :, 3].masked_fill(~lepton, 0)
        per_event_conversion = conversions.sum(dim=1) / lepton.sum(dim=1).clamp_min(1)
        auxiliary = self.origin_loss_weight * per_event_origin + self.conversion_loss_weight * per_event_conversion
        return event_loss + (weight * auxiliary).mean() / mean_train_weight, scores


class TruthEventWithAuxiliaryHeads(DistilledEventWithLeptonOrigins):
    """Truth-supervised event scores with separate EveNet and generator-process heads."""

    def __init__(self, teacher_loss_weight: float, process_loss_weight: float, origin_loss_weight: float,
                 conversion_loss_weight: float, balance_origin_classes: bool = False, num_process_classes: int = 8):
        super().__init__(0.0, origin_loss_weight, conversion_loss_weight, balance_origin_classes)
        if not (teacher_loss_weight >= 0 and process_loss_weight >= 0 and num_process_classes > 1):
            raise ValueError("Invalid auxiliary event loss coefficients or process class count")
        self.teacher_loss_weight = teacher_loss_weight
        self.process_loss_weight = process_loss_weight
        self.register_buffer("process_class_weights", torch.ones(num_process_classes) if process_loss_weight else None)

    @torch.no_grad()
    def fit_process_class_weights(self, process_info, event_labels):
        """Give each process 1/K of the total training weight, with mean event weight one."""
        info = torch.as_tensor(process_info, device="cpu")
        if info.ndim != 2 or info.shape[1] != 2:
            raise ValueError("Expected process_info columns [process_id, abs_weight]")
        ids = info[:, 0].long()
        nclasses = self.process_class_weights.numel()
        if not ((info[:, 0] == ids) & (ids >= 0) & (ids < nclasses)).all():
            raise ValueError("Unknown or invalid training process ID. Check the DSID catalogue.")
        if not torch.equal(ids == 0, torch.as_tensor(event_labels, device="cpu") == 0):
            raise ValueError("The HH process label and signal truth class disagree")
        absolute = info[:, 1].double()
        if not (torch.isfinite(absolute) & (absolute >= 0)).all():
            raise ValueError("Process balancing needs finite, nonnegative absolute event weights")
        total = torch.bincount(ids, weights=absolute, minlength=nclasses)
        if not (total > 0).all():
            raise ValueError(f"Missing training weight for process indices {(total == 0).nonzero().flatten().tolist()}")
        self.process_class_weights.copy_(len(ids) / (nclasses * total))

    def forward(self, pred, target, weight, mean_train_weight, extra_event_info=None, extra_obj_info=None,
                process_info=None):
        # The parent uses teacher_fraction=0, so only MC truth supervises the primary event logits.
        loss, scores = super().forward(pred, target, weight, mean_train_weight, extra_obj_info=extra_obj_info)
        if not self.training:
            return loss, scores
        if self.teacher_loss_weight:
            teacher_logits = pred[1]["teacher"]
            if extra_event_info is None or extra_event_info.shape != (len(target), teacher_logits.shape[1] + 1):
                raise ValueError("Expected the archived EveNet probabilities and validity flag")
            valid = extra_event_info[:, -1:].bool()
            teacher = torch.where(valid, extra_event_info[:, :-1].detach(), 0.)
            per_event = F.kl_div(teacher_logits.log_softmax(dim=1), teacher, reduction="none").sum(dim=1)
            loss = loss + self.teacher_loss_weight * (weight * per_event).mean() / mean_train_weight
        if self.process_loss_weight:
            if process_info is None:
                raise ValueError("The process loss needs process_info columns [process_id, abs_weight]")
            ids = process_info[:, 0].long()
            per_event = F.cross_entropy(pred[1]["process"], ids, reduction="none")
            process_weight = process_info[:, 1] * self.process_class_weights[ids]
            loss = loss + self.process_loss_weight * (process_weight * per_event).mean()
        return loss, scores

"""Losses for reconstructed Higgs, WZ and heavy-flavour candidates."""
import torch
from torch.nn import functional as F

from .distillation import TruthEventWithAuxiliaryHeads


class PhysicsCandidateLoss(TruthEventWithAuxiliaryHeads):
    def __init__(
        self,
        hh_pair_weight=0.12,
        wz_pair_weight=0.12,
        wz_control_weight=0.10,
        heavy_flavour_weight=0.08,
        hybrid_weight=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hh_pair_weight = hh_pair_weight
        self.wz_pair_weight = wz_pair_weight
        self.wz_control_weight = wz_control_weight
        self.heavy_flavour_weight = heavy_flavour_weight
        self.hybrid_weight = hybrid_weight
        self.register_buffer("pair_scale", torch.ones(3))
        self.register_buffer("base_fraction", torch.tensor(1.0))

    def fit_origin_class_weights(self, object_labels, object_mask, event_weights):
        super().fit_origin_class_weights(object_labels[:, :3], object_mask, event_weights)

    @torch.no_grad()
    def fit_physics_weights(self, physics_info, event_weights):
        # Normalize over eligible training events for each task. Pair categories retain their original priors.
        info = torch.as_tensor(physics_info)
        weights = torch.as_tensor(event_weights, dtype=torch.float64)
        control = info[:, 2].bool()
        self.base_fraction.copy_((~control).float().mean())
        raw = info[:, 3].double()
        sums = torch.stack(
            (
                weights[(info[:, 0] >= 0) & ~control].sum(),
                weights[(info[:, 1] >= 0) & ~control].sum(),
                raw[(info[:, 1] >= 0) & control].sum(),
            )
        )
        if not (sums.isfinite() & (sums > 0)).all():
            raise ValueError(f"Missing eligible training examples for pair tasks: {sums.tolist()}")
        self.pair_scale.copy_(len(info) / sums)

    def forward(
        self,
        pred,
        target,
        weight,
        mean_train_weight,
        extra_event_info=None,
        extra_obj_info=None,
        process_info=None,
        physics_info=None,
    ):
        if not self.training:
            return super().forward(
                pred,
                target,
                weight,
                mean_train_weight / self.base_fraction,
                extra_event_info=extra_event_info,
                extra_obj_info=extra_obj_info,
                process_info=process_info,
            )
        if physics_info is None or physics_info.shape[1] != 6:
            raise ValueError(
                "Expected higgsPair, zPair, in_wzcr, absolute nominal weight, jet count and signed weight"
            )
        extra, labels, mask = extra_obj_info
        if labels.shape[1] != 4:
            raise ValueError("Expected origin, lepton mask, conversion and heavy-flavour targets")
        orig_pred = (pred[0], pred[1], pred[2][:, :4])
        loss, scores = super().forward(
            orig_pred,
            target,
            weight,
            mean_train_weight,
            extra_event_info=extra_event_info,
            extra_obj_info=(extra, labels[:, :3], mask),
            process_info=process_info,
        )
        h, z = physics_info[:, 0].long(), physics_info[:, 1].long()
        control = physics_info[:, 2].bool()
        hce = F.cross_entropy(pred[1]["hh_pair"], h, ignore_index=-1, reduction="none")
        zce = F.cross_entropy(pred[1]["wz_flavour_blind"], z, ignore_index=-1, reduction="none")
        loss = loss + self.hh_pair_weight * (hce * weight * ~control).mean() * self.pair_scale[0]
        loss = loss + self.wz_pair_weight * (zce * weight * ~control).mean() * self.pair_scale[1]
        loss = (
            loss + self.wz_control_weight * (zce * physics_info[:, 3] * control).mean() * self.pair_scale[2]
        )
        if self.hybrid_weight:
            valid_reference = extra_event_info[:, 3].bool()
            hybrid_ce = F.cross_entropy(pred[1]["hybrid"], target, reduction="none")
            loss = (
                loss + self.hybrid_weight * (hybrid_ce * weight * valid_reference).mean() / mean_train_weight
            )
        lepton = labels[:, 1].bool() & mask.squeeze(1).bool()
        hf = F.binary_cross_entropy_with_logits(
            pred[2][:, 4].reshape_as(lepton), labels[:, 3], reduction="none"
        )
        per_event = (hf * lepton).sum(1) / lepton.sum(1).clamp_min(1)
        return loss + self.heavy_flavour_weight * (per_event * weight).mean() / mean_train_weight, scores

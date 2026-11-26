"""Fold-normalized MC supervision for the symmetric Higgs, top, W and Z relations."""
import numpy as np
import torch
from torch.nn import functional as F

from synapse.models.shared_parent import PARENTS, supervised_pair_loss
from .physics_candidates import PhysicsCandidateLoss


class SharedParentLoss(PhysicsCandidateLoss):
    def __init__(
        self,
        relation_weight=0.10,
        control_relation_weight=0.10,
        ttbar_other_fraction=0.10,
        recoil_weight=0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.relation_weight = relation_weight
        self.control_relation_weight = control_relation_weight
        self.recoil_weight = recoil_weight
        if not 0 <= ttbar_other_fraction < 1:
            raise ValueError("The non-lepton fraction of the ttbar relation loss must be in [0,1)")
        self.ttbar_other_fraction = ttbar_other_fraction
        self.register_buffer("relation_scale", torch.zeros(2, len(PARENTS)))
        self.register_buffer("ttbar_scale", torch.zeros(3))
        self.register_buffer("recoil_scale", torch.tensor(0.0))

    @torch.no_grad()
    def fit_relation_weights(self, identities, physics_info, event_weights, object_mask):
        # The five main-region tasks get equal shares. The control sample trains Z and recoil.
        totals = np.zeros((2, len(PARENTS)), dtype=np.float64)
        ttbar_totals = np.zeros(3, dtype=np.float64)
        control_total = 0.0
        ii, jj = np.triu_indices(3, 1)
        for start in range(0, len(identities), 16000):
            batch = slice(start, start + 16000)
            known = (identities[batch, :4] >= 0).sum(axis=-1) >= 2
            control = physics_info[batch, 2].astype(bool)
            base_weight = event_weights[batch].astype(np.float64) * ~control
            totals[0] += known.T @ base_weight
            totals[1, -1] += np.sum(physics_info[batch, 3] * control * known[:, -1], dtype=np.float64)
            control_total += np.sum(physics_info[batch, 3] * control, dtype=np.float64)
            leps = identities[batch, 4, :3]
            a, b = leps[:, ii], leps[:, jj]
            yes = (a > 0) & (a == b)
            valid = object_mask[batch, 0].astype(bool)
            eligible = np.stack(
                (
                    yes.any(1),
                    ((a >= 0) & (b >= 0) & ~yes).any(1),
                    valid[:, 3:].any(1) & (valid.sum(1) >= 2),
                ),
                axis=1,
            )
            ttbar_totals += eligible.T @ base_weight
        if (
            not np.isfinite(totals).all()
            or (totals[0] <= 0).any()
            or totals[1, -1] <= 0
            or not np.all(np.isfinite(ttbar_totals) & (ttbar_totals > 0))
            or control_total <= 0
        ):
            raise ValueError(f"Missing or invalid training weight for a shared-parent task: {totals}")
        scale = np.zeros_like(totals)
        one_task = self.relation_weight * len(identities) / 5
        scale[0] = one_task / totals[0]
        scale[1, -1] = self.control_relation_weight * len(identities) / totals[1, -1]
        self.relation_scale.copy_(torch.as_tensor(scale))
        other = self.ttbar_other_fraction
        portions = np.array([(1 - other) / 2, (1 - other) / 2, other])
        self.ttbar_scale.copy_(torch.as_tensor(one_task * portions / ttbar_totals))
        self.recoil_scale.fill_(self.recoil_weight * len(identities) / control_total)

    def ttbar_loss(self, logits, identities, valid, event_weight):
        i, j = torch.triu_indices(logits.shape[-1], logits.shape[-1], offset=1, device=logits.device)
        a, b = identities[:, i], identities[:, j]
        leptons = (i < 3) & (j < 3)
        yes = leptons & (a > 0) & (a == b)
        no = leptons & (a >= 0) & (b >= 0) & ~yes
        domain = valid[:, i, j]
        costs = F.binary_cross_entropy_with_logits(
            logits[:, i, j], yes.to(logits.dtype), reduction="none"
        )
        per_group = []
        for group in (yes, no, ~leptons):
            select = domain & group
            per_group.append(costs.masked_fill(~select, 0).sum(1) / select.sum(1).clamp_min(1))
        return ((torch.stack(per_group, dim=1) @ self.ttbar_scale) * event_weight).mean()

    def forward(
        self, pred, target, weight, mean_train_weight, physics_info=None, relation_info=None, **kwargs
    ):
        loss, scores = super().forward(
            pred, target, weight, mean_train_weight, physics_info=physics_info, **kwargs
        )
        if not self.training:
            return loss, scores
        logits = pred[1]["shared_parent_logits"]
        if relation_info is None or relation_info.shape != logits.shape[:-1] or logits.shape[1] != 5:
            raise ValueError("Expected aligned Higgs, top, W, Z and dileptonic ttbar targets")
        # -1 is unknown, 0 is a confirmed nonmember, positive IDs identify distinct parents within this event.
        parents = relation_info[:, :4]
        a, b = parents.unsqueeze(-1), parents.unsqueeze(-2)
        known = (a >= 0) & (b >= 0)
        same = (a > 0) & (a == b)
        control = physics_info[:, 2].bool()
        weights = (
            torch.where(control, physics_info[:, 3], weight)[:, None]
            * self.relation_scale[control.long()]
        )
        loss = loss + supervised_pair_loss(
            logits[:, :4], same, known, pred[1]["shared_parent_valid"], weights
        )
        loss = loss + self.ttbar_loss(
            logits[:, 4], relation_info[:, 4], pred[1]["ttbar_pair_valid"], weight * ~control
        )
        wz = (kwargs["process_info"][:, 0] == 1).long()
        recoil = F.cross_entropy(pred[1]["wz_recoil"], wz, reduction="none")
        return loss + (recoil * physics_info[:, 3] * control).mean() * self.recoil_scale, scores

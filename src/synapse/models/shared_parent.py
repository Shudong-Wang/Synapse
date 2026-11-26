"""Symmetric shared-parent predictions that guide the last ParT object-attention block."""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .physics_candidates import PhysicsParticleTransformer, embedding


PARENTS = ("Higgs", "top", "W", "Z")


class SharedParentHead(nn.Module):
    def __init__(self, width=128, rank=16, attention_heads=4, dileptonic_top=False):
        super().__init__()
        self.rank = rank
        self.num_relations = len(PARENTS) + int(dileptonic_top)
        self.projection = nn.Linear(width, 2 * self.num_relations * rank, bias=False)
        self.intercept = nn.Parameter(torch.zeros(self.num_relations))
        self.feedback = nn.Linear(self.num_relations, attention_heads, bias=False)
        nn.init.zeros_(self.feedback.weight)

    def forward(self, tokens, eligible_objects, ttbar_valid=None):
        """Project [batch, object, width] tokens and mix relation probabilities per attention head."""
        batch, n, _ = tokens.shape
        projected = self.projection(tokens).view(batch, n, 2, self.num_relations, self.rank)
        u, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
        product = u @ v.transpose(-1, -2)
        logits = (product + product.transpose(-1, -2)) / (2 * math.sqrt(self.rank))
        logits = logits + self.intercept[None, :, None, None]
        valid = eligible_objects[:, :, None] & eligible_objects[:, None, :]
        valid = valid & ~torch.eye(n, dtype=torch.bool, device=tokens.device)
        domain = valid[:, None]
        if ttbar_valid is not None:
            domain = torch.cat((domain.expand(-1, len(PARENTS), -1, -1), ttbar_valid[:, None]), dim=1)
        probabilities = logits.sigmoid() * domain
        bias = self.feedback(probabilities.movedim(1, -1)).movedim(-1, 1)
        return logits, bias, valid


class WZRecoil(nn.Module):
    """Evaluate each SFOS hypothesis independently using only transverse recoil observables."""

    def __init__(self):
        super().__init__()
        self.rank = nn.Sequential(embedding(13, 32), nn.Linear(32, 1))

    def forward(self, z, remaining, met, jets, jet_valid, sfos):
        # All momentum inputs are transverse (px, py) in GeV. The individual Z daughters never enter.
        def pt(v):
            return torch.linalg.vector_norm(v, dim=-1)

        def cos(a, b):
            return ((a * b).sum(-1) / (pt(a) * pt(b)).clamp_min(1.0e-6)).clamp(-1, 1)

        jets = jets * jet_valid[..., None]
        recoil = jets.sum(1)[:, None]
        ht = pt(jets).sum(1, keepdim=True)
        count = jet_valid.sum(1, keepdim=True)
        w = remaining + met
        mt = (2 * pt(remaining) * pt(met) * (1 - cos(remaining, met))).clamp_min(0).sqrt()
        zpt = pt(z)
        feats = [
            zpt.log1p(),
            pt(w).log1p(),
            pt(remaining).log1p(),
            pt(met).expand_as(zpt).log1p(),
            mt.log1p(),
            cos(z, w),
            cos(remaining, met),
            cos(z, recoil),
            cos(w, recoil),
            ht.expand_as(zpt).log1p(),
            count.expand_as(zpt),
            pt(z + w).log1p(),
            pt(z + w + recoil).log1p(),
        ]
        candidates = self.rank(torch.stack(feats, -1)).squeeze(-1)
        # Uniform hypotheses for WZ and background. No mass-based or truth-based candidate selection.
        return (candidates * sfos).sum(1) / sfos.sum(1).clamp_min(1)


class SharedParentParticleTransformer(PhysicsParticleTransformer):
    """Read the relations after four object blocks and apply them in the fifth by default."""

    def __init__(
        self, *, num_layers=5, relation_rank=16, dileptonic_top=False, event_aux_outputs, **kwargs
    ):
        if num_layers < 2 or relation_rank < 1:
            raise ValueError(
                "Shared-parent attention needs at least two object blocks and a positive rank"
            )
        outputs = dict(event_aux_outputs)
        recoil_outputs = outputs.pop("wz_recoil", None)
        super().__init__(num_layers=num_layers, event_aux_outputs=outputs, **kwargs)
        self.event_aux_outputs = event_aux_outputs
        self.dileptonic_top = dileptonic_top
        last = self.blocks[-1]
        self.parent_head = SharedParentHead(
            last.embed_dim, relation_rank, last.num_heads, dileptonic_top
        )
        self.recoil = None
        if recoil_outputs is not None:
            if len(recoil_outputs) != 2:
                raise ValueError("WZ recoil needs two output names in other, WZ order")
            self.recoil = WZRecoil()
            self.recoil_readout = nn.Linear(1, 2 * last.embed_dim, bias=False)

    def classify_tokens(self, event, tokens, raw_features, four_vectors, mask):
        if self.recoil is None:
            return super().classify_tokens(event, tokens, raw_features, four_vectors, mask)
        transverse = four_vectors[:, :2].transpose(1, 2).float() * 0.001
        pdg = raw_features[:, 4]
        valid = mask[:, 0].bool()
        sfos = (
            (pdg[:, self.pair_i] == -pdg[:, self.pair_j]) & valid[:, self.pair_i] & valid[:, self.pair_j]
        )
        jet_valid = valid & raw_features[:, 7].bool()
        z = transverse[:, self.pair_i] + transverse[:, self.pair_j]
        score = self.recoil(
            z, transverse[:, self.other], transverse[:, 9:10], transverse, jet_valid, sfos
        )
        event = event + self.recoil_readout(score.tanh()[:, None])
        scores, aux, objects = super().classify_tokens(event, tokens, raw_features, four_vectors, mask)
        aux["wz_recoil"] = torch.stack((torch.zeros_like(score), score), dim=1)
        return scores, aux, objects

    @contextmanager
    def frozen_batch_norm(self):
        bns = [
            m for m in self.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training
        ]
        try:
            for m in bns:
                m.eval()
            yield
        finally:
            for m in bns:
                m.train()

    def forward_with_control(self, inputs, control):
        """Keep control events out of encoder BatchNorm updates and stop control gradients at block four."""
        base_idx, cr_idx = (~control).nonzero().flatten(), control.nonzero().flatten()
        if not len(cr_idx):
            if len(base_idx) > 1:
                return self(*inputs)
            with self.frozen_batch_norm():
                return self(*inputs)
        groups = []
        if len(base_idx) > 1:
            groups.append(self(*(x[base_idx] for x in inputs)))
        with self.frozen_batch_norm():
            if len(base_idx) == 1:
                groups.append(self(*(x[base_idx] for x in inputs)))
            groups.append(self(*(x[cr_idx] for x in inputs), detach_relation_inputs=True))
        if not len(base_idx):
            return groups[0]
        order = torch.cat((base_idx, cr_idx)).argsort()
        scores, aux, objects = zip(*groups)
        slots = inputs[0].shape[-1]
        return (
            torch.cat(scores)[order],
            {key: torch.cat([g[key] for g in aux])[order] for key in aux[0]},
            torch.cat([g.reshape(-1, slots, self.num_obj_classes) for g in objects])[order].reshape(
                -1, self.num_obj_classes
            ),
        )

    def final_object_attention_bias(self, tokens, padding_mask, pair_bias):
        visible = ~padding_mask
        ttbar_valid = None
        if self.dileptonic_top:
            ttbar_valid = (
                visible[:, :, None]
                & visible[:, None, :]
                & ~torch.eye(visible.shape[1], dtype=torch.bool, device=visible.device)
            )
        visible = visible.clone()
        visible[:, 9] = False  # Slot 9 is MET. It still participates in ordinary attention.
        logits, relation_bias, valid = self.parent_head(tokens.transpose(0, 1), visible, ttbar_valid)
        attention_bias = relation_bias.flatten(0, 1)  # [batch × heads, objects, objects]
        if pair_bias is not None:
            attention_bias = pair_bias + attention_bias
        aux = {"shared_parent_logits": logits, "shared_parent_valid": valid}
        if ttbar_valid is not None:
            aux["ttbar_pair_valid"] = ttbar_valid
        return attention_bias, aux


def supervised_pair_loss(logits, same_parent, known, valid, event_relation_weights=None):
    """Count each unordered pair once and average known pairs within each event and parent."""
    i, j = torch.triu_indices(logits.shape[-1], logits.shape[-1], offset=1, device=logits.device)
    use = known[:, :, i, j] & valid[:, None, i, j]
    target = torch.where(use, same_parent[:, :, i, j], 0).to(logits.dtype)
    costs = F.binary_cross_entropy_with_logits(logits[:, :, i, j], target, reduction="none")
    counts = use.sum(-1)
    per_event_relation = costs.masked_fill(~use, 0).sum(-1) / counts.clamp_min(1)
    if event_relation_weights is not None:
        return (per_event_relation * event_relation_weights).sum(1).mean()
    return per_event_relation.sum() / (counts > 0).sum().clamp_min(1)

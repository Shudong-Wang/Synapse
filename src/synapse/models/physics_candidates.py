"""Competing three-lepton decay assignments and lepton–jet candidates for HH classification."""
import torch
from torch import nn

from .ParticleTransformer import ParticleTransformerSimple


def embedding(inputs, width):
    return nn.Sequential(
        nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width), nn.GELU()
    )


def transverse(vectors):
    return torch.linalg.vector_norm(vectors[..., :2], dim=-1)


def mass(vectors):
    return (vectors[..., 3].square() - vectors[..., :3].square().sum(-1)).clamp_min(0).sqrt()


def kin_cos(a, b):
    return ((a[..., :2] * b[..., :2]).sum(-1) / (transverse(a) * transverse(b)).clamp_min(1.0e-6)).clamp(
        -1, 1
    )


def angular_distance(eta_a, phi_a, eta_b, phi_b):
    delta = phi_a - phi_b
    phi = torch.atan2(delta.sin(), delta.cos())
    return torch.sqrt((eta_a - eta_b).square() + phi.square())


class PhysicsParticleTransformer(ParticleTransformerSimple):
    """Three symmetric pair assignments, an unassigned state, and six central-jet candidates per lepton."""

    def __init__(self, *, embed_dims, event_aux_outputs, num_obj_classes=5, trim=False, **kwargs):
        if trim or num_obj_classes != 5:
            raise ValueError("Physics reconstruction requires aligned slots and five object outputs")
        regular = {
            k: v
            for k, v in event_aux_outputs.items()
            if k not in ("hh_pair", "wz_pair", "wz_flavour_blind", "hybrid")
        }
        for name in ("hh_pair", "wz_pair"):
            if len(event_aux_outputs.get(name, ())) != 4:
                raise ValueError(f"{name} requires three candidate scores and one null score")
        super().__init__(
            embed_dims=embed_dims, event_aux_outputs=regular, num_obj_classes=4, trim=False, **kwargs
        )
        self.event_aux_outputs = event_aux_outputs
        self.num_obj_classes = 5
        dim = embed_dims[-1]
        self.register_buffer("pair_i", torch.tensor([0, 0, 1]))
        self.register_buffer("pair_j", torch.tensor([1, 2, 2]))
        self.register_buffer("other", torch.tensor([2, 1, 0]))
        # Globals already have fold-fitted normalization. Avoid a second running estimate.
        self.global_embed.input_bn = None
        self.wz_embed = embedding(14, 64)
        self.wz_rank = nn.Linear(64, 1)
        self.wz_null = nn.Linear(64, 1)
        self.hh_embed = embedding(3 * dim + 14, 64)
        self.hh_rank = nn.Linear(64, 1)
        self.hh_null = nn.Linear(2 * dim, 1)
        self.lep_proj = nn.Linear(dim, 24)
        self.jet_proj = nn.Linear(dim, 24)
        self.lj_embed = embedding(48 + 5, 48)
        self.lj_rank = nn.Linear(48, 1)
        self.lj_null = nn.Linear(dim, 1)
        self.hf = nn.Sequential(nn.Linear(dim + 48, 32), nn.GELU(), nn.Linear(32, 1))
        self.candidate_fusion = nn.Sequential(
            nn.Linear(64 + 64 + 48 + 2, dim), nn.GELU(), nn.Linear(dim, 2 * dim)
        )
        self.reference_correction = None
        if "hybrid" in event_aux_outputs:
            self.reference_correction = nn.Sequential(nn.Linear(2 * dim, 64), nn.GELU(), nn.Linear(64, 3))
            nn.init.zeros_(self.reference_correction[-1].weight)
            nn.init.zeros_(self.reference_correction[-1].bias)

    def add_reference_prediction(self, predictions, reference):
        if self.reference_correction is None:
            return predictions
        main, auxiliary, obj = predictions
        if reference is None or reference.shape != (len(main), 4):
            raise ValueError("The hybrid output requires the three recorded EveNet scores and validity flag")
        valid = reference[:, 3:].bool()
        log_reference = torch.where(valid, reference[:, :3], torch.ones_like(main)).clamp_min(1.0e-8).log()
        delta = 2.0 * auxiliary.pop("_reference_delta").tanh()
        auxiliary["hybrid"] = torch.where(valid, log_reference + delta, main.detach())
        return main, auxiliary, obj

    def normalize_inputs(self, x, g):
        x, g = super().normalize_inputs(x, g)
        return x.clamp(-12, 12), g.clamp(-12, 12)

    def pair_kinematics(self, v, raw):
        a, b, c = v[:, self.pair_i], v[:, self.pair_j], v[:, self.other]
        met = v[:, 9:10]
        ll = a + b
        mt = (2 * transverse(c) * transverse(met) * (1 - kin_cos(c, met))).clamp_min(0).sqrt()
        mll = mass(ll)
        pa, pb = transverse(a), transverse(b)
        ab_dr = angular_distance(
            raw[:, self.pair_i, 1], raw[:, self.pair_i, 2], raw[:, self.pair_j, 1], raw[:, self.pair_j, 2]
        )
        w = c + met
        balance = (transverse(ll) - transverse(w)) / (transverse(ll) + transverse(w)).clamp_min(1.0e-6)
        feats = (
            mll.log1p(),
            ((mll - 91.2).abs()).log1p(),
            transverse(ll).log1p(),
            ab_dr,
            torch.minimum(pa, pb).log1p(),
            torch.maximum(pa, pb).log1p(),
            transverse(c).log1p(),
            transverse(met).expand_as(mll).log1p(),
            mt.log1p(),
            transverse(w).log1p(),
            kin_cos(c, met),
            kin_cos(ll, w),
            balance,
            transverse(ll + w).log1p(),
        )
        return torch.stack(feats, dim=-1)

    def lepton_jet_candidates(self, tokens, raw, v, valid):
        lep = tokens[:, :3]
        jet = tokens[:, 3:9]
        l = v[:, :3, None, :]
        j = v[:, None, 3:9, :]
        dr = angular_distance(
            raw[:, :3, None, 1], raw[:, :3, None, 2], raw[:, None, 3:9, 1], raw[:, None, 3:9, 2]
        )
        jet_pt = transverse(j).clamp_min(1.0e-4)
        jet_tag = raw[:, None, 3:9, 8].expand_as(dr)
        tag_known = raw[:, None, 3:9, 9].expand_as(dr)
        feats = torch.stack(
            (dr, mass(l + j).log1p(), (transverse(l) / jet_pt).clamp_max(50).log1p(), jet_tag, tag_known), -1
        )
        lp = self.lep_proj(lep).unsqueeze(2).expand(-1, -1, 6, -1)
        jp = self.jet_proj(jet).unsqueeze(1).expand(-1, 3, -1, -1)
        candidate = self.lj_embed(torch.cat((lp, jp, feats), dim=-1))
        logits = self.lj_rank(candidate).squeeze(-1).masked_fill(~valid[:, None, 3:9], -1.0e4)
        probabilities = torch.cat((logits, self.lj_null(lep)), dim=-1).softmax(-1)
        context = (probabilities[:, :, :6, None] * candidate).sum(dim=2)
        hf = self.hf(torch.cat((lep, context), -1))
        pooled = (hf.squeeze(-1).softmax(dim=1).unsqueeze(-1) * context).sum(dim=1)
        return pooled, hf

    def classify_tokens(self, event, tokens, raw_features, four_vectors, mask):
        raw = raw_features.transpose(1, 2)
        v = four_vectors.transpose(1, 2).float() * 0.001  # MeV to GeV
        kin = self.pair_kinematics(v, raw)
        # The weak assignment comparison only sees reco kinematics. Flavour constrains inference.
        wz = self.wz_embed(kin)
        wz_rank = self.wz_rank(wz).squeeze(-1)
        wz_null = self.wz_null(wz.mean(1))
        os = raw[:, self.pair_i, 5] * raw[:, self.pair_j, 5] < 0
        sfos = os & (raw[:, self.pair_i, 4].abs() == raw[:, self.pair_j, 4].abs())
        wz_out = torch.cat((wz_rank.masked_fill(~sfos, -1.0e4), wz_null), dim=1)
        wz_probs = wz_out.softmax(1)
        hi, hj = tokens[:, self.pair_i], tokens[:, self.pair_j]
        hh = self.hh_embed(torch.cat((hi + hj, (hi - hj).abs(), tokens[:, self.other], kin), dim=-1))
        hh_out = torch.cat((self.hh_rank(hh).squeeze(-1), self.hh_null(event)), dim=1)
        hh_probs = hh_out.softmax(1)
        top, hf = self.lepton_jet_candidates(tokens, raw, v, mask.squeeze(1).bool())
        event = event + self.candidate_fusion(
            torch.cat(
                (
                    (hh_probs[:, :3, None] * hh).sum(1),
                    (wz_probs[:, :3, None] * wz).sum(1),
                    top,
                    hh_probs[:, 3:],
                    wz_probs[:, 3:],
                ),
                dim=1,
            )
        )
        scores = self.fc(event)
        auxiliary = {name: head(event) for name, head in self.event_aux_heads.items()}
        if self.reference_correction is not None:
            # The reference readout has its own gradients. The standalone representation trains on its physics tasks.
            auxiliary["_reference_delta"] = self.reference_correction(event.detach())
        auxiliary.update(
            hh_pair=hh_out,
            wz_pair=wz_out,
            wz_flavour_blind=torch.cat((wz_rank.masked_fill(~os, -1.0e4), wz_null), dim=1),
        )
        original = self.obj_fc(tokens)
        hf = torch.cat((hf, hf.new_full((len(hf), tokens.shape[1] - 3, 1), -20.0)), dim=1)
        return scores, auxiliary, torch.cat((original, hf), -1).reshape(-1, 5)

import unittest

import torch
from torch.nn import functional as F

from synapse.models.shared_parent import SharedParentParticleTransformer, supervised_pair_loss
from synapse.loss_functions.shared_parent import SharedParentLoss


class SharedParentTests(unittest.TestCase):
    def test_unique_pair_normalization_and_unknown_targets(self):
        logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
        truth = torch.zeros_like(logits, dtype=torch.bool)
        known = torch.zeros_like(truth)
        known[:, :, 0, 1] = known[:, :, 1, 0] = True
        truth[:, :, 0, 1] = truth[:, :, 1, 0] = True
        valid = ~torch.eye(4, dtype=torch.bool)[None]
        loss = supervised_pair_loss(logits, truth, known, valid)
        torch.testing.assert_close(loss, torch.tensor(2.0).log())
        loss.backward()
        self.assertEqual(logits.grad.count_nonzero().item(), 1)
        self.assertEqual(logits.grad[0, 0, 0, 1].item(), -0.5)
        self.assertEqual(supervised_pair_loss(logits, truth, ~torch.ones_like(known), valid).item(), 0.0)

    def test_ttbar_uses_all_valid_nonleptons_and_balances_only_lepton_pairs(self):
        loss = SharedParentLoss(
            teacher_loss_weight=0, process_loss_weight=0, origin_loss_weight=0, conversion_loss_weight=0
        )
        truth = torch.tensor([[1, 1, 0] + [-1] * 9, [-1] * 12])
        valid_objects = torch.zeros(2, 12, dtype=torch.bool)
        valid_objects[:, [0, 1, 2, 3, 9]] = True
        valid = valid_objects[:, :, None] & valid_objects[:, None, :] & ~torch.eye(12, dtype=torch.bool)
        logits = torch.zeros(2, 12, 12, requires_grad=True)
        loss.ttbar_scale.copy_(torch.tensor([0.45, 0.45, 0.10]))
        measured = loss.ttbar_loss(logits, truth, valid, torch.ones(2))
        torch.testing.assert_close(measured, torch.tensor(0.55) * torch.tensor(2.0).log())
        measured.backward()
        gradients = logits.grad
        self.assertLess(gradients[0, 0, 1].item(), 0)
        self.assertGreater(gradients[0, 0, 2].item(), 0)
        self.assertEqual(gradients[1, :3, :3].count_nonzero().item(), 0)
        self.assertGreater(
            gradients[:, 0, 9].min().item(), 0
        )  # Valid MET is a known False for this task.
        self.assertGreater(gradients[:, 3, 9].min().item(), 0)
        self.assertEqual(gradients[:, 4].count_nonzero().item(), 0)
        self.assertEqual(gradients.tril().count_nonzero().item(), 0)

    def test_recoil_mass_blindness_and_control_gradient_isolation(self):
        torch.manual_seed(11)
        heads = {k: ["12", "13", "23", "none"] for k in ("hh_pair", "wz_pair", "wz_flavour_blind")}
        heads["wz_recoil"] = ["other", "wz"]
        model = SharedParentParticleTransformer(
            input_dim=13,
            global_input_dim=3,
            num_classes=3,
            num_heads=4,
            embed_dims=[16],
            global_embed_dims=[16],
            pair_embed_dims=[8],
            num_cls_layers=1,
            dileptonic_top=True,
            event_aux_outputs=heads,
        )
        model.setup_input_norm(torch.zeros(13), torch.ones(13), torch.zeros(3), torch.ones(3))
        x, g = torch.randn(4, 13, 12), torch.randn(4, 3)
        x[:, 4, :3] = torch.tensor([11, -11, 13])
        x[:, 5, :3] = torch.tensor([-1, 1, -1])
        x[:, 7, :] = 0
        x[:, 7, 3:5] = 1
        v = torch.randn(4, 4, 12) * 10000
        v[:, 3] = 1.1 * torch.linalg.vector_norm(v[:, :3], dim=1)
        mask = torch.zeros(4, 1, 12, dtype=torch.bool)
        mask[:, :, [0, 1, 2, 3, 4, 9]] = True
        model.eval()

        def recoil(vectors):
            return model(x, g, vectors, mask)[1]["wz_recoil"]

        varied = v.clone()
        varied[:, 0, 0] += 3500.0
        varied[:, 0, 1] -= 3500.0
        varied[:, 2, 0] += 90000.0
        varied[:, 3, 0] += 120000.0
        torch.testing.assert_close(recoil(v), recoil(varied), atol=1.0e-6, rtol=1.0e-6)
        model.train()
        before = {k: b.clone() for k, b in model.named_buffers() if "running_" in k}
        _, aux, _ = model.forward_with_control((x, g, v, mask), torch.ones(4, dtype=torch.bool))
        self.assertTrue(
            torch.equal(aux["shared_parent_logits"], aux["shared_parent_logits"].transpose(-1, -2))
        )
        self.assertFalse(aux["shared_parent_valid"][:, 9].any())
        self.assertTrue(aux["ttbar_pair_valid"][:, 9, 0].all())
        self.assertFalse(aux["ttbar_pair_valid"][:, 5].any())
        (
            F.cross_entropy(aux["wz_recoil"], torch.tensor([0, 1, 0, 1]))
            + aux["shared_parent_logits"][:, 3, 0, 1].sigmoid().mean()
        ).backward()
        self.assertTrue(all(p.grad is None for b in model.blocks for p in b.parameters()))
        self.assertGreater(model.parent_head.projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.recoil.rank[-1].weight.grad.abs().sum().item(), 0)
        for key, b in model.named_buffers():
            if key in before:
                torch.testing.assert_close(b, before[key])
        mixed_scores, mixed_aux, _ = model.forward_with_control(
            (x, g, v, mask), torch.tensor([True, False, True, True])
        )
        self.assertTrue(mixed_scores.isfinite().all())
        self.assertEqual(mixed_aux["shared_parent_logits"].shape, (4, 5, 12, 12))

    def test_fourth_block_predicts_relations_and_fifth_adds_per_head_bias(self):
        torch.manual_seed(31)
        outputs = {
            name: ["12", "13", "23", "none"] for name in ("hh_pair", "wz_pair", "wz_flavour_blind")
        }
        model = SharedParentParticleTransformer(
            input_dim=13,
            global_input_dim=3,
            num_classes=3,
            num_heads=4,
            embed_dims=[16],
            global_embed_dims=[16],
            pair_embed_dims=[8],
            num_cls_layers=1,
            event_aux_outputs=outputs,
        ).eval()
        model.setup_input_norm(torch.zeros(13), torch.ones(13), torch.zeros(3), torch.ones(3))
        x, g = torch.randn(2, 13, 12), torch.randn(2, 3)
        v = 10000 * torch.randn(2, 4, 12)
        v[:, 3] = 1.1 * torch.linalg.vector_norm(v[:, :3], dim=1)
        mask = torch.zeros(2, 1, 12, dtype=torch.bool)
        mask[:, :, :5] = mask[:, :, 9] = True
        inputs = x, g, v, mask

        mixing = model.parent_head.feedback.weight
        self.assertEqual(tuple(mixing.shape), (4, 4))
        self.assertEqual(mixing.count_nonzero().item(), 0)
        ordinary_scores, auxiliary, _ = model(*inputs)
        (event_gradient,) = torch.autograd.grad(ordinary_scores[:, 0].sum(), mixing, retain_graph=True)
        self.assertGreater(event_gradient.abs().sum().item(), 0)
        logits, valid = auxiliary["shared_parent_logits"], auxiliary["shared_parent_valid"]
        auxiliary_loss = supervised_pair_loss(
            logits, torch.ones_like(logits), torch.ones_like(logits, dtype=torch.bool), valid
        )
        (relation_gradient,) = torch.autograd.grad(auxiliary_loss, model.parent_head.projection.weight)
        self.assertGreater(relation_gradient.abs().sum().item(), 0)

        with torch.no_grad():
            mixing.copy_(torch.randn_like(mixing))
        biases, fourth_output, relation_input = [], [], []

        def capture_bias(block, args, kwargs):
            biases.append(kwargs["attn_mask"])

        handles = [
            block.register_forward_pre_hook(capture_bias, with_kwargs=True) for block in model.blocks
        ]
        handles.append(
            model.blocks[3].register_forward_hook(lambda block, args, out: fourth_output.append(out))
        )
        handles.append(
            model.parent_head.register_forward_pre_hook(
                lambda head, args: relation_input.append(args[0])
            )
        )
        try:
            scores, auxiliary, _ = model(*inputs)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(biases), 5)
        torch.testing.assert_close(relation_input[0], fourth_output[0].transpose(0, 1))
        valid = auxiliary["shared_parent_valid"]
        self.assertFalse(
            valid[:, 9].any() or valid[:, 5].any() or valid.diagonal(dim1=-2, dim2=-1).any()
        )
        probabilities = auxiliary["shared_parent_logits"].sigmoid() * valid[:, None]
        expected_bias = torch.einsum("hr,brij->bhij", mixing, probabilities).flatten(0, 1)
        for earlier in biases[1:4]:
            torch.testing.assert_close(earlier, biases[0])
        torch.testing.assert_close(biases[4], biases[0] + expected_bias)
        self.assertGreater((scores - ordinary_scores).abs().max().item(), 0)
        (event_to_relations,) = torch.autograd.grad(
            scores[:, 0].sum(), model.parent_head.projection.weight
        )
        self.assertGreater(event_to_relations.abs().sum().item(), 0)

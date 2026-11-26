import unittest

import torch
from torch.nn import functional as F

from synapse.models.ParticleTransformer import Block, ParticleTransformerSimple


class ParticleTransformerTests(unittest.TestCase):
    def test_global_features_reach_valid_objects_before_attention(self):
        torch.manual_seed(11)
        model = ParticleTransformerSimple(
            input_dim=4, global_input_dim=3, num_classes=3, embed_dims=[8], global_embed_dims=[8],
            num_layers=1, num_heads=2, num_cls_layers=1, pair_embed_dims=None,
            num_obj_classes=4, fuse_global_into_objects=True, trim=False,
        ).eval()
        model.setup_input_norm(torch.zeros(4), torch.ones(4), torch.zeros(3), torch.ones(3))
        x = torch.randn(2, 4, 3)
        g = torch.randn(2, 3, requires_grad=True)
        mask = torch.tensor([[[True, True, False]], [[True, False, True]]])
        encoder_inputs = []
        handle = model.blocks[0].register_forward_pre_hook(lambda block, args: encoder_inputs.append(args[0]))
        try:
            _, _, object_logits = model(x, g, mask=mask)
        finally:
            handle.remove()

        embedded = model.embed(x)
        event = model.global_embed(g).unsqueeze(0).expand_as(embedded)
        fused = embedded + model.object_global_fusion(torch.cat((embedded, event), dim=-1))
        torch.testing.assert_close(encoder_inputs[0], fused.masked_fill(~mask.permute(2, 0, 1), 0))
        self.assertEqual(object_logits.shape, (6, 4))
        real_logits = object_logits[mask.flatten()]
        global_grad, = torch.autograd.grad(real_logits.square().sum(), g)
        self.assertTrue((global_grad.abs().sum(dim=1) > 0).all())

    def test_head_scaling_before_output_projection(self):
        torch.manual_seed(7)
        block = Block(embed_dim=8, num_heads=2).double().eval()
        attn = block.attn
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        ordinary, _ = attn(x, x, x)
        torch.testing.assert_close(block._attention(x, x, None), ordinary)

        # Read the concatenated head outputs with an identity output projection.
        heads, _ = F.multi_head_attention_forward(
            x, x, x, 8, 2, attn.in_proj_weight, attn.in_proj_bias, None, None,
            False, 0.0, torch.eye(8, dtype=x.dtype), None, training=False,
        )
        with torch.no_grad():
            block.head_scale.copy_(torch.tensor([0.25, 1.5]))
        scaled = heads.unflatten(-1, (2, 4)) * block.head_scale[:, None]
        expected = F.linear(scaled.flatten(-2), attn.out_proj.weight, attn.out_proj.bias)
        torch.testing.assert_close(block._attention(x, x, None), expected)

        # Old checkpoints applied their scale after projection and transposed the two axes.
        old_state = block.state_dict()
        old_scale = old_state.pop("head_scale")
        old_state["c_attn"] = old_scale
        block.load_state_dict(old_state)
        legacy = (ordinary.unflatten(-1, (2, 4)) * old_scale[:, None]).transpose(-1, -2).flatten(-2)
        torch.testing.assert_close(block._attention(x, x, None), legacy)
        self.assertIn("c_attn", block.state_dict())

    def test_restore_training_fold_normalization(self):
        params = dict(input_dim=2, global_input_dim=3, num_classes=3, num_layers=1,
                      num_heads=2, embed_dims=[8], global_embed_dims=[8], pair_embed_dims=[4])
        model = ParticleTransformerSimple(**params)
        unfitted = model.state_dict()
        model.setup_input_norm(torch.tensor([1., 3.]), torch.tensor([2., 4.]),
                               torch.tensor([2., 3., 1.]), torch.tensor([1., 4., 2.]))
        restored = ParticleTransformerSimple(**params)
        restored.load_state_dict(dict(model.state_dict()))
        x, g = torch.randn(4, 2, 6), torch.randn(4, 3)
        torch.testing.assert_close(restored.normalize_inputs(x, g), model.normalize_inputs(x, g))
        restored.load_state_dict(unfitted)
        self.assertFalse(restored.has_input_norm)

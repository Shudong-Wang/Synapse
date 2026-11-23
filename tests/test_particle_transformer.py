import unittest

import torch
from torch.nn import functional as F

from synapse.models.ParticleTransformer import Block, ParticleTransformerSimple


class ParticleTransformerTests(unittest.TestCase):
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

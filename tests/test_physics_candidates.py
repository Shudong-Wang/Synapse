import unittest

import torch

from synapse.core.metrics import hh_rejection_at_20
from synapse.models.physics_candidates import PhysicsParticleTransformer


class CandidateTests(unittest.TestCase):
    def test_weak_wz_assignment_cannot_read_flavour(self):
        names = {task: ["12", "13", "23", "none"] for task in ("hh_pair", "wz_pair", "wz_flavour_blind")}
        names["hybrid"] = ["hh", "bkg", "conv"]
        model = PhysicsParticleTransformer(
            input_dim=15,
            global_input_dim=4,
            num_classes=3,
            embed_dims=[16],
            global_embed_dims=[16],
            num_heads=2,
            num_layers=1,
            num_cls_layers=1,
            pair_embed_dims=None,
            event_aux_outputs=names,
        ).eval()
        model.setup_input_norm(torch.zeros(15), torch.ones(15), torch.zeros(4), torch.ones(4))
        x = torch.zeros(2, 15, 12)
        x[:, 4, :3] = torch.tensor([11.0, -11.0, -13.0])
        x[:, 5, :3] = torch.tensor([-1.0, 1.0, 1.0])
        x[1, 4, 2] = -11.0
        v = torch.zeros(2, 4, 12)
        v[:, 0, :3] = torch.tensor([20000.0, -30000.0, 25000.0])
        v[:, 1, :3] = torch.tensor([40000.0, 10000.0, -15000.0])
        v[:, 1, 9] = 42000.0
        v[:, 3] = torch.linalg.vector_norm(v[:, :3], dim=1)
        mask = torch.zeros(2, 1, 12, dtype=torch.bool)
        mask[:, :, :3] = True
        mask[:, :, 9] = True
        with torch.no_grad():
            main, pairs, obj = model(x, torch.zeros(2, 4), v, mask)
        torch.testing.assert_close(pairs["wz_flavour_blind"][0], pairs["wz_flavour_blind"][1])
        self.assertLess(pairs["wz_pair"][0, 1].item(), -1000.0)
        self.assertGreater(pairs["wz_pair"][1, 1].item(), -1000.0)
        self.assertTrue(torch.isfinite(main).all())
        self.assertEqual(tuple(obj.shape), (24, 5))
        teacher = torch.tensor([[0.6, 0.3, 0.1, 1.0], [-1.0, -1.0, -1.0, 0.0]])
        _, extra, _ = model.add_reference_prediction((main, pairs, obj), teacher)
        torch.testing.assert_close(extra["hybrid"][0].softmax(0), teacher[0, :3])
        torch.testing.assert_close(extra["hybrid"][1], main[1])

    def test_rejection20_counts_common_preselection_and_score_ties(self):
        probs = torch.tensor(
            [
                [0.9, 0.09, 0.01],
                [0.1, 0.89, 0.01],
                [0.05, 0.01, 0.94],
                [0.9, 0.09, 0.01],
                [0.6, 0.39, 0.01],
                [0.99, 0.009, 0.001],
                [0.95, 0.04, 0.01],
            ]
        )
        true = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        info = torch.tensor([[1, 2], [1, 5], [1, 3], [1, 1], [1, 4], [0, 20], [1, -9]], dtype=torch.float)
        self.assertEqual(hh_rejection_at_20(probs.log(), true, torch.ones(7), selection_info=info), 5.0)


if __name__ == "__main__":
    unittest.main()

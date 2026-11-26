import unittest

import numpy as np
import torch

from synapse.core.working_point_significance import asimov_significance, hh_significance_20_30_50, working_point_significance


class HHWorkingPointTests(unittest.TestCase):
    def test_signed_counts_positive_efficiencies_veto_and_ties(self):
        score = np.array([.9,.8,.8,.65,.1,.85, .98,.94,.85,.99,.99])
        hh = np.array([True]*6+[False]*5)
        weights = np.array([2.,1.,2.,1.,4.,-.25, 3.,-1.,2.,900.,100.])
        pre = np.ones(len(hh), dtype=bool)
        sr = pre.copy()
        pre[-1] = False
        sr[4] = sr[-2] = False
        r = working_point_significance(score,hh,weights,pre,sr)
        np.testing.assert_equal(r["cuts"], [.9,.8,.8])
        np.testing.assert_allclose([x["positive_hh_efficiency"] for x in r["cumulative"]], [.2,.5,.5])
        np.testing.assert_equal([x["signal"] for x in r["cumulative"]], [2.,4.75,4.75])
        np.testing.assert_equal([x["background"] for x in r["cumulative"]], [2.,4.,4.])
        np.testing.assert_equal([x["background_mc_variance"] for x in r["cumulative"]], [10.,14.,14.])
        bands = r["exclusive"]
        np.testing.assert_equal([band["signal"] for band in bands], [2., 2.75, 0.])
        np.testing.assert_equal([band["background"] for band in bands], [2., 2., 0.])
        np.testing.assert_equal([band["background_mc_variance"] for band in bands], [10., 4., 0.])
        self.assertEqual(bands[-1]["z"], 0.)
        self.assertAlmostEqual(r["quadrature"], np.linalg.norm([band["z"] for band in bands]))
        self.assertEqual(r["quadrature"], r["exclusive_quadrature"])
        self.assertEqual(r,working_point_significance(score,hh,weights,pre,sr,r["cuts"]))

    def test_epoch_monitor_uses_nominal_weights_and_the_conversion_selection(self):
        prob = torch.tensor([[.90,.09,.01],[.80,.19,.01],[.70,.29,.01],[.05,.05,.90],
                             [.95,.04,.01],[.85,.14,.01],[.86,.13,.01],[.75,.24,.01],[.99,.005,.005]],dtype=torch.float64)
        true = torch.tensor([0,0,0,0,1,1,1,2,1])
        info = torch.tensor([[1,2],[1,1],[1,2],[1,5],[1,3],[1,4],[1,-1],[1,2],[0,99]],dtype=torch.float64)
        logits = prob.log()
        actual = hh_significance_20_30_50(logits,true,torch.zeros(len(true)),info)
        z = [asimov_significance(2, 3, 9), asimov_significance(1, 3, 17), asimov_significance(2, 2, 4)]
        self.assertAlmostEqual(actual,np.linalg.norm(z))
        self.assertEqual(hh_significance_20_30_50(logits,true,torch.ones(len(true)),info*0),0.)

    def test_known_background_limit_and_unreachable_cut(self):
        self.assertAlmostEqual(asimov_significance(2.,5.,0.),np.sqrt(2*(7*np.log(1.4)-2)))
        self.assertEqual(asimov_significance(1.,0.,1.),0.)
        r = working_point_significance([.9,.8],[True,True],[1,9],[True,True],[True,False])
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()

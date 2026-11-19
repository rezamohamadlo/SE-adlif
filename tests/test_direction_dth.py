"""Non-training tests of the opt-in directional response."""
import copy
import unittest
import torch
from models.ours.dth_adlif import DTHSEAdLIF
from tests.test_dth_dropout_consistency import dth_config, make


class DirectionDTHTests(unittest.TestCase):
    def test_initial_shared_weights_rng_and_forward(self):
        cfg=dth_config(True)
        torch.manual_seed(42);a=make(cfg).eval();rng=torch.get_rng_state()
        bcfg=copy.deepcopy(cfg)
        for layer in ('l1','l2'): bcfg[layer].direction_aware=True
        torch.manual_seed(42);b=make(bcfg).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        sa,sb=a.state_dict(),b.state_dict()
        self.assertEqual(set(sb)-set(sa),{'l1.offset_gain','l2.offset_gain'})
        for k,v in sa.items():torch.testing.assert_close(v,sb[k],rtol=0,atol=0)
        x=torch.rand(2,9,4)
        with torch.no_grad():torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)

    def test_formula_recovery_constraints_and_gradients(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            cfg=dth_config(True).l1;cfg.direction_aware=True
            m=DTHSEAdLIF(cfg).to(device)
            with torch.no_grad():m.homeostatic_gain.fill_(.7);m.transient_gain.fill_(1.3);m.offset_gain.fill_(-1.3)
            fast=torch.tensor([[.7],[.1],[.4]],device=device)
            slow=torch.tensor([[.3],[.6],[.4]],device=device)
            old=m.threshold_logits+m.homeostatic_gain*(slow-m.target_rate)-m.transient_gain*(fast-slow).abs()
            torch.testing.assert_close(m.dynamic_threshold(fast,slow),m.theta_min+(m.theta_max-m.theta_min)*old.sigmoid())
            out=m.dynamic_threshold(fast,slow);out.sum().backward()
            self.assertTrue(torch.isfinite(m.offset_gain.grad));self.assertGreater(m.offset_gain.grad.item(),0)
            self.assertTrue(torch.all(out>m.theta_min) and torch.all(out<m.theta_max))
            with torch.no_grad():m.offset_gain.fill_(-8);m.homeostatic_gain.fill_(-3);m.transient_gain.fill_(8)
            m.apply_parameter_constraints()
            self.assertEqual(m.offset_gain.item(),-5);self.assertEqual(m.homeostatic_gain.item(),0);self.assertEqual(m.transient_gain.item(),5)
            with torch.no_grad():m.offset_gain.fill_(8)
            m.apply_parameter_constraints();self.assertEqual(m.offset_gain.item(),5)

    def test_default_has_no_new_parameter(self):
        m=DTHSEAdLIF(dth_config().l1)
        self.assertFalse(m.direction_aware);self.assertFalse(hasattr(m,'offset_gain'))

if __name__=='__main__':unittest.main()

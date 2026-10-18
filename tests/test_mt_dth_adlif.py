import unittest

import torch
from omegaconf import OmegaConf

from models.ours.mt_adlif import MTSEAdLIF
from models.ours.dth_adlif import DTHSEAdLIF
from models.ours.mt_dth_adlif import MTDTHSEAdLIF
from tests.test_measure_shd_costs import tiny_model
from measure_shd_costs import operation_profiles, ScalarOperations, batch_scalar_operations
import runner


def config(**kwargs):
    return OmegaConf.create(dict(input_size=4, n_neurons=6, tau_u_range=[5, 25],
                                 tau_w_range=[60, 300], q=120, **kwargs))


class MTDTHTests(unittest.TestCase):
    def test_initial_threshold_reduces_to_mt(self):
        torch.manual_seed(42)
        mt = MTSEAdLIF(config())
        combined = MTDTHSEAdLIF(config())
        combined.load_state_dict(mt.state_dict(), strict=False)
        s_mt, s_new = mt.initial_state(3), combined.initial_state(3)
        for x in (torch.randn(3, 4) * 20 for _ in range(17)):
            z_mt, s_mt = mt(x, s_mt)
            z_new, s_new = combined(x, s_new)
            torch.testing.assert_close(z_mt, z_new, rtol=0, atol=0)
            for a, b in zip(s_mt, s_new[:4]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_equal_adaptation_branches_reduce_to_dth(self):
        combined = MTDTHSEAdLIF(config())
        dth = DTHSEAdLIF(config())
        dth.load_state_dict({k: v for k, v in combined.state_dict().items()
                             if k in dth.state_dict()}, strict=True)
        combined.tau_w_slow_trainer.load_state_dict(combined.tau_w_trainer.state_dict())
        for cell in [combined, dth]:
            with torch.no_grad():
                cell.homeostatic_gain.fill_(1)
                cell.transient_gain.fill_(0.4)
        a, b = combined.initial_state(2), dth.initial_state(2)
        for x in (torch.randn(2, 4) * 20 for _ in range(19)):
            za, a = combined(x, a)
            zb, b = dth(x, b)
            torch.testing.assert_close(za, zb, rtol=0, atol=0)
            torch.testing.assert_close(a[0], b[0])
            torch.testing.assert_close(a[2], b[2])
            torch.testing.assert_close(a[4], b[3])

    def test_scan_gradients_and_cost_counter(self):
        cell = MTDTHSEAdLIF(config())
        x = torch.randn(2, 13, 4) * 20
        state, spikes = cell.initial_state(2), []
        for frame in x.unbind(1):
            z, state = cell(frame, state)
            spikes.append(z)
        torch.testing.assert_close(cell.layer_forward(x), torch.stack(spikes, dim=1))
        cell.layer_forward(x).sum().backward()
        for name in ['mix_logits', 'threshold_logits', 'homeostatic_gain', 'transient_gain',
                     'tau_w_slow_trainer.weight']:
            grad = dict(cell.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
        with torch.inference_mode():
            model = tiny_model('mt_dth_se_adlif')
            profiles = operation_profiles(model)
            inputs = torch.randn(3, 7, 4)
            with ScalarOperations() as count:
                model(inputs)
            self.assertEqual(count.counts, batch_scalar_operations(profiles, 3, 7))

    def test_three_seed_baseline_plan_and_separate_paths(self):
        plan = runner.build_plan('MT_DTH_SE_adLIF', datasets=('SHD',))
        mt = runner.build_plan('MT_SE_adLIF', datasets=('SHD',))[0]['config']
        self.assertEqual([p['seed'] for p in plan], [42, 123, 456])
        for p in plan:
            self.assertIn('MT_DTH_SE_adLIF', p['directory'].parts)
            for k in ['dataset', 'lr', 'batch_size', 'n_epochs', 'early_stopping', 'dropout', 'loss_agg']:
                self.assertEqual(p['config'][k], mt[k], k)


if __name__ == '__main__':
    unittest.main()

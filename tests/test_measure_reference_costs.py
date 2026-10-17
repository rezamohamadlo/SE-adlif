import unittest

import torch

from measure_reference_costs import ReferenceCounter, measure, data_signature
from tests.test_measure_shd_costs import tiny_model
from omegaconf import OmegaConf


class ReferenceCostTests(unittest.TestCase):
    def test_analog_inputs_are_not_counted_as_spikes(self):
        with torch.inference_mode():
            model = tiny_model()
            inputs = torch.randn(3, 4, 4) * 100
            lengths = torch.tensor([4, 3, 2])
            counter = ReferenceCounter(model, lengths, False)
            actual = model(inputs)
            counts = counter.close()
            torch.testing.assert_close(actual, model(inputs), rtol=0, atol=0)
            # First projection contributes zero event operations; only the
            # previous hidden spikes drive layer-one recurrent connections.
            manual = []
            h = model.l1.register_forward_hook(lambda m, a, o: manual.append(o[0].clone()))
            model(inputs)
            h.remove()
            self.assertEqual(counts['l1'][0], float(torch.stack(manual[:-1]).sum()) * 6)
            self.assertGreaterEqual(counts['l1'][0], 0)

    def test_cost_scopes_and_analog_matrix_projection(self):
        with torch.inference_mode():
            model = tiny_model()
            x = torch.randn(3, 4, 4)
            target = torch.tensor([[-1, 0], [-1, 1], [-1, 0]])
            block = torch.ones(3, 4, dtype=torch.long)
            result = measure(model, [(x, target, block, torch.tensor([4, 4, 4]))],
                             torch.device('cpu'), False, 2, 1)
            self.assertEqual(result['totals']['samples'], 3)
            self.assertEqual(result['analog_input_dense_macs_per_sample'], 4 * 6 * 4)
            self.assertGreater(result['arithmetic_gflops_test_padded'], result['dense_gflops_test_padded'])
            self.assertGreater(result['logical_gops_test_padded'], result['arithmetic_gflops_test_padded'])

    def test_default_ecg_burn_in_and_split_guard(self):
        a = OmegaConf.create(dict(dataset=dict(_target_='datasets.ecg.ECGLDM', valid_fraction=0.05)))
        b = OmegaConf.create(dict(dataset=dict(_target_='datasets.ecg.ECGLDM', valid_fraction=0.05, burn_in_time=0)))
        self.assertEqual(data_signature(a), data_signature(b))
        b.dataset.valid_fraction = 0.1
        self.assertNotEqual(data_signature(a), data_signature(b))


if __name__ == '__main__':
    unittest.main()

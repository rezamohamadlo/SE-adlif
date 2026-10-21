import unittest

import torch

from evaluate_tta import batch_predictions
from measure_reference_costs import measure as ordinary_measure
from measure_tta_costs import measure, verify_result, test_fingerprint as data_fingerprint
from tests.test_measure_shd_costs import tiny_model


class TtaCostTests(unittest.TestCase):
    def batches(self):
        x = torch.randint(0, 3, (3, 4, 4)).float()
        target = torch.tensor([[-1, 0], [-1, 1], [-1, 0]])
        block = torch.ones(3, 4, dtype=torch.long)
        return [(x, target, block, torch.tensor([4, 3, 2]))]

    def test_identity_matches_existing_counter_and_accuracy(self):
        torch.manual_seed(42)
        model, batches = tiny_model('dth_se_adlif'), self.batches()
        with torch.inference_mode():
            expected = ordinary_measure(model, batches, torch.device('cpu'), True, 1, 1)
        actual = measure(model, batches, torch.device('cpu'), True, [1.0], 2, 1)
        self.assertAlmostEqual(actual['tta_synops_per_sample'], expected['synops_per_sample'])
        self.assertEqual(actual['tta_dense_gflops'], expected['dense_gflops_test_padded'])
        self.assertEqual(actual['tta_arithmetic_gflops'], expected['arithmetic_gflops_test_padded'])
        self.assertEqual(actual['tta_accuracy'], expected['reproduced_test_acc'])
        self.assertEqual(actual['gain_scaling_gflops'], 0)
        self.assertEqual(actual['samples'], 3)
        self.assertTrue(all(len(v) == 2 for v in actual['timing_repeats_ms_per_sample'].values()))

    def test_three_views_match_predictions_and_sum_activity(self):
        torch.manual_seed(123)
        model, batches = tiny_model('mt_se_adlif'), self.batches()
        gains = [1.0, .95, 1.05]
        actual = measure(model, batches, torch.device('cpu'), True, gains, 1, 1)
        _, probs, labels = batch_predictions(model, batches[0][:3], model.output_func, gains)
        self.assertEqual(actual['tta_accuracy'], (probs.argmax(-1)==labels).double().mean().item())
        self.assertEqual(actual['tta_dense_gflops'], 3*actual['single_dense_gflops'])
        self.assertAlmostEqual(actual['tta_synops_per_sample'], sum(v['synops_per_sample'] for v in actual['per_view']))
        # Check amplitude scaling does not scale the first-layer event count.
        x, y, block, lengths = batches[0]
        for gain, view in zip(gains, actual['per_view']):
            with torch.inference_mode():
                raw = ordinary_measure(model, [(x*gain, y, block, lengths)], torch.device('cpu'), True, 1, 1)
            valid = torch.arange(4)[None, :] < lengths[:, None]
            correction = ((x.sum(-1)*valid).sum().item()*model.l1.out_features*(1-gain))/3
            self.assertAlmostEqual(view['synops_per_sample'], raw['synops_per_sample']+correction, places=4)

    def test_analog_input_and_evidence_guards(self):
        model, batches = tiny_model(), self.batches()
        actual = measure(model, batches, torch.device('cpu'), False, [1.0, .95, 1.05], 1, 1)
        self.assertEqual(actual['tta_analog_input_dense_macs_per_sample'], 3*4*6*4)
        evidence = {'metrics': {k:actual[k] for k in ('data_sha256', 'evaluated_targets', 'baseline_accuracy', 'tta_accuracy')}}
        verify_result(actual, evidence)
        self.assertEqual(actual['data_sha256'], data_fingerprint(batches))
        for key, changed in [('tta_accuracy', -1), ('data_sha256', 'changed'), ('evaluated_targets', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_result(dict(actual, **{key:changed}), evidence)


if __name__ == '__main__':
    unittest.main()

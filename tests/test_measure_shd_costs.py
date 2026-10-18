import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from measure_shd_costs import (
    ScalarOperations, SpikeCounter, batch_scalar_operations,
    collate_lengths, operation_profiles, update_reports, MODELS, ROOT, dataset_signature,
)
from models.pl_module import MLPSNN


def tiny_model(cell='se_adlif'):
    layer = dict(cell=cell, input_size=4, n_neurons=6,
                 tau_u_range=[5, 25], tau_w_range=[60, 300], q=120,
                 adaptation_update_interval=2)
    cfg = OmegaConf.create(dict(two_layers=True, dataset=dict(num_classes=2, batch_size=3),
        tracking_metric='val_acc', tracking_mode='max', dropout=0.15,
        lr=0.01, factor=0.9, patience=9999, l1=layer,
        l2=dict(layer, input_size=6),
        l_out=dict(input_size=6, n_neurons=2, tau_u_range=[15, 15]),
        loss_agg='sum_softmax_over_time'))
    with contextlib.redirect_stdout(io.StringIO()):
        return MLPSNN(cfg).eval()


class MeasurementTests(unittest.TestCase):
    def test_legacy_omitted_default_is_equivalent_but_preprocessing_is_not(self):
        a = OmegaConf.create(dict(dataset=dict(batch_size=256, time_factor=0.001)))
        b = OmegaConf.create(dict(dataset=dict(batch_size=256, time_factor=0.001,
                                               additional_test_set_validation=False)))
        self.assertEqual(dataset_signature(a), dataset_signature(b))
        b.dataset.time_factor = 0.002
        self.assertNotEqual(dataset_signature(a), dataset_signature(b))

    def test_scalar_conventions(self):
        x = torch.ones(2, 4)
        with ScalarOperations() as c:
            (x * 2 + 1).mean(-1)
        self.assertEqual(c.counts, dict(mul=8, add=8, mean=8))

    def test_unknown_operation_fails(self):
        with self.assertRaises(NotImplementedError):
            with ScalarOperations():
                torch.linalg.vector_norm(torch.ones(3))

    def test_affine_profiles_match_executed_full_forward_all_variants(self):
        # Detect initial-state broadcasting, shared parameter transforms and
        # multi-rate phase mistakes against a fully instrumented forward.
        for cell in ['se_adlif', 'dth_se_adlif', 'lg_dth_se_adlif',
                     'da_se_adlif', 'fp_se_adlif', 'mr_se_adlif', 'mt_se_adlif']:
            with self.subTest(cell=cell), torch.inference_mode():
                model = tiny_model(cell)
                profiles = operation_profiles(model)
                x = torch.randn(3, 7, 4)
                with ScalarOperations() as direct:
                    model(x)
                self.assertEqual(direct.counts, batch_scalar_operations(profiles, 3, 7))

    def test_hooks_preserve_output_and_match_manual_spike_count(self):
        with torch.inference_mode():
            model = tiny_model()
            x = torch.randn(3, 5, 4) * 100
            expected = model(x)
            manual = {}
            handles = []
            for name in ['l1', 'l2']:
                manual[name] = []
                handles.append(getattr(model, name).register_forward_hook(
                    lambda m, a, o, name=name: manual[name].append(o[0].clone())))
            lengths = torch.tensor([5, 3, 2])
            counter = SpikeCounter(model, lengths, 5)
            actual = model(x)
            counts = counter.close()
            for h in handles:
                h.remove()
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            mask = torch.arange(5)[None, :] < lengths[:, None]
            for name in manual:
                spikes = torch.stack(manual[name], dim=1)
                self.assertEqual(counts[name][2], float(spikes.sum()))
                self.assertEqual(counts[name][3], float((spikes * mask[..., None]).sum()))
            # Recurrence consumes previous outputs, never the final output.
            z1 = torch.stack(manual['l1'], dim=1)
            expected_synops = (x.sum() * 6 + z1[:, :-1].sum() * 6)
            self.assertAlmostEqual(counts['l1'][0], float(expected_synops), places=2)

    def test_collation_preserves_prebatch_lengths(self):
        items = [(torch.ones(t, 4), torch.tensor(1), torch.ones(t, dtype=torch.long))
                 for t in [3, 5]]
        x, targets, block, lengths = collate_lengths(items)
        self.assertEqual(x.shape, (2, 5, 4))
        self.assertEqual(lengths.tolist(), [3, 5])
        self.assertEqual(block[0, 3:].tolist(), [0, 0])

    def test_report_update_preserves_paper_data_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = ['reports/reference-vs-models.md', 'reports/paper-vs-reference.md',
                       'results/not_improved/LG_DTH_SE_adLIF/RESULTS.md']
            for name in reports:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((ROOT / name).read_text(encoding='utf-8'), encoding='utf-8')
            output = root / 'measurements'
            fields = ['spikes_total_per_sample', 'spikes_l1_per_sample', 'spikes_l2_per_sample',
                      'firing_l1_percent', 'firing_l2_percent', 'synops_per_sample',
                      'nonmatrix_ops_per_sample', 'forward_ms_per_sample']
            for model in MODELS:
                (output / model).mkdir(parents=True)
                for seed in [42, 123, 456]:
                    (output / model / f'seed_{seed}.json').write_text(json.dumps(dict(
                        version='shd_costs_v1', measurement_identity='same', model=model, seed=seed,
                        measurements=dict.fromkeys(fields, 10.0))), encoding='utf-8')
            with patch('measure_shd_costs.ROOT', root):
                update_reports(output)
                first = [(root / p).read_text(encoding='utf-8') for p in reports]
                update_reports(output)
                self.assertEqual(first, [(root / p).read_text(encoding='utf-8') for p in reports])
            self.assertIn('| LG-DTH | 10', first[0])
            self.assertNotIn('| Pending |', first[0])
            self.assertIn('| SHD* | 93.79', first[1])
            original_ssc = [line for line in (ROOT / reports[1]).read_text(encoding='utf-8').splitlines()
                            if line.startswith('| SSC |')]
            self.assertEqual(original_ssc, [line for line in first[1].splitlines()
                                            if line.startswith('| SSC |')])


if __name__ == '__main__':
    unittest.main()

"""User-launched full-validation contracts; no training or real inference."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.adaptation_ssc_full import evaluate, runtime


class SSCFullValidationContracts(unittest.TestCase):
    def test_subset_flag_is_not_available(self):
        with self.assertRaises(SystemExit) as error:
            evaluate.main(['--samples', '256', '--check-only'])
        self.assertEqual(error.exception.code, 2)

    def test_feature_preprocessing_is_batched_with_complete_order(self):
        def rows(data, ids, condition, study):
            return [(sid, object(), dict(sample_id=sid)) for sid in ids]
        with patch.object(runtime, 'input_rows', side_effect=rows) as loader:
            records = runtime.feature_rows(None, list(range(11)), ('clean', 0, 'identity', 1.), {}, 4)
        self.assertEqual([r['sample_id'] for r in records], list(range(11)))
        self.assertEqual([c.args[1] for c in loader.call_args_list], [list(range(4)), list(range(4, 8)), list(range(8, 11))])

    def test_streamed_csv_preserves_evidence_and_retains_report_fields(self):
        rows = [dict(seed=42, condition='clean', replicate=0, sample_id=sid,
            parent_loss=.8, variant_loss=.7, benefit=.1, parent_correct=1, variant_correct=1,
            correctness_difference=0, mean_rate_hz=4., fast_slow_contrast=.3,
            autocorr_tau_ms=None, active_channel_fraction=.5, duration_ms=100.,
            input_sha256='input-identity', speaker='speaker-a') for sid in [3, 5]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paired.csv'
            retained = runtime.write_paired_csv(path, iter(rows))
            with path.open(newline='', encoding='utf-8') as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual([int(r['sample_id']) for r in saved], [3, 5])
        self.assertEqual(saved[0]['input_sha256'], 'input-identity')
        self.assertEqual(saved[0]['speaker'], 'speaker-a')
        self.assertNotIn('input_sha256', retained[0])
        self.assertIsNone(retained[0]['autocorr_tau_ms'])
        self.assertEqual(retained[0]['benefit'], .1)

    def test_empty_pair_stream_cannot_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paired.csv'
            with self.assertRaises(ValueError):
                runtime.write_paired_csv(path, iter(()))
            self.assertFalse(path.exists())

    def test_failed_clean_anchor_blocks_all_perturbations(self):
        condition = ['clean', 0, 'identity', 1.]
        study = dict(conditions=[condition, ['time_scale_0.9', 0, 'warp', .9]])
        run = dict(model='MT_SE_adLIF', seed=42, checkpoint_sha256='checkpoint',
                   config=dict(dataset=dict(batch_size=256)))
        manifest = dict(environment=dict(device='cpu'), sample_ids=[0],
                        data=dict(validation_indices=[0]), runs=[run])
        features = [dict(sample_id=0)]
        predictions = [dict(sample_id=0, model='MT_SE_adLIF', seed=42, condition='clean', replicate=0)]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runtime, 'feature_rows', return_value=features) as extract, \
                patch.object(runtime, 'load_model', return_value=object()), \
                patch.object(runtime.pilot, 'infer', return_value=predictions), \
                patch.object(runtime, 'check_anchor', side_effect=ValueError('anchor mismatch')):
            destination = Path(directory)
            with self.assertRaisesRegex(ValueError, 'anchor mismatch'):
                runtime.execute(study, destination, manifest, None)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(extract.call_args.args[2], condition)
            self.assertFalse((destination / 'COMPLETE.json').exists())
            self.assertFalse(any('time_scale' in p.name for p in destination.rglob('*.json')))

    def test_check_only_does_not_execute_workload(self):
        with patch.object(evaluate, 'prepare', return_value=({}, None, {}, None)), \
                patch.object(runtime, 'execute', side_effect=AssertionError('execution')):
            self.assertEqual(evaluate.main(['--check-only']), 0)


if __name__ == '__main__':
    unittest.main()

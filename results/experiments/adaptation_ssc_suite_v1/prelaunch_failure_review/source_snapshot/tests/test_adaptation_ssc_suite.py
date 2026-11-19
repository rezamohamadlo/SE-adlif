"""User-launched queue/preservation contracts. No training or actual neural inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.adaptation_analysis import pilot
from scripts.adaptation_ssc_suite import common, run, statistics
from scripts.adaptation_ssc_suite.preflight import validate_config_and_objective


class SSCQueueContracts(unittest.TestCase):
    def source_fixture(self, directory):
        source = Path(directory) / 'original'
        source.mkdir()
        manifest = dict(scope='ssc_full_validation', training=False, official_test_opened=False,
            ordinary_inference=True, sample_ids=list(range(9981)), seeds=[42, 123, 456],
            condition_definitions=[['clean', 0, 'identity', 1.]], data=dict(sha256='data'),
            resolved_study=dict(features=dict(bin_ms=4), parent='MT_SE_adLIF', variant='AMT_SE_adLIF'))
        pilot.write_json(source / 'MANIFEST.json', manifest)
        payload = source / 'diagnostic_rows/chunk.json'
        payload.parent.mkdir()
        payload.write_bytes(b'{"identity":"original bytes"}\r\n')
        complete = dict(status='complete', manifest_sha256=pilot.digest(manifest),
            outputs={p.as_posix(): pilot.sha256(p) for p in (source / 'MANIFEST.json', payload)})
        pilot.write_json(source / 'COMPLETE.json', complete)
        spec = dict(source_evaluation=dict(root=source.as_posix(), manifest_sha256=pilot.digest(manifest),
            completion_sha256=pilot.sha256(source / 'COMPLETE.json')), seeds=[42, 123, 456],
            conditions=manifest['condition_definitions'], features=manifest['resolved_study']['features'], dataset_sha256='data')
        return source, spec

    def test_snapshot_is_byte_exact_and_survives_missing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source, spec = self.source_fixture(directory)
            before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            target = common.snapshot_source(spec, Path(directory) / 'suite')
            for suffix, data in before.items():
                self.assertEqual((target / suffix).read_bytes(), data)
                self.assertEqual((source / suffix).read_bytes(), data)
            source.rename(source.with_name('original_unavailable'))
            common.validate_source(spec, target)
            self.assertEqual(common.snapshot_source(spec, Path(directory) / 'suite'), target)
            index = pilot.read_json(target / 'PATH_INDEX.json')
            self.assertIn((source / 'diagnostic_rows/chunk.json').as_posix(), index['relocations'])

    def test_changed_snapshot_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            _, spec = self.source_fixture(directory)
            root = Path(directory) / 'suite'
            target = common.snapshot_source(spec, root)
            payload = target / 'diagnostic_rows/chunk.json'
            payload.write_bytes(b'user change')
            with self.assertRaises(ValueError):
                common.snapshot_source(spec, root)
            self.assertEqual(payload.read_bytes(), b'user change')

    def test_manifest_guard_preserves_mismatched_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot.write_json(root / 'MANIFEST.json', dict(version='old'))
            with self.assertRaises(ValueError):
                common.guard(root, dict(version='new'))
            self.assertEqual(pilot.read_json(root / 'MANIFEST.json'), dict(version='old'))

    def test_completion_cannot_point_outside_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'package'
            root.mkdir()
            outside = Path(directory) / 'outside.json'
            outside.write_text('evidence')
            pilot.write_json(root / 'MANIFEST.json', dict(version='test'))
            common.seal(root, dict(version='test'), [outside])
            with patch.object(common, 'experiment_directory', side_effect=lambda value: Path(value)):
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    common.verify_completion(root, root)

    def test_censored_acf_is_excluded_without_imputation(self):
        valid = dict(fast_slow_contrast=.4, autocorr_tau_ms=20., duration_ms=100., mean_rate_hz=5.,
            fast_slow_valid=True, autocorr_valid=True, autocorr_censored=False, autocorr_reason=None,
            autocorr_horizon_ms=200., speaker='speaker-string', target=2)
        censored = dict(valid, autocorr_tau_ms=None, autocorr_valid=False,
                        autocorr_censored=True, autocorr_reason='no_crossing_within_horizon')
        features = {10: valid, 11: censored}
        original = copy.deepcopy(features)
        keep, excluded = statistics.regression_cases([10, 11], features, ['fast_slow_contrast', 'autocorr_tau_ms'])
        self.assertEqual(keep, [0])
        self.assertEqual(excluded[0]['sample_id'], 11)
        self.assertIn('right_censored_ACF_not_imputed', excluded[0]['reasons'])
        self.assertEqual(features, original)

    def test_missing_duration_is_an_explicit_exclusion(self):
        feature = dict(fast_slow_contrast=.4, autocorr_tau_ms=20., duration_ms=None, mean_rate_hz=5.,
            fast_slow_valid=True, autocorr_valid=True, autocorr_censored=False, autocorr_reason=None,
            autocorr_horizon_ms=200., speaker='speaker-string', target=2)
        keep, excluded = statistics.regression_cases([0], {0: feature}, ['fast_slow_contrast', 'autocorr_tau_ms'])
        self.assertEqual(keep, [])
        self.assertIn('undefined_duration_ms', excluded[0]['reasons'])

    def test_temporal_objective_requires_registered_protocol_and_weight(self):
        spec, baseline = common.load_spec()
        section = baseline['datasets']['SSC']
        cfg = dict(random_seed=42, model_variant='MT_TC_SE_adLIF', two_layers=True,
            paper_baseline_version=baseline['version'], paper_baseline_dataset='SSC',
            evaluation_protocol='ssc_mean_loss_v1_mt_temporal_consistency_v1', dropout=.15,
            loss_agg='mean_membrane_potentials', early_stopping=False,
            lr=section['lr'], batch_size=section['batch_size'], n_epochs=section['n_epochs'],
            training_objective='mt_temporal_consistency_v1', temporal_consistency=dict(weight=.1, window=5),
            l1=dict(cell='mt_se_adlif', input_size=140, n_neurons=720),
            l2=dict(cell='mt_se_adlif', input_size=720, n_neurons=720), l_out=dict(n_neurons=35),
            dataset=dict(_target_='datasets.ssc.SSCLDM', num_classes=35,
                **{k: section[k] for k in ('spatial_factor', 'time_factor', 'window_size', 'pad_to_min_size',
                                          'ignore_first_timesteps', 'batch_size')}))
        record = dict(model='MT_TC_SE_adLIF', seed=42)
        validate_config_and_objective(cfg, record, spec, baseline)
        for changes in [dict(evaluation_protocol='ssc_mean_loss_v1'), dict(temporal_consistency=dict(weight=.2, window=5))]:
            wrong = dict(cfg, **changes)
            with self.assertRaises(ValueError):
                validate_config_and_objective(wrong, record, spec, baseline)

    def test_check_only_never_executes_queue(self):
        with patch.object(run, 'preflight', return_value={}), \
                patch.object(run, 'execute', side_effect=AssertionError('execution')):
            self.assertEqual(run.main(['--check-only']), 0)

    def test_failed_statistics_stops_before_next_pair(self):
        pairs = [dict(id='amt_mt'), dict(id='lt_dth')]
        manifest = dict(jobs=['contract_tests', 'snapshot_source', 'amt_mt_statistics', 'lt_dth_evaluation', 'lt_dth_statistics', 'summary'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = dict(root=root, spec=dict(pairs=pairs), manifest=manifest)
            with patch.object(run, 'run_contract_tests'), \
                    patch.object(run, 'snapshot_source', return_value=root / 'shared'), \
                    patch.object(run, 'validate_source', return_value=({}, {}, {})), \
                    patch('scripts.adaptation_ssc_suite.evaluations.evaluate', return_value=(root / 'eval', {})) as evaluate, \
                    patch('scripts.adaptation_ssc_suite.statistics.execute', side_effect=ValueError('failed design')):
                with self.assertRaisesRegex(ValueError, 'failed design'):
                    run.execute(prepared)
            self.assertEqual(evaluate.call_count, 1)
            progress = pilot.read_json(root / 'PROGRESS.json')
            self.assertEqual(progress['status'], 'failed')
            self.assertEqual(progress['current_job'], 'amt_mt_statistics')
            self.assertFalse((root / 'COMPLETE.json').exists())

    def test_failed_contract_checks_stop_before_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(run, 'run_contract_tests', side_effect=ValueError('contract failure')), \
                    patch.object(run, 'snapshot_source') as snapshot:
                with self.assertRaisesRegex(ValueError, 'contract failure'):
                    run.execute(dict(root=root, spec={}, manifest=dict(jobs=[])))
            snapshot.assert_not_called()
            self.assertEqual(pilot.read_json(root / 'PROGRESS.json')['current_job'], 'contract_tests')
            self.assertFalse((root / 'COMPLETE.json').exists())

    def test_complete_queue_verifies_without_restarting_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(jobs=[])
            pilot.write_json(root / 'MANIFEST.json', manifest)
            pilot.write_json(root / 'COMPLETE.json', dict(marker='already complete'))
            with patch.object(run, 'verify') as verify, \
                    patch.object(run, 'snapshot_source', side_effect=AssertionError('restart')):
                run.execute(dict(root=root, spec={}, manifest=manifest))
            verify.assert_called_once_with(root, {})


if __name__ == '__main__':
    unittest.main()

"""Non-training tests: synthetic HDF5, composition, guards and queue ordering."""
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from datasets.shd import SHDLDM
from datasets import shd_speaker_split as split
import runner
import shd_speaker_holdout as study


def fixture(root):
    location = Path(root) / 'SHDWrapper'
    location.mkdir(parents=True)
    for name in ('shd_train.h5', 'shd_test.h5'):
        with h5py.File(location / name, 'w') as stream:
            stream['labels'] = np.tile(np.arange(20, dtype=np.uint16), 5)
            stream['extra/speaker'] = np.repeat(np.arange(5, dtype=np.uint16), 20)
            stream['extra/keys'] = np.array([str(i).encode() for i in range(20)])
            for key in ('spikes/times', 'spikes/units'):
                stream.create_dataset(key, (100,), dtype=h5py.vlen_dtype(np.dtype('float64')))
        (location / (name + '.zip')).touch()  # Tonic's existence check only.


@contextmanager
def isolated_runner(manifest):
    with tempfile.TemporaryDirectory(dir=study.ROOT, prefix='speaker-test-') as temporary:
        root = Path(temporary)
        output = root.relative_to(study.ROOT).as_posix() + '/results'
        with patch.object(study, 'OUTPUT_ROOT', output), patch.object(study, 'SPLIT_PATH', output + '/SPLIT_MANIFEST.json'), patch.object(study, 'ARCHIVE_ROOT', output + '/archive'):
            with study.configured_runner(runner, manifest):
                yield root


class SpeakerMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)
        self.manifest = split.build_manifest(self.root)

    def test_deterministic_disjoint_complete_coverage(self):
        a = self.manifest
        self.assertEqual(a, split.build_manifest(self.root))
        train, valid = (a['subsets'][key] for key in ('train', 'validation'))
        self.assertEqual(len(valid['speaker_ids']), 2)
        self.assertFalse(set(train['speaker_ids']) & set(valid['speaker_ids']))
        self.assertFalse(set(train['indices']) & set(valid['indices']))
        self.assertEqual(sorted(train['indices'] + valid['indices']), list(range(100)))
        self.assertEqual(train['class_counts'], [3] * 20)
        self.assertEqual(valid['class_counts'], [2] * 20)
        self.assertEqual(a['validation_fraction'], .4)
        self.assertEqual(a['split_seed'], 42)

    def test_stable_numeric_order_and_row_index_alignment(self):
        speakers, labels = split.read_metadata(self.root / 'SHDWrapper/shd_train.h5')
        perm = np.random.default_rng(7).permutation(len(labels))
        shuffled = split.split_metadata(speakers[perm], labels[perm])
        self.assertEqual(shuffled['validation_speaker_ids'], self.manifest['validation_speaker_ids'])
        self.assertTrue(np.isin(speakers[perm][shuffled['subsets']['validation']['indices']],
                                shuffled['validation_speaker_ids']).all())

    def test_missing_misaligned_and_incomplete_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, 'aligned'):
            split.split_metadata(np.arange(3), np.arange(20))
        with self.assertRaisesRegex(ValueError, 'integer'):
            split.split_metadata(np.zeros(20, dtype=float), np.arange(20))
        with self.assertRaisesRegex(ValueError, 'three'):
            split.split_metadata(np.zeros(20, dtype=int), np.arange(20))
        speakers, labels = split.read_metadata(self.root / 'SHDWrapper/shd_train.h5')
        selected = self.manifest['validation_speaker_ids']
        labels[(np.isin(speakers, selected)) & (labels == 19)] = 0
        with self.assertRaisesRegex(ValueError, 'validation lacks classes'):
            split.split_metadata(speakers, labels)
        with h5py.File(self.root / 'SHDWrapper/shd_train.h5', 'a') as stream:
            del stream['extra/speaker']
        with self.assertRaisesRegex(ValueError, 'Missing SHD'):
            split.build_manifest(self.root)

    def test_manifest_tampering_and_dataset_changes_rejected(self):
        path = self.root / 'manifest.json'
        split.save_manifest(path, self.manifest)
        self.assertEqual(split.load_manifest(path, self.root, self.manifest['fingerprint']), self.manifest)
        changed = copy.deepcopy(self.manifest)
        changed['subsets']['train']['indices'].reverse()
        with self.assertRaisesRegex(ValueError, 'Frozen split'):
            split.save_manifest(path, changed)
        with self.assertRaisesRegex(ValueError, 'changed'):
            split.verify_manifest(changed, self.root, self.manifest['fingerprint'])
        with h5py.File(self.root / 'SHDWrapper/shd_test.h5', 'a') as stream:
            stream['labels'][0] = 1
        with self.assertRaisesRegex(ValueError, 'changed'):
            split.load_manifest(path, self.root, self.manifest['fingerprint'])

    def test_content_fingerprint_portable_across_checkout_locations(self):
        with tempfile.TemporaryDirectory() as other:
            shutil.copytree(self.root / 'SHDWrapper', Path(other) / 'SHDWrapper')
            self.assertEqual(split.build_manifest(other), self.manifest)

    def test_loader_subsets_and_cache_isolation(self):
        path = self.root / 'manifest.json'
        split.save_manifest(path, self.manifest)
        kwargs = dict(data_path=str(self.root), validate_on='speaker_holdout', random_seed=42,
                      cache_namespace=split.PROTOCOL, speaker_split_manifest=str(path),
                      speaker_split_fingerprint=self.manifest['fingerprint'])
        first, second = SHDLDM(**kwargs), SHDLDM(**kwargs)
        legacy = SHDLDM(data_path=str(self.root), validate_on=.2, cache_namespace='controlled_v2')
        self.assertEqual(first.data_train.indices, self.manifest['subsets']['train']['indices'])
        self.assertEqual(first.data_val.indices, second.data_val.indices)
        self.assertEqual(first.cache_path, second.cache_path)
        self.assertNotEqual(first.cache_path, legacy.cache_path)
        self.assertIn(self.manifest['fingerprint'], first.cache_path)
        self.assertEqual(len(legacy.data_val), 20)
        self.assertEqual(len(first.data_val), 40)
        self.assertIs(first.data_train.dataset, first.data_val.dataset)
        self.assertEqual(first.data_test.data_filename, 'shd_test.h5')
        changed = SHDLDM(**dict(kwargs, window_size=8))
        self.assertNotEqual(first.cache_path, changed.cache_path)
        self.assertFalse(Path(first.cache_path).exists())
        with self.assertRaisesRegex(ValueError, 'split seed 42'):
            SHDLDM(**dict(kwargs, random_seed=123))
        with self.assertRaisesRegex(ValueError, 'no test validation'):
            SHDLDM(**dict(kwargs, additional_test_set_validation=True))
        with self.assertRaisesRegex(ValueError, 'changed'):
            SHDLDM(**dict(kwargs, speaker_split_fingerprint='different'))

    def test_loader_metadata_disagreement_rejected(self):
        from datasets.shd import SHDWrapper
        loader = SHDWrapper(save_to=str(self.root), train=True)
        loader._speaker = loader._speaker[::-1]
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            split.check_loader(loader, self.manifest)

    def test_missing_raw_file_and_misaligned_hdf5_fail_without_fallback(self):
        with h5py.File(self.root / 'SHDWrapper/shd_train.h5', 'a') as stream:
            del stream['extra/speaker']
            stream['extra/speaker'] = np.arange(99, dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, 'not aligned'):
            split.build_manifest(self.root)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, 'Raw SHD file unavailable'):
                split.build_manifest(empty)

    def test_relative_paths_resolve_from_changed_working_directory(self):
        from project_paths import portable_path
        with tempfile.TemporaryDirectory(dir=study.ROOT) as checkout_data:
            target = Path(checkout_data)
            shutil.copytree(self.root / 'SHDWrapper', target / 'SHDWrapper')
            path = target / 'split.json'
            split.save_manifest(path, self.manifest)
            original_cwd = Path.cwd()
            try:
                os.chdir(self.root)
                module = SHDLDM(data_path=portable_path(target), validate_on='speaker_holdout',
                                cache_namespace=split.PROTOCOL, speaker_split_manifest=portable_path(path),
                                speaker_split_fingerprint=self.manifest['fingerprint'])
                self.assertEqual(module.data_val.indices, self.manifest['subsets']['validation']['indices'])
            finally:
                os.chdir(original_cwd)


class SpeakerRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        speakers = np.repeat(np.arange(5), 20)
        cls.manifest = split.split_metadata(speakers, np.tile(np.arange(20), 5))
        cls.manifest['fingerprint'] = split.digest(cls.manifest)

    def test_retired_workflow_fails_before_data_access_or_launch(self):
        with isolated_runner(self.manifest):
            marker = study.ROOT / study.ARCHIVE_ROOT / 'ARCHIVE.json'
            marker.parent.mkdir(parents=True)
            marker.write_text('{}', encoding='utf-8')
            with patch.object(study, 'preflight_data') as preflight, self.assertRaises(SystemExit) as result:
                study.main(['--shd-speaker-phase', '1', '--dry-run'], runner)
            self.assertEqual(result.exception.code, 1)
            preflight.assert_not_called()

    def test_matched_composition_fixed_split_and_output_isolation(self):
        with isolated_runner(self.manifest):
            configs = []
            for model, seed in study.ordered_jobs('all'):
                item = runner.build_plan(model, seeds=(seed,), datasets=('SHD',))[0]
                cfg = item['config']
                configs.append(cfg)
                self.assertEqual((cfg['n_epochs'], cfg['batch_size'], cfg['early_stopping']), (300, 256, False))
                self.assertEqual(cfg['dataset']['random_seed'], 42)
                self.assertEqual(cfg['random_seed'], seed)
                self.assertEqual(cfg['evaluation_protocol'], split.PROTOCOL)
                self.assertEqual(cfg['loss_agg'], 'sum_softmax_over_time')
                self.assertEqual(cfg['paper_baseline_dataset'], 'SHD')
                self.assertNotIn('results/valid', cfg['logdir'])
                self.assertFalse(Path(cfg['logdir']).is_absolute())
                if model == study.MODELS[1]:
                    for layer in ('l1', 'l2'):
                        self.assertTrue(cfg[layer]['local_slow_enabled'])
                        self.assertTrue(cfg[layer]['local_transient_enabled'])
            self.assertEqual(len({c['dataset']['speaker_split_fingerprint'] for c in configs}), 1)

    def test_full_manifest_settings_guards(self):
        with isolated_runner(self.manifest):
            item = runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))[0]
            directory = item['directory']
            directory.mkdir(parents=True)
            path = directory / 'FULL_RUN_CONFIG.json'
            path.write_text(json.dumps(item['config']), encoding='utf-8')
            self.assertEqual(runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))[0]['status'], 'new')
            mutations = [('evaluation_protocol', 'wrong'), ('random_seed', 123), ('lr', .1),
                         ('n_epochs', 200), ('early_stopping', True), ('model_variant', study.MODELS[1])]
            for key, value in mutations:
                bad = copy.deepcopy(item['config'])
                bad[key] = value
                path.write_text(json.dumps(bad), encoding='utf-8')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Configuration changed'):
                    runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))
            for key, value in [('speaker_split_fingerprint', 'wrong'), ('random_seed', 123),
                               ('cache_namespace', 'controlled_v2')]:
                bad = copy.deepcopy(item['config'])
                bad['dataset'][key] = value
                path.write_text(json.dumps(bad), encoding='utf-8')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Configuration changed'):
                    runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))

    def test_checkpoint_identity_full_resume_and_completed_budget(self):
        with isolated_runner(self.manifest):
            item = runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))[0]
            cfg, directory = item['config'], item['directory']
            split.save_manifest(study.SPLIT_PATH, self.manifest)
            split.save_manifest(directory / 'SPLIT_MANIFEST.json', self.manifest)
            (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps(cfg), encoding='utf-8')
            (directory / 'ckpt').mkdir()
            best = directory / 'ckpt/best.ckpt'
            state = dict(hyper_parameters={'cfg': cfg}, epoch=10, optimizer_states=[{'state': {}}],
                         lr_schedulers=[{}], callbacks={'selection': dict(monitor='val_acc',
                             best_model_path=str(best), best_model_score=.8)})
            torch.save(state, best)
            last = directory / 'ckpt/last.ckpt'
            torch.save(state, last)
            plan = runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))[0]
            self.assertEqual(plan['status'], 'resume')
            self.assertIn('last.ckpt', plan['overrides'][-1])
            with self.assertRaisesRegex(ValueError, 'full budget'):
                study.inspect_checkpoint(directory, cfg, complete=True)
            state['hyper_parameters']['cfg'] = dict(cfg, random_seed=123)
            torch.save(state, last)
            with self.assertRaisesRegex(ValueError, 'Incompatible checkpoint'):
                runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))
            state['hyper_parameters']['cfg'] = cfg
            state['optimizer_states'] = []
            torch.save(state, last)
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                study.inspect_checkpoint(directory, cfg)

    def test_queue_order_and_stop_on_failure_without_training(self):
        jobs = study.ordered_jobs('all')
        self.assertEqual(jobs, tuple((model, seed) for seed in (42, 123, 456) for model in study.MODELS))
        self.assertEqual(study.ordered_jobs('1'), jobs[:2])
        self.assertEqual(study.ordered_jobs('2'), jobs[2:])
        seen = []
        def operation(model, seed):
            seen.append((model, seed))
            if len(seen) == 2:
                raise RuntimeError('fixture failure')
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            study.execute_jobs(jobs, operation)
        self.assertEqual(seen, list(jobs[:2]))

    def test_phase_two_requires_completed_phase_one(self):
        with patch.object(runner, 'build_plan', return_value=[{'status': 'new'}]):
            with self.assertRaisesRegex(ValueError, 'Complete Phase 1'):
                study.require_phase_one(runner)
        with patch.object(runner, 'build_plan', return_value=[{'status': 'complete'}]) as build:
            study.require_phase_one(runner)
            self.assertEqual(build.call_count, 2)

    def test_speaker_aggregation_keeps_protocol_label_and_pending_values(self):
        with isolated_runner(self.manifest):
            root = runner.result_root(study.MODELS[0])
            for seed, value in [(42, .91), (123, .93)]:
                directory = root / 'SHD' / f'seed_{seed}'
                metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
                metrics.parent.mkdir(parents=True)
                metrics.write_text(f'test_acc\n{value}\n', encoding='utf-8')
                (directory / 'completed.txt').touch()
            runner.write_aggregate(study.MODELS[0])
            text = (root / 'RESULTS.md').read_text(encoding='utf-8')
            self.assertIn('| SHD speaker holdout v1 | 91.00 | 93.00 | pending | 2 | 92.00 ± 1.41 |', text)
            self.assertNotIn('| SHD* |', text)

    def test_shared_root_lock_excludes_second_launcher(self):
        from extended_seed_study import study_lock
        with tempfile.TemporaryDirectory() as root:
            with study_lock(Path(root)):
                with self.assertRaisesRegex(ValueError, 'Another launcher'):
                    with study_lock(Path(root)):
                        self.fail('overlapping launcher acquired the same lock')
            with study_lock(Path(root)):
                pass  # A stale file does not hold the OS lock.

    def test_legacy_runner_and_protected_output_guard(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as temporary:
            with patch.object(runner, 'RUN_LOGDIR', Path(temporary).relative_to(study.ROOT).as_posix()):
                cfg = runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))[0]['config']
                self.assertEqual(cfg['dataset']['validate_on'], .2)
                self.assertEqual(cfg['dataset']['cache_namespace'], 'controlled_v2')
                self.assertNotIn('speaker_split_fingerprint', cfg['dataset'])
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(['--models', study.MODELS[0], '--logdir', study.OUTPUT_ROOT, '--dry-run'])

    def test_dry_run_writes_nothing_and_restores_runner_globals(self):
        original = runner.EVALUATION, runner.SPEAKER_HOLDOUT
        with isolated_runner(self.manifest) as root:
            with patch.object(study, 'preflight_data', return_value=self.manifest), redirect_stdout(io.StringIO()):
                study.main(['--shd-speaker-phase', 'all', '--dry-run'], runner)
            self.assertFalse(any(root.iterdir()))
        self.assertEqual((runner.EVALUATION, runner.SPEAKER_HOLDOUT), original)

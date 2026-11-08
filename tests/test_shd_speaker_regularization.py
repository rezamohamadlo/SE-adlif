"""Non-training checks: queue order, effective configs, completion/resume and guards."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import argparse
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from datasets import shd_speaker_split as split
import extended_seed_study as evidence
from project_paths import portable_path
import runner
import shd_speaker_holdout as speaker
import shd_speaker_regularization as queue


class RegularizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=speaker.ROOT, prefix='regularization-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        output = portable_path(self.root / 'study')
        for target, key in ((speaker, 'REGULARIZATION_OUTPUT_ROOT'), (queue, 'OUTPUT_ROOT')):
            context = patch.object(target, key, output)
            context.start()
            self.addCleanup(context.stop)
        self.manifest = split.split_metadata(np.repeat(np.arange(10), 20), np.tile(np.arange(20), 10))
        self.manifest['fingerprint'] = split.digest(self.manifest)
        self.source = dict(git_commit='synthetic', tracked_diff_sha256='synthetic', source_files_sha256='synthetic')
        context = patch.object(evidence, 'source_identity', return_value=self.source)
        context.start()
        self.addCleanup(context.stop)

    def plan(self, profile=queue.PROFILES[0], model=speaker.MODELS[0], seed=42):
        with speaker.selected_profile(profile), speaker.configured_runner(runner, self.manifest):
            return runner.build_plan(model, seeds=(seed,), datasets=('SHD',))[0]

    def test_order_is_seed42_all_variants_then_remaining_seeds_per_variant(self):
        jobs = queue.ordered_jobs('all')
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len(set(jobs)), 18)
        self.assertEqual(jobs, queue.ordered_jobs('1') + queue.ordered_jobs('2'))
        self.assertEqual(jobs[:6], [(p, m, 42) for p in queue.PROFILES for m in speaker.MODELS])
        self.assertEqual(jobs[6:10], [(queue.PROFILES[0], m, seed)
                                     for seed in (123, 456) for m in speaker.MODELS])

    def test_effective_configs_only_change_one_regularizer_and_defer_test(self):
        before = (runner.RUN_LOGDIR, runner.SPEAKER_HOLDOUT, runner.VALIDATION_ONLY, speaker.ACTIVE_PROFILE)
        for profile, model, seed in queue.ordered_jobs('all'):
            with self.subTest(profile=profile, model=model, seed=seed):
                item = self.plan(profile, model, seed)
                cfg = item['config']
                evidence.validate_config(cfg)
                self.assertFalse(cfg['run_test'])
                self.assertTrue(cfg['validation_only'])
                self.assertEqual((cfg['n_epochs'], cfg['batch_size'], cfg['lr']), (300, 256, .01))
                self.assertFalse(cfg['early_stopping'])
                self.assertEqual(cfg['paper_baseline_version'], 'paper_classification_v1')
                self.assertEqual(cfg['dataset']['random_seed'], 42)
                self.assertEqual(cfg['random_seed'], seed)
                augmentation = cfg['dataset']['train_augmentation']
                self.assertEqual(augmentation['time_scale'], [.9, 1.1])
                self.assertEqual(augmentation['channel_shift'], 2)
                self.assertEqual(cfg['dropout'], .30 if profile == queue.PROFILES[0] else
                                 .35 if profile == queue.PROFILES[1] else .25)
                self.assertEqual(augmentation['spike_delete_probability'],
                                 .10 if profile == queue.PROFILES[2] else .05)
                self.assertEqual(cfg['dataset']['cache_namespace'], speaker.AUGMENTED_PROTOCOL)
                if model == speaker.MODELS[1]:
                    self.assertTrue(cfg['l1']['local_slow_enabled'])
                    self.assertTrue(cfg['l2']['local_transient_enabled'])
        self.assertEqual(before, (runner.RUN_LOGDIR, runner.SPEAKER_HOLDOUT,
                                  runner.VALIDATION_ONLY, speaker.ACTIVE_PROFILE))

    def test_config_mutation_test_leakage_and_foreign_resume_fail(self):
        cfg = self.plan()['config']
        for key, value in [('dropout', .4), ('lr', .1), ('run_test', True),
                           ('validation_only', False), ('init_ckpt_path', 'foreign.ckpt'),
                           ('ckpt_path', 'foreign/last.ckpt')]:
            changed = copy.deepcopy(cfg)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate_config(changed)
        changed = copy.deepcopy(cfg)
        changed['dataset']['train_augmentation']['channel_shift'] = 4
        with self.assertRaises(ValueError):
            evidence.validate_config(changed)
        directory = self.plan()['directory']
        directory.mkdir(parents=True)
        (directory / 'TEST_RESULTS.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Test artifacts'):
            self.plan()

    def test_actual_dry_run_dispatch_is_read_only_and_ordered(self):
        baseline = [(model, dict(status='complete', seed=seed))
                    for model in speaker.MODELS for seed in queue.SEEDS]
        with patch.object(speaker, 'preflight_data', return_value=self.manifest), \
                patch.object(queue, 'baseline_plans', return_value=baseline), \
                patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()) as output:
            runner.main(['--shd-speaker-phase', 'all', '--shd-speaker-profile',
                         'regularization_v1', '--dry-run'])
        launch.assert_not_called()
        self.assertFalse(any(self.root.iterdir()))
        text = output.getvalue()
        positions = [text.index(f'{p} / {m} / seed {seed}:') for p, m, seed in queue.ordered_jobs('all')]
        self.assertEqual(positions, sorted(positions))

    def test_queue_stops_on_failure_and_phase_two_requires_seed42(self):
        baseline = [(model, dict(status='complete', seed=seed))
                    for model in speaker.MODELS for seed in queue.SEEDS]
        args = SimpleNamespace(shd_speaker_phase='1', measure_existing_costs=False, dry_run=False)
        with patch.object(speaker, 'preflight_data', return_value=self.manifest), \
                patch.object(queue, 'baseline_plans', return_value=baseline), \
                patch.object(evidence, 'study_lock', side_effect=lambda _: nullcontext()), \
                patch.object(queue, 'write_summary'), \
                patch.object(runner, 'run', side_effect=[None, RuntimeError('synthetic failure')]) as launch, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            queue.run_queue(args, argparse.ArgumentParser(), runner)
        self.assertEqual(launch.call_count, 2)
        args.shd_speaker_phase = '2'
        with patch.object(speaker, 'preflight_data', return_value=self.manifest), \
                patch.object(queue, 'baseline_plans', return_value=baseline), \
                patch.object(evidence, 'study_lock', side_effect=lambda _: nullcontext()), \
                patch.object(runner, 'run') as launch, redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            queue.run_queue(args, argparse.ArgumentParser(), runner)
        launch.assert_not_called()

    def test_summary_waits_for_all_seeds_and_selects_by_validation_including_baseline(self):
        root = self.root / 'study'
        root.mkdir()
        scores = dict(zip(queue.PROFILES, (.89, .91, .88)))
        def record(directory):
            return dict(best_validation_accuracy=scores[directory.parents[2].name]), {}
        with patch.object(queue, 'baseline_validation', return_value=.90), \
                patch.object(evidence, 'read_completed', side_effect=record):
            queue.write_summary(self.manifest)
            self.assertFalse((root / 'VALIDATION_SELECTION.json').exists())
            for profile, model, seed in queue.ordered_jobs('all'):
                directory = root / profile / model / 'SHD' / f'seed_{seed}'
                directory.mkdir(parents=True)
                (directory / 'validation_completed.txt').touch()
            queue.write_summary(self.manifest)
        saved = json.loads((root / 'VALIDATION_SELECTION.json').read_text())
        for model in speaker.MODELS:
            self.assertEqual(saved['selections'][model]['profiles'], [queue.PROFILES[1]])
        self.assertIn('augmented_v1', (root / 'VALIDATION_COMPARISON.md').read_text())

    def test_completion_full_resume_split_copy_and_portability_without_training(self):
        item = self.plan()
        cfg, directory = item['config'], item['directory']
        directory.mkdir(parents=True)
        (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps(cfg), encoding='utf-8')
        split.save_manifest(cfg['dataset']['speaker_split_manifest'], self.manifest)
        with patch('torch.cuda.device_count', return_value=0):
            evidence.write_provenance(directory, speaker.ROOT)
        self.assertEqual(json.loads((directory / 'SPLIT_MANIFEST.json').read_text()), self.manifest)
        (directory / 'ckpt').mkdir()
        best = directory / 'ckpt/best.ckpt'
        checkpoint = dict(epoch=10, hyper_parameters={'cfg': cfg}, optimizer_states=[{'state': {}}],
                          lr_schedulers=[{}], callbacks={'selection': dict(
                              monitor='val_acc', best_model_path=portable_path(best), best_model_score=.8)})
        torch.save(checkpoint, best)
        torch.save(checkpoint, directory / 'ckpt/last.ckpt')
        metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
        metrics.parent.mkdir(parents=True)
        metrics.write_text('epoch,val_acc\n10,0.8\n')
        self.assertEqual(self.plan()['status'], 'resume')
        resumed = dict(cfg, ckpt_path=portable_path(directory / 'ckpt/last.ckpt'))
        previous = Path.cwd()
        try:
            os.chdir(self.root)  # Hydra-style CWD changes must not change artifact identity.
            evidence.validate_config(resumed)
            for epoch in range(11, 300):
                with metrics.open('a') as stream:
                    stream.write(f'{epoch},0.7\n')
            checkpoint['epoch'] = 299
            torch.save(checkpoint, directory / 'ckpt/last.ckpt')
            trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(
                best_model_path=portable_path(best), best_model_score=.8))
            evidence.finish_validation(OmegaConf.create(cfg), trainer)
            record, _ = evidence.read_completed(directory)
            self.assertEqual(record['completed_epochs'], 300)
            self.assertEqual(record['selected_epoch'], 10)
            self.assertEqual(self.plan()['status'], 'complete')
            # An explicit test field is forbidden even after validation completion.
            metrics.write_text('epoch,val_acc,test_acc\n10,0.8,\n299,0.7,\n300,,0.9\n')
            with self.assertRaisesRegex(ValueError, 'Test metrics'):
                evidence.read_completed(directory)
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

"""Non-training checks for speaker-disjoint augmentation, caching and resume."""
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from datasets.shd import SHDLDM
from datasets.shd_augmentation import SHDFrameAugmentation
from datasets import shd_speaker_split as split
import runner
import shd_speaker_holdout as study
from tests import test_shd_speaker_holdout as legacy


def augmentation(**overrides):
    settings = dict(version=SHDFrameAugmentation.VERSION, time_scale=[.9, 1.1],
                    channel_shift=2, spike_delete_probability=.05)
    return SHDFrameAugmentation(**dict(settings, **overrides))


class FrameAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.frames = torch.zeros(250, 140)
        self.frames[15:200:3, 65:70] = 3
        self.blocks = torch.ones(250, dtype=torch.long)
        self.blocks[:10] = 0

    def test_time_rebin_preserves_spikes_and_regenerates_aligned_blocks(self):
        original = self.frames.clone()
        for scale in (.9, 1., 1.1):
            with self.subTest(scale=scale):
                result, target, blocks = augmentation(
                    time_scale=[scale, scale], channel_shift=0,
                    spike_delete_probability=0)(self.frames, np.int64(7), self.blocks)
                self.assertEqual(result.sum(), self.frames.sum())
                self.assertTrue(torch.equal(result, result.round()))
                self.assertEqual(result.shape[1], 140)
                self.assertEqual(len(result), len(blocks))
                self.assertGreaterEqual(len(result), 250)
                self.assertTrue((blocks[:10] == 0).all())
                self.assertTrue((blocks[10:] == 1).all())
                self.assertEqual(target, 7)
        self.assertTrue(torch.equal(self.frames, original))

    def test_channel_shift_discards_boundary_spikes_without_wraparound(self):
        self.frames.zero_()
        self.frames[40, -1] = 5
        with patch('datasets.shd_augmentation.torch.randint', return_value=torch.tensor(2)):
            output, _, _ = augmentation(time_scale=[1., 1.], spike_delete_probability=0)(
                self.frames, 0, self.blocks)
        self.assertEqual(output.sum(), 0)
        self.assertEqual(self.frames.sum(), 5)

    def test_spike_deletion_uses_counts_and_preserves_labels(self):
        counts = torch.full((250, 140), 4.)
        torch.manual_seed(42)
        result, target, _ = augmentation(time_scale=[1., 1.], channel_shift=0)(
            counts, np.int64(17), self.blocks)
        self.assertEqual(target, 17)
        self.assertTrue(((result >= 0) & (result <= counts)).all())
        self.assertTrue(torch.equal(result, result.round()))
        self.assertAlmostEqual(float(result.sum() / counts.sum()), .95, delta=.005)
        deleted, _, _ = augmentation(spike_delete_probability=1)(self.frames, 0, self.blocks)
        self.assertEqual(deleted.sum(), 0)

    def test_training_seed_reproducibility_and_fresh_draws(self):
        transform = augmentation()
        torch.manual_seed(42)
        first = transform(self.frames, 0, self.blocks)[0]
        second = transform(self.frames, 0, self.blocks)[0]
        torch.manual_seed(42)
        repeated = transform(self.frames, 0, self.blocks)[0]
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, second))

    def test_invalid_settings_and_noncount_input_fail(self):
        for settings in (dict(version='other'), dict(time_scale=[0, 1]),
                         dict(time_scale=[1, float('nan')]), dict(channel_shift=-1),
                         dict(channel_shift=True), dict(spike_delete_probability=1.1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                augmentation(**settings)
        with self.assertRaisesRegex(ValueError, 'count frames'):
            augmentation()(self.frames + .1, 0, self.blocks)


class AugmentedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = split.split_metadata(np.repeat(np.arange(10), 20), np.tile(np.arange(20), 10))
        cls.manifest['fingerprint'] = split.digest(cls.manifest)

    def test_six_matched_jobs_have_new_protocol_and_only_intended_deviations(self):
        with study.selected_profile('augmented_v1'), legacy.isolated_runner(self.manifest):
            for model, seed in study.ordered_jobs('all'):
                cfg = runner.build_plan(model, seeds=(seed,), datasets=('SHD',))[0]['config']
                self.assertEqual(cfg['evaluation_protocol'], study.AUGMENTED_PROTOCOL)
                self.assertEqual(cfg['dropout'], .25)
                self.assertEqual((cfg['lr'], cfg['batch_size'], cfg['n_epochs']), (.01, 256, 300))
                self.assertFalse(cfg['early_stopping'])
                self.assertEqual(cfg['random_seed'], seed)
                self.assertEqual(cfg['dataset']['random_seed'], 42)
                self.assertEqual(cfg['dataset']['cache_namespace'], study.AUGMENTED_PROTOCOL)
                self.assertEqual(cfg['dataset']['speaker_split_fingerprint'], self.manifest['fingerprint'])
                self.assertEqual(cfg['dataset']['train_augmentation']['spike_delete_probability'], .05)
                self.assertEqual(cfg['paper_baseline_version'], 'paper_classification_v1')
                self.assertTrue(any('dropout=0.25' in d for d in cfg['paper_baseline_deviations']))
                for key, value in [('dropout', .15), ('lr', .1)]:
                    changed = copy.deepcopy(cfg)
                    changed[key] = value
                    with self.assertRaisesRegex(ValueError, 'training settings differ'):
                        study.validate_config(changed, model, seed, self.manifest)
                changed = copy.deepcopy(cfg)
                changed['dataset']['train_augmentation']['channel_shift'] = 4
                with self.assertRaisesRegex(ValueError, 'training settings differ'):
                    study.validate_config(changed, model, seed, self.manifest)

    def test_new_protocol_refuses_reusing_a_different_config(self):
        with study.selected_profile('augmented_v1'), legacy.isolated_runner(self.manifest):
            item = runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))[0]
            directory, cfg = item['directory'], item['config']
            directory.mkdir(parents=True)
            changed = copy.deepcopy(cfg)
            changed['dataset']['train_augmentation']['spike_delete_probability'] = .1
            (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                runner.build_plan(study.MODELS[0], seeds=(42,), datasets=('SHD',))

    def test_full_resume_requires_own_augmented_checkpoint(self):
        with study.selected_profile('augmented_v1'):
            legacy.SpeakerRunnerTests.test_checkpoint_identity_full_resume_and_completed_budget(self)

    def test_dry_run_is_ordered_without_writes_or_subprocesses(self):
        with study.selected_profile('augmented_v1'), legacy.isolated_runner(self.manifest) as root:
            with patch.object(study, 'preflight_data', return_value=self.manifest), \
                    patch.object(study, 'selected_profile', return_value=nullcontext()), \
                    patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()) as output:
                study.main(['--shd-speaker-phase', 'all', '--shd-speaker-profile', 'augmented_v1',
                            '--dry-run'], runner)
            self.assertFalse(any(root.iterdir()))
            launch.assert_not_called()
            text = output.getvalue()
            positions = [text.index(f'{model} seed {seed}:') for model, seed in study.ordered_jobs('all')]
            self.assertEqual(positions, sorted(positions))

    def test_default_profile_and_runner_are_restored(self):
        original = study.OUTPUT_ROOT, study.PROTOCOL, runner.EVALUATION, runner.SPEAKER_HOLDOUT
        with study.selected_profile('augmented_v1'), study.configured_runner(runner, self.manifest):
            self.assertIn(study.AUGMENTED_PROTOCOL, runner.RUN_LOGDIR)
        self.assertEqual((study.OUTPUT_ROOT, study.PROTOCOL, runner.EVALUATION, runner.SPEAKER_HOLDOUT), original)


class AugmentedLoaderTests(unittest.TestCase):
    def test_only_training_augmented_after_cache_with_portable_manifest(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as temporary:
            root = Path(temporary)
            legacy.fixture(root)
            # Give every synthetic recording the same nonempty count pattern.
            for name in ('shd_train.h5', 'shd_test.h5'):
                with h5py.File(root / 'SHDWrapper' / name, 'a') as stream:
                    for index in range(100):
                        stream['spikes/times'][index] = np.linspace(.01, .9, 300)
                        stream['spikes/units'][index] = np.full(300, 350.)
            manifest = split.build_manifest(root)
            path = root / 'split.json'
            split.save_manifest(path, manifest)
            from project_paths import portable_path
            kwargs = dict(data_path=portable_path(root), validate_on='speaker_holdout', random_seed=42,
                          spatial_factor=.2, window_size=4, pad_to_min_size=250,
                          cache_namespace=study.AUGMENTED_PROTOCOL,
                          speaker_split_manifest=portable_path(path),
                          speaker_split_fingerprint=manifest['fingerprint'],
                          train_augmentation=dict(version=SHDFrameAugmentation.VERSION,
                                                  time_scale=[.9, 1.1], channel_shift=2,
                                                  spike_delete_probability=.05), num_workers=0)
            original_cwd = Path.cwd()
            try:
                os.chdir(root)  # Simulate Hydra changing working directory.
                module = SHDLDM(**kwargs)
                train = module.train_dataloader().dataset
                valid = module.val_dataloader().dataset
                test = module.test_dataloader().dataset
                self.assertIsNotNone(train.full_transform)
                self.assertIsNone(valid.full_transform)
                self.assertIsNone(test.full_transform)
                torch.manual_seed(42)
                first = train[0]
                cache_path = Path(module.cache_path) / 'train/0_0'
                cached_before = cache_path.read_bytes()
                second = train[0]
                self.assertFalse(torch.equal(first[0], second[0]))
                self.assertEqual(cache_path.read_bytes(), cached_before)
                original_frames = train.dataset[0][0]
                self.assertTrue(torch.equal(torch.load(cache_path, weights_only=False)[0], original_frames))
                for dataset in (valid, test):
                    a, b = dataset[0], dataset[0]
                    self.assertTrue(torch.equal(a[0], b[0]))
                    self.assertEqual(a[1], b[1])
                self.assertFalse(set(module.data_train.indices) & set(module.data_val.indices))
                cfg = dict(kwargs, cache_namespace=split.PROTOCOL)
                cfg.pop('train_augmentation')
                ordinary = SHDLDM(**cfg)
                self.assertNotEqual(ordinary.cache_path, module.cache_path)
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()

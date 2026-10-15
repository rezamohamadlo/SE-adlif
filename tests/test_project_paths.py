import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from project_paths import (ROOT, portable_path, local_path, portable_config,
                           relocate_checkpoint_callbacks)
import runner
import tune_shd

OLD_ROOT = 'C:/Users/jafari.h.SPADANACO/Desktop/SE-adlif'


class ProjectPathTests(unittest.TestCase):
    def test_registered_roots_preserve_full_checkpoint_identity(self):
        suffix = 'results/valid/reference/SHD/seed_42/hydra/2026-10-12/07-17-03/ckpt/epoch=162-step=1956.ckpt'
        old = (OLD_ROOT + '/' + suffix).replace('/', '\\')
        self.assertEqual(portable_path(old), suffix)
        self.assertEqual(local_path(old), ROOT / suffix)
        self.assertEqual(portable_path(ROOT / suffix), suffix)
        self.assertEqual(portable_path('./' + suffix), suffix)
        self.assertEqual(portable_path(OLD_ROOT.lower() + '/' + suffix), suffix)

    def test_missing_file_and_unknown_root_do_not_fall_back_to_same_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'last.ckpt').touch()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing artifact'):
                local_path(OLD_ROOT + '/different/last.ckpt', root, must_exist=True)
        external = 'Z:/another-repository/checkpoint.ckpt'
        self.assertEqual(portable_path(external), external)
        with self.assertRaises(ValueError):
            portable_path(OLD_ROOT + '/../outside.ckpt')
        with self.assertRaises(ValueError):
            portable_path('C:relative.ckpt')

    def test_relative_paths_ignore_hydra_working_directory(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            current = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(local_path('data'), ROOT / 'data')
            finally:
                os.chdir(current)

    def test_normalization_does_not_modify_records_or_training_settings(self):
        source = {'config': {'init_ckpt_path': OLD_ROOT + '/weights/a.ckpt',
                             'dataset': {'data_path': OLD_ROOT + '/data'}, 'lr': .01,
                             'hydra': {'run': {'dir': OLD_ROOT + '/results/a'}}},
                  'reference': {'checkpoint': OLD_ROOT + '/weights/a.ckpt', 'sha256': 'abc'},
                  'note': OLD_ROOT + '/historical text'}
        original = copy.deepcopy(source)
        normalized = portable_config(source)
        self.assertEqual(source, original)
        self.assertEqual(normalized['config']['init_ckpt_path'], 'weights/a.ckpt')
        self.assertEqual(normalized['config']['hydra']['run']['dir'], 'results/a')
        self.assertEqual(normalized['reference']['sha256'], 'abc')
        self.assertEqual(normalized['note'], original['note'])

    def test_callback_relocation_keeps_scores_and_tensor_state_untouched(self):
        state_dict = {'weight': object()}
        saved = {'state_dict': state_dict, 'callbacks': {'ModelCheckpoint': {
            'dirpath': OLD_ROOT + '/results/a/ckpt',
            'best_model_path': OLD_ROOT + '/results/a/ckpt/best.ckpt',
            'best_k_models': {OLD_ROOT + '/results/a/ckpt/best.ckpt': .9}}}}
        relocate_checkpoint_callbacks(saved)
        self.assertIs(saved['state_dict'], state_dict)
        callback = saved['callbacks']['ModelCheckpoint']
        self.assertEqual(callback['dirpath'], str(ROOT / 'results/a/ckpt'))
        self.assertEqual(callback['best_k_models'], {str(ROOT / 'results/a/ckpt/best.ckpt'): .9})

    def test_sweep_guard_accepts_relocation_but_rejects_changed_hash_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            saved = {'reference': {'checkpoint': OLD_ROOT + '/weights/a.ckpt', 'sha256': 'abc'},
                     'config': {'init_ckpt_path': OLD_ROOT + '/weights/a.ckpt', 'lr': .01}}
            manifest = folder / 'SWEEP_CONFIG.json'
            manifest.write_text(json.dumps(saved))
            portable = portable_config(saved)
            self.assertEqual(tune_shd.check_directory(folder, portable), 'new')
            self.assertEqual(json.loads(manifest.read_text()), saved)
            for changed in [dict(portable['reference'], sha256='different'),
                            dict(portable['reference'], checkpoint='weights/b.ckpt')]:
                altered = copy.deepcopy(portable)
                altered['reference'] = changed
                with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                    tune_shd.check_directory(folder, altered)

    def test_runner_relocated_manifest_and_checkpoint_keep_training_guards(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            shutil.copytree(ROOT / 'config', Path(directory) / 'config')
            item = runner.build_plan('LG_DTH_SE_adLIF', seeds=(123,), datasets=('SHD',))[0]
            self.assertFalse(Path(item['config']['logdir']).is_absolute())
            saved = copy.deepcopy(item['config'])
            saved['logdir'] = OLD_ROOT + '/' + saved['logdir']
            saved['dataset']['data_path'] = OLD_ROOT + '/data'
            (item['directory'] / 'ckpt').mkdir(parents=True)
            (item['directory'] / 'ckpt/last.ckpt').touch()
            manifest = item['directory'] / 'FULL_RUN_CONFIG.json'
            manifest.write_text(json.dumps(saved))
            plan = runner.build_plan('LG_DTH_SE_adLIF', seeds=(123,), datasets=('SHD',))[0]
            self.assertEqual(plan['status'], 'resume')
            self.assertEqual(plan['compatibility']['original_config'], saved)
            self.assertEqual(json.loads(manifest.read_text()), saved)
            saved['dataset']['random_seed'] = 123
            manifest.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                runner.build_plan('LG_DTH_SE_adLIF', seeds=(123,), datasets=('SHD',))


if __name__ == '__main__':
    unittest.main()

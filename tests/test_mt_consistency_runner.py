import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

import runner
from measure_shd_costs import ScalarOperations, batch_scalar_operations, operation_profiles
from models.pl_module import MLPSNN, build_training_model
from models.ours.mt_consistency import MTConsistencyMLPSNN
from project_paths import portable_config
from tests.test_measure_shd_costs import tiny_model


class ConsistencyRunnerTests(unittest.TestCase):
    def test_nine_jobs_match_mt_and_child_overrides_preserve_objective_metadata(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'results') as temporary:
            output = Path(temporary).relative_to(runner.ROOT).as_posix()
            with patch.object(runner, 'RUN_LOGDIR', output):
                plan = runner.build_plan('MT_TC_SE_adLIF')
                baseline = runner.build_plan('MT_SE_adLIF')
        self.assertEqual(len(plan), 9)
        self.assertEqual({p['seed'] for p in plan}, {42, 123, 456})
        for item, original in zip(plan, baseline):
            cfg = item['config']
            for key in ['dataset', 'l1', 'l2', 'l_out', 'lr', 'batch_size', 'n_epochs',
                        'early_stopping', 'scheduler', 'dropout', 'loss_agg', 'paper_baseline_version']:
                self.assertEqual(cfg[key], original['config'][key], key)
            self.assertTrue(cfg['run_test'])
            self.assertIn('MT_TC_SE_adLIF', item['directory'].parts)
            self.assertTrue(cfg['evaluation_protocol'].endswith('_mt_temporal_consistency_v1'))
            self.assertIn('temporal_consistency.weight=0.1', cfg['paper_baseline_deviations'])
            with initialize_config_dir(config_dir=str(runner.ROOT / 'config'), version_base=None):
                child = compose(config_name='main', overrides=item['overrides'])
            self.assertEqual(portable_config(OmegaConf.to_container(child, resolve=True)), cfg)

    def test_factory_is_opt_in_and_inference_and_cost_measurement_match_mt(self):
        ordinary = tiny_model('mt_se_adlif')
        cfg = OmegaConf.create(OmegaConf.to_container(ordinary.hparams.cfg, resolve=True))
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = build_training_model(cfg)
            self.assertIs(type(baseline), MLPSNN)
            cfg.training_objective = 'mt_temporal_consistency_v1'
            cfg.temporal_consistency = dict(weight=.1, window=5)
            variant = build_training_model(cfg).eval()
        self.assertIsInstance(variant, MTConsistencyMLPSNN)
        variant.load_state_dict(ordinary.state_dict(), strict=True)
        inputs = torch.randn(3, 7, 4)
        torch.testing.assert_close(variant(inputs), ordinary(inputs), rtol=0, atol=0)
        with torch.inference_mode():
            profiles = operation_profiles(variant)
            with ScalarOperations() as count:
                variant(inputs)
            self.assertEqual(count.counts, batch_scalar_operations(profiles, 3, 7))
        cfg.training_objective = 'typo'
        with self.assertRaises(ValueError):
            build_training_model(cfg)

    def test_resume_uses_own_last_checkpoint_and_changed_objective_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(runner.ROOT / 'config', root / 'config')
            with patch.object(runner, 'ROOT', root), patch.object(
                    runner, 'RUN_LOGDIR', 'results/experiments/mt_consistency_runner_v1'):
                item = runner.build_plan('MT_TC_SE_adLIF', seeds=(42,), datasets=('SHD',))[0]
                directory = item['directory']
                directory.mkdir(parents=True)
                (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps(item['config']))
                (directory / 'ckpt').mkdir()
                (directory / 'ckpt/last.ckpt').touch()
                resumed = runner.build_plan('MT_TC_SE_adLIF', seeds=(42,), datasets=('SHD',))[0]
                self.assertEqual(resumed['status'], 'resume')
                self.assertTrue(any('last.ckpt' in option for option in resumed['overrides']))
                config = root / 'config/experiment/SHD_MT_TC_SE_adLIF.yaml'
                config.write_text(config.read_text().replace('weight: 0.1', 'weight: 0.2'))
                with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                    runner.build_plan('MT_TC_SE_adLIF', seeds=(42,), datasets=('SHD',))


if __name__ == '__main__':
    unittest.main()

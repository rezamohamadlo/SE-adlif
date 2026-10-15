from pathlib import Path
import unittest
import shutil
import tempfile
from unittest.mock import patch

from hydra import compose, initialize_config_dir


class PaperConfigurationTests(unittest.TestCase):
    def test_single_runner_paper_preflight(self):
        import runner
        captured = []
        def inspect_run(*args, **kwargs):
            captured.extend(runner.build_plan(*args, **kwargs))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(runner.ROOT / 'config', root / 'config')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'run', side_effect=inspect_run):
                runner.main(['--models', 'DTH_SE_adLIF',
                             '--datasets', 'SHD', '--seeds', '42', '123', '456'])
        self.assertEqual([item['seed'] for item in captured], [42, 123, 456])
        for item in captured:
            cfg = item['config']
            self.assertEqual(cfg['batch_size'], 256)
            self.assertEqual(cfg['dataset']['batch_size'], 256)
            self.assertEqual(cfg['n_epochs'], 300)
            self.assertFalse(cfg['early_stopping'])
            self.assertEqual(cfg['paper_baseline_version'], 'paper_classification_v1')
            self.assertIn('protocol_v2', str(item['directory']))

    def test_dth_matches_paper_training_and_preprocessing(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'config'), version_base=None):
            reference = compose(config_name='main', overrides=['experiment=SHD_SE_adLIF_paper'])
            proposed = compose(config_name='main', overrides=['experiment=SHD_DTH_SE_adLIF_paper'])
        for key in ('dataset', 'batch_size', 'lr', 'n_epochs', 'early_stopping',
                    'scheduler', 'patience', 'factor', 'gradient_clip_val', 'dropout',
                    'loss_agg', 'tracking_metric', 'tracking_mode', 'paper_baseline_version'):
            self.assertEqual(reference[key], proposed[key], key)
        self.assertEqual(proposed.l1.cell, 'dth_se_adlif')
        self.assertEqual(proposed.l2.cell, 'dth_se_adlif')
        self.assertEqual(proposed.paper_baseline_dataset, 'SHD')

    def test_paper_settings_and_original_config_are_isolated(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'config'), version_base=None):
            paper = compose(config_name='main', overrides=['experiment=SHD_SE_adLIF_paper'])
            original = compose(config_name='main', overrides=['experiment=SHD_SE_adLIF'])
        self.assertEqual(paper.batch_size, 256)
        self.assertEqual(paper.dataset.batch_size, 256)
        self.assertEqual(paper.n_epochs, 300)
        self.assertFalse(paper.early_stopping)
        self.assertEqual(paper.lr, .01)
        self.assertEqual(paper.dropout, .15)
        self.assertEqual(paper.paper_baseline_version, 'paper_classification_v1')
        self.assertNotIn('paper_baseline', original)
        paper.paper_baseline.datasets.SHD.batch_size = 128
        self.assertEqual(paper.batch_size, 128)
        self.assertEqual(paper.dataset.batch_size, 128)
        self.assertEqual(paper.dataset.validate_on, .2)
        self.assertEqual(paper.dataset.random_seed, 42)
        self.assertTrue(paper.run_test)
        self.assertIsNone(paper.init_ckpt_path)
        self.assertIsNone(paper.ckpt_path)
        self.assertNotIn('evaluation', paper)
        self.assertEqual(paper.l1, original.l1)
        self.assertEqual(paper.l2, original.l2)
        self.assertEqual(paper.l_out, original.l_out)
        self.assertEqual(original.exp_name, 'SHD_SE_adLIF')


if __name__ == '__main__':
    unittest.main()

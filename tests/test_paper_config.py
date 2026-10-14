from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir


class PaperConfigurationTests(unittest.TestCase):
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

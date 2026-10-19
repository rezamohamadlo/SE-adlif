import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runner


class RunnerSSCMeanTests(unittest.TestCase):
    def test_ssc_mean_is_automatic_and_explicit_flag_remains_compatible(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'results') as temp:
            output = Path(temp).relative_to(runner.ROOT).as_posix()
            protocol = copy.deepcopy(runner.EVALUATION)
            protocol['name'] = 'ssc_mean_loss_v1'
            protocol['datasets']['SSC']['loss_agg'] = 'mean_membrane_potentials'
            with patch.object(runner, 'RUN_LOGDIR', output):
                original = runner.build_plan('DTH_SE_adLIF', seeds=[42], datasets=['SSC'])[0]
                self.assertEqual(original['config']['loss_agg'], 'mean_membrane_potentials')
                self.assertEqual(original['config']['evaluation_protocol'], 'ssc_mean_loss_v1')
                for dataset, loss in [('SHD', 'sum_softmax_over_time'), ('ECG', 'summed_membrane_potentials')]:
                    cfg = runner.build_plan('DTH_SE_adLIF', seeds=[42], datasets=[dataset])[0]['config']
                    self.assertEqual(cfg['loss_agg'], loss)
                    self.assertEqual(cfg['evaluation_protocol'], 'paper_classification_v1')
                with patch.object(runner, 'EVALUATION', protocol), patch.object(runner, 'PROTOCOL', protocol['datasets']):
                    for model in ['DTH_SE_adLIF', 'MT_SE_adLIF']:
                        cfg = runner.build_plan(model, seeds=[42], datasets=['SSC'], resume=0)[0]['config']
                        self.assertEqual(cfg['loss_agg'], 'mean_membrane_potentials')
                        self.assertEqual(cfg['evaluation_protocol'], 'ssc_mean_loss_v1')
                        self.assertEqual((cfg['lr'], cfg['batch_size'], cfg['n_epochs']), (.006, 256, 40))
                        self.assertTrue(cfg['run_test'])
                        self.assertIsNone(cfg['ckpt_path'])
                        self.assertIsNone(cfg['init_ckpt_path'])
                        self.assertFalse(cfg['early_stopping'])

    def test_mean_flag_rejects_other_datasets(self):
        for datasets in [[], ['--datasets', 'SHD'], ['--datasets', 'SSC', 'ECG']]:
            with self.subTest(datasets=datasets), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main(['--models', 'DTH_SE_adLIF', '--ssc-mean-loss', '--dry-run', *datasets])
                self.assertEqual(error.exception.code, 1)


if __name__ == '__main__':
    unittest.main()

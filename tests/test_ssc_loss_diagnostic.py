import csv
from pathlib import Path
import tempfile
import unittest

from ssc_loss_diagnostic import ARMS, resolve_job, validation_result


class SSCLossDiagnosticTests(unittest.TestCase):
    def test_arms_differ_only_in_loss_and_output_metadata(self):
        configs = [resolve_job('results/experiments/test_ssc_loss', arm, 123)[2]
                   for arm in ARMS]
        for cfg in configs:
            self.assertEqual(cfg['paper_baseline_version'], 'paper_classification_v1')
            self.assertEqual((cfg['lr'], cfg['batch_size'], cfg['n_epochs']), (.006, 256, 40))
            self.assertFalse(cfg['run_test'])
            self.assertFalse(cfg['early_stopping'])
            self.assertIsNone(cfg['ckpt_path'])
            self.assertIsNone(cfg['init_ckpt_path'])
            for key in ('loss_agg', 'logdir', 'paper_baseline_deviations'):
                cfg.pop(key)
        self.assertEqual(configs[0], configs[1])

    def metrics(self, root, rows):
        path = root / 'logs/mlp_snn/version_0/metrics.csv'
        path.parent.mkdir(parents=True)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'val_acc', 'val_loss', 'test_acc'])
            writer.writeheader()
            writer.writerows(rows)

    def test_partial_budget_is_not_a_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.metrics(root, [dict(epoch=0, val_acc=.8, val_loss=1)])
            self.assertIsNone(validation_result(root, 40))

    def test_conflicting_history_and_test_evaluation_are_rejected(self):
        for rows, message in [
            ([dict(epoch=0, val_acc=.8), dict(epoch=0, val_acc=.9)], 'Duplicate'),
            ([dict(epoch=0, test_acc=.8)], 'Unexpected test'),
        ]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.metrics(root, rows)
                with self.assertRaisesRegex(ValueError, message):
                    validation_result(root, 40)


if __name__ == '__main__':
    unittest.main()

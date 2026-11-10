import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import search_samt_shd as search

class ContinuationTests(unittest.TestCase):
    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                search.checkpoint_source(Path(temp)/'missing.ckpt')

    def test_plan_is_weights_only_and_source_hashed(self):
        import runner
        old=runner.RUN_LOGDIR
        try:
            runner.RUN_LOGDIR=search.DEFAULT_OUTPUT
            cfg=runner.build_plan('SA_MT_SE_adLIF',seeds=[123],datasets=['SHD'],inspect_existing=False)[0]['config']
        finally:
            runner.RUN_LOGDIR=old
        origin=dict(checkpoint='results/example.ckpt',epoch_index=188,sha256='a'*64)
        with tempfile.TemporaryDirectory() as temp, patch.object(search,'checkpoint_source',return_value=(cfg,origin)):
            plan=search.build_plan(Path(temp)/'out','unused')
            self.assertEqual(len(plan),8)
            for item in plan:
                c=item['manifest']['config']
                self.assertEqual(c['init_ckpt_path'],origin['checkpoint'])
                self.assertIsNone(c['ckpt_path'])
                self.assertFalse(c['early_stopping'])
                self.assertTrue(c['run_test'])
                self.assertEqual(c['n_epochs'],80)
                self.assertEqual(item['manifest']['origin'],origin)
                self.assertIn(c['lr'], (.0004, .0001))
                self.assertEqual(c['scheduler'], 'plateau')
                self.assertEqual(c['tracking_metric'], 'val_acc')
                self.assertEqual(c['tracking_mode'], 'max')
                self.assertEqual(c['factor'], .9)
                self.assertEqual(c['patience'], 10)

import tempfile
import unittest
import json
from pathlib import Path
import confirm_samt_shd as confirm

class ConfirmationTests(unittest.TestCase):
    def test_fresh_configs_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'out'; plan=confirm.build_plan(output)
            self.assertEqual([p['manifest']['config']['random_seed'] for p in plan],[42,123,456])
            self.assertFalse(output.exists())
            for p in plan:
                c=p['manifest']['config']
                self.assertIsNone(c['init_ckpt_path'])
                self.assertIsNone(c['ckpt_path'])
                self.assertEqual(c['n_epochs'],300)
                self.assertFalse(c['early_stopping'])
                self.assertTrue(c['run_test'])
                self.assertEqual(c['dataset']['random_seed'],42)
            d=plan[0]['directory'];(d/'ckpt').mkdir(parents=True)
            (d/'SWEEP_CONFIG.json').write_text(json.dumps(plan[0]['manifest']))
            (d/'ckpt/last.ckpt').touch()
            self.assertEqual(confirm.build_plan(output)[0]['status'],'resume')
            changed=plan[0]['manifest'];changed['config']['lr']=.01
            (d/'SWEEP_CONFIG.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Configuration changed'):
                confirm.build_plan(output)

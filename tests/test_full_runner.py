import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import full_runner


class FullRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(full_runner.ROOT / "config", self.root / "config")
        self.root_patch = patch.object(full_runner, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_all_model_plans(self):
        for model in ("SE_adLIF", "DTH_SE_adLIF", "MT_SE_adLIF"):
            plan = full_runner.build_plan(model)
            self.assertEqual([(x['dataset'], x['seed']) for x in plan],
                             [(d, s) for d in ("SHD", "SSC", "ECG") for s in (42,123,456)])
            self.assertEqual([x['config']['batch_size'] for x in plan[::3]], [512,256,64])
            self.assertEqual([x['config']['n_epochs'] for x in plan[::3]], [300,40,400])
            self.assertEqual([x['config']['early_stopping'] for x in plan[::3]],
                             [True, False, True])
            self.assertTrue(all(x['config']['early_stopping_patience'] == 50 for x in plan))
        with self.assertRaises(ValueError):
            full_runner.build_plan("../bad")
        with self.assertRaises(ValueError):
            full_runner.build_plan("MissingModel")

    def test_resume_skip_and_configuration_guard(self):
        item = full_runner.build_plan("MT_SE_adLIF")[0]
        directory = item['directory']
        directory.mkdir(parents=True)
        manifest = directory / "FULL_RUN_CONFIG.json"
        manifest.write_text(json.dumps(item['config']), encoding='utf-8')
        (directory / 'ckpt').mkdir()
        (directory / 'ckpt' / 'last.ckpt').touch()
        resumed = full_runner.build_plan("MT_SE_adLIF")[0]
        self.assertEqual(resumed['status'], 'resume')
        self.assertTrue(resumed['overrides'][-1].startswith('ckpt_path='))
        (directory / 'completed.txt').touch()
        self.assertEqual(full_runner.build_plan("MT_SE_adLIF")[0]['status'], 'complete')
        manifest.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            full_runner.build_plan("MT_SE_adLIF")

    def test_aggregate_only_completed_seeds(self):
        for seed, value, complete in ((42, .90, True), (123, .92, True), (456, .99, False)):
            directory = self.root / 'results/full_runs/MT_SE_adLIF/SHD' / f'seed_{seed}'
            metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
            metrics.parent.mkdir(parents=True)
            with metrics.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['test_acc'])
                writer.writeheader()
                writer.writerow({'test_acc': value})
            if complete:
                (directory / 'completed.txt').touch()
        full_runner.write_aggregate('MT_SE_adLIF')
        text = (self.root / 'results/full_runs/MT_SE_adLIF/RESULTS.md').read_text(encoding='utf-8')
        self.assertIn('| SHD | 90.00% | 92.00% | pending | 2 | 91.00% ± 1.41% |', text)


if __name__ == '__main__':
    unittest.main()

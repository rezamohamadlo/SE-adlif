import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import contextlib
import io
import torch

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
        for model in ("SE_adLIF", "DTH_SE_adLIF", "MT_SE_adLIF", "DA_SE_adLIF", "MR_SE_adLIF"):
            plan = full_runner.build_plan(model)
            self.assertEqual([(x['dataset'], x['seed']) for x in plan],
                             [(d, s) for d in ("SHD", "SSC", "ECG") for s in (42,123,456)])
            self.assertEqual([x['config']['batch_size'] for x in plan[::3]], [512,256,64])
            self.assertEqual([x['config']['n_epochs'] for x in plan[::3]], [300,40,400])
            self.assertEqual([x['config']['early_stopping'] for x in plan[::3]],
                             [True, False, True])
            self.assertTrue(all(x['config']['early_stopping_patience'] == 50 for x in plan))
            self.assertTrue(all(x['config']['patience'] == 9999 for x in plan))
            self.assertTrue(all(x['config']['evaluation_protocol'] == 'controlled_v2' for x in plan))
            self.assertEqual([x['config']['lr'] for x in plan[::3]], [.01, .006, .01])
            for item in plan[:3]:
                self.assertEqual(item['config']['dataset']['validate_on'], .2)
                self.assertEqual(item['config']['dataset']['random_seed'], 42)
                self.assertEqual(item['config']['dataset']['cache_namespace'], 'controlled_v2')
            for item in plan[6:]:
                self.assertEqual(item['config']['dataset']['random_seed'], 42)
                self.assertEqual(item['config']['dataset']['valid_fraction'], .05)
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

    def test_two_stage_seed_selection_preserves_config_and_paths(self):
        for model in ('DTH_SE_adLIF', 'DA_SE_adLIF', 'MR_SE_adLIF'):
            complete = full_runner.build_plan(model)
            first = full_runner.build_plan(model, seeds=(42, 42))
            second = full_runner.build_plan(model, seeds=(123, 456))
            self.assertEqual(len(first), 3)
            self.assertEqual(len(second), 6)
            self.assertEqual({x['seed'] for x in first}, {42})
            self.assertEqual({x['seed'] for x in second}, {123, 456})
            for item in first + second:
                original = next(x for x in complete if x['directory'] == item['directory'])
                self.assertEqual(item['config'], original['config'])
                self.assertEqual(item['overrides'], original['overrides'])
        for seeds in ((), (0,), (True,)):
            with self.assertRaises(ValueError):
                full_runner.build_plan('DTH_SE_adLIF', seeds=seeds)

    def test_aggregate_only_completed_seeds(self):
        for seed, value, complete in ((42, .90, True), (123, .92, True), (456, .99, False)):
            directory = full_runner.result_root('MT_SE_adLIF') / 'SHD' / f'seed_{seed}'
            metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
            metrics.parent.mkdir(parents=True)
            with metrics.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['test_acc'])
                writer.writeheader()
                writer.writerow({'test_acc': value})
            if complete:
                (directory / 'completed.txt').touch()
        full_runner.write_aggregate('MT_SE_adLIF')
        text = (full_runner.result_root('MT_SE_adLIF') / 'RESULTS.md').read_text(encoding='utf-8')
        self.assertIn('| SHD | 90.00% | 92.00% | pending | 2 | 91.00% ± 1.41% |', text)

    def test_mr_interval_isolation(self):
        first = full_runner.build_plan('MR_SE_adLIF', 2)[0]
        other = full_runner.build_plan('MR_SE_adLIF', 4)[0]
        self.assertNotEqual(first['directory'], other['directory'])
        self.assertIn('K_2', first['directory'].parts)
        for item in full_runner.build_plan('MR_SE_adLIF', 2):
            self.assertEqual(item['config']['l1']['adaptation_update_interval'], 2)
            self.assertEqual(item['config']['l2']['adaptation_update_interval'], 2)
        for k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                full_runner.build_plan('MR_SE_adLIF', k)

    def test_candidate_models_construct_on_each_dataset(self):
        from models.pl_module import MLPSNN
        from omegaconf import OmegaConf
        torch.manual_seed(42)
        for variant in ('SE_adLIF', 'DTH_SE_adLIF', 'DA_SE_adLIF', 'MR_SE_adLIF'):
            for item in full_runner.build_plan(variant)[::3]:
                cfg = OmegaConf.create(item['config'])
                with contextlib.redirect_stdout(io.StringIO()):
                    model = MLPSNN(cfg)
                output = model(torch.randn(2, 14, cfg.l1.input_size)*5)
                self.assertEqual(output.shape, (2, 14, cfg.dataset.num_classes))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(model.l1.weight.grad).all())
                self.assertTrue(torch.isfinite(model.l2.weight.grad).all())


if __name__ == '__main__':
    unittest.main()

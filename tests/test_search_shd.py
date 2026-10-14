import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
import search_shd
import tune_shd


class ScratchSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_nine_fresh_configs_and_original_control_first(self):
        plan = search_shd.build_plan(self.root)
        self.assertEqual(len(plan), 9)
        self.assertEqual(plan[0]['manifest']['candidate']['id'], 'r06')
        self.assertEqual({(item['manifest']['config']['lr'], item['manifest']['config']['batch_size'])
                          for item in plan}, {(lr, batch) for lr in (.005, .01, .015)
                                             for batch in (128, 256, 512)})
        for item in plan:
            cfg = item['manifest']['config']
            self.assertIsNone(cfg['init_ckpt_path'])
            self.assertIsNone(cfg['ckpt_path'])
            self.assertFalse(cfg['run_test'])
            self.assertEqual(cfg['dataset']['batch_size'], cfg['batch_size'])
            self.assertEqual(cfg['dataset']['validate_on'], .2)
            self.assertEqual(cfg['dataset']['random_seed'], 42)
            self.assertEqual(cfg['dataset']['cache_namespace'], 'controlled_v2')
            self.assertEqual(cfg['dropout'], .15)
            self.assertEqual(cfg['patience'], 9999)
            self.assertEqual(cfg['early_stopping_patience'], 50)
            self.assertEqual(cfg['early_stopping_min_delta'], .001)
            self.assertEqual(cfg['n_epochs'], 300)
        confirmed = search_shd.build_plan(self.root, ['r06'], [123, 456])
        self.assertEqual(len(confirmed), 2)
        self.assertEqual([p['manifest']['config']['random_seed'] for p in confirmed], [123, 456])
        self.assertTrue(all(p['manifest']['config']['dataset']['random_seed'] == 42 for p in confirmed))

    def test_duplicate_launch_is_rejected_and_lock_is_released(self):
        with search_shd.search_lock(self.root):
            with self.assertRaisesRegex(ValueError, 'Another search is running'):
                with search_shd.search_lock(self.root):
                    self.fail('Duplicate launcher acquired the lock')
        with search_shd.search_lock(self.root):
            pass

    def test_resume_uses_run_state_and_config_changes_are_rejected(self):
        item = search_shd.build_plan(self.root, ['r06'])[0]
        directory = item['directory']
        directory.mkdir(parents=True)
        (directory/'SWEEP_CONFIG.json').write_text(json.dumps(item['manifest']))
        (directory/'ckpt').mkdir()
        (directory/'ckpt/last.ckpt').touch()
        resumed = search_shd.build_plan(self.root, ['r06'])[0]
        self.assertEqual(resumed['status'], 'resume')
        with patch.object(tune_shd.subprocess, 'check_call'):
            tune_shd.launch(directory, resumed['manifest'], 'resume')
        cfg = OmegaConf.load(directory/'trial.yaml')
        self.assertIsNone(cfg.init_ckpt_path)
        self.assertEqual(cfg.ckpt_path, (directory/'ckpt/last.ckpt').as_posix())
        self.assertEqual(search_shd.build_plan(self.root, ['r06'])[0]['status'], 'complete')
        manifest = item['manifest']
        manifest['config']['lr'] = .2
        with self.assertRaises(ValueError):
            tune_shd.check_directory(directory, manifest)

    def test_validation_ranking_ignores_test_and_keeps_first_best_epoch(self):
        for trial, acc, test in [('r01', .90, .99), ('r06', .95, .80)]:
            item = search_shd.build_plan(self.root, [trial])[0]
            directory = item['directory']
            metrics = directory/'logs/mlp_snn/version_0/metrics.csv'
            metrics.parent.mkdir(parents=True)
            (directory/'SWEEP_CONFIG.json').write_text(json.dumps(item['manifest']))
            (directory/'completed.txt').touch()
            (directory/'TEST_RESULTS.json').write_text(json.dumps(dict(metrics=dict(test_acc=test))))
            with metrics.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['epoch', 'val_acc', 'val_loss'])
                writer.writeheader()
                writer.writerow(dict(epoch=0, val_acc=acc, val_loss=.2))
                writer.writerow(dict(epoch=1, val_acc=acc, val_loss=.1))
            (directory/'TRAINING_SUMMARY.md').write_text('## Resume command\n\n```powershell\nold command\n```\n')
        rows = search_shd.report(self.root)
        self.assertEqual(rows[0]['id'], 'r06')
        self.assertEqual(rows[0]['best_epoch'], 0)
        self.assertEqual(rows[0]['val_loss'], .2)
        self.assertIn('search_shd.py', (self.root/'r06/seed_42/TRAINING_SUMMARY.md').read_text())

    def test_test_evaluation_requires_explicit_completed_trials_and_correct_checkpoint(self):
        with self.assertRaises(SystemExit) as failure:
            search_shd.main(['--output', str(self.root), '--test-selected'])
        self.assertEqual(failure.exception.code, 1)
        with self.assertRaises(SystemExit):
            search_shd.main(['--output', str(self.root), '--test-selected', '--trials', 'r06'])
        directory = self.root/'r06/seed_42'
        (directory/'ckpt').mkdir(parents=True)
        best = directory/'ckpt/epoch=3-step=20.ckpt'
        best.touch()
        torch.save(dict(callbacks={'ModelCheckpoint': dict(best_model_path='E:\\old\\ckpt\\epoch=3-step=20.ckpt')}),
                   directory/'ckpt/last.ckpt')
        with patch.object(search_shd.subprocess, 'check_call') as call:
            search_shd.evaluate(directory)
        self.assertEqual(call.call_args.args[0][-1], str(best))


if __name__ == '__main__':
    unittest.main()

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import runner
import measure_run_costs
from tests.test_measure_shd_costs import tiny_model


class RunnerCostTests(unittest.TestCase):
    def test_completed_checkpoint_measurement_and_cache_without_training(self):
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            directory = Path(tmp)
            model = tiny_model()
            cfg = model.hparams.cfg
            cfg.random_seed = 42
            cfg.device = 'cpu'
            cfg.dataset._target_ = 'datasets.shd.SHDLDM'
            ckpt = directory / 'best.ckpt'
            torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(cfg=cfg), epoch=2), ckpt)
            (directory / 'completed.txt').touch()
            (directory / 'TRAINING_SUMMARY.md').write_text('# Training\nOriginal epoch history\n')
            x = torch.ones(3, 4, 4)
            block = torch.ones(3, 4, dtype=torch.long)
            prediction = model(x).softmax(-1).sum(1).argmax(-1)
            targets = torch.stack([torch.full_like(prediction, -1), prediction], dim=1)
            batch = (x, targets, block, torch.tensor([4, 4, 4]))
            with patch.object(measure_run_costs.shared, 'selected_checkpoint', return_value=ckpt), \
                 patch.object(measure_run_costs, 'prepare_batches', return_value=([batch], 'same-data')) as data, \
                 patch.object(measure_run_costs.shared, 'saved_test_accuracy', return_value=1.0), \
                 contextlib.redirect_stdout(io.StringIO()):
                record = measure_run_costs.collect(directory, 'SHD')
                cached = measure_run_costs.collect(directory, 'SHD')
            self.assertEqual(record, cached)
            data.assert_called_once()
            self.assertTrue((directory / 'COST_RESULTS.md').exists())
            summary = (directory / 'TRAINING_SUMMARY.md').read_text()
            self.assertIn('Original epoch history', summary)
            self.assertIn('Hidden spikes/sample', summary)
            self.assertIn('Dense matrix GFLOPs/sample', summary)
            self.assertEqual(summary.count('<!-- test-costs:start -->'), 1)
            self.assertEqual(record['measurements']['totals']['samples'], 3)
            self.assertEqual(record['measurements']['reproduced_test_acc'], 1)
            self.assertGreater(record['measurements']['arithmetic_gflops_test_padded'],
                               record['measurements']['dense_gflops_test_padded'])

    def test_cost_failure_preserves_completed_training_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            marker = path / 'completed.txt'
            marker.write_text('finished')
            with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                runner.collect_costs(path, 'SSC')
            self.assertEqual(marker.read_text(), 'finished')
            self.assertEqual(json.loads((path / 'COST_ERROR.json').read_text())['status'], 'failed')
            self.assertFalse((path / 'COST_RESULTS.json').exists())

    def test_completed_runs_skipped_unless_backfill_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            item = dict(directory=path, dataset='SHD', seed=42, status='complete')
            with patch.object(runner, 'build_plan', return_value=[item]), \
                 patch.object(runner, 'write_aggregate'), patch.object(runner, 'collect_costs') as costs, \
                 patch.object(runner.subprocess, 'run') as train:
                runner.run('SE_adLIF')
                costs.assert_not_called()
                runner.run('SE_adLIF', measure_existing_costs=True)
                costs.assert_called_once_with(path, 'SHD')
                train.assert_not_called()

    def test_costs_follow_training_completion_and_opt_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'seed_42'
            item = dict(directory=path, dataset='SHD', seed=42, status='new', config={}, overrides=[])
            def completed(*args):
                self.assertTrue((path / 'completed.txt').exists())
            with patch.object(runner, 'build_plan', return_value=[item]), patch.object(runner, 'write_aggregate'), \
                 patch.object(runner, 'collect_costs', side_effect=completed) as costs, \
                 patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
                runner.run('SE_adLIF')
                costs.assert_called_once()
                runner.run('SE_adLIF', measure_costs=False)
                self.assertEqual(costs.call_count, 1)

    def test_aggregate_requires_matching_test_data_and_marks_mixed_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in [42, 123]:
                directory = root / 'SHD' / f'seed_{seed}'
                directory.mkdir(parents=True)
                (directory / 'completed.txt').touch()
                metrics = dict.fromkeys(['total_parameters', 'trainable_parameters', 'spikes_total_per_sample',
                    'spikes_l1_per_sample', 'spikes_l2_per_sample', 'firing_l1_percent', 'firing_l2_percent',
                    'synops_per_sample', 'dense_gflops_test_padded', 'arithmetic_gflops_test_padded',
                    'logical_gops_test_padded', 'forward_ms_per_sample'], 1.)
                record = dict(version='runner_test_costs_v1', status='complete', dataset='SHD', seed=seed,
                              measurements=metrics, data_sha256='same', environment=dict(gpu=str(seed)))
                (directory / 'COST_RESULTS.json').write_text(json.dumps(record))
            text = '\n'.join(runner.cost_aggregate_lines(root))
            self.assertIn('mixed hardware', text)
            record['data_sha256'] = 'different'
            (directory / 'COST_RESULTS.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'different test data'):
                runner.cost_aggregate_lines(root)

    def test_refresh_preserves_provenance_without_duplicate_generated_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / 'RESULTS.md'
            report.write_text('# Old aggregate\n\n## Provenance\nRetain this research evidence.\n')
            with patch.object(runner, 'result_root', return_value=root):
                runner.write_aggregate('SE_adLIF')
                runner.write_aggregate('SE_adLIF')
            text = report.read_text(encoding='utf-8')
            self.assertEqual(text.count('## Provenance'), 1)
            self.assertIn('Retain this research evidence.', text)
            self.assertEqual(text.count('<!-- runner-costs:start -->'), 1)

    def test_dry_run_never_measures_or_starts_training(self):
        with patch.object(runner, 'build_plan', return_value=[]), \
             patch.object(runner, 'run') as train, patch.object(runner, 'collect_costs') as costs, \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main(['--models', 'SE_adLIF', '--dry-run', '--measure-existing-costs'])
            train.assert_not_called()
            costs.assert_not_called()


if __name__ == '__main__':
    unittest.main()

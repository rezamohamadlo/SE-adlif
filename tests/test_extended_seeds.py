"""Infrastructure checks only: no fitting, optimizer steps or official-test inference."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import runner
import project_paths
import extended_seed_study as study
from aggregate_extended_seeds import aggregate, paired_statistics


class ExtendedSeedsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(runner.ROOT / 'config', self.root / 'config')
        paths_patch = patch.object(project_paths, 'ROOT', self.root)
        paths_patch.start()
        self.addCleanup(paths_patch.stop)
        for key, value in [('ROOT', self.root), ('RUN_LOGDIR', 'results/experiments/study'),
                           ('SEED_COHORT', 'extension_v1'), ('VALIDATION_ONLY', True)]:
            p = patch.object(runner, key, value)
            p.start()
            self.addCleanup(p.stop)

    def plan(self, model='MT_SE_adLIF', dataset='ECG', seeds=(1009,)):
        return runner.build_plan(model, seeds=seeds, datasets=(dataset,))

    def completed(self, model='MT_SE_adLIF', dataset='ECG', seed=1009, accuracy=.8):
        item = self.plan(model, dataset, (seed,))[0]
        directory, cfg = item['directory'], item['config']
        (directory / 'ckpt').mkdir(parents=True)
        best = directory / 'ckpt/best.ckpt'
        torch.save({'epoch': 2}, best)
        torch.save({'epoch': cfg['n_epochs'] - 1}, directory / 'ckpt/last.ckpt')
        metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
        metrics.parent.mkdir(parents=True)
        metrics.write_text(f'epoch,val_acc\n1,{accuracy - .1}\n2,{accuracy}\n')
        (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps(cfg))
        (directory / 'LAUNCH_ENVIRONMENTS.json').write_text(json.dumps([
            dict(git_commit='abc', tracked_diff_sha256='clean', python='fixture')]))
        trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path=str(best), best_model_score=accuracy))
        study.finish_validation(OmegaConf.create(cfg), trainer)
        return item

    def test_canonical_and_extended_seeds(self):
        self.assertEqual(runner.SEEDS, (42, 123, 456))
        with patch.object(runner, 'SEED_COHORT', None), patch.object(runner, 'VALIDATION_ONLY', False):
            self.assertEqual(len(runner.build_plan('MT_SE_adLIF', datasets=('ECG',))), 3)
            with self.assertRaises(ValueError):
                self.plan()
        seeds = study.cohort_definition()['additional_seeds']
        self.assertEqual([i['seed'] for i in self.plan(seeds=seeds)], seeds)
        for seed in (1, 7080, True):
            with self.assertRaises(ValueError):
                self.plan(seeds=(seed,))

    def test_pc_plans_and_split_integrity(self):
        for assignment in study.cohort_definition()['assignments'].values():
            items = [i for m in assignment['models'] for i in self.plan(m, assignment['dataset'], study.cohort_definition()['additional_seeds'])]
            self.assertEqual(len(items), 14)
            for i in items:
                cfg = i['config']
                self.assertEqual(cfg['dataset']['random_seed'], 42)
                self.assertFalse(cfg['early_stopping'])
                self.assertFalse(cfg['run_test'])
                self.assertIsNone(cfg['init_ckpt_path'])
                self.assertIn('run_test=false', i['overrides'])
                self.assertEqual(cfg['n_epochs'], 400 if assignment['dataset'] == 'ECG' else 300)
                self.assertEqual(cfg['batch_size'], 64 if assignment['dataset'] == 'ECG' else 256)
                self.assertEqual(cfg['lr'], .01)

    def test_completion_skip_resume_and_manifest_preservation(self):
        item = self.completed()
        directory = item['directory']
        before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        self.assertEqual(self.plan()[0]['status'], 'complete')
        with patch.object(runner.subprocess, 'run') as launch, patch.object(runner, 'collect_costs') as costs:
            runner.run('MT_SE_adLIF', seeds=(1009,), datasets=('ECG',), measure_existing_costs=True)
            launch.assert_not_called()
            costs.assert_not_called()
        self.assertEqual(before, {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()})
        self.assertFalse((directory / 'completed.txt').exists())
        (directory / 'validation_completed.txt').unlink()
        resumed = self.plan()[0]
        self.assertEqual(resumed['status'], 'resume')
        self.assertIn('last.ckpt', resumed['overrides'][-1])
        changed = copy.deepcopy(item['config'])
        changed['dataset']['random_seed'] = 123
        (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'Configuration changed'):
            self.plan()

    def test_unknown_artifacts_and_full_test_markers_rejected(self):
        directory = self.plan()[0]['directory']
        directory.mkdir(parents=True)
        (directory / 'unknown.txt').touch()
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            self.plan()
        (directory / 'completed.txt').touch()
        with self.assertRaisesRegex(ValueError, 'Full-test artifacts'):
            self.plan()

    def test_full_budget_and_metrics_required(self):
        item = self.completed()
        cfg, directory = item['config'], item['directory']
        trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(
            best_model_path=str(directory / 'ckpt/best.ckpt'), best_model_score=.8))
        (directory / 'validation_completed.txt').unlink()
        torch.save({'epoch': 1}, directory / 'ckpt/last.ckpt')
        with self.assertRaisesRegex(RuntimeError, 'full epoch budget'):
            study.finish_validation(OmegaConf.create(cfg), trainer)
        self.assertFalse((directory / 'validation_completed.txt').exists())
        torch.save({'epoch': cfg['n_epochs'] - 1}, directory / 'ckpt/last.ckpt')
        trainer.checkpoint_callback.best_model_score = .9
        with self.assertRaisesRegex(ValueError, 'metrics disagree'):
            study.finish_validation(OmegaConf.create(cfg), trainer)

    def test_guarded_test_call_and_validation_branch_has_no_costs(self):
        source = Path(study.ROOT / 'run.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "cfg.get('run_test', True)"]
        self.assertEqual(len(calls), 1)
        self.assertIn('trainer.test(', ast.unparse(calls[0]))
        cfg = self.plan()[0]['config']
        for key, value in [('run_test', True), ('early_stopping', True), ('init_ckpt_path', 'trained.ckpt')]:
            bad = copy.deepcopy(cfg)
            bad[key] = value
            with self.assertRaises(ValueError):
                study.validate_config(bad)
        item = self.plan()[0]
        with patch.object(runner, 'build_plan', return_value=[item]), \
             patch.object(study, 'write_provenance'), patch.object(study, 'read_completed'), \
             patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
             patch.object(runner, 'collect_costs') as costs, patch.object(runner, 'write_aggregate') as reports:
            runner.run('MT_SE_adLIF', seeds=(1009,), datasets=('ECG',))
            costs.assert_not_called()
            reports.assert_not_called()
        self.assertFalse((item['directory'] / 'completed.txt').exists())

    def test_dry_run_no_writes_and_cli_restrictions(self):
        args = ['--models', 'MT_SE_adLIF', 'SA_MT_SE_adLIF', '--datasets', 'ECG',
                '--seed-cohort', 'extension_v1', '--validation-only',
                '--logdir', 'results/experiments/study', '--dry-run']
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        with patch.object(runner, 'run') as train, contextlib.redirect_stdout(io.StringIO()) as out:
            runner.main(args)
            train.assert_not_called()
        self.assertIn('Validated 14 runs', out.getvalue())
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))
        for extra in (['--measure-existing-costs'], ['--epochs', '1'], ['--early-stopping', 'true']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                runner.main(args + extra)

    def test_paired_aggregation_only_completed_matching_seeds(self):
        self.completed(accuracy=.8)
        self.completed('SA_MT_SE_adLIF', accuracy=.83)
        self.completed(seed=2027, accuracy=.82)
        directory = self.plan('SA_MT_SE_adLIF', seeds=(3037,))[0]['directory']
        directory.mkdir(parents=True)
        root = self.root / runner.RUN_LOGDIR
        record = aggregate([root])
        pair = record['paired_comparisons'][0]
        self.assertEqual([p['seed'] for p in pair['pairs']], [1009])
        self.assertAlmostEqual(pair['mean_pp'], 3)
        self.assertIsNone(pair['ci95_pp'])
        self.assertEqual(sum(r['status'] == 'complete' for r in record['runs']), 3)
        self.assertEqual(sum(r['status'] == 'incomplete' for r in record['runs']), 1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate([root, root])
        stats = paired_statistics([1, 2, 3])
        self.assertEqual(stats['mean_pp'], 2)
        self.assertEqual(stats['sample_sd_pp'], 1)
        self.assertAlmostEqual(stats['ci95_pp'][1], 4.4841377, places=6)

    def test_lock_excludes_second_launcher(self):
        with study.study_lock(self.root):
            with self.assertRaises(ValueError):
                with study.study_lock(self.root):
                    self.fail('Second launcher acquired held lock')
        with study.study_lock(self.root):
            pass

    def test_import_without_weights_and_reject_modified_architecture_or_source(self):
        item = self.completed()
        imported = self.root / 'imported'
        original = self.root / runner.RUN_LOGDIR
        shutil.copytree(original, imported)
        directory = imported / item['directory'].relative_to(original)
        shutil.rmtree(directory / 'ckpt')
        self.assertEqual(aggregate([imported])['summaries'][0]['n'], 1)
        manifest = directory / 'FULL_RUN_CONFIG.json'
        changed = copy.deepcopy(item['config'])
        changed['l1']['hidden_size'] = 999
        manifest.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'frozen study'):
            aggregate([imported])
        manifest.write_text(json.dumps(item['config']))
        environments = directory / 'LAUNCH_ENVIRONMENTS.json'
        environments.write_text(json.dumps([dict(git_commit='a', tracked_diff_sha256='x'),
                                            dict(git_commit='b', tracked_diff_sha256='x')]))
        with self.assertRaisesRegex(ValueError, 'Mixed source'):
            aggregate([imported])

    def test_portable_paths_with_changed_working_directory(self):
        item = self.completed()
        import os
        previous = Path.cwd()
        try:
            os.chdir(item['directory'])
            record, _ = study.read_completed(item['directory'])
            self.assertEqual(record['selected_epoch'], 2)
            self.assertFalse(Path(record['last_checkpoint']).is_absolute())
        finally:
            os.chdir(previous)

    def test_environment_provenance_is_preserved_and_source_changes_rejected(self):
        directory = self.root / 'provenance'
        directory.mkdir()
        with patch('torch.cuda.device_count', return_value=0):
            study.write_provenance(directory, study.ROOT)
            first = (directory / 'LAUNCH_ENVIRONMENTS.json').read_bytes()
            study.write_provenance(directory, study.ROOT)
            self.assertEqual((directory / 'LAUNCH_ENVIRONMENTS.json').read_bytes(), first)
            record = json.loads(first)
            record[0]['git_commit'] = 'different'
            study.atomic_json(directory / 'LAUNCH_ENVIRONMENTS.json', record)
            with self.assertRaisesRegex(ValueError, 'Source revision'):
                study.write_provenance(directory, study.ROOT)


if __name__ == '__main__':
    unittest.main()

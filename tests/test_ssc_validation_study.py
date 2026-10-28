"""SSC workflow integrity checks using synthetic checkpoint metadata; no training."""
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
from hydra import compose, initialize_config_dir
import torch

import extended_seed_study as validation
import project_paths
import runner
import ssc_validation_study as study


class SSCStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(runner.ROOT / 'config', self.root / 'config')
        for module in (runner, study, validation, project_paths):
            p = patch.object(module, 'ROOT', self.root)
            p.start()
            self.addCleanup(p.stop)
        for key, value in [('RUN_LOGDIR', 'results/valid'), ('VALIDATION_ONLY', False),
                           ('SEED_COHORT', None), ('SSC_STUDY_JOB', None)]:
            p = patch.object(runner, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.source = dict(git_commit='a' * 40, tracked_diff_sha256='b' * 64,
                           source_files_sha256='c' * 64)
        p = patch.object(validation, 'source_identity', return_value=self.source)
        p.start()
        self.addCleanup(p.stop)

    def plan(self, j):
        with study.runner_job(runner, j):
            return runner.build_plan(j['candidate']['model'], seeds=[j['seed']], datasets=['SSC'])[0]

    def completed(self, j, accuracy=.82):
        item = self.plan(j)
        directory, cfg = item['directory'], item['config']
        data = study.study_root() / j['pc'] / 'DATA_PROVENANCE.json'
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(json.dumps(dict(schema='ssc_training_data_v1', files={'train': '1' * 64, 'valid': '2' * 64})))
        (directory / 'ckpt').mkdir(parents=True)
        best = directory / 'ckpt/best.ckpt'
        state = dict(hyper_parameters={'cfg': cfg}, callbacks={'checkpoint': dict(
            monitor='val_acc', best_model_path=str(best), best_model_score=accuracy,
            dirpath=str(best.parent), best_k_models={str(best): accuracy})})
        torch.save(dict(state, epoch=15), best)
        torch.save(dict(state, epoch=39), directory / 'ckpt/last.ckpt')
        metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
        metrics.parent.mkdir(parents=True)
        metrics.write_text('epoch,val_acc\n' + ''.join(
            f'{e},{accuracy if e == 15 else accuracy-.01}\n' for e in range(40)))
        (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps(cfg))
        (directory / 'TRAINING_SUMMARY.md').write_text('## Best checkpoint\n\n```text\nckpt/best.ckpt\n```\n')
        (directory / 'LAUNCH_ENVIRONMENTS.json').write_text(json.dumps([self.source]))
        trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(
            best_model_path=str(best), best_model_score=accuracy))
        validation.finish_validation(OmegaConf.create(cfg), trainer)
        return item

    def select(self, stage):
        with contextlib.redirect_stdout(io.StringIO()):
            return study.select(stage, False)

    def lr_results(self):
        for j in study.all_jobs('lr'):
            c = j['candidate']
            # Different models advance; LR 0.003 improves on the baseline.
            base = {'MT_SE_adLIF': .823, 'SA_MT_SE_adLIF': .822, 'AMT_SE_adLIF': .819}[c['model']]
            self.completed(j, base + (.004 if c['lr'] == .003 else 0))
        return self.select('select-lr')

    def dropout_results(self):
        self.lr_results()
        for j in study.all_jobs('dropout'):
            self.completed(j, .835 if j['candidate']['dropout'] == .1 else .82)
        return self.select('select-dropout')

    def confirmation_results(self):
        self.dropout_results()
        for j in study.all_jobs('check'):
            self.completed(j, .83)
        self.select('select-winner')
        for j in study.all_jobs('confirm'):
            self.completed(j, .83)
        with contextlib.redirect_stdout(io.StringIO()):
            return study.freeze_test(False)

    def test_pc_screening_jobs_are_disjoint_and_effective_configs_match(self):
        jobs = study.all_jobs('lr')
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({study.job_directory(j) for j in jobs}), 18)
        for j in jobs:
            item = self.plan(j)
            cfg = item['config']
            self.assertEqual(j['seed'], 42 if j['pc'] == 'pc3' else 123)
            self.assertEqual(cfg['lr'], j['candidate']['lr'])
            self.assertEqual(cfg['dropout'], .15)
            self.assertEqual(cfg['batch_size'], 256)
            self.assertEqual(cfg['n_epochs'], 40)
            self.assertEqual(cfg['loss_agg'], 'mean_membrane_potentials')
            self.assertFalse(cfg['run_test'])
            self.assertFalse(cfg['early_stopping'])
            self.assertIsNone(cfg['init_ckpt_path'])
            self.assertIn('run_test=false', item['overrides'])
            self.assertIn(f'hydra.run.dir="{cfg["logdir"]}"', item['overrides'])
            with initialize_config_dir(config_dir=str(self.root / 'config'), version_base=None):
                child = compose(config_name='main', overrides=item['overrides'])
            self.assertEqual(project_paths.portable_config(OmegaConf.to_container(child, resolve=True)), cfg)
        self.assertIsNone(runner.SSC_STUDY_JOB)
        self.assertFalse(runner.VALIDATION_ONLY)

    def test_dry_run_never_launches_or_creates_results(self):
        with patch.object(runner.subprocess, 'run') as launch, patch('pytorch_lightning.Trainer') as trainer:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(['--ssc-study-stage', 'lr', '--study-pc', 'pc4', '--dry-run'])
            launch.assert_not_called()
            trainer.assert_not_called()
        self.assertFalse(study.study_root().exists())

    def test_unknown_options_and_unfrozen_later_stage_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(['--ssc-study-stage', 'lr', '--study-pc', 'pc3', '--lr', '.02'])
        for stage in ('dropout', 'check', 'confirm'):
            with self.assertRaisesRegex(ValueError, 'Missing frozen'):
                study.jobs(stage, 'pc3')
        with self.assertRaisesRegex(ValueError, 'Pending'):
            self.select('select-lr')

    def test_undeclared_settings_test_flags_and_seed_owner_are_rejected(self):
        cfg = self.plan(study.jobs('lr', 'pc3')[0])['config']
        for key, value in [('run_test', True), ('random_seed', 123), ('n_epochs', 80),
                           ('batch_size', 128), ('scheduler', 'cosine'), ('init_ckpt_path', 'bad.ckpt'),
                           ('loss_agg', 'sum_softmax_over_time'), ('dropout', .2),
                           ('datadir', 'elsewhere'), ('auto_regression', True)]:
            bad = copy.deepcopy(cfg)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate_training_config(bad)

    def test_resume_complete_and_changed_manifest_guards(self):
        j = study.jobs('lr', 'pc3')[0]
        item = self.completed(j)
        self.assertEqual(self.plan(j)['status'], 'complete')
        (item['directory'] / 'validation_completed.txt').unlink()
        resumed = self.plan(j)
        self.assertEqual(resumed['status'], 'resume')
        self.assertIn('last.ckpt', resumed['overrides'][-1])
        cfg = copy.deepcopy(item['config'])
        cfg['lr'] = .009
        (item['directory'] / 'FULL_RUN_CONFIG.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'Configuration changed'):
            self.plan(j)

    def test_missing_exact_checkpoint_and_overlapping_histories_rejected(self):
        j = study.jobs('lr', 'pc3')[0]
        item = self.completed(j)
        checkpoint = item['directory'] / 'ckpt/best.ckpt'
        checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan(j)
        metrics = item['directory'] / 'logs/mlp_snn/version_0/metrics.csv'
        with metrics.open('a') as stream:
            stream.write('39,.82\n')
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            study.evidence(j)

    def test_screening_test_evidence_is_rejected(self):
        j = study.jobs('lr', 'pc3')[0]
        item = self.completed(j)
        (item['directory'] / 'TEST_RESULTS.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            self.plan(j)

    def test_selections_are_distinct_models_immutable_and_need_both_pcs(self):
        selected = self.lr_results()
        self.assertEqual(len({c['model'] for c in selected['candidates']}), 2)
        self.assertEqual(len(study.jobs('dropout', 'pc3')), 4)
        self.assertEqual(len(study.jobs('dropout', 'pc4')), 4)
        before = study.manifest_path('select-lr').read_bytes()
        self.select('select-lr')
        self.assertEqual(before, study.manifest_path('select-lr').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Frozen selection changed'):
            study.freeze('select-lr', dict(candidates=[]), False)
        j = study.jobs('lr', 'pc4')[0]
        metrics = study.job_directory(j) / 'logs/mlp_snn/version_0/metrics.csv'
        metrics.write_text(metrics.read_text().replace('0.827', '0.829'))
        with self.assertRaises(ValueError):
            study.evidence(j)

    def test_all_stages_and_final_freeze_without_training_or_inference(self):
        frozen = self.confirmation_results()
        self.assertEqual(len(study.all_jobs('check')), 2)
        self.assertEqual(len(study.jobs('confirm', 'pc3')), 6)
        self.assertEqual(len(study.jobs('confirm', 'pc4')), 4)
        for pc in ('pc3', 'pc4'):
            assigned = study.jobs('confirm', pc)
            for seed in {j['seed'] for j in assigned}:
                self.assertEqual(len([j for j in assigned if j['seed'] == seed]), 2)
        self.assertEqual(len(frozen['evidence']), 10)
        with patch('pytorch_lightning.Trainer') as trainer:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(['--ssc-study-stage', 'test', '--study-pc', 'pc3', '--dry-run'])
            trainer.assert_not_called()
        for r in frozen['evidence']:
            directory = project_paths.local_path(r['directory'])
            value = .81 if r['candidate']['model'] != 'SE_adLIF' else .805
            record = dict(schema='ssc_frozen_test_v1', freeze_sha256=frozen['sha256'],
                          checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
                          seed=r['seed'], model=r['candidate']['model'], inference='ordinary',
                          metrics=dict(test_acc=value, test_loss=.5), test_data_sha256='f' * 64,
                          environment=self.source)
            (directory / 'TEST_RESULTS.json').write_text(json.dumps(record))
            study.validate_final_evidence(directory)
            validation.read_completed(directory)
        import measure_run_costs
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(measure_run_costs.collect(directory, 'SSC', dry_run=True))
        with contextlib.redirect_stdout(io.StringIO()):
            study.report(False)
        text = (study.study_root() / 'CONFIRMATION_RESULTS.md').read_text(encoding='utf-8')
        self.assertIn('+0.500 pp', text)
        self.assertIn('95% paired t interval', text)
        j = study.jobs('confirm', 'pc3')[0]
        with patch.object(runner, 'collect_costs') as costs, patch('pytorch_lightning.Trainer') as trainer:
            study.evaluate(j, frozen, False, runner)
            trainer.assert_not_called()
            costs.assert_not_called()
            study.evaluate(j, frozen, False, runner, measure_existing=True)
            costs.assert_called_once()
        bad = json.loads((study.job_directory(j) / 'TEST_RESULTS.json').read_text())
        bad['checkpoint_sha256'] = '0' * 64
        (study.job_directory(j) / 'TEST_RESULTS.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'Invalid frozen'):
            study.validate_final_evidence(study.job_directory(j))

    def test_ties_and_duplicate_seeds(self):
        a = study.candidate('AMT_SE_adLIF', .006)
        b = study.candidate('MT_SE_adLIF', .006)
        records = [dict(candidate=c, seed=s, accuracy=.82) for c in (a, b) for s in (42, 123)]
        self.assertEqual(study.ranking(records)[0]['candidate'], a)
        with self.assertRaisesRegex(ValueError, 'Repeated seed'):
            study.ranking(records + [records[0]])

    def test_mismatched_pc_data_is_rejected(self):
        records = [dict(candidate=study.candidate('MT_SE_adLIF', .006), seed=seed,
                        accuracy=.82, training_data_sha256=str(seed)) for seed in (42, 123)]
        with self.assertRaisesRegex(ValueError, 'different training/validation data'):
            study.ranking(records)

    def test_no_validation_improvement_stops_confirmation(self):
        candidates = [study.candidate('MT_SE_adLIF', .006), study.candidate('SA_MT_SE_adLIF', .006)]
        records = [dict(candidate=c, seed=s, accuracy=.82) for c in candidates for s in (42, 123)]
        checks = [dict(candidate=c, seed=456, accuracy=.82) for c in candidates]
        prior = dict(candidates=candidates, evidence=records, sha256='a' * 64)
        with patch.object(study, 'read_frozen', return_value=prior), \
                patch.object(study, 'all_jobs', return_value=checks), \
                patch.object(study, 'evidence', side_effect=lambda r: r), \
                patch.object(study, 'verify_snapshots'):
            with self.assertRaisesRegex(ValueError, 'Neither finalist improved'):
                study.select('select-winner', True)

    def test_cost_measurement_uses_exact_frozen_checkpoint_before_loading(self):
        import measure_run_costs as costs
        directory = self.root / 'final'
        directory.mkdir()
        checkpoint = directory / 'best.ckpt'
        checkpoint.write_bytes(b'trusted-fixture-identity')
        (directory / 'FULL_RUN_CONFIG.json').write_text(json.dumps({'ssc_study': {'stage': 'confirm'}}))
        frozen = dict(checkpoint=project_paths.portable_path(checkpoint), checkpoint_sha256='0' * 64)
        cfg = dict(dataset={'_target_': 'datasets.ssc.SSCLDM'}, two_layers=True,
                   l1={'cell': 'mt_se_adlif'}, l2={'cell': 'mt_se_adlif'})
        with patch.object(study, 'validate_final_evidence', return_value=frozen), \
                patch.object(costs.shared, 'selected_checkpoint', return_value=checkpoint), \
                patch.object(costs.torch, 'load', return_value={'hyper_parameters': {'cfg': cfg}}) as load:
            with self.assertRaisesRegex(ValueError, 'differs from frozen'):
                costs.collect(directory, 'SSC', dry_run=True)
            load.assert_not_called()
            frozen['checkpoint_sha256'] = study.sha256(checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(costs.collect(directory, 'SSC', dry_run=True))
            load.assert_called_once()


if __name__ == '__main__':
    unittest.main()

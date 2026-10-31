"""SSC wrapper tests: preflight and mocks only; no training subprocesses."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import runner
import runner_vs_mt_ssc as study


class VSMTSSCRunnerTests(unittest.TestCase):
    def test_baseline_inheritance_and_seeds(self):
        new, old = study.plans([42, 123, 456])
        self.assertEqual(len(new), 3)
        for a, b in zip(new, old):
            self.assertEqual(a['status'], 'new')
            self.assertEqual(a['config']['paper_baseline_dataset'], 'SSC')
            self.assertEqual(a['config']['loss_agg'], 'mean_membrane_potentials')
            self.assertIsNone(a['config']['init_ckpt_path'])
            self.assertIsNone(a['config']['ckpt_path'])
            for key in ('lr', 'batch_size', 'n_epochs', 'dataset', 'early_stopping',
                        'scheduler', 'factor', 'patience', 'loss_agg'):
                self.assertEqual(a['config'][key], b['config'][key])
            for layer in ('l1', 'l2'):
                self.assertEqual(a['config'][layer], dict(b['config'][layer], cell='vs_mt_se_adlif'))
            self.assertIn('model_variant=VS_MT_SE_adLIF', a['config']['paper_baseline_deviations'])

    def test_dry_run_has_no_writes_or_training(self):
        def snapshot():
            return {p.as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                    for p in Path(study.OUTPUT).rglob('*') if p.is_file()}
        before = snapshot()
        with patch.object(runner, 'run', side_effect=AssertionError('Training forbidden')), \
             patch.object(study, 'write_comparison', side_effect=AssertionError('Report write forbidden')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            study.main(['--dry-run', '--seeds', '123'])
        self.assertIn('Validated 1 runs; no training or writes.', output.getvalue())
        self.assertEqual(before, snapshot())

    def test_test_records_must_agree_and_partial_is_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            item = dict(directory=d, status='resume')
            self.assertIsNone(study.test_accuracy(item))
            item['status'] = 'complete'
            logs = d/'logs/mlp_snn/version_0'
            logs.mkdir(parents=True)
            (logs/'metrics.csv').write_text('epoch,test_acc\n39,0.8\n')
            self.assertEqual(study.test_accuracy(item), 80.)
            (d/'TEST_RESULTS.json').write_text(json.dumps(dict(metrics=dict(test_acc=.9))))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                study.test_accuracy(item)

    def test_report_only_never_dispatches_training(self):
        with patch.object(study, 'plans', return_value=([], [])), patch.object(study, 'write_comparison') as report, \
             patch.object(runner, 'main', side_effect=AssertionError('Training dispatch forbidden')):
            study.main(['--report-only', '--seeds', '42'])
        report.assert_called_once_with([42])

    def test_reject_other_datasets_and_seeds(self):
        for args in (['--datasets', 'SHD'], ['--datasets', 'SSC', 'ECG'], ['--seeds', '1009']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                study.main(args+['--dry-run'])

    def test_guarded_resume_and_configuration_mismatch(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'results') as temp:
            with patch.object(study, 'OUTPUT', Path(temp).relative_to(runner.ROOT).as_posix()):
                item = study.plans([42])[0][0]
                d = item['directory']
                (d/'ckpt').mkdir(parents=True)
                (d/'ckpt/last.ckpt').touch()
                manifest = d/'FULL_RUN_CONFIG.json'
                manifest.write_text(json.dumps(item['config']))
                resumed = study.plans([42])[0][0]
                self.assertEqual(resumed['status'], 'resume')
                self.assertIn('last.ckpt', resumed['overrides'][-1])
                changed = dict(item['config'], lr=.2)
                manifest.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                    study.plans([42])

    def test_portability_from_relocated_checkout_and_hydra_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'checkout'
            shutil.copytree(runner.ROOT/'config', root/'config')
            hydra = Path(temp)/'hydra_job'
            hydra.mkdir()
            try:
                os.chdir(hydra)
                with patch.object(runner, 'ROOT', root):
                    new, _ = study.plans([42])
            finally:
                os.chdir(previous)
        self.assertTrue(new[0]['config']['logdir'].startswith(study.OUTPUT+'/'))
        self.assertNotIn(str(root), json.dumps(new[0]['config']))

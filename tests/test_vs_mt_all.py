"""Non-training checks for sequential SSC/SHD*/ECG VS-MT execution."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import runner
import runner_vs_mt_all as sequence
import runner_vs_mt_ssc as ssc
from tests.test_measure_shd_costs import tiny_model


class VSMTAllTests(unittest.TestCase):
    def test_dataset_specific_baseline_inheritance(self):
        with ssc.output_root(sequence.OUTPUT):
            variant = runner.build_plan(ssc.MODEL, seeds=[42, 123, 456], inspect_existing=False)
            mt = runner.build_plan('MT_SE_adLIF', seeds=[42, 123, 456], inspect_existing=False)
        self.assertEqual(len(variant), 9)
        for new, old in zip(variant, mt):
            a, b = copy.deepcopy(new['config']), copy.deepcopy(old['config'])
            for cfg in (a, b):
                for key in ('exp_name', 'model_variant', 'logdir', 'paper_baseline_deviations'):
                    cfg.pop(key, None)
                for layer in ('l1', 'l2'):
                    cfg[layer]['cell'] = 'mt_se_adlif'
            self.assertEqual(a, b)
            self.assertEqual(new['config']['paper_baseline_dataset'], new['dataset'])
            self.assertIsNone(new['config']['init_ckpt_path'])
            self.assertTrue(new['config']['run_test'])
        shd = next(i['config'] for i in variant if i['dataset'] == 'SHD')
        self.assertEqual(shd['dataset']['validate_on'], .2)
        self.assertEqual(shd['dataset']['random_seed'], 42)
        self.assertFalse(shd['dataset']['additional_test_set_validation'])

    def test_order_resume_identity_and_cost_forwarding(self):
        jobs = sequence.commands(['ECG', 'SHD', 'SSC'], [42], cost_option='--measure-existing-costs')
        self.assertEqual([ds for ds, _ in jobs], ['SSC', 'SHD', 'ECG'])
        self.assertNotIn('--logdir', jobs[0][1])  # Original SSC wrapper owns its root.
        for _, args in jobs:
            self.assertIn('--measure-existing-costs', args)
        self.assertIn(sequence.OUTPUT, jobs[1][1])
        self.assertEqual(sequence.commands(['ECG'], [123])[0][0], 'ECG')

    def test_dry_run_never_trains_or_exports(self):
        with patch.object(runner, 'run', side_effect=AssertionError('Training forbidden')), \
             patch.object(sequence, 'save_parameters', side_effect=AssertionError('Export forbidden')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            sequence.main(['--dry-run'])
        self.assertEqual(output.getvalue().count('Validated 3 runs; no training or writes.'), 3)

    def test_missing_data_stops_preflight_before_dispatch(self):
        item = dict(config=dict(dataset=dict(data_path='missing_vs_mt_test_data')))
        with patch.object(ssc, 'plans', return_value=([item], [])), \
             patch.object(ssc, 'main', side_effect=AssertionError('Dispatch forbidden')):
            with self.assertRaisesRegex(FileNotFoundError, 'Missing SSC data'):
                sequence.main(['--dry-run'])

    def test_saved_checkpoint_parameter_export_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            model = tiny_model('vs_mt_se_adlif')
            model.hparams.cfg.random_seed = 42
            model.hparams.cfg.dataset._target_ = 'datasets.ecg.ECGLDM'
            checkpoint = d/'best.ckpt'
            torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(cfg=model.hparams.cfg)), checkpoint)
            (d/'completed.txt').touch()
            item = dict(directory=d, dataset='ECG', seed=42,
                        config=dict(dataset=dict(_target_='datasets.ecg.ECGLDM')))
            with patch('measure_shd_costs.selected_checkpoint', return_value=checkpoint):
                sequence.save_parameters([item])
            r = json.loads((d/'NEURON_PARAMETERS.json').read_text())
            self.assertEqual(r['dataset'], 'ECG')
            self.assertEqual(len(r['layers']['l1']['a_slow']), 6)

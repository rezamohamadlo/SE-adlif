"""Non-training checks for the LT-DTH/DTH evaluation preparation."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from project_paths import ROOT
from scripts.adaptation_pairs.lt_dth import load_study,compare_configs
from scripts.adaptation_pairs.parameters import trace_record,extract
from scripts.adaptation_pairs.reports import write_reports


class PairTests(unittest.TestCase):
    def test_frozen_cohort_and_inherited_conditions(self):
        study=load_study()
        self.assertEqual(study['seeds'],[42,123,456])
        self.assertEqual(study['samples'],1631)
        self.assertEqual(len(study['runs']),6)
        self.assertEqual(len(study['conditions']),6)
        self.assertEqual(study['conditions'][0],['clean',0,'identity',1.])
        self.assertEqual(study['parent'],'DTH_SE_adLIF')
        self.assertEqual(study['variant'],'LT_DTH_SE_adLIF')
        self.assertFalse(study['training'])
        self.assertFalse(study['test_set_access'])
        self.assertEqual(len({r['checkpoint_sha256'] for r in study['runs']}),6)

    def test_only_intended_layer_differences_are_allowed(self):
        parent=dict(l1={'cell':'dth','rho_fast':.9,'n_neurons':360},l2={'cell':'dth','rho_fast':.9},lr=.01)
        variant=copy.deepcopy(parent)
        for layer in ('l1','l2'):
            variant[layer].update(cell='lt_dth',trace_rho_fast_range=[.5,.98],trace_rho_slow_range=[.985,.9995])
        compare_configs(parent,variant)
        for path,value in [('lr',.02),('l1',dict(variant['l1'],rho_fast=.8)),('l2',dict(variant['l2'],n_neurons=720))]:
            wrong=copy.deepcopy(variant);wrong[path]=value
            with self.assertRaisesRegex(ValueError,'settings differ'):compare_configs(parent,wrong)

    def test_trace_units_use_input_bin_not_neuron_dt(self):
        value=trace_record(.9,4.)
        self.assertAlmostEqual(value['e_folding_frames'],9.491221581,places=7)
        self.assertAlmostEqual(value['e_folding_ms'],4*value['e_folding_frames'])
        for rho in (0,1,-1,float('nan')):
            with self.assertRaises(ValueError):trace_record(rho,4)

    def test_layer_parameters_are_not_duplicated_over_samples(self):
        layer=SimpleNamespace(rho_fast=.9,rho_slow=.99)
        learned=SimpleNamespace(trace_decays=lambda:(.95,.995))
        run=dict(model='test',seed=42,checkpoint='same.ckpt',checkpoint_sha256='same')
        record=extract(SimpleNamespace(l1=layer,l2=learned),run,4)
        self.assertEqual(len(record['layers']),2)
        self.assertFalse(record['layers'][0]['learned'])
        self.assertTrue(record['layers'][1]['learned'])
        self.assertEqual(record['checkpoint_sha256'],'same')

    def test_report_uses_correct_variant_name(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'SUMMARY.md';path.write_text('Positive favors AMT.')
            with patch('scripts.adaptation_pairs.reports.original_reports',return_value=[path]):
                write_reports(Path(folder),[],{})
            self.assertEqual(path.read_text(),'Positive favors LT-DTH.')

    def test_cli_is_independent_of_current_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            p=subprocess.run([sys.executable,str(ROOT/'scripts/adaptation_pairs/lt_dth.py'),'--help'],cwd=folder,
                capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('--check-only',p.stdout)
        self.assertNotIn('--pc',p.stdout)


if __name__=='__main__':unittest.main()

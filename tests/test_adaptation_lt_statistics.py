"""User-run checks of LT-DTH identity, hypothesis labels and preflight boundaries."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts.adaptation_lt_statistics import analyze


class SpecificationTests(unittest.TestCase):
    def test_pair_and_hypotheses_are_not_silently_relabelled_amt_results(self):
        spec=analyze.load_spec()
        self.assertEqual((spec['parent'],spec['variant']),('DTH_SE_adLIF','LT_DTH_SE_adLIF'))
        self.assertNotEqual(spec['autocorrelation_hypothesis'],'exploratory_two_sided_association')
        self.assertNotEqual(spec['fast_slow_hypothesis'],'positive_adjusted_association')
        self.assertEqual(spec['bootstrap']['draws'],2000)
        self.assertFalse(spec['new_inference'])
        self.assertFalse(spec['training'])

    def test_trace_summary_counts_models_not_samples(self):
        spec=analyze.load_spec();records=[]
        for model in (spec['parent'],spec['variant']):
            for seed in spec['training_seeds']:
                records.append(dict(model=model,seed=seed,layers=[dict(layer=layer,
                    fast=dict(rho=.9,e_folding_frames=10,e_folding_ms=40),
                    slow=dict(rho=.99,e_folding_frames=100,e_folding_ms=400)) for layer in ('l1','l2')]))
        result=analyze.trace_summary(records,spec)
        self.assertEqual(len(result['groups']),8)
        self.assertTrue(all(row['n_models']==3 for row in result['groups']))
        with self.assertRaises(ValueError):analyze.trace_summary(records[:-1],spec)

    def test_preflight_never_fits_bootstraps_writes_or_locks(self):
        inputs=dict(ids=[1,2],keep=[0,1],all_speakers=np.array([0,1]),source_manifest={})
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'not_created'
            with patch.object(analyze,'experiment_directory',side_effect=lambda v:Path(v).resolve()), \
                 patch.object(analyze.original,'load_inputs',return_value=inputs), \
                 patch.object(analyze,'load_traces',return_value=[]), \
                 patch.object(analyze,'make_manifest',return_value={}), \
                 patch.object(analyze,'cluster_analysis',side_effect=AssertionError('Fit called')), \
                 patch.object(analyze.pilot,'write_json',side_effect=AssertionError('Write called')), \
                 patch.object(analyze.pilot,'study_lock',side_effect=AssertionError('Lock called')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(['--check-only','--output-root',str(output)]),0)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()

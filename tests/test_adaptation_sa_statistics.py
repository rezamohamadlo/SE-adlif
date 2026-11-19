"""User-run tests; agents must not execute this experiment test suite."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts.adaptation_sa_statistics import analyze


class SpecificationTests(unittest.TestCase):
    def test_explicit_pair_and_mechanism_proxy(self):
        spec=analyze.load_spec()
        self.assertEqual((spec['parent'],spec['variant']),('MT_SE_adLIF','SA_MT_SE_adLIF'))
        self.assertIn('transient_sustained',spec['fast_slow_hypothesis'])
        self.assertIn('proxy',spec['mechanism_hypothesis'])
        self.assertEqual(spec['bootstrap']['draws'],2000)
        self.assertEqual(spec['expected_samples'],1631)
        self.assertFalse(spec['training'])
        self.assertFalse(spec['new_inference'])

    def test_wrong_pair_is_rejected_before_loading_reuse_files(self):
        spec=analyze.load_spec()
        inputs=dict(source_manifest=dict(resolved_study=dict(parent='MT_SE_adLIF',variant='AMT_SE_adLIF')))
        with self.assertRaisesRegex(ValueError,'Source model pair differs'):
            analyze.validate_pair(spec,inputs)

    def test_preflight_does_not_fit_resample_write_or_lock(self):
        inputs=dict(ids=[1,2],keep=[0,1],all_speakers=np.array([0,1]),source_manifest={})
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'not_created'
            with patch.object(analyze,'experiment_directory',side_effect=lambda v:Path(v).resolve()), \
                 patch.object(analyze.original,'load_inputs',return_value=inputs), \
                 patch.object(analyze,'validate_pair'), \
                 patch.object(analyze,'make_manifest',return_value={}), \
                 patch.object(analyze,'cluster_analysis',side_effect=AssertionError('Fit called')), \
                 patch.object(analyze.pilot,'write_json',side_effect=AssertionError('Write called')), \
                 patch.object(analyze.pilot,'study_lock',side_effect=AssertionError('Lock called')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(['--check-only','--output-root',str(output)]),0)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()

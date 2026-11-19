"""Numerical/statistical contracts. No neural model training or inference."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from project_paths import ROOT
from scripts.adaptation_statistics import methods, analyze


def fixture():
    rng = np.random.default_rng(12)
    n = 400
    speakers = np.repeat(np.arange(10), n // 10)
    classes = np.tile(np.arange(4), n // 4)
    rate = np.exp(rng.normal(size=n) + 5)
    duration = np.exp(rng.normal(size=n) + 3)
    features = np.column_stack((np.log1p(rate) + rng.normal(size=n), rng.normal(size=n)))
    x, scaling = methods.make_design(features, duration, rate, classes, speakers)
    coefficients = np.zeros((x.shape[1], 3))
    coefficients[:4] = [[.2, .4, -.1], [-.3, .2, .1], [.5, .5, .5], [2., 2., 2.]]
    y = x @ coefficients + methods.within_speaker(rng.normal(size=(n,3)) * .05, speakers)
    effects = rng.normal(size=(n,3,3))
    spec=dict(draws=100,seed=91,minimum_speakers=8,minimum_valid_fraction=.95,rank_relative_tolerance=1e-10)
    return x, y, speakers, effects, spec


class RegressionTests(unittest.TestCase):
    def test_speaker_demeaning_removes_offsets_without_changing_slopes(self):
        x,y,speakers,_,_=fixture()
        shifted=y+speakers[:,None]*100
        np.testing.assert_allclose(methods.within_speaker(shifted,speakers),y,atol=1e-12)
        for group in np.unique(speakers):
            np.testing.assert_allclose(x[speakers==group].mean(axis=0),0,atol=1e-14)

    def test_sufficient_statistic_fit_matches_explicit_weighted_least_squares(self):
        x,y,speakers,_,spec=fixture()
        gram,cross=methods.sufficient_statistics(x,y,speakers,np.arange(10))
        weights=np.array([0,2,1,0,3,1,1,0,1,1])
        observed=methods.solve_clusters(gram,cross,weights,spec['rank_relative_tolerance'])
        expanded=np.repeat(np.arange(len(x)),weights[speakers])
        expected=np.linalg.lstsq(x[expanded],y[expanded],rcond=None)[0]
        np.testing.assert_allclose(observed,expected,rtol=1e-10,atol=1e-10)

    def test_within_fit_matches_full_speaker_fixed_effect_fit(self):
        x,y,speakers,_,_=fixture()
        offset=speakers[:,None]*10
        dummies=np.column_stack([speakers==g for g in range(10)])
        full=np.linalg.lstsq(np.column_stack((x,dummies)),y+offset,rcond=None)[0][:x.shape[1]]
        within=np.linalg.lstsq(x,methods.within_speaker(y+offset,speakers),rcond=None)[0]
        np.testing.assert_allclose(full,within,atol=1e-12)

    def test_activity_confounding_does_not_become_feature_effect(self):
        x,_,speakers,_,spec=fixture()
        y=np.column_stack([3*x[:,3]]*3)
        gram,cross=methods.sufficient_statistics(x,y,speakers,np.arange(10))
        fitted=methods.solve_clusters(gram,cross,np.ones(10),spec['rank_relative_tolerance'])
        np.testing.assert_allclose(fitted[:2],0,atol=1e-12)
        np.testing.assert_allclose(fitted[3],3,atol=1e-12)

    def test_rank_deficient_design_is_rejected(self):
        x,y,speakers,effects,spec=fixture()
        x[:,1]=x[:,0]
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            methods.cluster_analysis(x,y,speakers,effects,speakers,spec)


class BootstrapTests(unittest.TestCase):
    def test_reproducible_cluster_draws_preserve_fixed_seed_pairing(self):
        x,y,speakers,effects,spec=fixture()
        a=methods.cluster_analysis(x,y,speakers,effects,speakers,spec)
        b=methods.cluster_analysis(x,y,speakers,effects,speakers,spec)
        np.testing.assert_array_equal(a['cluster_weights'],b['cluster_weights'])
        np.testing.assert_array_equal(a['bootstrap'],b['bootstrap'])
        for weights,estimate,effect in zip(a['cluster_weights'][:5],a['bootstrap'][:5],a['effect_bootstrap'][:5]):
            expanded=np.repeat(np.arange(len(x)),weights[speakers])
            fit=np.linalg.lstsq(x[expanded],y[expanded],rcond=None)[0][:2]
            np.testing.assert_allclose(estimate,fit,atol=1e-11)
            np.testing.assert_allclose(effect,effects[expanded].mean(axis=0),atol=1e-12)
        self.assertEqual(len(a['loo']),10)
        self.assertEqual(a['bootstrap'].shape,(100,2,3))

    def test_seed_average_is_not_treated_as_more_speaker_draws(self):
        x,y,speakers,effects,spec=fixture()
        y[:]=y[:,0,None]
        result=methods.cluster_analysis(x,y,speakers,effects,speakers,spec)
        np.testing.assert_allclose(result['bootstrap'][:,:,0],result['bootstrap'][:,:,1],atol=1e-12)
        np.testing.assert_allclose(result['bootstrap'].mean(axis=2),result['bootstrap'][:,:,0],atol=1e-12)

    def test_too_few_speakers_are_rejected(self):
        x,y,speakers,effects,spec=fixture()
        mask=speakers<7
        with self.assertRaisesRegex(ValueError,'Too few speakers'):
            methods.cluster_analysis(x[mask],y[mask],speakers[mask],effects[mask],speakers[mask],spec)

    def test_family_adjustment_uses_two_tails(self):
        low,high=methods.interval(np.arange(1001),.05,2)
        self.assertEqual(low,12.5)
        self.assertEqual(high,987.5)
        self.assertEqual(methods.direction(low,high),'positive_within_fixed_cohort')
        self.assertEqual(methods.direction(-2,2),'inconclusive')

    def test_nonfinite_inputs_are_rejected(self):
        x,y,speakers,effects,spec=fixture()
        y[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            methods.cluster_analysis(x,y,speakers,effects,speakers,spec)


class IntegrityTests(unittest.TestCase):
    def test_source_manifest_pin_cannot_be_bypassed(self):
        spec=yaml.safe_load((ROOT/analyze.SPEC).read_text(encoding='utf-8'))
        spec.update(source='results/experiments/nonexistent_stats_fixture',source_manifest_sha256='bad')
        with patch.object(analyze,'verify_package'),patch.object(analyze.pilot,'read_json',return_value={}):
            with self.assertRaisesRegex(ValueError,'Source manifest differs'):
                analyze.load_inputs(spec)

    def test_help_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            process=subprocess.run([sys.executable,str(ROOT/'scripts/adaptation_statistics/analyze.py'),'--help'],
                cwd=folder,capture_output=True,text=True,timeout=30)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn('--check-only',process.stdout)
        self.assertNotIn('--pc',process.stdout)


if __name__=='__main__':
    unittest.main()

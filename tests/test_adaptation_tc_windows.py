"""User-run window metric and preflight contracts; do not run as agent tests."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scripts.adaptation_tc_windows.metrics import describe
from scripts.adaptation_tc_windows import evaluate
from scripts.adaptation_tc_statistics.analyze import load_spec


class WindowTests(unittest.TestCase):
    def test_constant_evidence_and_excluded_burn_in_padding(self):
        p=np.tile([.8,.2],(30,1))
        first=describe(p,0,20)
        p[:10]=[0,1];p[20:]=[0,1]
        second=describe(p,0,20)
        self.assertEqual(first,second)
        self.assertAlmostEqual(first['adjacent_js_nats'],0)
        self.assertAlmostEqual(first['true_class_concentration'],0)
        self.assertEqual(first['valid_windows'],2)

    def test_brief_class_evidence_and_switching_have_known_limits(self):
        p=np.array([[1.,0.]]*5+[[0.,1.]]*5)
        result=describe(p,0,10,burn_in=0)
        self.assertAlmostEqual(result['adjacent_js_nats'],np.log(2))
        self.assertAlmostEqual(result['true_class_concentration'],1)
        self.assertEqual(result['window_accuracy'],.5)
        self.assertEqual(result['top_class_switch_fraction'],1)

    def test_stable_wrong_evidence_is_not_marked_as_useful(self):
        result=describe(np.tile([0.,1.],(20,1)),0,20)
        self.assertEqual(result['adjacent_js_nats'],0)
        self.assertEqual(result['window_accuracy'],0)
        self.assertEqual(result['true_class_mean_probability'],0)
        self.assertIsNone(result['true_class_concentration'])

    def test_incomplete_tail_does_not_enter_windows(self):
        p=np.tile([.8,.2],(21,1));p[-1]=[0,1]
        result=describe(p,0,21)
        self.assertEqual(result['dropped_tail_frames'],1)
        self.assertAlmostEqual(result['true_class_mean_probability'],.8)
        self.assertEqual(describe(p,0,14)['undefined_reason'],'fewer_than_two_complete_windows')

    def test_statistics_are_explicitly_input_associations_not_window_hypothesis_test(self):
        spec=load_spec()
        self.assertEqual(spec['variant'],'MT_TC_SE_adLIF')
        self.assertIn('two_sided',spec['fast_slow_hypothesis'])
        self.assertIn('Window diagnostics are separate',spec['mechanism_hypothesis'])
        self.assertIn('JS_excluded',spec['comparison_loss'])

    def test_window_preflight_does_not_run_diagnostics(self):
        with patch.object(evaluate,'prepare',return_value=(Path('unused'),{},{})), \
             patch.object(evaluate,'execute',side_effect=AssertionError('Diagnostic ran')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(['--check-only']),0)


if __name__=='__main__':unittest.main()

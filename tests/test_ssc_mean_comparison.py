import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ssc_mean_comparison import MODELS, job, verify_reference


class SSCMeanComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        reference = Path(self.temp.name)
        _, _, cfg = job(MODELS[0])
        cfg = copy.deepcopy(cfg)
        for layer in ('l1', 'l2'):
            cfg[layer] = {key: value for key, value in cfg[layer].items()
                          if key not in ('rho_fast', 'rho_slow', 'target_rate', 'theta_min', 'theta_max')}
            cfg[layer]['cell'] = 'se_adlif'
        (reference / 'FULL_RUN_CONFIG.json').write_text(json.dumps(cfg))
        (reference / 'DIAGNOSTIC_MANIFEST.json').write_text(json.dumps(dict(packages={}, source_sha256={})))
        checkpoint = reference / 'selected.ckpt'
        checkpoint.write_bytes(b'identity-only fixture; never deserialized')
        (reference / 'TEST_EVALUATION_METADATA.json').write_text(json.dumps(dict(
            checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
        self.reference_patch = patch('ssc_mean_comparison.REFERENCE', str(reference))
        self.reference_patch.start()
        self.addCleanup(self.reference_patch.stop)

    def test_configs_match_promoted_reference(self):
        for model in MODELS:
            with self.subTest(model=model):
                _, _, cfg = job(model)
                self.assertEqual(cfg['loss_agg'], 'mean_membrane_potentials')
                self.assertEqual(cfg['evaluation_protocol'], 'ssc_mean_loss_v1')
                self.assertEqual(cfg['random_seed'], 42)
                self.assertFalse(cfg['run_test'])
                self.assertIsNone(cfg['init_ckpt_path'])
                verify_reference(cfg)

    def test_mismatched_optimization_is_rejected(self):
        _, _, cfg = job(MODELS[0])
        cfg = copy.deepcopy(cfg)
        cfg['lr'] = .01
        with self.assertRaisesRegex(ValueError, 'Mismatch'):
            verify_reference(cfg)


if __name__ == '__main__':
    unittest.main()

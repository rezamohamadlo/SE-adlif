"""User-launched queue contracts. No training or checkpoint inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.adaptation_analysis import pilot
from scripts.adaptation_ecg_suite.preflight import numerical_contract, PAIRS
from scripts.adaptation_ecg_suite.reporting import describe, paired_records
from scripts.adaptation_ecg_suite.runtime import prediction_keys, require_all_anchors, snapshot_failure


def record(model, seed=42, condition='delete_0.1', loss=1.):
    return dict(model=model, seed=seed, condition=condition, replicate=0, sample_id=7,
        input_sha256='input_'+condition, targets_sha256='labels', blocks_sha256='blocks',
        valid_steps=5, loss=loss, accuracy=.6)


class ECGSuiteContracts(unittest.TestCase):
    def test_precision_matches_training_and_is_recorded(self):
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('highest')
            contract = numerical_contract()
            self.assertEqual(torch.get_float32_matmul_precision(), 'high')
            self.assertEqual(contract['float32_matmul_precision'], 'high')
            self.assertFalse(contract['amp'])
            self.assertFalse(contract['tolerances_relaxed'])
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_four_pairs_use_72_unique_chunks_and_share_mt(self):
        spec = dict(pairs=[dict(id=i, parent=p, variant=v) for i, p, v in PAIRS], seeds=[42, 123, 456],
                    conditions=[['clean', 0], ['delete_0.1', 0], ['delete_0.1', 1], ['delete_0.1', 2]])
        keys = prediction_keys(spec)
        self.assertEqual(len(keys), 72)
        self.assertEqual(sum(k[0] == 'MT_SE_adLIF' for k in keys), 12)
        self.assertEqual(sum(k[2] == 'clean' for k in keys), 18)

    def test_pairing_rejects_missing_or_duplicate_records(self):
        p, v = record('parent'), record('variant')
        cp, cv = record('parent', condition='clean'), record('variant', condition='clean')
        with self.assertRaises(ValueError): paired_records([p], [], [cp], [cv])
        with self.assertRaises(ValueError): paired_records([p, p], [v, v], [cp, cp], [cv, cv])

    def test_pairing_rejects_seed_or_label_mismatch(self):
        p, v = record('parent'), record('variant')
        cp, cv = record('parent', condition='clean'), record('variant', condition='clean')
        for key, replacement in [('seed', 123), ('targets_sha256', 'other'), ('input_sha256', 'other')]:
            damaged = dict(v, **{key: replacement})
            with self.assertRaises(ValueError): paired_records([p], [damaged], [cp], [cv])

    def test_pairing_uses_shared_supervised_loss_and_correct_delta(self):
        p, v = record('parent', loss=1.), record('variant', loss=.5)
        cp, cv = record('parent', condition='clean', loss=2.), record('variant', condition='clean', loss=1.75)
        row = paired_records([p], [v], [cp], [cv])[0]
        self.assertEqual(row['loss_benefit'], .5)
        self.assertEqual(row['clean_loss_benefit'], .25)
        self.assertEqual(row['benefit_change_from_clean'], .25)

    def test_realizations_do_not_inflate_training_seed_count(self):
        rows = []
        for seed, value in [(42, 1.), (123, 2.), (456, 3.)]:
            for rep in range(3):
                rows.append(dict(seed=seed, condition='delete_0.1', replicate=rep,
                    **{key: value+(rep-1)*.1 for key in ('parent_loss', 'variant_loss', 'parent_accuracy',
                    'variant_accuracy', 'accuracy_gap_pp', 'loss_benefit', 'benefit_change_from_clean')}))
        group = describe(rows, [42, 123, 456])[0]
        self.assertEqual(group['n_training_seeds'], 3)
        self.assertAlmostEqual(group['aggregate']['loss_benefit']['mean'], 2.)
        self.assertAlmostEqual(group['aggregate']['loss_benefit']['sample_sd'], 1.)

    def test_all_18_clean_anchors_are_required(self):
        anchors = [dict(model=str(i), seed=42, status='passed') for i in range(18)]
        require_all_anchors(anchors)
        with self.assertRaises(ValueError): require_all_anchors(anchors[:-1])
        damaged = deepcopy(anchors)
        damaged[0]['status'] = 'failed'
        with self.assertRaises(ValueError): require_all_anchors(damaged)
        with self.assertRaises(ValueError): require_all_anchors([anchors[0]]*18)

    def test_failed_source_is_preserved_and_can_resume_from_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, root = base/'old', base/'new'
            source.mkdir()
            path = source/'evidence.json'
            original = b'{"meaning": "failed evidence"}\n'
            path.write_bytes(original)
            spec = dict(failed_pilot=dict(root=str(source), files={'evidence.json': pilot.sha256(path)}))
            first = snapshot_failure(root, spec, pilot.write_json)
            self.assertEqual(path.read_bytes(), original)
            path.rename(source/'original_renamed.json')
            second = snapshot_failure(root, spec, pilot.write_json)
            self.assertEqual(first, second)
            self.assertEqual((root/'reviews/failed_alignment_pilot_v1/evidence.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()

"""Non-training tests of full-split coverage, resume guards and portable evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from project_paths import ROOT, portable_path
from scripts.adaptation_analysis import evaluate, pilot
from scripts.adaptation_analysis.records import experiment_directory, resolve_historical_path, verify_completion, verify_package


class StudyTests(unittest.TestCase):
    def test_full_split_and_conditions_are_frozen_and_machine_independent(self):
        study = evaluate.load_study()
        self.assertEqual(json.loads(json.dumps(study)), study)
        self.assertNotIn('arms', study)
        self.assertEqual(study['baseline_version'], 'paper_classification_v1')
        self.assertEqual(study['seeds'], [42, 123, 456])
        conditions = evaluate.all_conditions(study)
        self.assertEqual([x[0] for x in conditions], ['clean', 'time_scale_0.9', 'time_scale_1.1',
                                                     'delete_0.1', 'delete_0.1', 'delete_0.1'])
        self.assertEqual([x[1] for x in conditions[-3:]], [0, 1, 2])
        ids = list(range(1631))
        self.assertEqual(evaluate.full_sample_ids({'validation_indices': ids}, study), ids)
        for bad in (ids[:256], ids[:-1] + [0]):
            with self.assertRaises(ValueError):
                evaluate.full_sample_ids({'validation_indices': bad}, study)

    def test_paths_are_independent_of_cwd_and_checkout(self):
        relative = 'results/experiments/adaptation_mechanism_full_validation_v1'
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                self.assertEqual(experiment_directory(relative), ROOT / relative)
            finally:
                os.chdir(previous)
        from project_paths import local_path
        with tempfile.TemporaryDirectory() as folder:
            other = Path(folder)
            self.assertEqual(local_path(relative, root=other), (other / relative).resolve())
        for bad in ('results/valid/test', 'results/experiments/../valid/test', ROOT.parent / 'outside'):
            with self.assertRaises(ValueError):
                experiment_directory(bad)

    def test_source_hash_is_line_ending_independent(self):
        study = {'inherits': 'config/studies/adaptation_mechanism_pilot_v1.yaml'}
        with patch.object(pilot, 'source_identity', return_value={'model.py': 'raw'}):
            with patch.object(Path, 'read_bytes', return_value=b'x\r\ny\r\n'):
                windows = evaluate.source_identity(study)
            with patch.object(Path, 'read_bytes', return_value=b'x\ny\n'):
                linux = evaluate.source_identity(study)
        self.assertEqual(windows, linux)

    def test_failed_anchor_is_rejected(self):
        study = evaluate.load_study()
        with self.assertRaisesRegex(ValueError, 'anchor'):
            evaluate.check_anchor([{'correct': 0, 'loss': 3.}],
                                  {'validation': {'accuracy': 1., 'loss': 0.}}, study)


class RecordTests(unittest.TestCase):
    def test_historical_aliases_preserve_suffix_and_checkpoint_identity(self):
        index = {'path_aliases': {'results/old': 'results/new'}}
        self.assertEqual(resolve_historical_path('results/old/model/seed_42/file.json', index),
                         'results/new/model/seed_42/file.json')
        self.assertEqual(resolve_historical_path('results/older/file.json', index), 'results/older/file.json')

    def test_relocated_outputs_verify_without_original_files_and_reject_tampering(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/experiments') as folder:
            root = Path(folder)
            manifest = {'historical_identity': 'unchanged'}
            pilot.write_json(root / 'MANIFEST.json', manifest)
            output = root / 'rows.json'
            output.write_bytes(b'original')
            pilot.write_json(root / 'PILOT_COMPLETE.json', dict(status='complete',
                manifest_sha256=pilot.digest(manifest), outputs={'old/missing/rows.json': pilot.sha256(output)}))
            mapping = {'old/missing/rows.json': portable_path(output)}
            verify_completion(root, mapping)
            output.write_bytes(b'damaged')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_completion(root, mapping)

    def test_index_rejects_changed_completion(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/experiments') as folder:
            root = Path(folder)
            child = root / 'clean'
            child.mkdir()
            (child / 'PILOT_COMPLETE.json').write_text('{}')
            pilot.write_json(root / 'ARTIFACT_INDEX.json', dict(analyses={'clean': dict(
                directory=portable_path(child), completion_sha256='wrong', relocations={})}))
            with self.assertRaisesRegex(ValueError, 'completion changed'):
                verify_package(root)


class EvaluationFlowTests(unittest.TestCase):
    def test_report_averages_replicates_inside_seed_and_has_no_pc_identity(self):
        from scripts.adaptation_analysis.analysis_reporting import write_reports
        rows = []
        for seed in (42, 123, 456):
            for condition, rep, benefit in [('clean', 0, 0.), ('delete_0.1', 0, .1),
                                             ('delete_0.1', 1, .2), ('delete_0.1', 2, .3)]:
                rows.append(dict(seed=seed, sample_id=1, condition=condition, replicate=rep,
                    parent_loss=1., variant_loss=1.-benefit, benefit=benefit,
                    parent_correct=1, variant_correct=1, correctness_difference=0))
        with tempfile.TemporaryDirectory() as folder, \
             patch('scripts.adaptation_analysis.reporting._plot', return_value=(MagicMock(), MagicMock())):
            root = Path(folder)
            write_reports(root, rows, dict(analyses=['spike_deletion'], seeds=[42,123,456]))
            summary = pilot.read_json(root / 'analyses/spike_deletion/SUMMARY.json')
            self.assertNotIn('pc', summary)
            effect = next(x for x in summary['replicate_averaged_across_seed_effects'] if x['condition']=='delete_0.1')
            self.assertEqual(effect['n_training_seeds'], 3)
            self.assertAlmostEqual(effect['benefit_change']['mean'], .2)
            self.assertEqual(effect['benefit_change']['sample_sd'], 0.)

    def test_one_clean_inference_per_model_resume_and_pairing(self):
        # All inference and model loading are mocked. No training/inference runs.
        from scripts.adaptation_analysis import analysis_reporting, shd_data
        from models import pl_module
        study = evaluate.load_study()
        study['seeds'] = [42]
        runs = [dict(model=name, seed=42, checkpoint='unused.ckpt', checkpoint_sha256=name,
                     config={}, validation={'accuracy': 1., 'loss': .5})
                for name in ('MT_SE_adLIF', 'AMT_SE_adLIF')]
        manifest = dict(runs=runs, seeds=[42], sample_ids=[1, 2], environment={'device': 'cpu'})
        calls = []
        class FakeModel:
            def load_state_dict(self, *a, **k): pass
            def requires_grad_(self, *a): return self
            def eval(self): return self
            def to(self, *a): return self
        def fake_infer(model, cfg, data, inputs, device, name):
            calls.append(name)
            return [dict(row, loss=.5, correct=1) for _, _, row in inputs]
        def fake_inputs(data, ids, condition, study, features=True):
            name, rep, _, _ = condition
            return [(i, None, dict(sample_id=i, target=0, condition=name, replicate=rep,
                                  input_sha256=f'{name}-{rep}-{i}')) for i in ids]
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/experiments') as folder:
            root = Path(folder)
            with patch.object(pl_module, 'MLPSNN', return_value=FakeModel()), \
                 patch('torch.load', return_value={'state_dict': {}}), \
                 patch.object(pilot, 'input_rows', side_effect=fake_inputs), \
                 patch.object(pilot, 'infer', side_effect=fake_infer), \
                 patch.object(analysis_reporting, 'write_reports', return_value=[]):
                class FakeData:
                    sample_ids = [1, 2]
                    def __len__(self): return 2
                with patch.object(shd_data, 'SHDValidationData', return_value=FakeData()):
                    evaluate.evaluate(study, root, manifest)
                    self.assertEqual(len(calls), 12)
                    self.assertEqual(calls.count('clean'), 2)
                    verify_package(root)
                    evaluate.evaluate(study, root, manifest)
                    self.assertEqual(len(calls), 12)
                    changed = copy.deepcopy(manifest)
                    changed['sample_ids'] = [1]
                    with self.assertRaisesRegex(ValueError, 'manifest differs'):
                        evaluate.evaluate(study, root, changed)


if __name__ == '__main__':
    unittest.main()

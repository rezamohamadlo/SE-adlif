"""User-run checks for SA-MT identity and lossless parent-cache import."""
import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.adaptation_analysis import pilot
from scripts.adaptation_sa_mt import evaluate,reuse


class SAMTTests(unittest.TestCase):
    def test_registered_cohort_excludes_tuned_first_trial(self):
        study=evaluate.load_study()
        self.assertEqual(study['parent'],'MT_SE_adLIF')
        self.assertEqual(study['variant'],'SA_MT_SE_adLIF')
        self.assertEqual(study['samples'],1631)
        self.assertEqual(study['seeds'],[42,123,456])
        self.assertEqual(len(study['runs']),6)
        variants=[r for r in study['runs'] if r['model']==study['variant']]
        self.assertTrue(all('results/experiments/sa_mt_v1/' in r['checkpoint'] for r in variants))
        self.assertEqual([r['selected_epoch'] for r in variants],[188,188,99])

    def test_only_neuron_type_may_differ_in_configs(self):
        parent=dict(l1={'cell':'mt','n_neurons':360},l2={'cell':'mt'},lr=.01,dropout=.15)
        variant=copy.deepcopy(parent)
        for layer in ('l1','l2'):variant[layer]['cell']='sa_mt'
        evaluate.compare_configs(parent,variant)
        variant['dropout']=.1
        with self.assertRaises(ValueError):evaluate.compare_configs(parent,variant)

    def test_import_preserves_source_and_value_identity_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.json';output=root/'output';output.mkdir()
            rows=[dict(sample_id=1,model='MT',loss=.5)]
            old=dict(identity={'manifest_sha256':'old'},value=rows,value_sha256=pilot.digest(rows))
            pilot.write_json(source,old);before=source.read_bytes()
            manifest=dict(version='new',reuse=dict(chunks={'MT.json':dict(source_path=str(source),
                source_sha256=pilot.sha256(source),original_identity=old['identity'],value_sha256=old['value_sha256'])}))
            reuse.import_reuse(output,manifest)
            target=output/'diagnostic_rows/MT.json';current=pilot.read_json(target)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(current['value_sha256'],old['value_sha256'])
            self.assertEqual(current['identity']['manifest_sha256'],pilot.digest(manifest))
            current['value'][0]['loss']=999;pilot.write_json(target,current)
            with self.assertRaises(ValueError):reuse.import_reuse(output,manifest)

    def test_check_only_does_not_call_runtime(self):
        with patch.object(evaluate,'prepare',return_value=({},Path('unused'),{})), \
             patch('scripts.adaptation_sa_mt.runtime.evaluate',side_effect=AssertionError('Inference called')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(['--check-only']),0)


if __name__=='__main__':unittest.main()

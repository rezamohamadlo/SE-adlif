"""User-run contracts: no saved-checkpoint inference or training in tests."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_batch_precision.contracts import (
    MODES, backend_settings, batch_identity, precision_mode, repeated_target,
    reversed_batch, singleton_from_original, require_original_anchor,
)
from scripts.adaptation_batch_precision.runtime import compare_rows, evaluate


def batch():
    x = torch.zeros(3, 8, 140)
    x[1, :5, 2] = torch.arange(1., 6.)
    targets = torch.tensor([[-1, 0], [-1, 1], [-1, 2]], dtype=torch.long)
    blocks = torch.ones(3, 8, dtype=torch.long)
    blocks[:, :2] = 0
    blocks[1, 5:] = 0
    return x, targets, blocks


class LayoutContracts(unittest.TestCase):
    def test_singleton_truncates_only_synthetic_padding(self):
        old = batch();value = singleton_from_original(old, 1, 5, False)
        self.assertEqual(value[0].shape, (1, 5, 140))
        for a,b in zip(value, (old[0][1:2,:5], old[1][1:2], old[2][1:2,:5])):self.assertTrue(torch.equal(a,b))
        value[0].zero_();self.assertNotEqual(old[0].sum(), 0)

    def test_singleton_preserves_original_padding_and_mask(self):
        old=batch();value=singleton_from_original(old,1,5,True)
        self.assertEqual(value[0].shape,(1,8,140))
        self.assertTrue(torch.equal(value[2],old[2][1:2]))
        self.assertEqual(value[2][0,:2].sum(),0)

    def test_nonpadding_cannot_be_truncated(self):
        for component in (0,2):
            old=batch();old[component][1,6]=1
            with self.assertRaises(ValueError):singleton_from_original(old,1,5,False)

    def test_repetition_preserves_every_target_and_mask(self):
        old=batch();value=repeated_target(old,1)
        for a,b in zip(value,old):
            self.assertEqual(a.shape,b.shape)
            for slot in range(3):self.assertTrue(torch.equal(a[slot],b[1]))
        self.assertEqual(value[0].sum(),3*old[0][1].sum())

    def test_reverse_moves_target_slot_with_input_mask(self):
        old=batch();value=reversed_batch(old)
        for a,b in zip(value,old):self.assertTrue(torch.equal(a[2],b[0]))
        for a,b in zip(reversed_batch(value),old):self.assertTrue(torch.equal(a,b))

    def test_batch_identity_includes_masks_and_targets(self):
        old=batch();before=batch_identity(old)
        old[2][1,2]=0;self.assertNotEqual(before['blocks_sha256'],batch_identity(old)['blocks_sha256'])
        old[1][1,1]=2;self.assertNotEqual(before['targets_sha256'],batch_identity(old)['targets_sha256'])

    def test_wrong_label_dtype_rejected(self):
        x,y,b= batch()
        with self.assertRaises(ValueError):batch_identity((x,y.int(),b))


class PrecisionContracts(unittest.TestCase):
    def test_modes_and_restore(self):
        before=backend_settings()
        for mode in MODES:
            with precision_mode(mode,before) as settings:
                self.assertEqual(settings['matmul_allow_tf32'],mode==MODES[0])
                if mode==MODES[2]:self.assertFalse(settings['cudnn_allow_tf32'])
                else:self.assertEqual(settings['cudnn_allow_tf32'],before['cudnn_allow_tf32'])
            self.assertEqual(backend_settings(),before)

    def test_restore_on_failure(self):
        before=backend_settings()
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with precision_mode(MODES[2],before):raise RuntimeError('test failure')
        self.assertEqual(backend_settings(),before)


class ReportingAndExecutionContracts(unittest.TestCase):
    def test_original_anchor_cannot_pass_prediction_or_loss_mismatch(self):
        anchor=dict(mode='source_high',samples=1631,source_prediction_mismatches=0,accuracy_difference=0.,loss_difference=0.)
        spec=dict(accuracy_tolerance=1e-6,loss_tolerance=.001)
        require_original_anchor(anchor,spec)
        for key,value in [('source_prediction_mismatches',1),('loss_difference',.0189),('accuracy_difference',-.00525)]:
            with self.assertRaises(ValueError):require_original_anchor(dict(anchor,**{key:value}),spec)
        require_original_anchor(dict(anchor,mode='highest_matmul',loss_difference=.0189),spec)

    def test_comparisons_keep_fixed_precision_layout_pairs(self):
        rows=[]
        for i,mode in enumerate(MODES):
            for layout,offset in [('original_batch',0),('singleton_natural',.3),('singleton_original_padding',.1)]:
                rows.append(dict(model='MT',mode=mode,layout=layout,sample_id=3896,
                    loss=i+offset,prediction=int(offset>.2),logits=[i+offset]*20))
        values=compare_rows(rows)
        pad=next(r for r in values if r['mode']==MODES[0] and r['contrast']=='padding_at_batch_one')
        self.assertAlmostEqual(pad['loss_difference'],-.2)
        precision=next(r for r in values if r['mode']==MODES[1] and r['layout']=='singleton_natural' and r['contrast']=='precision_at_fixed_layout')
        self.assertAlmostEqual(precision['loss_difference'],1)
        self.assertEqual(precision['reference_layout'],'singleton_natural')

    def test_evaluate_uses_no_grad_and_requested_slot(self):
        seen=[]
        def fake(model,x,target,blocks,cfg):
            seen.append(torch.is_grad_enabled())
            return torch.tensor([.4,.7,.9]),torch.tensor([[2.,0.,0.],[0.,2.,0.],[0.,0.,2.]])
        with patch('scripts.adaptation_batch_precision.runtime.supervised',side_effect=fake):
            value=evaluate(None,batch(),{},torch.device('cpu'),[1])
        self.assertEqual(seen,[False]);self.assertEqual(len(value),1);self.assertEqual(value[0]['prediction'],1)

    def test_check_only_never_calls_execute(self):
        from scripts.adaptation_batch_precision.run import main
        with patch('scripts.adaptation_batch_precision.run.prepare',return_value={}), patch('scripts.adaptation_batch_precision.runtime.execute') as execute:
            self.assertEqual(main(['--check-only']),0);execute.assert_not_called()

    def test_verify_only_does_not_prepare_or_execute(self):
        from scripts.adaptation_batch_precision.run import main
        with patch('scripts.adaptation_batch_precision.run.prepare') as prepare, patch('scripts.adaptation_batch_precision.runtime.execute') as execute, patch('scripts.adaptation_batch_precision.runtime.verify',return_value={'outputs':{}}):
            self.assertEqual(main(['--verify-only','--output-root','unused']),0)
            prepare.assert_not_called();execute.assert_not_called()

    def test_manifest_guard_rejects_changed_scientific_settings(self):
        from scripts.adaptation_analysis import pilot
        from scripts.adaptation_gradient_pilot.preflight import guard
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);pilot.write_json(p/'MANIFEST.json',{'mode':'high'})
            guard(p,{'mode':'high'})
            with self.assertRaises(ValueError):guard(p,{'mode':'highest'})

    def test_portable_paths_resolve_against_checkout_root(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout=Path(directory)/'new_checkout';checkout.mkdir()
            target=checkout/'results/experiments/example/COMPLETE.json'
            self.assertEqual(portable_path(target,checkout),'results/experiments/example/COMPLETE.json')
            self.assertEqual(local_path('results/experiments/example/COMPLETE.json',checkout),target.resolve())
        self.assertEqual(local_path('config/studies/adaptation_batch_precision_v1.yaml'),ROOT/'config/studies/adaptation_batch_precision_v1.yaml')


if __name__ == '__main__':unittest.main()

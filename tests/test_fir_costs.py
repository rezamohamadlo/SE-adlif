"""CPU-only synthetic inference/accounting checks; no training or test data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from models.pl_module import build_training_model
from measure_reference_costs import input_fir_operations, measure, ReferenceCounter
import measure_run_costs
from tests.test_measure_shd_costs import tiny_model


def fixture(lags=(1,3), timesteps=5):
    base=tiny_model()
    cfg=base.hparams.cfg.copy();cfg.input_fir={'lags':list(lags)}
    with contextlib.redirect_stdout(io.StringIO()):
        fir=build_training_model(cfg).eval()
    missing=fir.load_state_dict(base.state_dict(),strict=False)
    assert missing.missing_keys==['input_fir.coefficients'] and not missing.unexpected_keys
    x=torch.ones(3,timesteps,4)
    block=torch.ones(3,timesteps,dtype=torch.long)
    with torch.inference_mode():pred=fir(x).softmax(-1).sum(1).argmax(-1)
    target=torch.stack([torch.full_like(pred,-1),pred],dim=1)
    return base,fir,(x,target,block,torch.full((3,),timesteps,dtype=torch.long))


class FIRCostTests(unittest.TestCase):
    def test_executed_identity_filter_costs_and_short_sequence_boundary(self):
        base,fir,batch=fixture()
        with torch.inference_mode():
            torch.testing.assert_close(fir(batch[0]),base(batch[0]),rtol=0,atol=0)
        # Two active taps multiply/add every padded slot, even at identity init.
        self.assertEqual(input_fir_operations(fir,batch[0]),{'tanh':8,'mul':120,'add':120})
        self.assertEqual(input_fir_operations(fir,batch[0][:,:1]),{'tanh':8})
        self.assertEqual(input_fir_operations(base,batch[0]),{})

    def test_fir_adds_arithmetic_and_uses_analog_first_projection(self):
        base,fir,batch=fixture()
        with torch.inference_mode():
            ordinary=measure(base,[batch],torch.device('cpu'),True,1,0)
            filtered=measure(fir,[batch],torch.device('cpu'),True,1,0)
        self.assertEqual(filtered['input_fir_arithmetic_ops_per_sample'],80)
        self.assertEqual(filtered['input_fir_logical_ops_per_sample'],248/3)
        self.assertEqual(filtered['analog_input_dense_macs_per_sample'],120)
        self.assertEqual(ordinary['analog_input_dense_macs_per_sample'],0)
        self.assertAlmostEqual(filtered['arithmetic_gflops_test_padded']-ordinary['arithmetic_gflops_test_padded'],80/1e9)
        self.assertAlmostEqual(filtered['logical_gops_test_padded']-ordinary['logical_gops_test_padded'],248/3/1e9)
        self.assertEqual(filtered['total_parameters']-ordinary['total_parameters'],8)
        self.assertEqual(filtered['raw_input_events_per_sample'],20)
        self.assertEqual(filtered['reproduced_test_acc'],ordinary['reproduced_test_acc'])
        self.assertNotIn('input_fir_scalar_operations_by_kind',ordinary)

    def test_signed_filter_output_is_not_counted_as_negative_events(self):
        _,fir,batch=fixture()
        with torch.no_grad():fir.input_fir.coefficients.fill_(-2)
        with torch.inference_mode():
            metrics=measure(fir,[batch],torch.device('cpu'),True,1,0)
            counter=ReferenceCounter(fir,batch[3],False)
            try:fir(batch[0])
            finally:activity=counter.close()
        expected=sum(v[1] for v in activity.values())/3
        self.assertEqual(metrics['synops_per_sample'],expected)
        self.assertGreaterEqual(expected,0)
        self.assertEqual(metrics['raw_input_events_per_sample'],20)

    def test_registered_checkpoint_factory_and_idempotent_measurement(self):
        _,fir,batch=fixture()
        cfg=fir.hparams.cfg;cfg.random_seed=42;cfg.device='cpu';cfg.dataset._target_='datasets.shd.SHDLDM'
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=d/'best.ckpt'
            torch.save(dict(state_dict=fir.state_dict(),hyper_parameters=dict(cfg=cfg),epoch=2),p)
            (d/'completed.txt').touch();(d/'TRAINING_SUMMARY.md').write_text('# Training\nPreserved history\n')
            with patch.object(measure_run_costs.shared,'selected_checkpoint',return_value=p), \
                 patch.object(measure_run_costs,'prepare_batches',return_value=([batch],'fixture')) as data, \
                 patch.object(measure_run_costs.shared,'saved_test_accuracy',return_value=1.), \
                 contextlib.redirect_stdout(io.StringIO()):
                first=measure_run_costs.collect(d,'SHD');second=measure_run_costs.collect(d,'SHD')
            self.assertEqual(first,second);data.assert_called_once()
            self.assertEqual(first['version'],'runner_test_costs_v2_fir')
            self.assertEqual(first['measurements']['input_fir_arithmetic_ops_per_sample'],80)
            self.assertIn('incompatible',first['scopes']['synops'])
            self.assertIn('Preserved history',(d/'TRAINING_SUMMARY.md').read_text())
            self.assertIn('first projection is analog',(d/'TRAINING_SUMMARY.md').read_text())
            self.assertIn('functional/count_thinning.py',first['environment']['source_sha256'])
            # An older record must not be silently reused for the new FIR scope.
            old=dict(first,version='runner_test_costs_v1')
            (d/'COST_RESULTS.json').write_text(json.dumps(old))
            before=(d/'COST_RESULTS.json').read_bytes()
            with patch.object(measure_run_costs.shared,'selected_checkpoint',return_value=p), \
                 patch.object(measure_run_costs,'prepare_batches') as data:
                with self.assertRaisesRegex(ValueError,'Preserve the older FIR cost record'):
                    measure_run_costs.collect(d,'SHD')
                data.assert_not_called()
            self.assertEqual((d/'COST_RESULTS.json').read_bytes(),before)


if __name__=='__main__':unittest.main()

"""Stage-7 common objective, native domains and exact matched relocations."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.adaptation_gradient_pilot.methods import state_hash
from scripts.adaptation_stage7.methods import objective,backward,propose,random_relocation,validate_native,budget_for

SPEC=dict(score_atol=2e-5,score_rtol=1e-5,profile_bins=20)


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[.3,.2],[-.2,.4],[.1,-.3]]))
    def forward(self,x):return x@self.weight.T


class ObjectiveTests(unittest.TestCase):
    def test_classification_objective_matches_explicit_aggregation(self):
        for agg in ['sum_softmax_over_time','mean_membrane_potentials']:
            model=Linear().requires_grad_(False).eval();x=torch.tensor([[[1.,0.],[0.,1.],[2.,1.],[0.,0.]]])
            y=torch.tensor([[-1,2]],dtype=torch.long);blocks=torch.tensor([[0,1,1,1]])
            loss,_,rows=objective(model,(x,y,blocks),{'loss_agg':agg},torch.device('cpu'))
            output=model(x)[:,1:]
            logits=output.softmax(-1).sum(1) if agg=='sum_softmax_over_time' else output.mean(1)
            torch.testing.assert_close(loss,F.cross_entropy(logits,y[:,1],reduction='none'))
            self.assertEqual(rows[0]['valid_steps'],1)

    def test_temporal_objective_matches_valid_step_mean(self):
        model=Linear().requires_grad_(False).eval();x=torch.ones(1,4,2)
        y=torch.tensor([[-1,0,1,2,0]],dtype=torch.long);blocks=torch.tensor([[1,2,3,4]])
        losses,_,rows=objective(model,(x,y,blocks),{'loss_agg':'summed_membrane_potentials'},torch.device('cpu'))
        expected=F.cross_entropy(model(x).transpose(1,2),y[:,1:],reduction='none').mean(1)
        torch.testing.assert_close(losses,expected);self.assertEqual(rows[0]['valid_steps'],4)

    def test_input_backward_finite_frozen_on_available_devices(self):
        for device in [torch.device('cpu')]+([torch.device('cuda:0')] if torch.cuda.is_available() else []):
            model=Linear().requires_grad_(False).eval().to(device);expected=state_hash(model)
            x=torch.tensor([[[1.,0.],[0.,1.],[2.,1.],[0.,0.]]]);y=torch.tensor([[-1,2]],dtype=torch.long)
            value,g=backward(model,(x,y,torch.tensor([[0,1,1,1]])),{'loss_agg':'sum_softmax_over_time'},device,x[0].numpy(),1,SPEC,expected)
            self.assertTrue(np.isfinite(g).all());self.assertTrue(value['autograd_forward_check']['passed'])
            self.assertEqual(state_hash(model),expected);self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_integer_target_dtype_required(self):
        with self.assertRaises(ValueError):objective(Linear(),(torch.ones(1,2,2),torch.tensor([[-1,1]],dtype=torch.int32),torch.ones(1,2,dtype=torch.long)),{'loss_agg':'mean_membrane_potentials'},torch.device('cpu'))


class ProjectionTests(unittest.TestCase):
    def test_counts_preserve_channel_totals_caps_and_exact_budget(self):
        original=np.array([[2,0],[0,2],[0,0],[1,0]],dtype=np.float32)
        gradient=np.array([[-1.,1.],[2.,-1.],[3.,4.],[-2.,2.]])
        proposed,allocation=propose(original,gradient,'counts',3)
        self.assertEqual(validate_native(original,proposed,'counts',3),3)
        np.testing.assert_array_equal(original.sum(0),proposed.sum(0))
        self.assertTrue(np.all(proposed<=original.max(0)))
        for seed in [1,2,3]:
            control=random_relocation(original,allocation,'counts',seed)
            self.assertEqual(validate_native(original,control,'counts',3),3)

    def test_signed_relocation_preserves_each_polarity(self):
        original=np.array([[1,0],[-1,1],[0,-1],[0,0],[0,0]],dtype=np.float32)
        gradient=np.array([[-3.,1.],[3.,-2.],[4.,3.],[-4.,4.],[5.,-5.]])
        proposed,allocation=propose(original,gradient,'signed',3)
        self.assertEqual(validate_native(original,proposed,'signed',3),3)
        for seed in [1,2,3]:
            control=random_relocation(original,allocation,'signed',seed)
            self.assertEqual(validate_native(original,control,'signed',3),3)

    def test_zero_gradient_does_not_fabricate_changes(self):
        original=np.array([[1.],[0.],[0.]])
        x,allocation=propose(original,np.zeros_like(original),'counts',1)
        np.testing.assert_array_equal(x,original);self.assertEqual(allocation,[])

    def test_budget_restarts_from_original_each_step(self):
        original=np.array([[1.],[0.],[0.],[0.]])
        first,_=propose(original,np.array([[-1.],[3.],[2.],[1.]]),'counts',1)
        second,_=propose(original,np.array([[-1.],[1.],[3.],[2.]]),'counts',1)
        self.assertEqual(validate_native(original,first,'counts',1),1)
        self.assertEqual(validate_native(original,second,'counts',1),1)

    def test_domain_or_budget_violations_fail(self):
        original=np.array([[1.],[0.],[0.]])
        with self.assertRaises(ValueError):validate_native(original,np.array([[0.],[.5],[.5]]),'counts',1)
        with self.assertRaises(ValueError):validate_native(original,np.array([[-1.],[1.],[1.]]),'signed',1)
        with self.assertRaises(ValueError):validate_native(original,np.array([[0.],[1.],[0.]]),'counts',0)

    def test_random_relocation_reproducible(self):
        x=np.array([[2.],[1.],[0.],[0.],[0.]])
        _,allocation=propose(x,np.array([[-1.],[-2.],[3.],[4.],[5.]]),'counts',2)
        np.testing.assert_array_equal(random_relocation(x,allocation,'counts',42),random_relocation(x,allocation,'counts',42))

    def test_budget_uses_event_units_not_active_cells_for_counts(self):
        self.assertEqual(budget_for(np.array([[150.],[50.]]),'counts',.01),2)
        self.assertEqual(budget_for(np.array([[1.],[-1.]]),'signed',.01),1)


class CacheAndReportingTests(unittest.TestCase):
    def test_existing_candidate_array_resumes_without_new_backward(self):
        from scripts.adaptation_stage7.runtime import Cache
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'study';cache=Cache(root,{'version':'toy'})
            identity=dict(iteration=0,current_sha256='original')
            cache.array('candidate.npz',identity,lambda:({'accepted':False},{'input':np.ones((3,2)),'gradient':np.ones((3,2))}))
            cache2=Cache(root,{'version':'toy'})
            value,_=cache2.array('candidate.npz',identity,lambda:self.fail('must not repeat backward'))
            self.assertFalse(value['accepted'])

    def test_cache_refuses_unmanifested_evidence(self):
        from scripts.adaptation_stage7.runtime import Cache
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evidence.txt').write_text('preserve')
            with self.assertRaises(ValueError):Cache(root,{'version':'toy'})

    def test_seed_mean_sd_consumes_generator_once(self):
        from scripts.adaptation_stage7.reporting import mean_sd
        r=mean_sd(v for v in [1.,2.,3.])
        self.assertEqual(r,dict(mean=2.,sample_sd=1.,n_training_seeds=3))


if __name__=='__main__':unittest.main()

"""Loss/gradient/config tests only; no Trainer.fit or optimizer steps."""
import contextlib
import copy
import io
import unittest

import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

from models.pl_module import build_training_model, validate_training_objective
from models.ours.dropout_consistency import symmetric_kl_from_logits, OBJECTIVE
from module.study_control import load_study, study_overrides, validate_config
from project_paths import ROOT
from tests.test_label_smoothing import config


def make(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_training_model(cfg)


class DropoutConsistencyTests(unittest.TestCase):
    def test_manual_symmetric_kl_value_and_gradients(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            x=torch.tensor([[.2,1.,-.5],[2.,-.1,.4]],device=device,dtype=torch.float64,requires_grad=True)
            y=torch.tensor([[.1,-.3,.4],[.3,1.,-.2]],device=device,dtype=torch.float64,requires_grad=True)
            actual=symmetric_kl_from_logits(x,y)
            p=x.softmax(-1);q=y.softmax(-1)
            expected=.5*((p*(p.log()-q.log())).sum(-1)+(q*(q.log()-p.log())).sum(-1)).mean()
            torch.testing.assert_close(actual,expected)
            for a,b in zip(torch.autograd.grad(actual,(x,y),retain_graph=True),torch.autograd.grad(expected,(x,y))):
                torch.testing.assert_close(a,b)
                self.assertTrue(torch.isfinite(a).all());self.assertGreater(float(a.abs().sum()),0)
            self.assertEqual(float(symmetric_kl_from_logits(x,x)),0.)
            torch.testing.assert_close(symmetric_kl_from_logits(x,y),symmetric_kl_from_logits(y,x))
        with self.assertRaises(ValueError):symmetric_kl_from_logits(torch.empty(0,3),torch.empty(0,3))

    def test_two_pass_terms_masks_and_first_view_metric(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for eps,alpha in [(0.,1.),(.1,1.),(0.,0.)]:
                cfg=config();cfg.training_objective=OBJECTIVE;cfg.dropout_consistency_alpha=alpha;cfg.label_smoothing_epsilon=eps
                torch.manual_seed(42);m=make(cfg).to(device).train()
                x=torch.rand(3,8,4,device=device);y=torch.tensor([[-1,0],[-1,1],[-1,2]],device=device,dtype=torch.int32)
                blocks=torch.tensor([[0,0,1,1,1,1,1,1]]*3,device=device)
                seed=117;torch.manual_seed(seed)
                a,l1,_=m.process_predictions_and_compute_losses(m(x),y,blocks)
                b,l2,_=m.process_predictions_and_compute_losses(m(x),y,blocks)
                mask=y.flatten()!=-1
                expected=.5*(l1+l2)+alpha*symmetric_kl_from_logits(a[mask].float(),b[mask].float())
                torch.manual_seed(seed);first,total,diag=m.training_terms((x,y,blocks))
                torch.testing.assert_close(first,a,rtol=0,atol=0)
                torch.testing.assert_close(total,expected,rtol=0,atol=0)
                hard=.5*(F.cross_entropy(a[mask].float(),y.flatten()[mask].long())+F.cross_entropy(b[mask].float(),y.flatten()[mask].long()))
                torch.testing.assert_close(diag['train_hard_ce'],hard)
                self.assertFalse(any(v.requires_grad for v in diag.values()))
                total.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                m.eval()
                with self.assertRaises(ValueError):m.training_terms((x,y,blocks))

    def test_no_extra_state_rng_or_inference_behavior(self):
        cfg=config();torch.manual_seed(42);baseline=make(cfg).eval();before=torch.get_rng_state()
        cfg.training_objective=OBJECTIVE;cfg.dropout_consistency_alpha=1.;cfg.label_smoothing_epsilon=.1
        torch.manual_seed(42);candidate=make(cfg).eval();after=torch.get_rng_state()
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(set(baseline.state_dict()),set(candidate.state_dict()))
        for k,v in baseline.state_dict().items():torch.testing.assert_close(v,candidate.state_dict()[k],rtol=0,atol=0)
        x=torch.rand(3,8,4);rng=torch.get_rng_state()
        torch.testing.assert_close(baseline(x),candidate(x),rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        scores=torch.randn(3,3);targets=torch.tensor([0,1,2])
        torch.testing.assert_close(candidate.loss(scores,targets),F.cross_entropy(scores,targets),rtol=0,atol=0)

    def test_invalid_or_undeclared_settings_rejected(self):
        cfg=config();cfg.training_objective=OBJECTIVE;cfg.dropout_consistency_alpha=1.;cfg.label_smoothing_epsilon=0.
        for key,value in [('dropout_consistency_alpha',-1),('dropout_consistency_alpha',True),('dropout_consistency_alpha',float('nan')),('label_smoothing_epsilon',1),('dropout',0),('auto_regression',True),('spike_delete_probability',.1)]:
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_training_objective(bad)

    def test_all_six_declared_arm_configs_and_study_guards(self):
        study=load_study('config/studies/complementary_C005_dropout_consistency_v1.yaml')
        initial=None
        for name in study['models']:
            with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
                cfg=compose(config_name='main',overrides=[f'experiment=SHD_{name}','+comparison=SHD',f'++model_variant={name}','early_stopping=false']+study_overrides(study,30,name,'SHD',42))
            full=OmegaConf.to_container(cfg,resolve=True);validate_config(full)
            self.assertEqual(full['paper_baseline_version'],'paper_classification_v1')
            invalid=copy.deepcopy(full);invalid['dropout_consistency_alpha']=.5
            with self.assertRaisesRegex(ValueError,'alpha'):validate_config(invalid)
            torch.manual_seed(42);model=make(cfg)
            if initial is None:initial={k:v.clone() for k,v in model.state_dict().items()}
            self.assertEqual(set(initial),set(model.state_dict()))
            for k,v in initial.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)

    def test_ssc_inherits_its_own_baseline_and_keeps_reference_initial_state(self):
        configs=[]
        for experiment in ('SSC_SE_adLIF','SSC_RDROP_SE_adLIF'):
            with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
                cfg=compose(config_name='main',overrides=[f'experiment={experiment}','+comparison=SSC'])
            cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));configs.append(cfg)
            self.assertEqual(cfg.paper_baseline_dataset,'SSC')
            self.assertEqual(cfg.lr,cfg.paper_baseline.datasets.SSC.lr)
            self.assertEqual(cfg.n_epochs,40)
            self.assertEqual(cfg.loss_agg,'mean_membrane_potentials')
            self.assertEqual(cfg.dataset.num_classes,35)
            self.assertEqual(cfg.l1.n_neurons,720)
            self.assertEqual(cfg.l2.n_neurons,720)
        self.assertEqual(validate_training_objective(configs[1]),OBJECTIVE)
        torch.manual_seed(42);baseline=make(configs[0]);rng=torch.get_rng_state()
        torch.manual_seed(42);candidate=make(configs[1])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(baseline.state_dict()),set(candidate.state_dict()))
        for key,value in baseline.state_dict().items():
            torch.testing.assert_close(value,candidate.state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()

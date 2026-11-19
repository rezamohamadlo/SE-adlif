"""Non-training objective/backbone checks; no Trainer or optimizer steps."""
import contextlib
import copy
import io
import unittest

import torch
from torch.nn import functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from models.pl_module import build_training_model
from models.ours.dropout_consistency import validate_dropout_consistency_config
from project_paths import ROOT
from tests.test_label_smoothing import config


def make(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_training_model(cfg)


def dth_config(objective=False):
    cfg=config()
    for layer in ('l1','l2'):
        cfg[layer].cell='dth_se_adlif'
    if objective:
        cfg.training_objective='classification_dropout_consistency_v1'
        cfg.dropout_consistency_backbone='dth_se_adlif'
        cfg.dropout_consistency_alpha=1.
        cfg.label_smoothing_epsilon=0.
    return cfg


class DTHDropoutConsistencyTests(unittest.TestCase):
    def test_explicit_backbone_and_matching_layers_required(self):
        cfg=dth_config(True)
        self.assertEqual(validate_dropout_consistency_config(cfg),(1.,0.))
        for key,value in [('dropout_consistency_backbone','se_adlif'),('dropout_consistency_backbone','unknown')]:
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):validate_dropout_consistency_config(bad)
        del cfg.dropout_consistency_backbone
        with self.assertRaises(ValueError):validate_dropout_consistency_config(cfg)
        cfg=dth_config(True);cfg.l2.cell='se_adlif'
        with self.assertRaises(ValueError):validate_dropout_consistency_config(cfg)

    def test_initial_state_rng_and_ordinary_inference_match_DTH(self):
        torch.manual_seed(42);baseline=make(dth_config()).eval();rng=torch.get_rng_state()
        torch.manual_seed(42);candidate=make(dth_config(True)).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(baseline.state_dict()),set(candidate.state_dict()))
        for key,value in baseline.state_dict().items():
            torch.testing.assert_close(value,candidate.state_dict()[key],rtol=0,atol=0)
        x=torch.rand(3,8,4);rng=torch.get_rng_state()
        with torch.no_grad():
            expected=baseline(x)
            torch.testing.assert_close(candidate(x),expected,rtol=0,atol=0)
            torch.testing.assert_close(candidate(x),expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_two_views_original_masks_loss_and_finite_DTH_gradients(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(42);m=make(dth_config(True)).to(device).train()
            x=torch.rand(3,8,4,device=device)
            y=torch.tensor([[-1,0],[-1,1],[-1,2]],dtype=torch.int32,device=device)
            blocks=torch.tensor([[0,0,1,1,1,1,1,1]]*3,device=device)
            torch.manual_seed(117)
            a,l1,_=m.process_predictions_and_compute_losses(m(x),y,blocks)
            b,l2,_=m.process_predictions_and_compute_losses(m(x),y,blocks)
            valid=y.flatten().ne(-1);pa=a[valid].float().softmax(-1);pb=b[valid].float().softmax(-1)
            kl=.5*((pa*(pa.log()-pb.log())).sum(-1)+(pb*(pb.log()-pa.log())).sum(-1)).mean()
            expected=.5*(l1+l2)+kl
            torch.manual_seed(117);first,total,diagnostics=m.training_terms((x,y,blocks))
            torch.testing.assert_close(first,a,rtol=0,atol=0)
            torch.testing.assert_close(total,expected,rtol=1e-5,atol=1e-7)
            hard=.5*(F.cross_entropy(a[valid].float(),y.flatten()[valid].long())+F.cross_entropy(b[valid].float(),y.flatten()[valid].long()))
            torch.testing.assert_close(diagnostics['train_hard_ce'],hard)
            total.backward()
            gradients=[p.grad for p in m.parameters() if p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
            thresholds=[p.grad for n,p in m.named_parameters() if n.endswith('threshold_logits')]
            self.assertEqual(len(thresholds),2)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in thresholds))

    def test_SSC_config_inherits_DTH_and_dataset_specific_paper_values(self):
        with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
            cfg=compose(config_name='main',overrides=['experiment=SSC_DTH_RDROP_SE_adLIF','+comparison=SSC'])
        cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        self.assertEqual(cfg.paper_baseline_dataset,'SSC')
        self.assertEqual(cfg.lr,cfg.paper_baseline.datasets.SSC.lr)
        self.assertEqual(cfg.n_epochs,40)
        self.assertEqual(cfg.loss_agg,'mean_membrane_potentials')
        self.assertEqual(cfg.l1.n_neurons,720)
        self.assertEqual(cfg.l2.n_neurons,720)
        self.assertEqual(cfg.dataset.num_classes,35)
        self.assertEqual(cfg.dropout_consistency_backbone,'dth_se_adlif')
        for layer in ('l1','l2'):
            self.assertEqual(cfg[layer].rho_fast,.9)
            self.assertEqual(cfg[layer].rho_slow,.99)
        self.assertEqual(validate_dropout_consistency_config(cfg),(1.,0.))


if __name__=='__main__':unittest.main()

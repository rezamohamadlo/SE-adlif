"""Objective/gradient/config checks only; no Trainer, optimizer or datasets."""
import contextlib
import copy
import io
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from models.ours.label_smoothing import TrainingOnlyLabelSmoothingLoss, OBJECTIVE
from models.pl_module import build_training_model, validate_training_objective
from module.study_control import load_study, study_overrides, validate_config
from project_paths import ROOT


def config(gain=False, smoothed=False):
    name='SHD_GAIN_SE_adLIF' if gain else 'SHD_SE_adLIF'
    with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
        cfg=compose(config_name='main',overrides=[f'experiment={name}','+comparison=SHD'])
    cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    cfg.l1.input_size=4;cfg.l1.n_neurons=5;cfg.l2.input_size=5;cfg.l2.n_neurons=5
    cfg.l_out.input_size=5;cfg.l_out.n_neurons=3;cfg.dataset.num_classes=3
    if smoothed:
        cfg.training_objective=OBJECTIVE;cfg.label_smoothing_epsilon=.1
    return cfg


class LabelSmoothingTests(unittest.TestCase):
    def test_manual_soft_target_loss_and_finite_gradients_cpu_and_cuda(self):
        devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                x=torch.tensor([[.2,1.,-.5],[2.,-.1,.4],[.3,.2,.1]],device=device,requires_grad=True)
                target=torch.tensor([1,2,-1],dtype=torch.long,device=device)
                loss=TrainingOnlyLabelSmoothingLoss(.1).to(device).train()
                actual=loss(x,target)
                logp=x[:2].log_softmax(-1)
                expected=-((.9*logp[torch.arange(2,device=device),target[:2]])+.1*logp.mean(-1)).mean()
                torch.testing.assert_close(actual,expected)
                grad=torch.autograd.grad(actual,x,retain_graph=True)[0]
                expected_grad=torch.autograd.grad(expected,x)[0]
                torch.testing.assert_close(grad,expected_grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertEqual(int(grad[2].count_nonzero()),0)
                torch.testing.assert_close(loss.last_hard_ce,F.cross_entropy(x,target,ignore_index=-1))
                self.assertFalse(loss.last_hard_ce.requires_grad)
                loss.eval()
                evaluated=loss(x,target)
                ordinary=F.cross_entropy(x,target,ignore_index=-1)
                torch.testing.assert_close(evaluated,ordinary,rtol=0,atol=0)
                torch.testing.assert_close(torch.autograd.grad(evaluated,x,retain_graph=True)[0],
                                           torch.autograd.grad(ordinary,x)[0],rtol=0,atol=0)

    def test_eval_and_zero_epsilon_equal_plain_ce(self):
        x=torch.tensor([[.2,.4,1.],[2.,1.,0.]],requires_grad=True);y=torch.tensor([2,-1])
        plain=F.cross_entropy(x,y,ignore_index=-1)
        for criterion in (TrainingOnlyLabelSmoothingLoss(.1).eval(),TrainingOnlyLabelSmoothingLoss(0).train()):
            actual=criterion(x,y)
            torch.testing.assert_close(actual,plain,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],
                                       torch.autograd.grad(plain,x,retain_graph=True)[0],rtol=0,atol=0)
            self.assertIsNone(criterion.last_hard_ce)
            self.assertFalse(criterion.state_dict())

    def test_factory_initialization_rng_state_keys_and_eval_parity(self):
        for gain in (False,True):
            with self.subTest(gain=gain),contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(123);ordinary=build_training_model(config(gain));rng=torch.get_rng_state()
                torch.manual_seed(123);smoothed=build_training_model(config(gain,True));other_rng=torch.get_rng_state()
                self.assertTrue(torch.equal(rng,other_rng))
                self.assertEqual(set(ordinary.state_dict()),set(smoothed.state_dict()))
                self.assertEqual(sum(p.numel() for p in ordinary.parameters()),sum(p.numel() for p in smoothed.parameters()))
                for key,value in ordinary.state_dict().items():
                    torch.testing.assert_close(value,smoothed.state_dict()[key],rtol=0,atol=0)
                smoothed.load_state_dict(ordinary.state_dict(),strict=True)
                x=torch.randn(2,7,4);ordinary.eval();smoothed.eval()
                with torch.no_grad():
                    a=ordinary(x);b=smoothed(x)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                targets=torch.tensor([[-1,0],[-1,2]],dtype=torch.int32)
                blocks=torch.ones(2,7,dtype=torch.long);blocks[:,:2]=0
                _,aloss,_=ordinary.process_predictions_and_compute_losses(a,targets,blocks)
                _,bloss,_=smoothed.process_predictions_and_compute_losses(b,targets,blocks)
                torch.testing.assert_close(aloss,bloss,rtol=0,atol=0)

    def test_training_changes_loss_only_after_original_aggregation_and_mask(self):
        with contextlib.redirect_stdout(io.StringIO()):model=build_training_model(config(True,True))
        model.train();scores=torch.randn(2,7,3,requires_grad=True)
        target=torch.tensor([[-1,0],[-1,2]],dtype=torch.int32)
        blocks=torch.ones(2,7,dtype=torch.long);blocks[:,:2]=0
        reduced,loss,_=model.process_predictions_and_compute_losses(scores,target,blocks)
        valid=target.flatten().ne(-1)
        expected=F.cross_entropy(reduced[valid].float(),target.flatten()[valid].long(),label_smoothing=.1)
        torch.testing.assert_close(loss,expected)
        loss.backward();self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.equal(target,torch.tensor([[-1,0],[-1,2]],dtype=torch.int32)))

    def test_invalid_or_undeclared_objective_combinations_rejected(self):
        for epsilon in (True,None,'0.1',float('nan'),float('inf'),-.1,1.):
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):
                TrainingOnlyLabelSmoothingLoss(epsilon)
        cfg=config(True,True)
        for values in ({'auto_regression':True},{'label_smoothing_epsilon':0},{'spike_delete_probability':.1}):
            invalid=copy.deepcopy(cfg)
            for key,value in values.items():invalid[key]=value
            with self.subTest(values=values),self.assertRaises(ValueError):validate_training_objective(invalid)
        cfg.input_fir.mode='causal'
        with self.assertRaisesRegex(ValueError,'gain-only'):validate_training_objective(cfg)

    def test_declared_four_arm_configs_and_epsilon_guards(self):
        study=load_study('config/studies/complementary_B002_primary_v1.yaml')
        frozen=dict(validation_speaker_ids=[0,3],role='development')
        with patch('datasets.shd_speaker_split.load_manifest',return_value=frozen):
            for model in study['models']:
                overrides=study_overrides(study,30,model,'SHD',42)
                with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
                    cfg=compose(config_name='main',overrides=[f'experiment=SHD_{model}','+comparison=SHD',f'++model_variant={model}','early_stopping=false']+overrides)
                resolved=OmegaConf.to_container(cfg,resolve=True);validate_config(resolved)
                self.assertEqual(resolved['n_epochs'],300)
                self.assertEqual(resolved['dataset']['validate_on'],'speaker_holdout')
                self.assertEqual(resolved['training_objective'],study['training_objectives'][model])
                invalid=copy.deepcopy(resolved);invalid['label_smoothing_epsilon']=.2
                with self.assertRaisesRegex(ValueError,'epsilon'):validate_config(invalid)

    def test_all_dataset_objective_configs_inherit_baseline(self):
        for dataset in ('SHD','SSC','ECG'):
            for model in ('LS_SE_adLIF','GAIN_LS_SE_adLIF'):
                exp=f'{dataset}_{model}'+('_2layer' if dataset=='ECG' else '')
                with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
                    cfg=compose(config_name='main',overrides=[f'experiment={exp}',f'+comparison={dataset}'])
                resolved=OmegaConf.to_container(cfg,resolve=True)
                self.assertEqual(resolved['lr'],resolved['paper_baseline']['datasets'][dataset]['lr'])
                self.assertEqual(resolved['n_epochs'],resolved['paper_baseline']['datasets'][dataset]['n_epochs'])
                self.assertEqual(validate_training_objective(cfg),OBJECTIVE)

    def test_runner_child_preserves_speaker_protocol_and_objective_metadata(self):
        import runner
        study=load_study('config/studies/complementary_B002_primary_v1.yaml')
        with patch.object(runner,'RUN_LOGDIR',study['output_root']), \
             patch.object(runner,'VALIDATION_ONLY',True), \
             patch.object(runner,'COMPLEMENTARY_STUDY',study), \
             patch.object(runner,'STAGE_CEILING',30), \
             patch('datasets.shd_speaker_split.load_manifest',return_value=dict(validation_speaker_ids=[0,3],role='development')):
            job=runner.build_plan('GAIN_LS_SE_adLIF',seeds=(42,),datasets=('SHD',),inspect_existing=False)[0]
        cfg=job['config']
        self.assertTrue(cfg['evaluation_protocol'].startswith('shd_speaker_label_smoothing_v1_primary_0_3'))
        self.assertTrue(cfg['evaluation_protocol'].endswith(OBJECTIVE))
        self.assertEqual(cfg['label_smoothing_epsilon'],.1)
        self.assertTrue(any('training_objective=' in x for x in cfg['paper_baseline_deviations']))
        self.assertFalse(cfg['run_test'])


if __name__=='__main__':unittest.main()

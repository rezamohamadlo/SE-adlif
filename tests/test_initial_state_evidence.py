"""Initial-state logging/config tests without Trainer.fit or optimizer steps."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

from models.pl_module import build_training_model
from module.initial_state import state_fingerprint, record_initial_state
from module.study_control import StudyProgress, load_study, study_overrides, validate_config
from project_paths import ROOT


STUDY = 'config/studies/complementary_C003_static_factorial_v1.yaml'


def configured(model):
    study = load_study(STUDY)
    with patch('datasets.shd_speaker_split.load_manifest',
               return_value=dict(validation_speaker_ids=[0, 3], role='development')):
        overrides = study_overrides(study, 30, model, 'SHD', 42)
        with initialize_config_dir(config_dir=str(ROOT/'config'), version_base=None):
            cfg = compose(config_name='main', overrides=[f'experiment=SHD_{model}',
                '+comparison=SHD', f'++model_variant={model}', 'early_stopping=false']+overrides)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        validate_config(cfg)
    return cfg


class StateFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)
        self.register_buffer('counter', torch.tensor(7, dtype=torch.int64))
        self.register_buffer('noncontiguous', torch.arange(6).reshape(2,3).t())
        self.register_buffer('bf16', torch.tensor([.25, .5], dtype=torch.bfloat16))


class InitialStateEvidenceTests(unittest.TestCase):
    def test_logging_preserves_values_rng_and_flags_cpu_and_cuda(self):
        devices = ['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(117)
                model = StateFixture().to(device)
                cfg = dict(model_variant='fixture', random_seed=42, paper_baseline_dataset='SHD',
                    dataset=dict(speaker_split_fingerprint='frozen'), evaluation_protocol='fixture')
                before = copy.deepcopy(model.state_dict())
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith('cuda') else None
                flags = (torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
                         torch.backends.cuda.matmul.allow_tf32, model.training)
                directory = Path(temporary)/'nested'/'seed_42'
                record = record_initial_state(directory, model, cfg, '32-true')
                self.assertEqual(record['optimizer_updates'], 0)
                self.assertEqual(record['tensor_count'], len(before))
                for name, value in before.items():
                    torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
                self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                if cuda_rng is not None:
                    self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
                self.assertEqual(flags, (torch.get_float32_matmul_precision(),
                    torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32, model.training))
                path = directory/'INITIAL_MODEL_STATE.json'
                original_bytes = path.read_bytes()
                self.assertEqual(record, record_initial_state(directory, model, cfg, '32-true'))
                self.assertEqual(path.read_bytes(), original_bytes)
                with torch.no_grad():
                    model.projection.weight.add_(1)
                with self.assertRaisesRegex(ValueError, 'conflicts'):
                    record_initial_state(directory, model, cfg, '32-true')
                self.assertEqual(path.read_bytes(), original_bytes)

    def test_tensor_values_and_trainability_are_distinguished(self):
        model = StateFixture()
        first = state_fingerprint(model)
        model.projection.weight.requires_grad_(False)
        second = state_fingerprint(model)
        self.assertEqual(first['model_state_sha256'], second['model_state_sha256'])
        self.assertNotEqual(first['tensors']['projection.weight']['requires_grad'],
                            second['tensors']['projection.weight']['requires_grad'])
        with torch.no_grad():
            model.counter.add_(1)
        third = state_fingerprint(model)
        self.assertNotEqual(second['model_state_sha256'], third['model_state_sha256'])
        self.assertNotEqual(second['tensors']['counter']['sha256'], third['tensors']['counter']['sha256'])
        self.assertEqual(second['tensors']['projection.weight']['sha256'], third['tensors']['projection.weight']['sha256'])

    def test_initial_evidence_rejects_resume_or_weights_only_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(model_variant='fixture', random_seed=42, paper_baseline_dataset='SHD')
            for key in ('ckpt_path','init_ckpt_path'):
                invalid = dict(cfg, **{key:'results/own/last.ckpt'})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'fresh training only'):
                    record_initial_state(temporary, StateFixture(), invalid, '32-true')
            self.assertFalse((Path(temporary)/'INITIAL_MODEL_STATE.json').exists())

    def test_four_actual_architectures_share_initial_common_tensors(self):
        records = {}
        for model in load_study(STUDY)['models']:
            cfg = OmegaConf.create(configured(model))
            with contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(42)
                instance = build_training_model(cfg)
            records[model] = state_fingerprint(instance)
        base = records['SE_adLIF']
        self.assertEqual(base['model_state_sha256'], records['LS_SE_adLIF']['model_state_sha256'])
        self.assertEqual(records['STATIC_SE_adLIF']['model_state_sha256'],
                         records['STATIC_LS_SE_adLIF']['model_state_sha256'])
        for name, value in base['tensors'].items():
            for model in records:
                self.assertEqual(value['sha256'], records[model]['tensors'][name]['sha256'], (model,name))

    def test_config_requires_its_declared_recording_flag(self):
        cfg = configured('STATIC_SE_adLIF')
        self.assertTrue(cfg['study_record_initial_state'])
        self.assertEqual(cfg['n_epochs'],300)
        self.assertEqual(cfg['l1']['cell'],'static_threshold_se_adlif')
        self.assertEqual(cfg['training_objective'],'classification')
        for flag in (False, 1, 'true'):
            invalid = copy.deepcopy(cfg)
            invalid['study_record_initial_state'] = flag
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'Initial-state recording'):
                validate_config(invalid)

    def test_callback_records_only_fresh_opted_in_zero_step_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            for enabled, resumed in ((False,False),(True,False),(True,True)):
                cfg = OmegaConf.create(dict(logdir=temporary, study_file=STUDY, evaluation_protocol='fixture',
                    study_record_initial_state=enabled, ckpt_path='results/own/last.ckpt' if resumed else None,
                    init_ckpt_path=None, study_stage=1, study_idea='fixture', paper_baseline_dataset='SHD',
                    model_variant='fixture', random_seed=42, dataset=dict(random_seed=42),
                    batch_size=1024, lr=.01, stage_ceiling=30, n_epochs=300, study_hypothesis='fixture'))
                study = dict(output_root='results/experiments/fixture/primary',plan='fixture.md')
                trainer = SimpleNamespace(global_step=20 if resumed else 0, precision='32-true')
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch('module.study_control.load_study', return_value=study), \
                     patch('module.study_control.atomic_json'), \
                     patch('module.study_control.validation_rows',return_value=[]), \
                     patch('module.study_control.forward_identity',return_value={}), \
                     patch('module.study_control.checkpoint',return_value={'epoch':4}), \
                     patch('module.initial_state.record_initial_state',return_value=dict(model_state_sha256='abc',tensor_count=1)) as recorder:
                    StudyProgress(cfg).on_fit_start(trainer, nn.Linear(2,2))
                    self.assertEqual(recorder.call_count, int(enabled and not resumed))


if __name__ == '__main__':
    unittest.main()

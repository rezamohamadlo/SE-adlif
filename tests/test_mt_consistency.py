import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from functional.temporal_consistency import temporal_js_loss
from models.ours.mt_consistency import MTConsistencyMLPSNN
from tests.test_measure_shd_costs import tiny_model
import train_mt_consistency as launcher
import project_paths


class TemporalConsistencyTests(unittest.TestCase):
    def test_constant_predictions_and_short_sequences_have_zero_loss(self):
        x = torch.randn(2, 1, 3).expand(2, 12, 3).clone().requires_grad_()
        targets = torch.tensor([[-1, 1], [-1, 2]])
        blocks = torch.ones(2, 12, dtype=torch.long)
        loss = temporal_js_loss(x, targets, blocks)
        self.assertAlmostEqual(loss.item(), 0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(temporal_js_loss(x[:, :3], targets, blocks[:, :3]).item(), 0)

    def test_ignored_padding_and_class_boundaries_have_no_penalty(self):
        # Abrupt predictions only change across target labels, not within one.
        x = torch.tensor([[[20., -20.]] * 4 + [[-20., 20.]] * 4 + [[99., -99.]] * 3],
                         requires_grad=True)
        targets = torch.tensor([[-1, 0, 1]])
        blocks = torch.tensor([[1] * 4 + [2] * 4 + [0] * 3])
        loss = temporal_js_loss(x, targets, blocks, window=3)
        self.assertAlmostEqual(loss.item(), 0, places=6)
        loss.backward()
        torch.testing.assert_close(x.grad[:, 8:], torch.zeros_like(x.grad[:, 8:]))

    def test_variation_within_same_label_has_positive_loss_and_gradients(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                torch.manual_seed(42)
                x = (torch.randn(2, 15, 3, device=device) * 3).requires_grad_()
                targets = torch.tensor([[-1, 1], [-1, 2]], device=device, dtype=torch.int32)
                blocks = torch.ones(2, 15, device=device, dtype=torch.long)
                loss = temporal_js_loss(x, targets, blocks)
                self.assertGreater(loss.item(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_all_ignored_and_extreme_logits_remain_finite(self):
        x = torch.tensor([[[1.e6, -1.e6], [-1.e6, 1.e6]] * 6], requires_grad=True)
        loss = temporal_js_loss(x, torch.tensor([[-1]]), torch.zeros(1, 12, dtype=torch.long))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):
            temporal_js_loss(x, torch.tensor([[-1]]), torch.zeros(1, 12), window=0)

    def test_inference_checkpoint_and_zero_weight_loss_match_ordinary_mt(self):
        baseline = tiny_model('mt_se_adlif')
        cfg = OmegaConf.create(OmegaConf.to_container(baseline.hparams.cfg, resolve=True))
        cfg.temporal_consistency = dict(weight=0., window=5)
        with contextlib.redirect_stdout(io.StringIO()):
            candidate = MTConsistencyMLPSNN(cfg).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(2, 15, 4) * 20
        torch.testing.assert_close(candidate(x), baseline(x), rtol=0, atol=0)
        targets = torch.tensor([[-1, 0], [-1, 1]], dtype=torch.int32)
        blocks = torch.ones(2, 15, dtype=torch.long)
        with patch.object(candidate, 'update_and_log_metrics'):
            actual = candidate.training_step((x, targets, blocks), 0)
        _, expected, _ = baseline.process_predictions_and_compute_losses(baseline(x), targets, blocks)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_configs_keep_baseline_and_mt_architecture_with_separate_protocol(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / 'config'), version_base=None):
            for ds in ['SHD', 'SSC', 'ECG']:
                suffix = '_2layer' if ds == 'ECG' else ''
                cfg = compose(config_name='main', overrides=[f'experiment={ds}_MT_consistency{suffix}'])
                mt = compose(config_name='main', overrides=[f'experiment={ds}_MT_SE_adLIF{suffix}',
                                                          f'+comparison={ds}'])
                for key in ['dataset', 'l1', 'l2', 'lr', 'n_epochs', 'batch_size', 'early_stopping',
                            'loss_agg', 'paper_baseline_version']:
                    self.assertEqual(cfg[key], mt[key], key)
                self.assertFalse(cfg.run_test)
                self.assertEqual(cfg.evaluation_protocol, 'mt_temporal_consistency_v1')
                self.assertEqual(cfg.temporal_consistency.weight, .1)
                self.assertTrue(cfg.logdir.startswith('results/experiments/mt_consistency_v1/'))
                record = launcher.manifest_for(cfg)
                self.assertEqual(record['paper_baseline_dataset'], ds)
                self.assertIn('functional/temporal_consistency.py', record['source_sha256'])

    def test_training_adds_only_auxiliary_loss_with_single_forward(self):
        baseline = tiny_model('mt_se_adlif')
        cfg = OmegaConf.create(OmegaConf.to_container(baseline.hparams.cfg, resolve=True))
        cfg.temporal_consistency = dict(weight=.1, window=5)
        with contextlib.redirect_stdout(io.StringIO()):
            model = MTConsistencyMLPSNN(cfg)
        outputs = torch.randn(2, 15, 2, requires_grad=True)
        targets = torch.tensor([[-1, 0], [-1, 1]], dtype=torch.int32)
        blocks = torch.ones(2, 15, dtype=torch.long)
        _, supervised, _ = model.process_predictions_and_compute_losses(outputs, targets, blocks)
        expected = supervised + .1 * temporal_js_loss(outputs, targets, blocks)
        with patch.object(model, 'forward', return_value=outputs) as forward, patch.object(
                model, 'update_and_log_metrics'), patch.object(model, 'log'):
            actual = model.training_step((torch.zeros(2, 15, 4), targets, blocks), 0)
        forward.assert_called_once()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(outputs.grad).all())

    def test_launcher_guards_resume_and_preserves_saved_config_on_mismatch(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / 'config'), version_base=None):
            cfg = compose(config_name='main', overrides=['experiment=SHD_MT_consistency'])
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            directory = checkout / cfg.logdir
            (directory / '.hydra').mkdir(parents=True)
            record = dict(identity='frozen')
            (directory / 'CONSISTENCY_MANIFEST.json').write_text(json.dumps(record))
            (directory / 'ckpt').mkdir()
            (directory / 'ckpt/last.ckpt').touch()  # Never deserialized in this test.
            result = dict(best_epoch=299, final_test='not_evaluated')
            try:
                os.chdir(directory)
                with patch.object(launcher, 'ROOT', checkout), patch.object(
                        launcher, 'local_path', side_effect=lambda path: checkout / path), patch.object(
                        launcher, 'manifest_for', return_value=record), patch.object(
                        launcher, 'completed_validation', side_effect=[None, result]), patch.object(
                        launcher.training.main, '__wrapped__') as train:
                    launcher.run_job(cfg)
                    train.assert_called_once()
                    self.assertTrue(str(train.call_args.args[0].ckpt_path).endswith('last.ckpt'))
                cfg.ckpt_path = None
                saved = (directory / '.hydra/config.yaml').read_text()
                with patch.object(launcher, 'ROOT', checkout), patch.object(
                        launcher, 'local_path', side_effect=lambda path: checkout / path), patch.object(
                        launcher, 'manifest_for', return_value=dict(identity='changed')), patch.object(
                        launcher.training.main, '__wrapped__') as train:
                    with self.assertRaises(ValueError):
                        launcher.run_job(cfg)
                    train.assert_not_called()
                self.assertEqual(saved, (directory / '.hydra/config.yaml').read_text())
            finally:
                os.chdir(original_cwd)

    def test_completion_rejects_duplicate_epochs_and_test_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'logs/mlp_snn/version_0/metrics.csv'
            path.parent.mkdir(parents=True)
            def save(rows):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, ['epoch', 'val_acc', 'val_loss', 'test_acc'])
                    writer.writeheader()
                    writer.writerows(rows)
            rows = [dict(epoch=i, val_acc=.8+i*.01, val_loss=.4) for i in range(3)]
            save(rows)
            self.assertEqual(launcher.completed_validation(directory, 3)['best_epoch'], 2)
            self.assertIsNone(launcher.completed_validation(directory, 4))
            save(rows + [rows[0]])
            with self.assertRaises(ValueError):
                launcher.completed_validation(directory, 3)
            save(rows + [dict(epoch=2, test_acc=.9)])
            with self.assertRaises(ValueError):
                launcher.completed_validation(directory, 3)

    def test_portability_from_other_checkout_and_hydra_working_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'checkout'
            shutil.copytree(launcher.ROOT / 'config', root / 'config')
            hydra_cwd = Path(temporary) / 'hydra_job'
            hydra_cwd.mkdir()
            try:
                os.chdir(hydra_cwd)
                with patch.object(project_paths, 'ROOT', root), initialize_config_dir(
                        config_dir=str(root / 'config'), version_base=None):
                    cfg = compose(config_name='main', return_hydra_config=True,
                                  overrides=['experiment=SHD_MT_consistency'])
                    self.assertEqual(Path(cfg.hydra.run.dir).resolve(), (root / cfg.logdir).resolve())
                    self.assertEqual(project_paths.local_path(cfg.dataset.data_path), (root / 'data').resolve())
            finally:
                os.chdir(original_cwd)

    def test_summary_uses_consistency_launcher(self):
        summary = object.__new__(launcher.ConsistencySummary)
        summary.cfg = OmegaConf.create(dict(exp_name='SHD_MT_consistency', random_seed=42))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'TRAINING_SUMMARY.md'
            summary._replace_summary(path, '## Resume command\n\n```powershell\nwrong command\n```\n')
            text = path.read_text()
        self.assertIn('train_mt_consistency.py', text)
        self.assertNotIn('wrong command', text)


if __name__ == '__main__':
    unittest.main()

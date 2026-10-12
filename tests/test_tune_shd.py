import contextlib
import copy
import csv
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.pl_module import MLPSNN
import run
import tune_shd


class TinyData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        inputs = torch.randn(4, 14, 3)
        targets = torch.tensor([[-1, 0], [-1, 1], [-1, 0], [-1, 1]])
        blocks = torch.tensor([[0] * 10 + [1] * 4] * 4)
        self.samples = TensorDataset(inputs, targets, blocks)

    def train_dataloader(self):
        return DataLoader(self.samples, batch_size=2)

    def val_dataloader(self):
        return DataLoader(self.samples, batch_size=2)

    def test_dataloader(self):
        raise AssertionError('Screening must never load test data')


class SweepTests(unittest.TestCase):
    def setUp(self):
        with initialize_config_dir(config_dir=str(tune_shd.ROOT / 'config'), version_base=None):
            self.source = OmegaConf.to_container(compose(config_name='main', overrides=[
                'experiment=SHD_SE_adLIF', 'dataset.validate_on=0.2']), resolve=True)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_twenty_recipes_keep_split_and_architecture_and_reset_training(self):
        recipes = tune_shd.candidates()
        self.assertEqual(len(recipes), 20)
        self.assertEqual(len({(r['lr'], r['batch_size'], r['scheduler']) for r in recipes}), 20)
        for candidate in recipes:
            cfg = tune_shd.trial_config(self.source, self.root/'source.ckpt', candidate,
                                       self.root/candidate['id'], 10)
            self.assertEqual(cfg['dataset']['validate_on'], .2)
            self.assertEqual(cfg['dataset']['random_seed'], 42)
            self.assertEqual(cfg['l1'], self.source['l1'])
            self.assertEqual(cfg['dropout'], self.source['dropout'])
            self.assertEqual(cfg['batch_size'], cfg['dataset']['batch_size'])
            self.assertFalse(cfg['run_test'])
            self.assertFalse(cfg['early_stopping'])
            self.assertIsNone(cfg['ckpt_path'])
            self.assertIsNotNone(cfg['init_ckpt_path'])
        full = tune_shd.trial_config(self.source, self.root/'source.ckpt', recipes[0],
                                    self.root/'full_train/c01/seed_123', 300, 123, True)
        self.assertIsNone(full['init_ckpt_path'])
        self.assertTrue(full['run_test'])
        self.assertEqual(full['dataset']['random_seed'], 42)
        self.assertEqual(full['random_seed'], 123)

    def test_ranking_excludes_test_and_baseline_and_rejects_changed_resume(self):
        for trial, val, test in [('c01', .80, .99), ('c02', .85, .50)]:
            directory = self.root/trial
            metrics = directory/'logs/mlp_snn/version_0/metrics.csv'
            metrics.parent.mkdir(parents=True)
            with metrics.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['step', 'val_acc', 'val_loss', 'test_acc'])
                writer.writeheader()
                writer.writerow(dict(step=0, val_acc=.9, val_loss=.1))
                writer.writerow(dict(step=2, val_acc=val, val_loss=.2))
                writer.writerow(dict(step=3, test_acc=test))
            (directory/'SWEEP_BASELINE.json').write_text(json.dumps(dict(val_acc=.9)))
            (directory/'completed.txt').touch()
        rows = tune_shd.report(self.root)
        self.assertEqual(rows[0]['id'], 'c02')
        self.assertAlmostEqual(rows[0]['delta_pp'], -5)
        directory = self.root/'new'
        directory.mkdir()
        manifest = dict(config=self.source)
        (directory/'SWEEP_CONFIG.json').write_text(json.dumps(manifest))
        self.assertEqual(tune_shd.check_directory(directory, manifest), 'new')
        (directory/'ckpt').mkdir()
        (directory/'ckpt/last.ckpt').touch()
        self.assertEqual(tune_shd.check_directory(directory, manifest), 'resume')
        with self.assertRaises(ValueError):
            tune_shd.check_directory(directory, dict(config={}))

    def test_scheduler_and_learning_rate_choices_reach_optimizer(self):
        for recipe in tune_shd.candidates()[:4]:
            cfg = tune_shd.trial_config(self.source, self.root/'source.ckpt', recipe,
                                       self.root/recipe['id'], 10)
            with contextlib.redirect_stdout(io.StringIO()):
                model = MLPSNN(OmegaConf.create(cfg))
            result = model.configure_optimizers()
            optimizer = result if recipe['scheduler'] == 'constant' else result['optimizer']
            self.assertEqual(optimizer.param_groups[0]['lr'], recipe['lr'])
            if recipe['scheduler'] == 'plateau':
                scheduler = result['lr_scheduler']['scheduler']
                for _ in range(4):
                    scheduler.step(.9)
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], recipe['lr'] * .5)
            elif recipe['scheduler'] == 'cosine':
                scheduler = result['lr_scheduler']['scheduler']
                for _ in range(10):
                    optimizer.step()
                    scheduler.step()
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], recipe['lr'] * .1)

    def test_real_weights_initialization_and_one_epoch_without_test(self):
        cfg = copy.deepcopy(self.source)
        cfg['l1'].update(input_size=3, n_neurons=4)
        cfg['l2'].update(input_size=4, n_neurons=4)
        cfg['l_out']['input_size'] = 4
        cfg['l_out']['n_neurons'] = 2
        cfg['dataset']['num_classes'] = 2
        cfg['device'] = 'cpu'
        cfg['batch_size'] = cfg['dataset']['batch_size'] = 2
        checkpoint = self.root/'source.ckpt'
        with contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(OmegaConf.create(cfg))
        torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(cfg=cfg), epoch=120), checkpoint)
        source, provenance = tune_shd.reference(checkpoint)
        self.assertEqual(provenance['epoch'], 120)
        trial = tune_shd.trial_config(source, checkpoint, tune_shd.candidates()[3], self.root, 1)
        trial['batch_size'] = trial['dataset']['batch_size'] = 2
        before = Path.cwd()
        handlers = list(logging.getLogger().handlers)
        try:
            os.chdir(self.root)
            with patch.object(run.hydra.utils, 'instantiate', return_value=TinyData()):
                with contextlib.redirect_stdout(io.StringIO()):
                    run.main.__wrapped__(OmegaConf.create(trial))
        finally:
            os.chdir(before)
            for handler in list(logging.getLogger().handlers):
                if handler not in handlers:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
        baseline = json.loads((self.root/'SWEEP_BASELINE.json').read_text())
        self.assertIn('val_acc', baseline)
        saved = torch.load(self.root/'ckpt/last.ckpt', weights_only=False, map_location='cpu')
        self.assertEqual(saved['epoch'], 0)  # source epoch 120 must not carry over
        self.assertEqual(saved['global_step'], 2)
        self.assertEqual(saved['lr_schedulers'][0]['T_max'], 1)
        self.assertAlmostEqual(saved['optimizer_states'][0]['param_groups'][0]['lr'], .00001)
        self.assertIn('val_acc', tune_shd.scores(self.root))
        self.assertNotIn('test_acc', (self.root/'logs/mlp_snn/version_0/metrics.csv').read_text())


if __name__ == '__main__':
    unittest.main()

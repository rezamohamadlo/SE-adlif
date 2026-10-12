import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from evaluation_protocol import apply_protocol, load_protocol, protocol_settings


class EvaluationProtocolTests(unittest.TestCase):
    def test_direct_runner_matches_shared_settings_for_all_models(self):
        root = Path(__file__).resolve().parents[1]
        protocol = load_protocol()
        with initialize_config_dir(config_dir=str(root / 'config'), version_base=None):
            for model in ('SE_adLIF', 'DTH_SE_adLIF', 'DA_SE_adLIF', 'MR_SE_adLIF', 'MT_SE_adLIF'):
                for dataset in protocol['datasets']:
                    experiment = f'{dataset}_{model}' + ('_2layer' if dataset == 'ECG' else '')
                    cfg = compose(config_name='main', overrides=[
                        f'experiment={experiment}',
                        'n_epochs=1', 'dataset.batch_size=2'])
                    with open_dict(cfg):
                        cfg.evaluation = protocol
                    cell = cfg.l1.cell
                    with patch('evaluation_protocol.OmegaConf.save'):
                        apply_protocol(cfg)
                    for key, value in protocol_settings(protocol, dataset).items():
                        self.assertEqual(OmegaConf.select(cfg, key), value)
                    self.assertEqual(cfg.l1.cell, cell)

    def test_rejects_resume_with_different_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.hydra').mkdir()
            OmegaConf.save(OmegaConf.create({'exp_name': 'SHD_SE_adLIF',
                                            'random_seed': 42,
                                            'dataset': {'validate_on': 'test'}}),
                           root / '.hydra/config.yaml')
            cfg = OmegaConf.create({'exp_name': 'SHD_SE_adLIF', 'random_seed': 42,
                                    'ckpt_path': str(root / 'ckpt/last.ckpt'),
                                    'evaluation': load_protocol()})
            with self.assertRaisesRegex(ValueError, 'does not match'):
                apply_protocol(cfg)

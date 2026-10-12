"""Shared evaluation settings for Hydra and experiment launchers."""
import json
from pathlib import Path

from omegaconf import OmegaConf, open_dict


PROTOCOL_PATH = Path(__file__).resolve().parent / 'evaluation.yaml'


def load_protocol(path=PROTOCOL_PATH):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def protocol_settings(protocol, dataset):
    if dataset not in protocol['datasets']:
        raise ValueError(f'Unsupported protocol dataset: {dataset}')
    return {**protocol['common'], **protocol['datasets'][dataset]}


def protocol_overrides(protocol, dataset):
    return [f'++{key}={json.dumps(value)}'
            for key, value in protocol_settings(protocol, dataset).items()]


def apply_protocol(cfg):
    """Explicitly selected protocols take precedence over runner overrides."""
    if 'evaluation_file' in cfg:
        with open_dict(cfg):
            cfg.evaluation = load_protocol(cfg.evaluation_file)
    if 'evaluation' not in cfg:
        return
    protocol = OmegaConf.to_container(cfg.evaluation, resolve=True)
    dataset = cfg.exp_name.split('_', 1)[0].upper()
    if cfg.random_seed not in protocol['seeds']:
        raise ValueError(f"Protocol training seeds must be one of {protocol['seeds']}")
    settings = protocol_settings(protocol, dataset)
    if cfg.get('ckpt_path'):
        saved_path = Path(cfg.ckpt_path).resolve().parent.parent / '.hydra/config.yaml'
        if not saved_path.exists():
            raise ValueError('Cannot verify resumed checkpoint protocol: missing saved Hydra config')
        saved = OmegaConf.load(saved_path)
        mismatches = [key for key, value in settings.items()
                      if OmegaConf.select(saved, key) != value]
        if saved.exp_name != cfg.exp_name or saved.random_seed != cfg.random_seed or mismatches:
            raise ValueError(f'Resumed checkpoint does not match selected protocol: {mismatches}')
    with open_dict(cfg):
        for key, value in settings.items():
            OmegaConf.update(cfg, key, value, force_add=True)
        cfg.evaluation_protocol = protocol['name']
    # Persist the effective settings, including values applied after composition.
    OmegaConf.save(cfg, '.hydra/config.yaml')

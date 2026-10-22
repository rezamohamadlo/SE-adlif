"""Isolated validation-first MT temporal-consistency experiment; user launches only."""

import csv
import hashlib
import json
from pathlib import Path
import platform
import re
from unittest.mock import patch

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

import run as training
from models.ours.mt_consistency import MTConsistencyMLPSNN
from project_paths import local_path, portable_config, portable_path
from search_shd import search_lock

ROOT = Path(__file__).resolve().parent
SOURCES = ('train_mt_consistency.py', 'functional/temporal_consistency.py',
           'models/ours/mt_consistency.py', 'models/ours/mt_adlif.py',
           'models/pl_module.py', 'models/alif.py', 'models/li.py',
           'models/helpers.py', 'module/tau_trainers.py', 'run.py',
           'config/paper_baseline.yaml')
if not OmegaConf.has_resolver('mt_tc_repo'):
    OmegaConf.register_new_resolver('mt_tc_repo', lambda value: str(local_path(value)))


class ConsistencySummary(training.TrainingSummaryCallback):
    def _replace_summary(self, path, text):
        overrides = (list(HydraConfig.get().overrides.task) if HydraConfig.initialized()
                     else [f'experiment={self.cfg.exp_name}', f'random_seed={self.cfg.random_seed}'])
        # Quote each complete override for a visible PowerShell terminal.
        command = '.\\.venv\\Scripts\\python.exe train_mt_consistency.py ' + ' '.join(
            "'" + str(item).replace("'", "''") + "'" for item in overrides)
        text = re.sub(r'(## Resume command\s+```powershell\s*\n).*?(\n```)',
                      lambda match: match[1] + command + match[2], text, flags=re.S)
        super()._replace_summary(path, text)


def completed_validation(directory, budget):
    path = directory / 'logs/mlp_snn/version_0/metrics.csv'
    if not path.exists():
        return None
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if any(row.get('test_acc') for row in rows):
        raise ValueError('Unexpected official test results in validation-only experiment')
    values = [row for row in rows if row.get('val_acc')]
    epochs = [int(float(row['epoch'])) for row in values]
    if len(epochs) != len(set(epochs)):
        raise ValueError('Duplicate validation epochs; inspect for mixed histories')
    if set(epochs) != set(range(budget)):
        return None
    best = max(values, key=lambda row: float(row['val_acc']))
    return dict(best_validation_accuracy=float(best['val_acc']),
                best_validation_loss=float(best['val_loss']),
                best_epoch=int(float(best['epoch'])), completed_epochs=budget,
                final_test='not_evaluated')


def manifest_for(cfg):
    resolved = portable_config(OmegaConf.to_container(cfg, resolve=True))
    resolved['ckpt_path'] = None
    return dict(protocol='mt_temporal_consistency_v1', resolved_config=resolved,
                training_seed=int(cfg.random_seed),
                split_seed=cfg.dataset.get('random_seed'),
                initialization='fresh_random_weights; own last.ckpt for interruption resume',
                paper_baseline_version=cfg.paper_baseline_version,
                paper_baseline_dataset=cfg.paper_baseline_dataset,
                deviations=list(cfg.paper_baseline_deviations),
                environment=dict(python=platform.python_version(), torch=torch.__version__,
                                 cuda=torch.version.cuda),
                source_hash_format='utf8_lf_v1',
                source_sha256={name: hashlib.sha256(
                    (ROOT / name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
                               for name in SOURCES})


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_job(cfg):
    if (cfg.run_test or cfg.get('init_ckpt_path') or cfg.get('ckpt_path')
            or cfg.evaluation_protocol != 'mt_temporal_consistency_v1'
            or cfg.random_seed not in (42, 123, 456)):
        raise ValueError('Use validation-only fresh initialization or automatic own-checkpoint resume')
    if cfg.get('evaluation_file') or cfg.get('evaluation'):
        raise ValueError('Historical evaluation overrides are not supported')
    directory = local_path(cfg.logdir)
    if not directory.is_relative_to(ROOT / 'results/experiments'):
        raise ValueError('Use a separate repository-relative results/experiments output')
    if Path.cwd().resolve() != directory.resolve():
        raise ValueError('Hydra working directory must match logdir')
    if not cfg.two_layers or any(layer.cell != 'mt_se_adlif' for layer in (cfg.l1, cfg.l2)):
        raise ValueError('This launcher is for two-layer MT only')
    with search_lock(directory):
        manifest = manifest_for(cfg)
        path = directory / 'CONSISTENCY_MANIFEST.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('Config/source changed; use a new experiment output')
        else:
            # Hydra creates its config and launcher log before entering here.
            if any((directory / name).exists() for name in ('ckpt', 'logs', 'VALIDATION_RESULTS.json')):
                raise ValueError('Existing artifacts without a matching manifest')
            atomic_json(path, manifest)
        result = completed_validation(directory, int(cfg.n_epochs))
        if result:
            atomic_json(directory / 'VALIDATION_RESULTS.json', result)
            print(f'Already complete: {portable_path(directory)}')
            return
        last = directory / 'ckpt/last.ckpt'
        if last.exists():
            cfg.ckpt_path = portable_path(last)
        else:
            ckpts = list((directory / 'ckpt').glob('*.ckpt'))
            if ckpts or (directory / 'logs/mlp_snn/version_0/metrics.csv').exists():
                raise ValueError('Partial artifacts without last.ckpt; use a new output for fresh training')
        # Reuse the established trainer inside this process only. No shared
        # entrypoint or model behavior is changed for ordinary runner jobs.
        (directory / '.hydra').mkdir(exist_ok=True)
        OmegaConf.save(cfg, directory / '.hydra/config.yaml')
        with patch.object(training, 'MLPSNN', MTConsistencyMLPSNN), patch.object(
                training, 'TrainingSummaryCallback', ConsistencySummary):
            training.main.__wrapped__(cfg)
        result = completed_validation(directory, int(cfg.n_epochs))
        if result is None:
            raise RuntimeError('Full validation budget is incomplete; rerun to resume last.ckpt')
        atomic_json(directory / 'VALIDATION_RESULTS.json', result)
        summary = directory / 'TRAINING_SUMMARY.md'
        with summary.open('a', encoding='utf-8') as stream:
            stream.write('\n## Temporal consistency experiment\n\n'
                         'Training-only auxiliary objective; official test not evaluated.\n'
                         'Resume with the same train_mt_consistency.py command, not run.py.\n'
                         'Configuration and source identity: CONSISTENCY_MANIFEST.json.\n')


@hydra.main(config_path='config', config_name='main', version_base=None)
def main(cfg):
    run_job(cfg)


if __name__ == '__main__':
    main()

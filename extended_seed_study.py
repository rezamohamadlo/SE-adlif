"""Validation-only study integrity, completion evidence and portable provenance."""
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from omegaconf import OmegaConf
from project_paths import local_path, portable_path

ROOT = Path(__file__).resolve().parent


def cohort_definition():
    return OmegaConf.to_container(OmegaConf.load(ROOT / 'config/seed_cohorts/extension_v1.yaml'))


def validate_config(cfg):
    dataset = cfg['paper_baseline_dataset']
    baseline = cfg['paper_baseline']['datasets'][dataset]
    if (dataset not in ('SHD', 'ECG') or cfg['run_test'] is not False
            or cfg['early_stopping'] or cfg.get('init_ckpt_path')
            or cfg['tracking_metric'] != 'val_acc' or cfg['tracking_mode'] != 'max'
            or cfg['dataset']['random_seed'] != 42):
        raise ValueError('Unsafe validation-only configuration')
    for key in ('n_epochs', 'batch_size', 'lr'):
        if cfg[key] != baseline[key]:
            raise ValueError(f'Validation study changed baseline {key}')
    for key in ('dropout', 'alpha', 'dt', 'gradient_clip_val', 'tracking_metric', 'tracking_mode'):
        if cfg[key] != cfg['paper_baseline']['common'][key]:
            raise ValueError(f'Validation study changed shared baseline {key}')
    for key in ('scheduler', 'patience', 'factor'):
        if cfg[key] != cfg['paper_baseline']['reproduction_policy'][key]:
            raise ValueError(f'Validation study changed optimizer schedule {key}')
    split = 'validate_on' if dataset == 'SHD' else 'valid_fraction'
    if cfg['dataset'][split] != baseline[split] or cfg['dataset']['batch_size'] != cfg['batch_size']:
        raise ValueError('Validation study changed dataset split/batch')
    if dataset == 'SHD' and cfg['dataset'].get('additional_test_set_validation'):
        raise ValueError('Official test validation is forbidden')


def atomic_json(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def write_provenance(directory, root):
    import torch
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    record = dict(git_commit=git('rev-parse', 'HEAD'),
                  tracked_diff_sha256=hashlib.sha256(git('diff', 'HEAD', '--', '.', ':!AGENTS.md', ':!command.txt').encode()).hexdigest(),
                  python=sys.version, platform=platform.platform(),
                  cuda_version=torch.version.cuda,
                  gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  packages={p: importlib.metadata.version(p) for p in
                            ('torch', 'pytorch-lightning', 'hydra-core', 'omegaconf', 'numpy', 'scipy')})
    # Preserve launch environments for every resume, without replacing the first record.
    path = directory / 'LAUNCH_ENVIRONMENTS.json'
    records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    if record not in records:
        if records and any((e['git_commit'], e['tracked_diff_sha256']) !=
                           (record['git_commit'], record['tracked_diff_sha256']) for e in records):
            raise ValueError('Source revision changed; refusing to resume this study')
        records.append(record)
        atomic_json(path, records)


@contextmanager
def study_lock(root):
    """Serialize launchers per output root; stale lock files are harmless."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.extension.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f'Another launcher holds {root}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def finish_validation(cfg, trainer):
    """Called after fit only. Check checkpoint evidence before publishing completion."""
    import torch
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_config(config)
    directory = local_path(cfg.logdir)
    callback = trainer.checkpoint_callback
    best = local_path(callback.best_model_path, must_exist=True)
    last = directory / 'ckpt/last.ckpt'
    # These are trusted checkpoints created by this run, never imported arbitrary weights.
    last_epoch = int(torch.load(last, map_location='cpu', weights_only=False)['epoch'])
    best_epoch = int(torch.load(best, map_location='cpu', weights_only=False)['epoch'])
    if last_epoch + 1 != int(cfg.n_epochs):
        raise RuntimeError('Validation-only training did not complete the full epoch budget')
    accuracy = float(callback.best_model_score)
    record = dict(schema='validation_only_v1', status='complete', metric='validation_accuracy',
                  model=cfg.model_variant, dataset=cfg.paper_baseline_dataset, seed=int(cfg.random_seed),
                  seed_cohort=cfg.get('seed_cohort'), seed_cohort_order=list(cfg.get('seed_cohort_order', [])),
                  best_validation_accuracy=accuracy, selected_epoch=best_epoch,
                  completed_epochs=last_epoch + 1, best_checkpoint=portable_path(best),
                  last_checkpoint=portable_path(last))
    atomic_json(directory / 'VALIDATION_RESULTS.json', record)
    check_record(directory, record, config)
    (directory / 'VALIDATION_SUMMARY.md').write_text(
        f'# Validation-only completion\n\nBest validation accuracy: {100 * accuracy:.6f}% '
        f'at epoch {best_epoch} (zero based).\n\nCompleted {last_epoch + 1} epochs. '
        'No official-test evaluation or cost measurement.\n', encoding='utf-8')
    (directory / 'validation_completed.txt').write_text('Full-budget validation-only training completed.\n', encoding='utf-8')


def check_record(directory, record, config):
    validate_config(config)
    if any((directory / name).exists() for name in ('completed.txt', 'TEST_RESULTS.json', 'COST_RESULTS.json')):
        raise ValueError(f'Test artifacts in validation-only results: {directory}')
    accuracy = record.get('best_validation_accuracy')
    if (record.get('schema') != 'validation_only_v1' or record.get('status') != 'complete'
            or record.get('metric') != 'validation_accuracy'
            or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1
            or record.get('seed') != config['random_seed']
            or record.get('dataset') != config['paper_baseline_dataset']
            or record.get('model') != config['model_variant']
            or record.get('seed_cohort') != config.get('seed_cohort')
            or record.get('seed_cohort_order') != config.get('seed_cohort_order', [])
            or record.get('completed_epochs') != config['n_epochs']
            or not isinstance(record.get('selected_epoch'), int)
            or not 0 <= record['selected_epoch'] < config['n_epochs']):
        raise ValueError(f'Invalid validation completion record: {directory}')
    metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
    with metrics.open(newline='', encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream) if r.get('val_acc')]
    if not rows or not any(int(float(r['epoch'])) == record['selected_epoch']
                           and math.isclose(float(r['val_acc']), accuracy, abs_tol=1e-7) for r in rows):
        raise ValueError(f'Validation metrics disagree with selected checkpoint: {directory}')
    if not math.isclose(max(float(r['val_acc']) for r in rows), accuracy, abs_tol=1e-7):
        raise ValueError(f'Best validation score disagrees with metrics: {directory}')
    checkpoint_dir = local_path(config['logdir']) / 'ckpt'
    if (local_path(record['best_checkpoint']).parent != checkpoint_dir
            or local_path(record['last_checkpoint']) != checkpoint_dir / 'last.ckpt'):
        raise ValueError(f'Checkpoint identity is outside this run: {directory}')


def read_completed(directory, require_checkpoints=True):
    config = json.loads((directory / 'FULL_RUN_CONFIG.json').read_text(encoding='utf-8'))
    record = json.loads((directory / 'VALIDATION_RESULTS.json').read_text(encoding='utf-8'))
    if not (directory / 'validation_completed.txt').exists():
        raise ValueError(f'No validation completion marker: {directory}')
    check_record(directory, record, config)
    if require_checkpoints:
        for key in ('best_checkpoint', 'last_checkpoint'):
            local_path(record[key], must_exist=True)
    return record, config

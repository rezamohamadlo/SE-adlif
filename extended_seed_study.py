"""Validation-only study integrity, completion evidence and portable provenance."""
from contextlib import contextmanager
import copy
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
from project_paths import local_path, portable_path, portable_config, relocate_checkpoint_callbacks

ROOT = Path(__file__).resolve().parent


def cohort_definition():
    return OmegaConf.to_container(OmegaConf.load(ROOT / 'config/seed_cohorts/extension_v1.yaml'))


def baseline_definition():
    return OmegaConf.to_container(OmegaConf.load(ROOT / 'config/paper_baseline.yaml'))['paper_baseline']


def validate_config(cfg):
    if cfg.get('ssc_study'):
        from ssc_validation_study import validate_training_config
        return validate_training_config(cfg)
    dataset = cfg['paper_baseline_dataset']
    authority = baseline_definition()
    if (dataset not in ('SHD', 'ECG') or cfg['run_test'] is not False
            or cfg.get('validation_only') is not True
            or cfg['paper_baseline'] != authority or cfg['paper_baseline_version'] != authority['version']
            or cfg['early_stopping'] or cfg.get('init_ckpt_path')
            or cfg['tracking_metric'] != 'val_acc' or cfg['tracking_mode'] != 'max'
            or cfg['dataset']['random_seed'] != 42):
        raise ValueError('Unsafe validation-only configuration')
    baseline = authority['datasets'][dataset]
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


def source_identity(root):
    """Fingerprint tracked changes and actual source files, including untracked modules."""
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True, encoding='utf-8').strip()
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
    digest = hashlib.sha256()
    for name in sorted(set(names.decode('utf-8').split('\0')) - {''}):
        path = Path(name)
        if path.parts[0] in ('results', 'data', 'docs', 'reports', 'archive'):
            continue
        if path.suffix not in ('.py', '.yaml', '.yml') and name != 'requirements.txt':
            continue
        source = Path(root) / path
        content = source.read_bytes().replace(b'\r\n', b'\n') if source.is_file() else b'<deleted>'
        digest.update(name.encode('utf-8') + b'\0' + hashlib.sha256(content).digest())
    return dict(git_commit=git('rev-parse', 'HEAD'), source_files_sha256=digest.hexdigest(),
                tracked_diff_sha256=hashlib.sha256(git('diff', 'HEAD', '--', '.', ':!AGENTS.md', ':!command.txt').encode()).hexdigest())


def read_provenance(directory, root=None):
    path = directory / 'LAUNCH_ENVIRONMENTS.json'
    if not path.is_file():
        raise ValueError(f'Missing source/environment provenance: {directory}')
    records = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(records, list) or not records
            or any(not isinstance(e, dict) or not e.get('git_commit') or not e.get('tracked_diff_sha256') for e in records)):
        raise ValueError(f'Invalid source/environment provenance: {directory}')
    keys = ('git_commit', 'tracked_diff_sha256', 'source_files_sha256')
    source_hashes = {e['source_files_sha256'] for e in records if e.get('source_files_sha256')}
    if len({tuple(e[k] for k in keys[:2]) for e in records}) != 1 or len(source_hashes) > 1:
        raise ValueError(f'Mixed source revisions in provenance: {directory}')
    if root is not None:
        current = source_identity(root)
        if any(e[k] != current[k] for e in records for k in keys if k in e):
            raise ValueError(f'Source revision changed; refusing to resume this study: {directory}')
    return records


def write_provenance(directory, root):
    import torch
    record = dict(**source_identity(root),
                  python=sys.version, platform=platform.platform(),
                  cuda_version=torch.version.cuda,
                  gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  packages={p: importlib.metadata.version(p) for p in
                            ('torch', 'pytorch-lightning', 'hydra-core', 'omegaconf', 'numpy', 'scipy')})
    # Preserve launch environments for every resume, without replacing the first record.
    path = directory / 'LAUNCH_ENVIRONMENTS.json'
    records = read_provenance(directory, root) if path.exists() else []
    if record not in records:
        records.append(record)
        atomic_json(path, records)


def portable_checkpoint(checkpoint):
    """Copy callback metadata only; never duplicate large model/optimizer tensors."""
    result = dict(checkpoint, callbacks=copy.deepcopy(checkpoint.get('callbacks', {})))
    for state in result['callbacks'].values():
        if not isinstance(state, dict):
            continue
        for key in ('dirpath', 'best_model_path', 'last_model_path', 'kth_best_model_path'):
            if state.get(key):
                state[key] = portable_path(state[key])
        if isinstance(state.get('best_k_models'), dict):
            state['best_k_models'] = {portable_path(path): score for path, score in state['best_k_models'].items()}
    return result


def load_run_checkpoint(path, config):
    """Inspect a trusted local checkpoint before using it for continuation/evidence."""
    import torch
    checkpoint = torch.load(local_path(path, must_exist=True), map_location='cpu', weights_only=False)
    saved = checkpoint.get('hyper_parameters', {}).get('cfg')
    if OmegaConf.is_config(saved):
        saved = OmegaConf.to_container(saved, resolve=True)
    if not isinstance(saved, dict):
        raise ValueError(f'Missing checkpoint configuration: {path}')
    saved, expected = portable_config(saved), portable_config(copy.deepcopy(config))
    # Full resume changes this routing field, but never experiment settings.
    saved.pop('ckpt_path', None)
    expected.pop('ckpt_path', None)
    if saved != expected:
        raise ValueError(f'Incompatible checkpoint configuration: {path}')
    epoch = checkpoint.get('epoch')
    if type(epoch) is not int or not 0 <= epoch < config['n_epochs']:
        raise ValueError(f'Invalid checkpoint epoch: {path}')
    return relocate_checkpoint_callbacks(checkpoint)


def checkpoint_selection(checkpoint, directory):
    states = [s for s in checkpoint.get('callbacks', {}).values()
              if isinstance(s, dict) and s.get('monitor') == 'val_acc']
    if len(states) != 1:
        raise ValueError(f'Missing validation checkpoint selection state: {directory}')
    state = states[0]
    if not state.get('best_model_path') or state.get('best_model_score') is None:
        raise ValueError(f'Missing best validation checkpoint: {directory}')
    best = local_path(state['best_model_path'], must_exist=True)
    if best.parent != (directory / 'ckpt').resolve():
        raise ValueError(f'Best checkpoint belongs to a different run: {directory}')
    return best, float(state['best_model_score'])


def inspect_resume(directory, config, root):
    reject_test_artifacts(directory)
    read_provenance(directory, root)
    rows = validation_rows(directory, config)
    last = load_run_checkpoint(directory / 'ckpt/last.ckpt', config)
    best, score = checkpoint_selection(last, directory)
    selected = load_run_checkpoint(best, config)
    if selected['epoch'] > last['epoch']:
        raise ValueError(f'Best checkpoint is ahead of last checkpoint: {directory}')
    if rows and rows[-1][0] > last['epoch']:
        raise ValueError(f'Validation metrics are ahead of last checkpoint; preserve and reconcile the interrupted history: {directory}')
    if not any(epoch == selected['epoch'] and math.isclose(value, score, abs_tol=1e-7)
               for epoch, value in rows):
        raise ValueError(f'Resume checkpoint disagrees with validation metrics: {directory}')


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
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_config(config)
    directory = local_path(cfg.logdir)
    callback = trainer.checkpoint_callback
    best = local_path(callback.best_model_path, must_exist=True)
    last = directory / 'ckpt/last.ckpt'
    # These are trusted checkpoints created by this run, never imported arbitrary weights.
    last_state = load_run_checkpoint(last, config)
    best_state = load_run_checkpoint(best, config)
    recorded_best, recorded_score = checkpoint_selection(last_state, directory)
    if recorded_best != best or not math.isclose(recorded_score, float(callback.best_model_score), abs_tol=1e-7):
        raise ValueError('Last checkpoint disagrees with validation checkpoint selection')
    last_epoch, best_epoch = last_state['epoch'], best_state['epoch']
    if last_epoch + 1 != int(cfg.n_epochs):
        raise RuntimeError('Validation-only training did not complete the full epoch budget')
    accuracy = float(callback.best_model_score)
    record = dict(schema='validation_only_v1', status='complete', metric='validation_accuracy',
                  model=cfg.model_variant, dataset=cfg.paper_baseline_dataset, seed=int(cfg.random_seed),
                  seed_cohort=cfg.get('seed_cohort'), seed_cohort_order=list(cfg.get('seed_cohort_order', [])),
                  best_validation_accuracy=accuracy, selected_epoch=best_epoch,
                  completed_epochs=last_epoch + 1, best_checkpoint=portable_path(best),
                  last_checkpoint=portable_path(last))
    if config.get('ssc_study'):
        from ssc_validation_study import sha256
        record.update(ssc_study=config['ssc_study'], checkpoint_sha256=sha256(best),
                      last_checkpoint_sha256=sha256(last))
    check_record(directory, record, config)
    atomic_json(directory / 'VALIDATION_RESULTS.json', record)
    (directory / 'VALIDATION_SUMMARY.md').write_text(
        f'# Validation-only completion\n\nBest validation accuracy: {100 * accuracy:.6f}% '
        f'at epoch {best_epoch} (zero based).\n\nCompleted {last_epoch + 1} epochs. '
        'No official-test evaluation or cost measurement.\n', encoding='utf-8')
    (directory / 'validation_completed.txt').write_text('Full-budget validation-only training completed.\n', encoding='utf-8')


def reject_test_artifacts(directory):
    if any((directory / name).exists() for name in ('completed.txt', 'TEST_RESULTS.json', 'COST_RESULTS.json')):
        config = directory / 'FULL_RUN_CONFIG.json'
        if config.is_file() and json.loads(config.read_text(encoding='utf-8')).get('ssc_study'):
            from ssc_validation_study import validate_final_evidence
            validate_final_evidence(directory)
            return
        raise ValueError(f'Test artifacts in validation-only results: {directory}')


def validation_rows(directory, config):
    metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
    if not metrics.is_file():
        return []
    rows = []
    with metrics.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if any(value not in (None, '') and key and key.lower().startswith('test')
                   for key, value in row.items()):
                raise ValueError(f'Test metrics in validation-only CSV: {directory}')
            if not row.get('val_acc'):
                continue
            try:
                epoch, value = float(row['epoch']), float(row['val_acc'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'Invalid validation metrics: {directory}') from error
            if (not math.isfinite(epoch) or not epoch.is_integer() or not 0 <= epoch < config['n_epochs']
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'Invalid validation epoch/accuracy: {directory}')
            if rows and epoch <= rows[-1][0]:
                raise ValueError(f'Repeated or non-monotonic validation epochs; reject mixed histories: {directory}')
            rows.append((int(epoch), value))
    return rows


def check_record(directory, record, config):
    validate_config(config)
    reject_test_artifacts(directory)
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
    rows = validation_rows(directory, config)
    if not rows or not any(epoch == record['selected_epoch']
                           and math.isclose(value, accuracy, abs_tol=1e-7) for epoch, value in rows):
        raise ValueError(f'Validation metrics disagree with selected checkpoint: {directory}')
    if not math.isclose(max(value for _, value in rows), accuracy, abs_tol=1e-7):
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
    read_provenance(directory)
    if require_checkpoints:
        best = load_run_checkpoint(record['best_checkpoint'], config)
        last = load_run_checkpoint(record['last_checkpoint'], config)
        selected_path, selected_score = checkpoint_selection(last, directory)
        if (best['epoch'] != record['selected_epoch'] or last['epoch'] + 1 != record['completed_epochs']
                or selected_path != local_path(record['best_checkpoint'])
                or not math.isclose(selected_score, record['best_validation_accuracy'], abs_tol=1e-7)):
            raise ValueError(f'Completion record disagrees with checkpoint metadata: {directory}')
    return record, config

"""Separate, sequential validation-only SSC sum/mean reproduction comparison."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from project_paths import local_path, portable_config, portable_path
from search_shd import search_lock

ROOT = Path(__file__).resolve().parent
OUTPUT = 'results/experiments/ssc_loss_diagnostic_v1'
ARMS = ('summed_membrane_potentials', 'mean_membrane_potentials')


def resolve_job(output, arm, seed):
    directory = local_path(output) / arm / f'seed_{seed}'
    overrides = [
        'experiment=SSC_SE_adLIF_loss_diagnostic',
        f'loss_agg={arm}', f'random_seed={seed}',
        f'logdir={json.dumps(portable_path(directory))}',
    ]
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = compose(config_name='main', overrides=overrides)
    resolved = portable_config(OmegaConf.to_container(cfg, resolve=True))
    assert resolved['run_test'] is False
    assert resolved['ckpt_path'] is None and resolved['init_ckpt_path'] is None
    return directory, overrides, resolved


def validation_result(directory, budget):
    path = directory / 'logs/mlp_snn/version_0/metrics.csv'
    if not path.exists():
        return None
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if any(row.get('test_acc') for row in rows):
        raise ValueError(f'Unexpected test evaluation in {portable_path(directory)}')
    values = [row for row in rows if row.get('val_acc')]
    epochs = [int(float(row['epoch'])) for row in values]
    if len(epochs) != len(set(epochs)):
        raise ValueError(f'Duplicate validation epochs in {portable_path(directory)}')
    if set(epochs) != set(range(budget)):
        return None
    best = max(values, key=lambda row: float(row['val_acc']))
    return dict(best_validation_accuracy=float(best['val_acc']),
                best_validation_loss=float(best['val_loss']),
                best_validation_epoch=int(float(best['epoch'])),
                completed_validation_epochs=len(values), final_test='not_evaluated')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def stream_training(command, log_path):
    """Show stdout/stderr immediately while retaining the per-run log."""
    with log_path.open('ab') as log, subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ) as child:
        while True:
            chunk = child.stdout.read1(4096)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return child.wait()


def write_report(output, jobs):
    lines = ['# SSC loss diagnostic - validation only', '',
             'Fresh initialization per arm/seed; full resume only for interrupted jobs. '
             'The canonical reference is unchanged. No official test evaluation.', '',
             '| Loss | Seed | Status | Best validation (%) | Best epoch (zero-based) |',
             '|---|---:|---|---:|---:|']
    groups = {arm: [] for arm in ARMS}
    for directory, _, cfg in jobs:
        result = validation_result(directory, cfg['n_epochs'])
        if result:
            groups[cfg['loss_agg']].append(result['best_validation_accuracy'] * 100)
        lines.append(f"| {cfg['loss_agg']} | {cfg['random_seed']} | " +
                     (f"complete | {result['best_validation_accuracy'] * 100:.2f} | "
                      f"{result['best_validation_epoch']} |" if result else
                      'pending/incomplete | pending | pending |'))
    lines += ['', '| Loss | Completed seeds | Mean best validation (%) | Sample SD (pp) |',
              '|---|---:|---:|---:|']
    for arm, values in groups.items():
        lines.append(f'| {arm} | {len(values)} | ' +
                     (f'{statistics.mean(values):.2f} | ' if values else 'pending | ') +
                     (f'{statistics.stdev(values):.2f} |' if len(values) > 1 else 'unavailable |'))
    lines += ['', 'These are validation-selection results, not final test accuracy. '
              'The experiment was designed after observing historical test results; '
              'do not describe a later evaluation as an untouched independent test.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted jobs from their own last.ckpt')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct')
    output = local_path(args.output)
    if (output / 'PROMOTED_RUNS.json').exists() and not args.dry_run:
        parser.error('This diagnostic queue is retired: reference seed 42 was promoted. '
                     'Use ssc_mean_comparison.py for the new DTH/MT comparison.')
    jobs = [resolve_job(args.output, arm, seed) for seed in args.seeds for arm in ARMS]
    for name in ('ssc_train.h5', 'ssc_valid.h5', 'ssc_test.h5'):
        if not (ROOT / 'data/SSCWrapper' / name).exists():
            parser.error(f'Missing local dataset: data/SSCWrapper/{name}')
    if args.dry_run:
        for directory, _, cfg in jobs:
            print(json.dumps(dict(output=portable_path(directory), loss=cfg['loss_agg'],
                                  seed=cfg['random_seed'], lr=cfg['lr'], batch=cfg['batch_size'],
                                  epochs=cfg['n_epochs'], early_stopping=cfg['early_stopping'],
                                  run_test=cfg['run_test'], baseline=cfg['paper_baseline_version'])))
        return
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'pytorch-lightning', 'tonic', 'h5py', 'torchmetrics')}
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ('run.py', 'models/pl_module.py', 'models/alif.py',
                                  'models/li.py', 'models/helpers.py', 'module/tau_trainers.py',
                                  'datasets/ssc.py', 'datasets/utils/pad_tensors.py',
                                  'config/paper_baseline.yaml')}
    with search_lock(output):
        for directory, overrides, cfg in jobs:
            identity = dict(config=cfg, packages=versions, source_sha256=source_hashes)
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            manifest = directory / 'DIAGNOSTIC_MANIFEST.json'
            if manifest.exists():
                if json.loads(manifest.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
                    raise ValueError(f'Config/environment mismatch: {portable_path(directory)}; use a new output')
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError(f'Nonempty unguarded directory: {portable_path(directory)}')
            directory.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                atomic_json(manifest, dict(fingerprint=fingerprint, **identity))
                atomic_json(directory / 'FULL_RUN_CONFIG.json', cfg)
            result = validation_result(directory, cfg['n_epochs'])
            if result:
                print(f'Complete: {portable_path(directory)}', flush=True)
                write_report(output, jobs)
                continue
            last = directory / 'ckpt/last.ckpt'
            run_overrides = list(overrides)
            if last.exists():
                if not args.resume:
                    raise ValueError('Interrupted job exists; use --resume to restore its own last.ckpt')
                run_overrides.append(f'ckpt_path={json.dumps(portable_path(last))}')
            elif (directory / 'logs/mlp_snn/version_0/metrics.csv').exists():
                raise ValueError('Partial metrics without last.ckpt; use a separate output directory')
            print(f"{'Resume' if last.exists() else 'Fresh'}: {portable_path(directory)}", flush=True)
            atomic_json(directory / 'RUN_STATUS.json', dict(status='running',
                        mode='full_resume' if last.exists() else 'fresh'))
            exit_code = stream_training(
                [sys.executable, '-u', str(ROOT / 'run.py'), *run_overrides],
                directory / 'console.log')
            result = validation_result(directory, cfg['n_epochs'])
            atomic_json(directory / 'RUN_STATUS.json', dict(status='complete' if exit_code == 0 and result else 'failed',
                        exit_code=exit_code))
            if exit_code or not result:
                write_report(output, jobs)
                raise RuntimeError(f'Job incomplete: see {portable_path(directory)}/console.log')
            atomic_json(directory / 'VALIDATION_RESULTS.json', result)
            write_report(output, jobs)


if __name__ == '__main__':
    main()

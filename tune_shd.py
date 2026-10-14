"""Validation-only SHD checkpoint screening and fresh confirmation training."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'results/experiments/shd_lr_sweep'
DEFAULT_CHECKPOINT = ROOT / 'results/valid/reference/SHD/seed_42/ckpt/epoch=261-step=6550.ckpt'


def test_baseline(output):
    saved = output / 'REFERENCE_BASELINE.json'
    if saved.exists():
        return json.loads(saved.read_text(encoding='utf-8'))
    result = DEFAULT_CHECKPOINT.parent.parent / 'REFERENCE_RESULT.json'
    if not result.exists():
        raise ValueError('Missing reference test baseline; supply REFERENCE_BASELINE.json for historical reports')
    record = json.loads(result.read_text(encoding='utf-8'))
    return dict(protocol=record['baseline_version'], reference_test_acc=record['test_acc'],
                reported_reference_percent=100 * record['test_acc'],
                checkpoint=str(DEFAULT_CHECKPOINT))


def candidates():
    """Five LRs crossed with four explicit optimization recipes."""
    recipes = [
        dict(batch_size=512, scheduler='constant', patience=9999, factor=0.9),
        dict(batch_size=256, scheduler='constant', patience=9999, factor=0.9),
        dict(batch_size=512, scheduler='plateau', patience=2, factor=0.5),
        dict(batch_size=512, scheduler='cosine', patience=9999, factor=0.9),
    ]
    return [dict(id=f'c{index + 1:02d}', lr=lr, **recipe)
            for index, (lr, recipe) in enumerate(
                (lr, recipe) for lr in (0.0001, 0.0003, 0.001, 0.003, 0.01)
                for recipe in recipes)]


def reference(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError(f'Missing checkpoint: {checkpoint}. Reports alone cannot initialize training.')
    # Only use trusted, locally produced Lightning checkpoints.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'state_dict' not in saved or 'cfg' not in saved.get('hyper_parameters', {}):
        raise ValueError('Expected a local Lightning checkpoint containing state_dict and hyper_parameters.cfg')
    cfg = OmegaConf.to_container(OmegaConf.create(saved['hyper_parameters']['cfg']), resolve=True)
    if cfg.get('exp_name') not in ('SHD_SE_adLIF', 'SHD_SE_adLIF_paper') or cfg.get('random_seed') != 42:
        raise ValueError('This sweep expects the seed-42 SHD_SE_adLIF reference checkpoint')
    if cfg['dataset'].get('validate_on') != 0.2 or cfg['dataset'].get('random_seed') != 42:
        raise ValueError('Reference must use the training-derived 20% validation split with split seed 42')
    if cfg.get('tracking_metric') != 'val_acc' or cfg.get('tracking_mode') != 'max':
        raise ValueError('Reference must select checkpoints using val_acc in max mode')
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return cfg, dict(checkpoint=str(checkpoint), sha256=digest, epoch=saved.get('epoch'))


def trial_config(source, checkpoint, candidate, directory, epochs, seed=42, full=False):
    cfg = copy.deepcopy(source)
    # Retain data preprocessing, architecture and split; do not reapply the
    # fixed comparison protocol over the deliberately tuned training settings.
    for key in ('evaluation', 'evaluation_file', 'hydra'):
        cfg.pop(key, None)
    cfg.update({key: value for key, value in candidate.items() if key != 'id'})
    cfg.update(n_epochs=epochs, random_seed=seed, ckpt_path=None,
               init_ckpt_path=None if full else str(checkpoint), run_test=full,
               scheduler_min_lr=candidate['lr'] * 0.1, gradient_clip_val=1.5,
               logdir=str(directory), launcher_script='tune_shd.py',
               evaluation_protocol='shd_validation_tuning', sweep_trial=candidate['id'])
    cfg['sweep_checkpoint'] = str(checkpoint)
    cfg['sweep_output'] = str(directory.parents[2] if full else directory.parent)
    if not full:
        cfg['early_stopping'] = False
    cfg['dataset']['batch_size'] = candidate['batch_size']
    cfg['dataset']['num_workers'] = 0
    cfg['dataset']['additional_test_set_validation'] = False
    cfg['hydra'] = dict(job=dict(chdir=True), run=dict(dir=directory.as_posix()))
    return cfg


def check_directory(directory, manifest):
    path = directory / 'SWEEP_CONFIG.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError(f'Configuration changed; choose another --output: {directory}')
        if (directory / 'completed.txt').exists():
            return 'complete'
        if (directory / 'ckpt/last.ckpt').is_file():
            return 'resume'
        if (directory / 'ckpt').exists() and any((directory / 'ckpt').iterdir()):
            raise ValueError(f'Cannot resume without last.ckpt: {directory}')
        # Retry failures before the first checkpoint in the same directory.
        if (directory / 'logs').exists():
            raise ValueError(f'Existing logs but no resumable checkpoint: {directory}')
        return 'new'
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Refusing to overwrite unrecognized results: {directory}')
    return 'new'


def launch(directory, manifest, status):
    if status == 'complete':
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'SWEEP_CONFIG.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cfg = copy.deepcopy(manifest['config'])
    if status == 'resume':
        cfg['init_ckpt_path'] = None
        cfg['ckpt_path'] = (directory / 'ckpt/last.ckpt').as_posix()
    OmegaConf.save(OmegaConf.create(cfg), directory / 'trial.yaml')
    subprocess.check_call([sys.executable, str(ROOT / 'run.py'), '--config-path',
                           str(directory), '--config-name', 'trial'], cwd=ROOT)
    (directory / 'completed.txt').write_text('complete\n', encoding='utf-8')


def scores(directory):
    baseline = json.loads((directory / 'SWEEP_BASELINE.json').read_text(encoding='utf-8'))
    rows = []
    with (directory / 'logs/mlp_snn/version_0/metrics.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            # Standalone baseline validation is logged at step zero. Rank only
            # validations after optimizer updates, including resumed epochs.
            if row.get('val_acc') and row.get('val_loss') and float(row.get('step') or 0) > 0:
                acc, loss = float(row['val_acc']), float(row['val_loss'])
                if math.isfinite(acc) and math.isfinite(loss):
                    rows.append((acc, loss))
    if not rows:
        raise ValueError(f'No finite validation results: {directory}')
    acc, loss = max(rows, key=lambda pair: (pair[0], -pair[1]))
    return dict(val_acc=acc, val_loss=loss, baseline=baseline['val_acc'],
                delta_pp=100 * (acc - baseline['val_acc']))


def report(output):
    baseline_record = test_baseline(output)
    baseline_percent = baseline_record.get('reported_reference_percent', 100 * baseline_record['reference_test_acc'])
    rows = []
    for candidate in candidates():
        directory = output / candidate['id']
        if (directory / 'completed.txt').exists():
            row = dict(**candidate, **scores(directory), test_acc=None)
            test_path = directory / 'TEST_RESULTS.json'
            if test_path.exists():
                value = json.loads(test_path.read_text(encoding='utf-8'))['metrics'].get('test_acc')
                if value is not None and math.isfinite(float(value)):
                    row['test_acc'] = float(value)
            row['acc_diffe'] = (100 * row['test_acc'] - baseline_percent
                                if row['test_acc'] is not None else None)
            rows.append(row)
    validation_best = min(rows, key=lambda row: (-row['val_acc'], row['val_loss'], row['id'])) if rows else None
    rows.sort(key=lambda row: (row['acc_diffe'] is None,
                              -row['acc_diffe'] if row['acc_diffe'] is not None else 0,
                              -row['val_acc'], row['val_loss'], row['id']))
    lines = ['# SHD validation sweep', '',
             'Sorted by acc_diffe descending; untested trials appear last. Ties use validation accuracy, then validation loss. BEST_CONFIG.json retains the best validation configuration.',
             f"`acc_diffe` uses this experiment's reference ({baseline_percent:.2f}%, {baseline_record['protocol']}), in percentage points. Historical baselines are not the current canonical reference.",
             'Delta is relative to the same saved reference weights evaluated before each trial.', '',
             '| Trial | LR | Batch | Scheduler | Val accuracy | Delta (pp) | Val loss | Test accuracy | acc_diffe (pp) |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        test_accuracy = f"{100 * row['test_acc']:.2f}%" if row['test_acc'] is not None else 'pending'
        acc_difference = f"{row['acc_diffe']:+.2f}" if row['acc_diffe'] is not None else 'pending'
        lines.append(f"| {row['id']} | {row['lr']} | {row['batch_size']} | {row['scheduler']} | "
                     f"{100*row['val_acc']:.3f}% | {row['delta_pp']:+.3f} | {row['val_loss']:.5f} | {test_accuracy} | {acc_difference} |")
    lines += ['', f'{len(rows)}/20 trials completed.',
              'Short continuation ranks fine-tuning settings; it does not establish the best settings for training from scratch.']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'LEADERBOARD.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    if rows:
        (output / 'BEST_CONFIG.json').write_text(json.dumps(validation_best, indent=2), encoding='utf-8')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
                        help='Trusted local seed-42 reference best checkpoint (default: canonical paper-baseline epoch-261 checkpoint)')
    parser.add_argument('--epochs', type=int, choices=(5, 10), default=10)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--trial', choices=[row['id'] for row in candidates()])
    parser.add_argument('--list', action='store_true', help='Print the 20 recipes without a checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Preflight configurations; no writes or training')
    parser.add_argument('--report', action='store_true', help='Refresh validation leaderboard only')
    parser.add_argument('--full-train', action='store_true', help='Fresh 300-epoch confirmation; requires --trial')
    parser.add_argument('--seeds', type=int, nargs='+', choices=(42, 123, 456), default=[42, 123, 456])
    args = parser.parse_args(argv)
    try:
        if args.list:
            print(json.dumps(candidates(), indent=2))
            return
        output = args.output.resolve()
        if args.report:
            report(output)
            return
        if args.full_train and not args.trial:
            raise ValueError('--full-train requires an explicitly chosen --trial')
        source, provenance = reference(args.checkpoint)
        plan = []
        for candidate in candidates():
            if args.trial and candidate['id'] != args.trial:
                continue
            for seed in dict.fromkeys(args.seeds if args.full_train else [42]):
                directory = output / candidate['id']
                if args.full_train:
                    directory = output / 'full_train' / candidate['id'] / f'seed_{seed}'
                cfg = trial_config(source, args.checkpoint.resolve(), candidate, directory,
                                   300 if args.full_train else args.epochs, seed, args.full_train)
                manifest = dict(reference=provenance, config=cfg)
                status = check_directory(directory, manifest)
                plan.append((directory, manifest, status))
                print(f"{candidate['id']} seed {seed}: {status}; LR={candidate['lr']}, "
                      f"batch={candidate['batch_size']}, scheduler={candidate['scheduler']}", flush=True)
        if args.dry_run:
            print(f'Validated {len(plan)} runs; no writes or training.')
            return
        if not args.full_train and not (output / 'REFERENCE_BASELINE.json').exists():
            if args.checkpoint.resolve() != DEFAULT_CHECKPOINT.resolve():
                raise ValueError('A custom checkpoint requires an explicit REFERENCE_BASELINE.json in --output')
            output.mkdir(parents=True, exist_ok=True)
            (output / 'REFERENCE_BASELINE.json').write_text(
                json.dumps(test_baseline(output), indent=2), encoding='utf-8')
        for directory, manifest, status in plan:
            try:
                launch(directory, manifest, status)
            finally:
                if not args.full_train:
                    report(output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; repeat the identical command to resume.\n')


if __name__ == '__main__':
    main()

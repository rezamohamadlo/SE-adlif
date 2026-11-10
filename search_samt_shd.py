"""Fresh SA-MT SHD* hyperparameter search, with per-trial test evaluation (exploratory)."""
import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import subprocess
import hashlib
import torch
from omegaconf import OmegaConf

import runner
from project_paths import local_path, portable_config, portable_path
from search_shd import atomic_text, read_metrics, search_lock
from tune_shd import check_directory, launch

DEFAULT_OUTPUT = 'results/experiments/samt_shd_seed123_epoch188_continuation_plateau80_v6'
DEFAULT_CHECKPOINT = 'results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_123/ckpt/epoch=188-step=4725.ckpt'


def candidates():
    rows = [dict(id=f'lr{lr:g}_drop{dropout:g}',
                 overrides=dict(lr=lr, dropout=dropout))
            for lr in (0.0004, 0.0001) for dropout in (0.10, 0.15, 0.25)]
    rows += [dict(id=f'lr0.0004_batch{batch}',
                  overrides=dict(lr=0.0004, batch_size=batch, dropout=0.15))
             for batch in (128, 512)]
    return rows


def checkpoint_source(checkpoint):
    checkpoint = local_path(checkpoint, must_exist=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = portable_config(OmegaConf.to_container(
        OmegaConf.create(saved['hyper_parameters']['cfg']), resolve=True))
    if (cfg['random_seed'] != 123 or cfg['l1']['cell'] != 'sa_mt_se_adlif'
            or cfg['l2']['cell'] != 'sa_mt_se_adlif'
            or cfg['dataset']['_target_'] != 'datasets.shd.SHDLDM'
            or cfg['dataset']['validate_on'] != .2
            or cfg['dataset']['random_seed'] != 42
            or cfg['tracking_metric'] != 'val_acc' or cfg['tracking_mode'] != 'max'
            or cfg['dataset'].get('additional_test_set_validation', False)):
        raise ValueError('Expected seed-123 SA-MT SHD* validation-selected training configuration')
    if saved['epoch'] != 188:
        raise ValueError(f"Expected source epoch index 188, got index {saved['epoch']}")
    cfg.pop('hydra', None)
    return cfg, dict(checkpoint=portable_path(checkpoint), epoch_index=saved['epoch'],
                     sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())


def build_plan(output, checkpoint, trials=None, epochs=80):
    output = local_path(output)
    source, origin = checkpoint_source(checkpoint)
    plan = []
    for candidate in candidates():
        if trials and candidate['id'] not in trials:
            continue
        directory = output / candidate['id'] / 'seed_123'
        cfg = copy.deepcopy(source)
        overrides = candidate['overrides']
        for key in ('lr', 'dropout', 'batch_size'):
            if key in overrides:
                cfg[key] = overrides[key]
        cfg['dataset']['batch_size'] = cfg['batch_size']
        if 'tau_w_slow_range' in overrides:
            for layer in ('l1', 'l2'):
                cfg[layer]['tau_w_slow_range'] = overrides['tau_w_slow_range']
        cfg.update(n_epochs=epochs, early_stopping=False,
                   scheduler="plateau", tracking_metric="val_acc", tracking_mode="max",
                   factor=0.9, patience=10, scheduler_min_lr=0.0,
                   run_test=True,
                   init_ckpt_path=origin['checkpoint'], ckpt_path=None, random_seed=123,
                   logdir=portable_path(directory), launcher_script='search_samt_shd.py',
                   launcher_logdir=portable_path(output),
                   evaluation_protocol='samt_shd_epoch188_continuation_exploratory_plateau80_v6')
        cfg['dataset'].update(validate_on=0.2, random_seed=42,
                              additional_test_set_validation=False, num_workers=0)
        cfg['paper_baseline_deviations'] = [
            'model_variant=SA_MT_SE_adLIF',
            f'tuning_overrides={json.dumps(overrides, sort_keys=True)}',
            f'tuning_epoch_budget={epochs}; training_seed=123; early_stopping=false',
            'scheduler=ReduceLROnPlateau; monitor=val_acc; mode=max; factor=0.9; patience=10',
            'test after each trial; repeated test inspection is exploratory']
        manifest = dict(search='samt_shd_epoch188_continuation_exploratory_plateau80_v6', origin=origin,
                        candidate=candidate, config=portable_config(cfg))
        status = check_directory(directory, manifest)
        plan.append(dict(directory=directory, manifest=manifest, status=status))
    return plan


def report(output):
    rows = []
    for candidate in candidates():
        directory = output / candidate['id'] / 'seed_123'
        if not (directory / 'SWEEP_CONFIG.json').exists():
            continue
        test_path = directory / 'TEST_RESULTS.json'
        test_acc = (json.loads(test_path.read_text())['metrics']['test_acc']
                    if test_path.exists() else None)
        if test_acc is None:
            import csv
            metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
            if metrics.exists():
                with metrics.open() as stream:
                    scores = [float(r['test_acc']) for r in csv.DictReader(stream) if r.get('test_acc')]
                if len(scores) == 1:
                    test_acc = scores[0]
        rows.append(dict(trial=candidate['id'], test_acc=test_acc,
                         status='complete' if (directory / 'completed.txt').exists() else 'incomplete',
                         overrides=candidate['overrides'], **read_metrics(directory)))
    rows.sort(key=lambda r: (r['status'] != 'complete',
                            -(r['val_acc'] if r['val_acc'] is not None else -1), r['trial']))
    atomic_text(output / 'LEADERBOARD.json', json.dumps(rows, indent=2)+'\n')
    lines = ['# SA-MT SHD* seed-123 validation search', '',
             'Weights-only continuation; optimizer resets; fixed 20% validation split, split seed 42. Test after each full-budget trial.',
             'Repeated test inspection makes this exploratory. Ranking remains validation-based.', '',
             '| Trial | Status | Best validation accuracy | Best epoch (zero based) | Test accuracy |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        score = 'pending' if row['val_acc'] is None else f"{100*row['val_acc']:.3f}%"
        test = 'pending' if row['test_acc'] is None else f"{100*row['test_acc']:.3f}%"
        lines.append(f"| {row['trial']} | {row['status']} | {score} | {row['best_epoch']} | {test} |")
    atomic_text(output / 'LEADERBOARD.md', '\n'.join(lines)+'\n')
    completed = [r for r in rows if r['status'] == 'complete' and r['val_acc'] is not None]
    atomic_text(output / 'BEST_VALIDATION.json', json.dumps(completed[0] if completed else None, indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--trials', nargs='+', choices=[r['id'] for r in candidates()])
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Seed-123 SA-MT SHD epoch-188 checkpoint')
    parser.add_argument('--epochs', type=int, default=80, help='Additional training epochs, optimizer resets')
    parser.add_argument('--pause-between-runs', action='store_true',
                        help='Wait for Enter after each trial; Ctrl+C stops the queue')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    output = local_path(args.output)
    try:
        if args.report:
            with search_lock(output):
                report(output)
            return
        with nullcontext() if args.dry_run else search_lock(output):
            plan = build_plan(output, args.checkpoint, args.trials, args.epochs)
            data = local_path(plan[0]['manifest']['config']['dataset']['data_path'])
            for name in ('shd_train.h5', 'shd_test.h5'):
                if not (data / 'SHDWrapper' / name).is_file():
                    raise FileNotFoundError(f'Missing local SHD data: {data / "SHDWrapper" / name}')
            for item in plan:
                cfg = item['manifest']['config']
                print(f"{item['manifest']['candidate']['id']}: {item['status']}; "
                      f"LR={cfg['lr']}, dropout={cfg['dropout']}, batch={cfg['batch_size']}, "
                      f"epochs={cfg['n_epochs']}, early_stopping=False, test_after_training=True", flush=True)
            if args.dry_run:
                print(f'Validated {len(plan)} trials; no training or writes.')
                return
            for item in plan:
                try:
                    launch(item['directory'], item['manifest'], item['status'])
                finally:
                    report(output)
                if args.pause_between_runs and item is not plan[-1]:
                    input('Trial finished; see LEADERBOARD.md. Enter continues, Ctrl+C stops: ')
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Repeat the identical command to resume last.ckpt.\n')


if __name__ == '__main__':
    main()

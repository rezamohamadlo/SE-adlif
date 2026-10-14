"""Train the nine SHD reference LR/batch configurations from scratch."""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from evaluation_protocol import load_protocol, protocol_overrides
from tune_shd import check_directory, launch

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'results/experiments/shd_scratch_search'


@contextmanager
def search_lock(output):
    """OS-managed lock; released automatically if the launcher exits or crashes."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.search.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'\0')
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
            raise ValueError(f'Another search is running in {output}. Do not launch a second copy.') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_text(path, text):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def candidates():
    return [dict(id=f'r{index+1:02d}', lr=lr, batch_size=batch,
                 baseline=lr == .01 and batch == 512)
            for index, (lr, batch) in enumerate(
                (lr, batch) for lr in (.005, .01, .015) for batch in (128, 256, 512))]


def source_config():
    protocol = load_protocol()
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = compose(config_name='main', overrides=[
            'experiment=SHD_SE_adLIF', *protocol_overrides(protocol, 'SHD')])
    return OmegaConf.to_container(cfg, resolve=True)


def configuration(source, candidate, directory, seed):
    cfg = copy.deepcopy(source)
    for key in ('evaluation', 'evaluation_file', 'hydra'):
        cfg.pop(key, None)
    cfg.update(lr=candidate['lr'], batch_size=candidate['batch_size'],
               random_seed=seed, n_epochs=300, early_stopping=True,
               early_stopping_patience=50, early_stopping_min_delta=.001,
               dropout=.15, scheduler='plateau', patience=9999, factor=.9,
               gradient_clip_val=1.5, init_ckpt_path=None, ckpt_path=None,
               run_test=False, logdir=str(directory), launcher_script='search_shd.py',
               evaluation_protocol='shd_scratch_lr_batch_search',
               search_trial=candidate['id'], search_output=str(directory.parents[1]))
    cfg['dataset'].update(batch_size=candidate['batch_size'], num_workers=0,
                          validate_on=.2, random_seed=42, cache_namespace='controlled_v2',
                          additional_test_set_validation=False)
    cfg['hydra'] = dict(job=dict(chdir=True), run=dict(dir=directory.as_posix()))
    return cfg


def build_plan(output, trials=None, seeds=(42,)):
    source = source_config()
    plan = []
    # Run the original LR/batch control first, while keeping stable trial IDs.
    selected = sorted(candidates(), key=lambda item: (not item['baseline'], item['id']))
    for candidate in selected:
        if trials and candidate['id'] not in trials:
            continue
        for seed in dict.fromkeys(seeds):
            directory = output / candidate['id'] / f'seed_{seed}'
            cfg = configuration(source, candidate, directory, seed)
            manifest = dict(search='shd_scratch_lr_batch_v1', candidate=candidate, config=cfg)
            plan.append(dict(directory=directory, manifest=manifest,
                             status=check_directory(directory, manifest)))
    return plan


def read_metrics(directory):
    path = directory / 'logs/mlp_snn/version_0/metrics.csv'
    if not path.exists():
        return dict(val_acc=None, val_loss=None, best_epoch=None, latest_epoch=None)
    rows = []
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('val_acc') and row.get('val_loss'):
                acc, loss = float(row['val_acc']), float(row['val_loss'])
                if math.isfinite(acc) and math.isfinite(loss):
                    rows.append(dict(val_acc=acc, val_loss=loss, epoch=int(float(row['epoch']))))
    if not rows:
        return dict(val_acc=None, val_loss=None, best_epoch=None, latest_epoch=None)
    # ModelCheckpoint keeps the first epoch with the highest val_acc.
    best = max(rows, key=lambda row: row['val_acc'])
    return dict(val_acc=best['val_acc'], val_loss=best['val_loss'],
                best_epoch=best['epoch'], latest_epoch=rows[-1]['epoch'])


def report(output):
    rows = []
    for candidate in candidates():
        for seed in (42, 123, 456):
            directory = output / candidate['id'] / f'seed_{seed}'
            if not (directory / 'SWEEP_CONFIG.json').exists():
                continue
            refresh_resume_command(directory, candidate['id'], seed, output)
            complete = (directory / 'completed.txt').exists()
            row = dict(**candidate, seed=seed, status='complete' if complete else 'incomplete',
                       **read_metrics(directory), test_acc=None)
            test = directory / 'TEST_RESULTS.json'
            if test.exists():
                value = json.loads(test.read_text(encoding='utf-8'))['metrics'].get('test_acc')
                if value is not None and math.isfinite(float(value)):
                    row['test_acc'] = float(value)
            rows.append(row)
    rows.sort(key=lambda row: (row['status'] != 'complete',
                              -(row['val_acc'] if row['val_acc'] is not None else -1),
                              row['val_loss'] if row['val_loss'] is not None else math.inf,
                              row['id'], row['seed']))
    def fmt(value, percent=False):
        return 'pending' if value is None else (f'{100*value:.3f}%' if percent else f'{value:.5f}')
    lines = ['# SHD training from scratch', '',
             'Ranked by completed-run validation accuracy, then validation loss. Test results are displayed only after explicit evaluation.',
             'r06 is the fresh reference control: LR 0.01, batch 512. All runs use the same 20% training-derived validation split (split seed 42).', '',
             '| Trial | Seed | LR | Batch | Status | Latest epoch | Best epoch | Val accuracy | Val loss | Test accuracy |',
             '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['id']} | {row['seed']} | {row['lr']} | {row['batch_size']} | "
                     f"{row['status']} | {row['latest_epoch']} | {row['best_epoch']} | "
                     f"{fmt(row['val_acc'], True)} | {fmt(row['val_loss'])} | {fmt(row['test_acc'], True)} |")
    lines += ['', '## Validation across completed seeds', '',
              '| Trial | Completed seeds | Mean val accuracy | Sample SD (pp) |',
              '|---|---|---:|---:|']
    aggregate = []
    for candidate in candidates():
        completed = [row for row in rows if row['id'] == candidate['id']
                     and row['status'] == 'complete' and row['val_acc'] is not None]
        if not completed:
            continue
        values = [row['val_acc'] for row in completed]
        mean = statistics.mean(values)
        deviation = 100 * statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(dict(id=candidate['id'], seeds=[row['seed'] for row in completed],
                              val_acc_mean=mean, val_loss_mean=statistics.mean(row['val_loss'] for row in completed),
                              val_acc_sd_pp=deviation))
        lines.append(f"| {candidate['id']} | {', '.join(str(row['seed']) for row in completed)} | "
                     f"{fmt(mean, True)} | {'pending' if deviation is None else f'{deviation:.3f}'} |")
    lines += ['', 'Compare configurations on matching seed sets. Epoch numbers are zero based.',
              'Repeat --report to refresh in-progress metrics. Training updates the report after each run.']
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / 'LEADERBOARD.md', '\n'.join(lines) + '\n')
    atomic_text(output / 'LEADERBOARD.json', json.dumps(rows, indent=2) + '\n')
    # Write empty aggregates too, so archived runs cannot leave stale results.
    atomic_text(output / 'VALIDATION_AGGREGATE.json', json.dumps(aggregate, indent=2) + '\n')
    return rows


def refresh_resume_command(directory, trial, seed, output):
    """Keep search instructions correct without changing the training runner."""
    path = directory / 'TRAINING_SUMMARY.md'
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8')
    marker = '## Resume command\n\n```powershell\n'
    if marker not in text:
        return
    before, rest = text.split(marker, 1)
    _, separator, after = rest.partition('\n```')
    if not separator:
        return
    command = (f'.\\.ven\\Scripts\\python.exe search_shd.py --output "{output}" '
               f'--trials {trial} --seeds {seed}')
    updated = before + marker + command + separator + after
    if updated != text:
        path.write_text(updated, encoding='utf-8')


def evaluate(directory):
    saved = torch.load(directory / 'ckpt/last.ckpt', map_location='cpu', weights_only=False)
    paths = [state.get('best_model_path') for key, state in saved.get('callbacks', {}).items()
             if 'ModelCheckpoint' in key and state.get('best_model_path')]
    if len(paths) != 1:
        raise ValueError(f'Cannot identify best validation checkpoint in {directory}')
    name = paths[0].replace('\\', '/').rsplit('/', 1)[-1]
    checkpoint = directory / 'ckpt' / name
    if not checkpoint.is_file():
        raise ValueError(f'Best validation checkpoint is missing: {checkpoint}')
    subprocess.check_call([sys.executable, str(ROOT / 'test_checkpoint.py'),
                           '--checkpoint', str(checkpoint)], cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--trials', nargs='+', choices=[row['id'] for row in candidates()])
    parser.add_argument('--seeds', type=int, nargs='+', choices=(42, 123, 456), default=[42])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--test-selected', action='store_true', help='Evaluate completed, explicitly selected trials without training')
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if args.report:
            report(output)
            return
        if args.test_selected and not args.trials:
            raise ValueError('--test-selected requires explicit --trials')
        with nullcontext() if args.dry_run else search_lock(output):
            plan = build_plan(output, args.trials, args.seeds)
            if args.test_selected and any(item['status'] != 'complete' for item in plan):
                raise ValueError('All selected runs must be completed before test evaluation')
            for item in plan:
                cfg = item['manifest']['config']
                print(f"{cfg['search_trial']} seed {cfg['random_seed']}: "
                      f"{item['status']}, LR={cfg['lr']}, batch={cfg['batch_size']}", flush=True)
            if args.dry_run:
                print(f'Validated {len(plan)} runs; no writes or training.')
                return
            for item in plan:
                try:
                    if args.test_selected:
                        evaluate(item['directory'])
                    else:
                        launch(item['directory'], item['manifest'], item['status'])
                finally:
                    report(output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; repeat the identical command to resume.\n')


if __name__ == '__main__':
    main()

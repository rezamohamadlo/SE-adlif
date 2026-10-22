"""Wait for the LR/batch search, then run dropout controls if time permits."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import search_shd
from tune_shd import check_directory, launch
from project_paths import local_path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'results/experiments/shd_dropout_search'


def build_plan(output, trials=None):
    source = search_shd.source_config()
    plan = []
    for name, dropout in [('d25', .25), ('d35', .35)]:
        if trials and name not in trials:
            continue
        directory = output / name / 'seed_42'
        candidate = dict(id=name, lr=.01, batch_size=512, dropout=dropout)
        cfg = search_shd.configuration(source, candidate, directory, 42)
        cfg.update(dropout=dropout, launcher_script='queue_shd_dropout.py',
                   evaluation_protocol='shd_dropout_search')
        manifest = dict(search='shd_dropout_v1', candidate=candidate, config=cfg)
        plan.append(dict(directory=directory, manifest=manifest,
                         status=check_directory(directory, manifest)))
    return plan


def source_finished(source):
    return all((source / row['id'] / 'seed_42/completed.txt').exists()
               for row in search_shd.candidates())


def fits_budget(deadline, epochs_left, seconds_per_epoch):
    # Allow a 25% margin for data loading, slower epochs, and validation.
    return deadline - time.time() >= epochs_left * seconds_per_epoch * 1.25 + 300


def report(output):
    lines = ['# SHD dropout search', '',
             'Fresh training at LR 0.01, batch 512; only dropout differs from the reference control.',
             'Validation-only selection. Test evaluation remains separate.', '',
             '| Trial | Dropout | Status | Latest epoch | Best validation | Validation loss |',
             '|---|---:|---|---:|---:|---:|']
    for name, dropout in [('d25', .25), ('d35', .35)]:
        directory = output / name / 'seed_42'
        metrics = search_shd.read_metrics(directory)
        status = 'complete' if (directory / 'completed.txt').exists() else (
            'incomplete' if (directory / 'SWEEP_CONFIG.json').exists() else 'queued')
        accuracy = 'pending' if metrics['val_acc'] is None else f"{100*metrics['val_acc']:.3f}%"
        loss = 'pending' if metrics['val_loss'] is None else f"{metrics['val_loss']:.5f}"
        lines.append(f"| {name} | {dropout} | {status} | {metrics['latest_epoch']} | {accuracy} | {loss} |")
        summary = directory / 'TRAINING_SUMMARY.md'
        if summary.exists():
            text = summary.read_text(encoding='utf-8')
            marker = '## Resume command\n\n```powershell\n'
            if marker in text:
                before, rest = text.split(marker, 1)
                _, separator, after = rest.partition('\n```')
                if separator:
                    command = f'.\\.venv\\Scripts\\python.exe queue_shd_dropout.py --output "{output}" --trials {name} --hours 1 --seconds-per-epoch 20'
                    search_shd.atomic_text(summary, before + marker + command + separator + after)
    search_shd.atomic_text(output / 'LEADERBOARD.md', '\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=search_shd.DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--hours', type=float, default=18)
    parser.add_argument('--trials', nargs='+', choices=('d25', 'd35'),
                        help='Select follow-up runs; default is both')
    parser.add_argument('--seconds-per-epoch', type=float, default=25,
                        help='Conservative dropout-run speed estimate including validation')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 48 or not 0 < args.seconds_per_epoch:
        parser.error('Hours must be in (0, 48]; seconds per epoch must be positive')
    source, output = local_path(args.source), local_path(args.output)
    try:
        with nullcontext() if args.dry_run else search_shd.search_lock(output):
            plan = build_plan(output, args.trials)
            for item in plan:
                print(f"{item['directory'].parent.name}: {item['status']}, dropout="
                      f"{item['manifest']['config']['dropout']}", flush=True)
            if args.dry_run:
                print(f'Validated {len(plan)} follow-up runs; no writes or training.')
                return
            deadline = time.time() + args.hours * 3600
            state = dict(deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                         status='waiting for nine LR/batch runs', source=str(source))
            def save_status(status):
                state['status'] = status
                search_shd.atomic_text(output / 'QUEUE_STATUS.json', json.dumps(state, indent=2) + '\n')
                print(status, flush=True)
            save_status(state['status'])
            report(output)
            while not source_finished(source):
                if not fits_budget(deadline, 300, args.seconds_per_epoch):
                    save_status('Time budget insufficient for a full dropout run; no extra training started.')
                    return
                time.sleep(30)
            # The source launcher releases its lock only after its final report.
            while True:
                try:
                    with search_shd.search_lock(source):
                        for item in build_plan(output, args.trials):
                            if item['status'] == 'complete':
                                continue
                            metrics = search_shd.read_metrics(item['directory'])
                            epochs_left = max(300 - ((metrics['latest_epoch'] or 0) + 1), 1) if item['status'] == 'resume' else 300
                            if not fits_budget(deadline, epochs_left, args.seconds_per_epoch):
                                save_status('Time budget insufficient for the next dropout run; remaining runs stay queued.')
                                return
                            save_status(f"Training {item['directory'].parent.name}")
                            try:
                                launch(item['directory'], item['manifest'], item['status'])
                            finally:
                                report(output)
                        save_status('Selected dropout runs completed.')
                        return
                except ValueError as error:
                    if 'Another search is running' not in str(error):
                        raise
                    if not fits_budget(deadline, 300, args.seconds_per_epoch):
                        save_status('Time budget expired while waiting for the source lock.')
                        return
                    time.sleep(30)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()

"""Standalone official-test evaluation of six completed speaker-holdout runs.

Selections are frozen from validation evidence before any test is evaluated.
This is partial tuning evidence, not a completed three-seed comparison.
"""
import json
import subprocess
import statistics
import sys
from pathlib import Path

from extended_seed_study import read_completed
from project_paths import ROOT, local_path, portable_path

OUTPUT = 'results/experiments/shd_speaker_regularization_test_v1'
JOBS = [
    (profile, model, seed)
    for profile, seeds in [('dropout035_v1', [456]), ('spike_delete010_v1', [123, 456])]
    for seed in seeds
    for model in ['DTH_SE_adLIF', 'LG_DTH_SE_adLIF']
]


def main():
    root = local_path(OUTPUT)
    selections = []
    for profile, model, seed in JOBS:
        directory = local_path(
            f'results/experiments/shd_speaker_regularization_v1/{profile}/{model}/SHD/seed_{seed}')
        record, _ = read_completed(directory)
        selections.append(dict(profile=profile, model=model, seed=seed,
                               validation_record=portable_path(directory / 'VALIDATION_RESULTS.json'),
                               checkpoint=record['best_checkpoint'],
                               selected_epoch=record['selected_epoch'],
                               validation_accuracy=record['best_validation_accuracy']))
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'SELECTIONS.json'
    frozen = dict(protocol='speaker_holdout_official_test_v1',
                  note='User-requested test of six completed candidates; no test-based selection. '
                       'Incomplete tuning grid; not the random-20%-validation SHD* protocol.',
                  selections=selections)
    if manifest.exists():
        if json.loads(manifest.read_text()) != frozen:
            raise ValueError('Frozen validation selections changed')
    else:
        manifest.write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
    rows = []
    for item in selections:
        output = root / item['profile'] / item['model'] / 'SHD' / f"seed_{item['seed']}"
        result = output / 'TEST_RESULTS.json'
        if not result.exists():
            subprocess.run([sys.executable, str(ROOT / 'test_checkpoint.py'),
                            '--checkpoint', item['checkpoint'], '--output-dir', portable_path(output)],
                           cwd=ROOT, check=True)
        evidence = json.loads(result.read_text())
        if portable_path(evidence['checkpoint']) != portable_path(item['checkpoint']):
            raise ValueError(f'Wrong checkpoint in {result}')
        rows.append(dict(**item, test_metrics=evidence['metrics']))
    (root / 'RESULTS.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    lines = ['# Speaker-holdout regularization: official test', '', frozen['note'], '',
             'Ordinary inference, no TTA. Best checkpoint chosen using validation only.', '',
             '| Profile | Model | Seed | Validation (%) | Test (%) |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['profile']} | {row['model']} | {row['seed']} | "
                     f"{100*row['validation_accuracy']:.4f} | {100*row['test_metrics']['test_acc']:.4f} |")
    lines += ['', '| Profile | Model | n | Test mean ± sample SD (%) |',
              '|---|---|---:|---:|']
    for profile, model in sorted({(r['profile'], r['model']) for r in rows}):
        scores = [100*r['test_metrics']['test_acc'] for r in rows
                  if (r['profile'], r['model']) == (profile, model)]
        spread = f'{statistics.stdev(scores):.4f}' if len(scores) > 1 else 'SD unavailable'
        lines.append(f'| {profile} | {model} | {len(scores)} | {statistics.mean(scores):.4f} ± {spread} |')
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

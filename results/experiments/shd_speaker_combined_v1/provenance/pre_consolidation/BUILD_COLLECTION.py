"""Build a portable, audited result collection for four speaker-holdout settings.

No training or inference. Original run/checkpoint paths remain authoritative.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import extended_seed_study as evidence
from project_paths import local_path, portable_path

OUTPUT = Path(__file__).resolve().parent
PROFILES = ('augmented_v1', 'dropout030_v1', 'dropout035_v1', 'spike_delete010_v1')
MODELS = ('DTH_SE_adLIF', 'LG_DTH_SE_adLIF')
SEEDS = (42, 123, 456)
TEST_ROOTS = ('shd_regularization_test_seed42_v1', 'shd_speaker_regularization_tests_v1',
              'shd_speaker_regularization_test_v1')
LABELS = dict(zip(PROFILES, ('Augmented dropout 0.25', 'Augmented dropout 0.30',
                            'Augmented dropout 0.35', 'Dropout 0.25 + deletion 10%')))
SCHEMA = 'shd_speaker_combined_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def serialized(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def write_checked(path, content):
    encoded = content.encode('utf-8') if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f'Existing collection artifact differs; use a new collection version: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def metric(rows, epoch, name):
    values = [float(row[name]) for row in rows if row.get(name) and int(float(row['epoch'])) == epoch]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f'Missing/invalid {name} at epoch {epoch}')
    if len(set(values)) != 1:
        raise ValueError(f'Conflicting {name} at epoch {epoch}')
    return values[-1]


def collect(profile, model, seed, frozen_split, owners):
    baseline = profile == 'augmented_v1'
    source = (ROOT / 'results/experiments/shd_speaker_augmented_v1' if baseline else
              ROOT / 'results/experiments/shd_speaker_regularization_v1' / profile)
    source = source / model / 'SHD' / f'seed_{seed}'
    cfg = read(source / 'FULL_RUN_CONFIG.json')
    split = read(source / 'SPLIT_MANIFEST.json')
    if split != frozen_split:
        raise ValueError(f'Run has a different speaker split: {source}')
    if (cfg['random_seed'] != seed or cfg['model_variant'] != model
            or cfg['dataset']['random_seed'] != 42 or cfg['dataset']['validate_on'] != 'speaker_holdout'
            or cfg['dataset']['speaker_split_fingerprint'] != frozen_split['fingerprint']
            or cfg['paper_baseline_version'] != 'paper_classification_v1'
            or (cfg['n_epochs'], cfg['batch_size'], cfg['lr'], cfg['early_stopping']) != (300, 256, .01, False)):
        raise ValueError(f'Mismatched run identity/settings: {source}')
    expected_dropout = .30 if profile == 'dropout030_v1' else .35 if profile == 'dropout035_v1' else .25
    expected_delete = .10 if profile == 'spike_delete010_v1' else .05
    aug = cfg['dataset']['train_augmentation']
    if (cfg['dropout'] != expected_dropout or aug['spike_delete_probability'] != expected_delete
            or aug['time_scale'] != [.9, 1.1] or aug['channel_shift'] != 2):
        raise ValueError(f'Regularization differs from its profile: {source}')
    csv_path = source / 'logs/mlp_snn/version_0/metrics.csv'
    with csv_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    vals = [(int(float(row['epoch'])), float(row['val_acc'])) for row in rows if row.get('val_acc')]
    if [epoch for epoch, _ in vals] != list(range(300)):
        raise ValueError(f'Validation history is not a single full-budget run: {source}')
    if baseline:
        cost = read(source / 'COST_RESULTS.json')
        if cost['status'] != 'complete' or not (source / 'completed.txt').is_file():
            raise ValueError('Baseline has no verified ordinary-test completion')
        checkpoint = local_path(cost['checkpoint'], must_exist=True)
        last_checkpoint = source / 'ckpt/last.ckpt'
        last_state = evidence.load_run_checkpoint(last_checkpoint, cfg)
        selected_path, selected_accuracy = evidence.checkpoint_selection(last_state, source)
        selected_state = evidence.load_run_checkpoint(checkpoint, cfg)
        selected_epoch = selected_state['epoch']
        if (selected_path != checkpoint or last_state['epoch'] + 1 != 300
                or selected_epoch != cost['best_epoch'] or sha(checkpoint) != cost['checkpoint_sha256']):
            raise ValueError('Baseline checkpoint differs from its recorded validation selection')
        test_rows = [row for row in rows if row.get('test_acc')]
        if len({(row['test_acc'], row['test_loss']) for row in test_rows}) != 1:
            raise ValueError('Missing/conflicting baseline test history')
        test_accuracy, test_loss = float(test_rows[-1]['test_acc']), float(test_rows[-1]['test_loss'])
        if abs(test_accuracy - cost['measurements']['reproduced_test_acc']) > 1e-6:
            raise ValueError('Baseline ordinary test and verified reproduction disagree')
        test_path = csv_path
        test_checkpoint = checkpoint
        test_kind = 'ordinary completion of augmented baseline'
        owner = 'baseline'
    else:
        validation, checked_cfg = evidence.read_completed(source)
        if checked_cfg != cfg:
            raise ValueError('Validation completion disagrees with resolved config')
        checkpoint = local_path(validation['best_checkpoint'], must_exist=True)
        last_checkpoint = local_path(validation['last_checkpoint'], must_exist=True)
        selected_epoch, selected_accuracy = validation['selected_epoch'], validation['best_validation_accuracy']
        matches = [ROOT / 'results/experiments' / name / profile / model / 'SHD' / f'seed_{seed}' / 'TEST_RESULTS.json'
                   for name in TEST_ROOTS]
        matches = [path for path in matches if path.is_file()]
        if len(matches) != 1:
            raise ValueError('Expected exactly one imported ordinary test record per run')
        test_path = matches[0]
        test = read(test_path)
        test_checkpoint = local_path(test['checkpoint'], must_exist=True)
        if sha(test_checkpoint) != sha(checkpoint):
            raise ValueError('Evaluated checkpoint differs from validation-selected checkpoint')
        test_accuracy, test_loss = test['metrics']['test_acc'], test['metrics']['test_loss']
        test_kind = 'exploratory test before merged settings selection'
        origin = owners[profile, model, seed]
        if origin['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('Selected checkpoint differs from verified PC transfer')
        owner = origin['owner']
    if (not math.isclose(max(value for _, value in vals), selected_accuracy, abs_tol=1e-7)
            or not math.isclose(metric(rows, selected_epoch, 'val_acc'), selected_accuracy, abs_tol=1e-7)
            or not 0 <= test_accuracy <= 1 or not math.isfinite(test_loss)):
        raise ValueError('Selected validation/test metrics are invalid')
    record = dict(schema=SCHEMA, collection_view=True, dataset='SHD', profile=profile,
                  label=LABELS[profile], model=model, seed=seed, origin=owner,
                  source_protocol=cfg['evaluation_protocol'], paper_baseline_version=cfg['paper_baseline_version'],
                  dropout=cfg['dropout'], spike_delete_probability=aug['spike_delete_probability'],
                  completed_epochs=300, selected_epoch=selected_epoch,
                  train_accuracy_selected=metric(rows, selected_epoch, 'train_acc_epoch'),
                  train_accuracy_final=metric(rows, 299, 'train_acc_epoch'),
                  best_validation_accuracy=selected_accuracy,
                  test_accuracy=test_accuracy, test_loss=test_loss, ordinary_inference=True,
                  test_evidence=test_kind, source_run=portable_path(source),
                  source_test_record=portable_path(test_path), source_test_checkpoint=portable_path(test_checkpoint),
                  selected_checkpoint=portable_path(checkpoint), selected_checkpoint_sha256=sha(checkpoint),
                  last_checkpoint=portable_path(last_checkpoint), last_checkpoint_sha256=sha(last_checkpoint),
                  split_fingerprint=frozen_split['fingerprint'])
    files = [source / name for name in ('FULL_RUN_CONFIG.json', 'SPLIT_MANIFEST.json', 'TRAINING_SUMMARY.md',
             'VALIDATION_RESULTS.json', 'VALIDATION_SUMMARY.md', 'LAUNCH_ENVIRONMENTS.json',
             'COST_RESULTS.json', 'COST_RESULTS.md') if (source / name).is_file()]
    snapshots = [(path, path.name) for path in files] + [(csv_path, 'metrics.csv')]
    if not baseline:
        snapshots += [(test_path, 'TEST_RESULTS.json')]
        for name, target in (('MANIFEST.json', 'TEST_MANIFEST.json'),
                             ('SOURCE_VALIDATION_RESULTS.json', 'SOURCE_VALIDATION_RESULTS.json')):
            if (test_path.parent / name).is_file():
                snapshots.append((test_path.parent / name, target))
    destination = OUTPUT / profile / model / 'SHD' / f'seed_{seed}'
    source_files = [dict(source=portable_path(path), snapshot=(destination / 'evidence' / name).relative_to(OUTPUT).as_posix(),
                         bytes=path.stat().st_size, sha256=sha(path)) for path, name in snapshots]
    return record, source_files


def summary(records):
    groups = []
    for profile in PROFILES:
        for model in MODELS:
            runs = [r for r in records if r['profile'] == profile and r['model'] == model]
            if sorted(r['seed'] for r in runs) != list(SEEDS):
                raise ValueError('Aggregation requires the same three independent training seeds')
            groups.append(dict(profile=profile, model=model, n=3, seeds=list(SEEDS),
                               mean_validation_accuracy=statistics.mean(r['best_validation_accuracy'] for r in runs),
                               sample_sd_validation=statistics.stdev(r['best_validation_accuracy'] for r in runs),
                               mean_test_accuracy=statistics.mean(r['test_accuracy'] for r in runs),
                               sample_sd_test=statistics.stdev(r['test_accuracy'] for r in runs)))
    return groups


def formatted_report(records, groups, selection):
    lines = ['# Combined SHD speaker-holdout results', '',
             'Four settings, two models, three training seeds: 24 completed runs. Ordinary inference without TTA.',
             'Values are accuracy percentages; variation is sample SD across seeds 42, 123 and 456.',
             'Checkpoints were selected by validation accuracy. Regularization test results are exploratory.', '',
             '| Setting | Model | Dropout | Spike deletion | n | Validation mean +/- SD (%) | Test mean +/- SD (%) | Selected by validation |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for item in groups:
        run = next(r for r in records if r['profile'] == item['profile'] and r['model'] == item['model'])
        chosen = item['profile'] in selection['selections'][item['model']]['profiles']
        lines.append(f'| {LABELS[item["profile"]]} | {item["model"]} | {run["dropout"]:.2f} | '
                     f'{100*run["spike_delete_probability"]:.0f}% | 3 | '
                     f'{100*item["mean_validation_accuracy"]:.2f} +/- {100*item["sample_sd_validation"]:.2f} | '
                     f'{100*item["mean_test_accuracy"]:.2f} +/- {100*item["sample_sd_test"]:.2f} | '
                     f'{"Yes" if chosen else ""} |')
    lines += ['', '## Per-seed results', '',
              '| Setting | Model | Seed | Selected epoch (zero-based) | Train at selected epoch (%) | Final train (%) | Best validation (%) | Test (%) | Record |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for run in records:
        relative = f'{run["profile"]}/{run["model"]}/SHD/seed_{run["seed"]}/RESULT.json'
        lines.append(f'| {LABELS[run["profile"]]} | {run["model"]} | {run["seed"]} | {run["selected_epoch"]} | '
                     f'{100*run["train_accuracy_selected"]:.2f} | {100*run["train_accuracy_final"]:.2f} | '
                     f'{100*run["best_validation_accuracy"]:.2f} | {100*run["test_accuracy"]:.2f} | [JSON]({relative}) |')
    lines += ['', 'The selection column uses the frozen three-seed validation rule, including the dropout-0.25 baseline.',
              'Test scores do not select settings. Previously observed test results remain exploratory.',
              'Original protocol metadata and checkpoint identities are retained in each record and evidence snapshot.',
              'These speaker-holdout results differ from random recording SHD*; paper accuracy is not a matched baseline.']
    return '\n'.join(lines) + '\n'


def build(dry_run):
    frozen_path = ROOT / 'results/experiments/shd_speaker_augmented_v1/SPLIT_MANIFEST.json'
    frozen_split = read(frozen_path)
    receipt = read(ROOT / 'results/provenance/shd-regularization-pc123-transfer-2026-11-10.json')
    owners = {(r['profile'], r['model'], r['seed']): r for r in receipt['results_verified']}
    selection_path = ROOT / 'results/experiments/shd_speaker_regularization_v1/VALIDATION_SELECTION.json'
    selection = read(selection_path)
    records, inputs = [], []
    for profile in PROFILES:
        for model in MODELS:
            for seed in SEEDS:
                record, snapshots = collect(profile, model, seed, frozen_split, owners)
                records.append(record)
                inputs.extend(snapshots)
    groups = summary(records)
    for model in MODELS:
        candidates = [g for g in groups if g['model'] == model]
        best = max(g['mean_validation_accuracy'] for g in candidates)
        if selection['selections'][model]['profiles'] != [g['profile'] for g in candidates if g['mean_validation_accuracy'] == best]:
            raise ValueError('Saved settings selection differs from the three-seed validation rule')
    print(f'Verified {len(records)} completed runs, {len(groups)} three-seed groups and {len(inputs)} source snapshots.', flush=True)
    if dry_run:
        print('No files copied, training or inference performed.')
        return
    source_identity = dict(schema=SCHEMA, records=records, snapshots=inputs,
                           split_fingerprint=frozen_split['fingerprint'], validation_selection=selection,
                           builder_sha256=sha(Path(__file__)))
    fingerprint = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()
    write_checked(OUTPUT / 'SOURCE_MANIFEST.json', serialized(dict(fingerprint=fingerprint, **source_identity)))
    for item in inputs:
        source = local_path(item['source'], must_exist=True)
        content = source.read_bytes()
        if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Source snapshot changed while building collection')
        write_checked(OUTPUT / item['snapshot'], content)
    for record in records:
        directory = OUTPUT / record['profile'] / record['model'] / 'SHD' / f'seed_{record["seed"]}'
        write_checked(directory / 'RESULT.json', serialized(record))
    write_checked(OUTPUT / 'SPLIT_MANIFEST.json', frozen_path.read_bytes())
    write_checked(OUTPUT / 'VALIDATION_SELECTION.json', selection_path.read_bytes())
    write_checked(OUTPUT / 'RESULTS.json', serialized(dict(schema=SCHEMA, fingerprint=fingerprint,
                  units='accuracy fractions; standard deviations in fraction units',
                  aggregation='mean and sample SD across matching training seeds', groups=groups, runs=records)))
    columns = ('profile', 'model', 'seed', 'origin', 'dropout', 'spike_delete_probability', 'completed_epochs',
               'selected_epoch', 'train_accuracy_selected', 'train_accuracy_final', 'best_validation_accuracy',
               'test_accuracy', 'test_loss', 'source_run', 'source_test_record', 'selected_checkpoint',
               'selected_checkpoint_sha256', 'last_checkpoint', 'last_checkpoint_sha256')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    write_checked(OUTPUT / 'RUNS.csv', stream.getvalue())
    write_checked(OUTPUT / 'RESULTS.md', formatted_report(records, groups, selection))
    readme = '''# SHD speaker-based experiments: combined result collection

This folder brings together the augmented dropout-0.25 baseline and all three
regularization variants for DTH and LG-DTH, with seeds 42, 123 and 456.
It contains **24 completed runs** and **8 matched three-seed summaries**.

- [Results and per-seed table](RESULTS.md)
- [Machine-readable results](RESULTS.json)
- [Flat CSV of all 24 runs](RUNS.csv)
- [Validation-based settings selection](VALIDATION_SELECTION.json)
- [Source paths, hashes and snapshot inventory](SOURCE_MANIFEST.json)
- [Frozen speaker split](SPLIT_MANIFEST.json)
- [Paper/reference/shuffled comparison](../../../reports/speaker-based-summary.md)

Each `<profile>/<model>/SHD/seed_<seed>/` contains a normalized `RESULT.json`
and an `evidence/` directory with byte-identical copies of the original config,
metrics, summaries and applicable validation/test/measurement records. The four
profiles are `augmented_v1`, `dropout030_v1`, `dropout035_v1` and
`spike_delete010_v1`. All have the same held-out validation speakers 0 and 3,
split seed 42, training seeds 42/123/456, batch 256 and 300-epoch budget.

Checkpoint files remain at their recorded original locations. Each result
contains repository-relative paths and SHA-256 identities for both the selected
checkpoint and its own last checkpoint. Resolve these paths with
`project_paths.py`; keep the original relative layout when transferring weights.
The snapshots are an audited result view, not a new resumable training run.
Original saved configs, protocols, directories and evaluation scripts remain
authoritative. No original evidence is moved or relabeled.

JSON/CSV accuracies are fractions; Markdown displays percentages. SD is sample
SD across the matching three seeds. Train accuracy at the selected epoch comes
from `metrics.csv`, because historical training summaries can lag. All test
values are ordinary inference without TTA. The regularization test results
were observed before the merged settings selection and remain exploratory.
Settings selection uses validation means only, including the original baseline.

`BUILD_COLLECTION.py --dry-run` checks local configs, full validation histories,
checkpoint identities and test provenance without training or inference. Running
it without that flag writes the collection and rejects changed existing output.
New or changed source evidence requires a new versioned collection directory.
'''
    write_checked(OUTPUT / 'README.md', readme)
    for item in inputs:
        if sha(local_path(item['source'])) != item['sha256'] or sha(OUTPUT / item['snapshot']) != item['sha256']:
            raise ValueError('Source or copied evidence failed final hash verification')
    files = [dict(path=p.relative_to(OUTPUT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'
             and '__pycache__' not in p.parts]
    write_checked(OUTPUT / 'MANIFEST.json', serialized(dict(schema=SCHEMA, fingerprint=fingerprint,
                  completed_runs=24, matched_groups=8, files=files,
                  note='Collection-file inventory excludes this manifest; original weights are referenced by identity.')))
    print(f'Saved: {portable_path(OUTPUT / "README.md")}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    build(parser.parse_args().dry_run)

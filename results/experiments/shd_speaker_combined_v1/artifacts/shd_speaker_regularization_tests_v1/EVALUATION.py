"""Explicitly requested ordinary test inference for six local tuning checkpoints.

Exploratory evaluation before full multi-PC hyperparameter selection. Original
validation-only evidence stays untouched; test scores do not select candidates.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import evaluate_tta
import extended_seed_study as evidence
from project_paths import local_path, portable_path
from search_shd import search_lock
import shd_speaker_holdout as speaker
import shd_speaker_regularization as regularization
from ssc_loss_diagnostic import atomic_json

OUTPUT = Path(__file__).resolve().parent
JOBS = (
    ('dropout030_v1', 'DTH_SE_adLIF', 123),
    ('dropout030_v1', 'LG_DTH_SE_adLIF', 123),
    ('dropout030_v1', 'DTH_SE_adLIF', 456),
    ('dropout030_v1', 'LG_DTH_SE_adLIF', 456),
    ('dropout035_v1', 'DTH_SE_adLIF', 123),
    ('dropout035_v1', 'LG_DTH_SE_adLIF', 123),
)
DISCLOSURE = ('Exploratory official-test evaluation explicitly requested before full '
              'three-seed hyperparameter selection. Every checkpoint was selected by '
              'validation accuracy; these test scores must not select settings.')


def preflight():
    with speaker.selected_profile('augmented_v1'):
        split = speaker.preflight_data()
    jobs = []
    for profile, model, seed in JOBS:
        directory = local_path(f'{regularization.OUTPUT_ROOT}/{profile}/{model}/SHD/seed_{seed}')
        record, config = evidence.read_completed(directory)
        if (record['model'] != model or record['seed'] != seed
                or record['completed_epochs'] != 300
                or config['shd_regularization_study']['profile'] != profile
                or config['dataset']['speaker_split_fingerprint'] != split['fingerprint']
                or json.loads((directory / 'SPLIT_MANIFEST.json').read_text(encoding='utf-8')) != split):
            raise ValueError('Checkpoint model/seed/profile/split or completion disagrees')
        checkpoint = local_path(record['best_checkpoint'], must_exist=True)
        job = dict(dataset='SHD', model=model, profile=profile, seed=seed,
                   checkpoint=portable_path(checkpoint),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   selected_epoch=record['selected_epoch'],
                   best_validation_accuracy=record['best_validation_accuracy'],
                   source_run=portable_path(directory), config=config,
                   split_fingerprint=split['fingerprint'])
        jobs.append(job)
        print(f'Ready: {profile} / {model} / seed {seed}, '
              f'validation-selected epoch {job["selected_epoch"]}', flush=True)
    return jobs


def write_report():
    lines = ['# SHD speaker-holdout regularization: exploratory ordinary test results', '',
             DISCLOSURE, '',
             'Frozen weights, one ordinary forward per batch, no TTA or training.',
             'Original validation-only run directories and selection evidence are preserved.', '',
             '| Dropout | Model | Seed | Selected epoch (zero-based) | Best validation (%) | Test (%) |',
             '|---:|---|---:|---:|---:|---:|']
    for profile, model, seed in JOBS:
        path = OUTPUT / profile / model / 'SHD' / f'seed_{seed}' / 'TEST_RESULTS.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        lines.append(f'| {result["dropout"]:.2f} | {model} | {seed} | '
                     f'{result["selected_epoch"]} | {100*result["best_validation_accuracy"]:.4f} | '
                     f'{100*result["metrics"]["test_acc"]:.4f} |')
    temporary = OUTPUT / 'RESULTS.md.tmp'
    temporary.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    temporary.replace(OUTPUT / 'RESULTS.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        jobs = preflight()
        print(f'Verified {len(jobs)} validation completions and exact selected checkpoints; no inference/training.')
        return
    with evidence.study_lock(local_path(regularization.OUTPUT_ROOT)), search_lock(OUTPUT):
        jobs = preflight()
        provenance = dict(source=evidence.source_identity(ROOT),
                          recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          environment={p: importlib.metadata.version(p) for p in
                                       ('torch', 'numpy', 'pytorch-lightning', 'tonic')})
        for job in jobs:
            directory = OUTPUT / job['profile'] / job['model'] / 'SHD' / f'seed_{job["seed"]}'
            identity = dict(**job, evaluation_protocol='shd_speaker_regularization_tests_v1',
                            split='official_test', ordinary_inference=True,
                            selection_disclosure=DISCLOSURE, provenance=provenance)
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            manifest = directory / 'MANIFEST.json'
            if manifest.exists():
                if json.loads(manifest.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
                    raise ValueError('Evaluation identity changed; use a new output directory')
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError('Refusing an unguarded evaluation directory')
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(manifest, dict(**identity, fingerprint=fingerprint))
            path = directory / 'TEST_RESULTS.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if (result.get('fingerprint') != fingerprint
                        or result.get('checkpoint_sha256') != job['checkpoint_sha256']):
                    raise ValueError('Cached result disagrees with selected checkpoint identity')
            else:
                # The existing inference evaluator also accepts identity alone:
                # one forward, the saved temporal aggregation and ignored-label mask.
                measured = evaluate_tta.evaluate(job, [1.0], 'test')
                if measured['baseline_accuracy'] != measured['tta_accuracy']:
                    raise ValueError('Identity-only predictions disagree')
                result = dict(model=job['model'], profile=job['profile'], seed=job['seed'],
                              dataset='SHD', dropout=job['config']['dropout'],
                              selected_epoch=job['selected_epoch'],
                              best_validation_accuracy=job['best_validation_accuracy'],
                              checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'],
                              source_run=job['source_run'], fingerprint=fingerprint,
                              evaluation_protocol=identity['evaluation_protocol'],
                              selection_disclosure=DISCLOSURE,
                              metrics=dict(test_acc=measured['baseline_accuracy'], test_loss=measured['baseline_loss']),
                              measurement={key: measured[key] for key in
                                           ('evaluated_targets', 'data_sha256', 'device', 'gpu', 'matmul_precision')})
                atomic_json(path, result)
            print(f'{job["profile"]} / {job["model"]} / seed {job["seed"]}: '
                  f'test accuracy {100*result["metrics"]["test_acc"]:.4f}%', flush=True)
            write_report()
    print(f'Saved: {portable_path(OUTPUT / "RESULTS.md")}', flush=True)


if __name__ == '__main__':
    main()

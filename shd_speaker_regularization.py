"""Validation-only regularization queue, dispatched by the official runner."""
from contextlib import ExitStack
import copy
import csv
import json
import math
import statistics

import shd_speaker_holdout as speaker
from project_paths import local_path, portable_path

VERSION = 'shd_speaker_regularization_v1'
OUTPUT_ROOT = f'results/experiments/{VERSION}'
PROFILES = speaker.REGULARIZATION_PROFILES
SEEDS = (42, 123, 456)


def ordered_jobs(phase):
    first = [(profile, model, 42) for profile in PROFILES for model in speaker.MODELS]
    remaining = [(profile, model, seed) for profile in PROFILES
                 for seed in SEEDS[1:] for model in speaker.MODELS]
    return first if phase == '1' else remaining if phase == '2' else first + remaining


def validate_training_config(cfg):
    """Run in the training child too, without relying on parent globals/CWD."""
    identity = cfg.get('shd_regularization_study', {})
    profile = identity.get('profile')
    if (identity.get('version') != VERSION or profile not in PROFILES
            or cfg.get('validation_only') is not True or cfg.get('run_test') is not False
            or cfg.get('init_ckpt_path') is not None):
        raise ValueError('Unsafe SHD regularization validation-only configuration')
    model, seed = cfg['model_variant'], cfg['random_seed']
    if model not in speaker.MODELS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Invalid SHD regularization model/seed')
    with speaker.selected_profile(profile):
        directory = local_path(f'{speaker.OUTPUT_ROOT}/{model}/SHD/seed_{seed}')
        if local_path(cfg['logdir']) != directory or local_path(cfg['launcher_logdir']) != local_path(speaker.OUTPUT_ROOT):
            raise ValueError('Regularization artifacts must stay in their own versioned directory')
        if cfg.get('ckpt_path') and local_path(cfg['ckpt_path']) != directory / 'ckpt/last.ckpt':
            raise ValueError('Full resume requires this run\'s own last.ckpt')
        checked = copy.deepcopy(cfg)
        checked['ckpt_path'] = None  # A resume changes routing, never experiment settings.
        speaker.validate_config(checked, model, seed,
                                {'fingerprint': cfg['dataset']['speaker_split_fingerprint']})


def profile_plans(runner, manifest, jobs):
    plans = []
    for profile, model, seed in jobs:
        with speaker.selected_profile(profile), speaker.configured_runner(runner, manifest):
            item = runner.build_plan(model, seeds=(seed,), datasets=('SHD',))[0]
            validate_training_config(item['config'])
            plans.append((profile, model, item))
    return plans


def baseline_plans(runner, manifest):
    with speaker.selected_profile('augmented_v1'), speaker.configured_runner(runner, manifest):
        return [(model, item) for model in speaker.MODELS
                for item in runner.build_plan(model, seeds=SEEDS, datasets=('SHD',))]


def frozen_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Frozen queue metadata changed: {path}')
    else:
        from extended_seed_study import atomic_json
        atomic_json(path, value)


def baseline_validation(directory):
    with (directory / 'logs/mlp_snn/version_0/metrics.csv').open(newline='', encoding='utf-8') as stream:
        values = [float(row['val_acc']) for row in csv.DictReader(stream) if row.get('val_acc')]
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError(f'Invalid baseline validation history: {directory}')
    return max(values)


def write_summary(manifest):
    """Rank only validation means; never read or rank by test scores."""
    from extended_seed_study import atomic_json, read_completed
    root = local_path(OUTPUT_ROOT)
    lines = ['# SHD speaker holdout: regularization tuning', '',
             'Validation accuracy (%) only. Official test evaluation is deferred.',
             f'Split seed 42; validation speakers {manifest["validation_speaker_ids"]}; '
             f'fingerprint `{manifest["fingerprint"]}`.',
             'Fresh training: 300 epochs, batch 256, baseline LR/schedule; full resume only for interrupted runs.', '',
             '| Model | Variant | Seed 42 | Seed 123 | Seed 456 | n | Mean +/- sample SD |',
             '|---|---|---:|---:|---:|---:|---:|']
    selections = {}
    all_done = True
    for model in speaker.MODELS:
        candidates = []
        for profile in ('augmented_v1', *PROFILES):
            scores = []
            cells = []
            with speaker.selected_profile(profile):
                for seed in SEEDS:
                    directory = local_path(f'{speaker.OUTPUT_ROOT}/{model}/SHD/seed_{seed}')
                    if profile == 'augmented_v1':
                        value = baseline_validation(directory)
                    elif (directory / 'validation_completed.txt').exists():
                        record, _ = read_completed(directory)
                        value = record['best_validation_accuracy']
                    else:
                        value = None
                        all_done = False
                    cells.append('pending' if value is None else f'{100 * value:.4f}')
                    if value is not None:
                        scores.append(value)
            mean = statistics.mean(scores) if scores else None
            spread = (f'{100 * mean:.4f} +/- {100 * statistics.stdev(scores):.4f}' if len(scores) > 1
                      else f'{100 * mean:.4f} (SD unavailable)' if scores else 'pending')
            lines.append(f'| {model} | {profile} | {" | ".join(cells)} | {len(scores)} | {spread} |')
            if len(scores) == len(SEEDS):
                candidates.append((mean, profile))
        # Do not publish a winner until every candidate has all matching seeds.
        if len(candidates) == len(PROFILES) + 1:
            best = max(value for value, _ in candidates)
            selections[model] = dict(
                profiles=[profile for value, profile in candidates if value == best],
                mean_validation_accuracy=best,
                selection_rule='maximum mean best-validation accuracy across seeds 42/123/456; ties retained')
    if all_done:
        atomic_json(root / 'VALIDATION_SELECTION.json', dict(
            study=VERSION, split_fingerprint=manifest['fingerprint'], selections=selections,
            note='Tuning on one fixed speaker pair; future independent confirmation is recommended. '
                 'No new official-test results have been used for selection.'))
    temporary = root / 'VALIDATION_COMPARISON.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(root / 'VALIDATION_COMPARISON.md')


def run_queue(args, parser, runner):
    from datasets.shd_speaker_split import save_manifest
    from extended_seed_study import study_lock
    if args.measure_existing_costs:
        parser.error('This queue is validation-only; test-cost measurement is deferred')
    try:
        with speaker.selected_profile('augmented_v1'):
            manifest = speaker.preflight_data()
        jobs = ordered_jobs(args.shd_speaker_phase)
        baselines = baseline_plans(runner, manifest)
        plans = profile_plans(runner, manifest, jobs)
        print(f'{VERSION}: validation speakers {manifest["validation_speaker_ids"]}; '
              f'split {manifest["fingerprint"]}', flush=True)
        for index, (profile, model, item) in enumerate(plans, 1):
            print(f'{index:02d}. {profile} / {model} / seed {item["seed"]}: '
                  f'{item["status"]} -> {portable_path(item["directory"])}', flush=True)
        pending = [f'{model} seed {item["seed"]}' for model, item in baselines if item['status'] != 'complete']
        if args.dry_run:
            print(f'Validated {len(plans)} sequential jobs; no training, downloads or writes.')
            if pending:
                print('Wait for the existing baseline queue before launching: ' + ', '.join(pending))
            return
        if pending:
            raise ValueError('Finish the existing augmented_v1 queue first: ' + ', '.join(pending))
        with ExitStack() as locks:
            # Exclude the current baseline launcher and competing regularization phases.
            locks.enter_context(study_lock(local_path(f'results/experiments/{speaker.AUGMENTED_PROTOCOL}')))
            locks.enter_context(study_lock(local_path(OUTPUT_ROOT)))
            for profile in PROFILES:
                with speaker.selected_profile(profile):
                    locks.enter_context(study_lock(local_path(speaker.OUTPUT_ROOT)))
                    save_manifest(speaker.SPLIT_PATH, manifest)
            profile_plans(runner, manifest, jobs)  # Recheck guards under the locks.
            if args.shd_speaker_phase == '2':
                first = profile_plans(runner, manifest, ordered_jobs('1'))
                if any(item['status'] != 'complete' for _, _, item in first):
                    raise ValueError('Complete seed 42 for all three variants and both models before Phase 2')
            root = local_path(OUTPUT_ROOT)
            frozen_json(root / 'QUEUE.json', dict(
                study=VERSION, baseline='shd_speaker_augmented_v1',
                split_fingerprint=manifest['fingerprint'], selection_metric='validation_accuracy',
                selection_rule='maximum three-seed mean best-validation accuracy per model, including baseline',
                jobs=[dict(profile=p, model=m, seed=s) for p, m, s in ordered_jobs('all')]))
            for profile, model, seed in jobs:
                with speaker.selected_profile(profile), speaker.configured_runner(runner, manifest):
                    runner.run(model, seeds=(seed,), datasets=('SHD',))
                write_summary(manifest)
            print(f'Finished phase {args.shd_speaker_phase}. Validation comparison: '
                  f'{OUTPUT_ROOT}/VALIDATION_COMPARISON.md', flush=True)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Rerun the same command for full resume from each run\'s last.ckpt.\n')
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')

"""Version 1.0.1: isolated speaker-holdout phases through the shared runner."""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import csv
import json
import math
from pathlib import Path
import statistics

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from datasets.shd_speaker_split import (
    PROTOCOL, build_manifest, check_loader, save_manifest, verify_manifest,
)
from project_paths import local_path, portable_config, portable_path

VERSION = '1.0.1'
AUGMENTED_PROTOCOL = 'shd_speaker_augmented_v1'
REGULARIZATION_PROFILES = ('dropout030_v1', 'dropout035_v1', 'spike_delete010_v1')
REGULARIZATION_OUTPUT_ROOT = portable_path('results/experiments/shd_speaker_regularization_v1')
ACTIVE_PROFILE = None
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = f'results/experiments/{PROTOCOL}'
ARCHIVE_ROOT = f'results/not_improved/{PROTOCOL}'
SPLIT_PATH = f'{OUTPUT_ROOT}/SPLIT_MANIFEST.json'
MODELS = ('DTH_SE_adLIF', 'LG_DTH_SE_adLIF')
PHASES = {'1': ((MODELS[0], 42), (MODELS[1], 42)),
          '2': ((MODELS[0], 123), (MODELS[1], 123), (MODELS[0], 456), (MODELS[1], 456))}


@contextmanager
def selected_profile(profile):
    """Scope the new opt-in study without modifying the shared runner/defaults."""
    keys = ('PROTOCOL', 'VERSION', 'OUTPUT_ROOT', 'ARCHIVE_ROOT', 'SPLIT_PATH', 'ACTIVE_PROFILE')
    original = {key: globals()[key] for key in keys}
    try:
        if profile == 'augmented_v1':
            output = portable_path(f'results/experiments/{AUGMENTED_PROTOCOL}')
            globals().update(PROTOCOL=AUGMENTED_PROTOCOL, VERSION='1.0.0', ACTIVE_PROFILE=profile,
                             OUTPUT_ROOT=output,
                             ARCHIVE_ROOT=f'results/not_improved/{AUGMENTED_PROTOCOL}',
                             SPLIT_PATH=f'{output}/SPLIT_MANIFEST.json')
        elif profile in REGULARIZATION_PROFILES:
            protocol = f'shd_speaker_{profile}'
            output = f'{REGULARIZATION_OUTPUT_ROOT}/{profile}'
            globals().update(PROTOCOL=protocol, VERSION='1.0.0', ACTIVE_PROFILE=profile,
                             OUTPUT_ROOT=output, ARCHIVE_ROOT=f'results/not_improved/{protocol}',
                             SPLIT_PATH=f'{output}/SPLIT_MANIFEST.json')
        elif profile is not None:
            raise ValueError(f'Unknown speaker study profile: {profile}')
        yield
    finally:
        globals().update(original)


def ordered_jobs(phase):
    return PHASES['1'] + PHASES['2'] if phase == 'all' else PHASES[phase]


def execute_jobs(jobs, execute):
    """Exceptions stop the queue immediately; no detached child processes."""
    for model, seed in jobs:
        execute(model, seed)


def require_phase_one(runner):
    for model, seed in PHASES['1']:
        item = runner.build_plan(model, seeds=(seed,), datasets=('SHD',))[0]
        if item['status'] != 'complete':
            raise ValueError('Complete Phase 1 before launching Phase 2')


def job_overrides(model, dataset, seed, manifest):
    if model not in MODELS or dataset != 'SHD' or seed not in (42, 123, 456):
        raise ValueError('Speaker holdout supports only DTH/LG-DTH SHD seeds 42/123/456')
    deviations = [f'model_variant={model}',
                  'validation=two held-out training speakers; replaces random-recording SHD* split']
    profile_overrides = []
    if ACTIVE_PROFILE:
        profile_overrides = [f'+shd_speaker_study={ACTIVE_PROFILE}']
        context = (nullcontext() if GlobalHydra.instance().is_initialized() else
                   initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None))
        with context:
            settings = compose(config_name='main', overrides=[f'+shd_speaker_study={ACTIVE_PROFILE}'])
        baseline = OmegaConf.load(ROOT / 'config/paper_baseline.yaml').paper_baseline
        deviations += [f'dropout={settings.dropout} (paper baseline {baseline.common.dropout})',
                       'training_augmentation=' + json.dumps(OmegaConf.to_container(
                           settings.dataset.train_augmentation, resolve=True), sort_keys=True)]
    return profile_overrides + [f'++evaluation_protocol={PROTOCOL}',
            f'++speaker_holdout_module_version={VERSION}',
            f'++paper_baseline_deviations={json.dumps(deviations)}',
            f'dataset.speaker_split_manifest={SPLIT_PATH}',
            f'dataset.speaker_split_fingerprint={manifest["fingerprint"]}']


def validate_config(cfg, model, seed, manifest):
    cfg = portable_config(cfg)
    baseline = OmegaConf.to_container(OmegaConf.load(ROOT / 'config/paper_baseline.yaml').paper_baseline,
                                      resolve=True)
    shd, policy = baseline['datasets']['SHD'], baseline['reproduction_policy']
    if (shd['n_epochs'], shd['batch_size'], policy['early_stopping']) != (300, 256, False):
        raise ValueError('SHD baseline budget changed; version this experiment before use')
    if cfg['paper_baseline'] != baseline or cfg['paper_baseline_version'] != baseline['version']:
        raise ValueError('Speaker holdout must inherit the current paper baseline')
    # Compare every effective setting against the ordinary model's resolved baseline.
    # Apply the explicit regularization overlay before checking all other fields.
    context = (nullcontext() if GlobalHydra.instance().is_initialized() else
               initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None))
    with context:
        expected_overrides = [f'experiment=SHD_{model}', '+comparison=SHD', f'random_seed={seed}']
        if ACTIVE_PROFILE:
            expected_overrides += [f'+shd_speaker_study={ACTIVE_PROFILE}']
        expected = compose(config_name='main', overrides=expected_overrides)
        expected = OmegaConf.to_container(expected, resolve=True)
    actual = copy.deepcopy(cfg)
    routing = ('evaluation_protocol', 'model_variant', 'launcher_script', 'launcher_logdir',
               'run_mode', 'paper_baseline_deviations', 'speaker_holdout_module_version', 'logdir')
    for key in routing:
        actual.pop(key, None)
        expected.pop(key, None)
    for key in ('speaker_split_manifest', 'speaker_split_fingerprint'):
        actual['dataset'].pop(key, None)
    actual['dataset']['validate_on'] = expected['dataset']['validate_on']
    actual['dataset']['cache_namespace'] = expected['dataset']['cache_namespace']
    if portable_config(actual) != portable_config(expected):
        raise ValueError('Speaker-holdout training settings differ from the matched model baseline')
    if (cfg['evaluation_protocol'] != PROTOCOL or cfg['dataset']['validate_on'] != 'speaker_holdout'
            or cfg['dataset']['cache_namespace'] != (expected['dataset']['cache_namespace'] if ACTIVE_PROFILE else PROTOCOL)
            or cfg['dataset']['speaker_split_manifest'] != SPLIT_PATH
            or cfg['dataset']['speaker_split_fingerprint'] != manifest['fingerprint']
            or cfg['model_variant'] != model or cfg['random_seed'] != seed):
        raise ValueError('Invalid speaker-holdout protocol/split/model/seed identity')
    if model == MODELS[1] and any(not cfg[layer][key] for layer in ('l1', 'l2')
                                 for key in ('local_slow_enabled', 'local_transient_enabled')):
        raise ValueError('Both LG-DTH local corrections must remain enabled')


def preflight_data():
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = compose(config_name='main', overrides=['experiment=SHD_DTH_SE_adLIF', '+comparison=SHD'])
        data_path = local_path(cfg.dataset.data_path)
    manifest = build_manifest(data_path)
    # Tonic checks archive presence even with extracted data. Fail instead of downloading.
    location = data_path / 'SHDWrapper'
    for name in ('shd_train.h5.zip', 'shd_test.h5.zip'):
        if not (location / name).is_file():
            raise FileNotFoundError(f'Installed Tonic loader requires {location / name}; preflight never downloads')
    from datasets.shd import SHDWrapper
    check_loader(SHDWrapper(save_to=str(data_path), train=True), manifest)
    test = SHDWrapper(save_to=str(data_path), train=False)
    if test.data_filename != 'shd_test.h5':
        raise ValueError('Installed SHD test loader does not use the official test file')
    if local_path(SPLIT_PATH).exists():
        saved = json.loads(local_path(SPLIT_PATH).read_text(encoding='utf-8'))
        verify_manifest(saved, data_path, manifest['fingerprint'])
    if ACTIVE_PROFILE and (len(manifest['subsets']['train']['speaker_ids']) != 8
                           or len(manifest['subsets']['validation']['speaker_ids']) != 2):
        raise ValueError('Augmented SHD study requires exactly 8 training / 2 validation speakers')
    return manifest


@contextmanager
def configured_runner(runner, manifest):
    keys = ('EVALUATION', 'PROTOCOL', 'RUN_LOGDIR', 'SPEAKER_HOLDOUT',
            'SEED_COHORT', 'VALIDATION_ONLY', 'SSC_STUDY_JOB')
    original = {key: getattr(runner, key) for key in keys}
    try:
        runner.EVALUATION = dict(name=PROTOCOL, seeds=[42, 123, 456], common={}, datasets={'SHD': {}})
        runner.PROTOCOL = runner.EVALUATION['datasets']
        runner.RUN_LOGDIR, runner.SPEAKER_HOLDOUT = OUTPUT_ROOT, manifest
        runner.SEED_COHORT = runner.SSC_STUDY_JOB = None
        runner.VALIDATION_ONLY = ACTIVE_PROFILE in REGULARIZATION_PROFILES
        yield
    finally:
        for key, value in original.items():
            setattr(runner, key, value)


def test_accuracy(directory):
    path = directory / 'logs/mlp_snn/version_0/metrics.csv'
    if not path.exists():
        raise ValueError(f'Missing final test metrics: {directory}')
    with path.open(newline='', encoding='utf-8') as stream:
        values = [float(row['test_acc']) for row in csv.DictReader(stream) if row.get('test_acc')]
    if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or len(set(values)) != 1:
        raise ValueError(f'Missing, invalid or conflicting final test metrics: {directory}')
    return values[-1] * 100


def inspect_checkpoint(directory, cfg, complete=False):
    from extended_seed_study import load_run_checkpoint, checkpoint_selection
    split = directory / 'SPLIT_MANIFEST.json'
    if not split.exists() or json.loads(split.read_text(encoding='utf-8'))['fingerprint'] != cfg['dataset']['speaker_split_fingerprint']:
        raise ValueError(f'Missing/mismatched per-run speaker split: {directory}')
    saved = json.loads(split.read_text(encoding='utf-8'))
    shared = json.loads(local_path(SPLIT_PATH).read_text(encoding='utf-8'))
    if saved != shared:
        raise ValueError(f'Per-run speaker split differs from frozen manifest: {directory}')
    last = load_run_checkpoint(directory / 'ckpt/last.ckpt', cfg)
    if not last.get('optimizer_states') or not last.get('lr_schedulers'):
        raise ValueError(f'Full resume requires optimizer and scheduler state: {directory}')
    best, _ = checkpoint_selection(last, directory)
    selected = load_run_checkpoint(best, cfg)
    if selected['epoch'] > last['epoch']:
        raise ValueError(f'Best checkpoint is ahead of last checkpoint: {directory}')
    if complete:
        if last['epoch'] + 1 != cfg['n_epochs']:
            raise ValueError(f'Completed run did not finish its full budget: {directory}')
        if cfg.get('shd_regularization_study'):
            from extended_seed_study import reject_test_artifacts
            reject_test_artifacts(directory)
        else:
            test_accuracy(directory)


def write_aggregate(model, runner):
    root, manifest = runner.result_root(model), runner.SPEAKER_HOLDOUT
    title = 'SHD* speaker holdout + augmentation' if ACTIVE_PROFILE else 'SHD speaker holdout'
    lines = [f'# {model}: {title} (two validation speakers)', '',
             f'Protocol: `{PROTOCOL}`; module {VERSION}. Baseline: `{runner._BASELINE.version}`.',
             'Final official-test accuracy (%) from speaker-validation-selected checkpoints; ordinary inference without TTA.',
             'This protocol differs from random-recording SHD*; paper accuracy is not a matched baseline.',
             f'Split fingerprint: `{manifest["fingerprint"]}`.',
             f'Validation speakers: {manifest["validation_speaker_ids"]}; validation fraction: {manifest["validation_fraction"]:.6%}.',
             'Per-run resolved settings: `SHD/seed_<seed>/FULL_RUN_CONFIG.json`; split counts/indices: `SPLIT_MANIFEST.json`.', '',
             '| Dataset / protocol | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |',
             '|---|---:|---:|---:|---:|---:|']
    values, cells = [], []
    for seed in (42, 123, 456):
        directory = root / 'SHD' / f'seed_{seed}'
        if (directory / 'completed.txt').exists():
            value = test_accuracy(directory)
            values.append(value)
            cells.append(f'{value:.2f}')
        else:
            cells.append('pending')
    aggregate = (f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}' if len(values) > 1
                 else f'{values[0]:.2f} (SD unavailable)' if values else 'pending')
    label = 'SHD* speaker holdout + augmentation v1' if ACTIVE_PROFILE else 'SHD speaker holdout v1'
    lines += [f'| {label} | {" | ".join(cells)} | {len(values)} | {aggregate} |', '',
              *(line.replace('| SHD* |', f'| {label} |')
                for line in runner.cost_aggregate_lines(root))]
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / 'RESULTS.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(root / 'RESULTS.md')


def main(argv, runner):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shd-speaker-phase', choices=('1', '2', 'all'), required=True)
    parser.add_argument('--shd-speaker-profile', choices=('augmented_v1', 'regularization_v1'),
                        help='New isolated augmentation/dropout study; omitted selects retired v1')
    parser.add_argument('--dry-run', action='store_true', help='Check actual data and every planned run; no training/writes')
    parser.add_argument('--measure-existing-costs', action='store_true')
    args = parser.parse_args(argv)
    if args.shd_speaker_profile == 'regularization_v1':
        from shd_speaker_regularization import run_queue
        return run_queue(args, parser, runner)
    with selected_profile(args.shd_speaker_profile):
        return run_phase(args, parser, runner)


def run_phase(args, parser, runner):
    try:
        if (ROOT / ARCHIVE_ROOT / 'ARCHIVE.json').is_file():
            raise ValueError(f'This experiment was retired by the user; results are preserved at {ARCHIVE_ROOT}. '
                             'Intentional reruns require a new experiment version and output directory.')
        manifest = preflight_data()
        print(f'{PROTOCOL}: validation speakers {manifest["validation_speaker_ids"]}; '
              f'fingerprint {manifest["fingerprint"]}', flush=True)
        for name, subset in manifest['subsets'].items():
            print(f'{name}: {subset["sample_count"]} recordings; speakers {subset["speaker_ids"]}; '
                  f'class counts {subset["class_counts"]}', flush=True)
        print(f'Validation fraction: {manifest["validation_fraction"]:.6%}', flush=True)
        with configured_runner(runner, manifest):
            jobs = ordered_jobs(args.shd_speaker_phase)
            def preflight_jobs():
                return [(model, runner.build_plan(model, seeds=(seed,), datasets=('SHD',))[0])
                        for model, seed in jobs]
            plans = preflight_jobs()
            for model, item in plans:
                print(f'{model} seed {item["seed"]}: {item["status"]} -> {portable_path(item["directory"])}', flush=True)
            if args.dry_run:
                print(f'Validated {len(plans)} sequential jobs and real SHD metadata; no training or writes.')
                return
            from extended_seed_study import study_lock
            with study_lock(local_path(OUTPUT_ROOT)):
                # The same OS lock protects both phases; stale files do not bypass it.
                save_manifest(SPLIT_PATH, manifest)
                preflight_jobs()
                if args.shd_speaker_phase == '2':
                    require_phase_one(runner)
                def execute(model, seed):
                    runner.run(model, seeds=(seed,), datasets=('SHD',),
                               measure_existing_costs=args.measure_existing_costs)
                    directory = runner.result_root(model) / 'SHD' / f'seed_{seed}'
                    if not (directory / 'COST_RESULTS.json').exists():
                        raise RuntimeError('Training/test remain complete; cost measurement is pending. '
                                           'Rerun this phase with --measure-existing-costs; no retraining.')
                execute_jobs(jobs, execute)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Rerun the same phase to resume from its own last.ckpt.\n')
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')

"""Evaluate models with shared settings: python runner.py --models MODEL --datasets SHD --seeds 42."""

import argparse
import copy
import hashlib
import csv
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from evaluation_protocol import load_protocol, protocol_overrides, protocol_settings
from project_paths import portable_config, portable_path
from cost_records import normalized_record, VERSIONS as COST_VERSIONS


ROOT = Path(__file__).resolve().parent
_BASELINE = OmegaConf.load(ROOT / 'config/paper_baseline.yaml').paper_baseline
EVALUATION = {'name': _BASELINE.version,
              'seeds': list(_BASELINE.reproduction_policy.training_seeds),
              'common': {}, 'datasets': {name: {} for name in _BASELINE.datasets}}
SEEDS = tuple(EVALUATION['seeds'])
RUN_LOGDIR = "results/valid"
# One explicit protocol shared by reference and proposed models.
PROTOCOL = EVALUATION['datasets']


def result_root(model, adaptation_update_interval=2):
    root = ROOT / RUN_LOGDIR / model
    return root / f"K_{adaptation_update_interval}" if model == "MR_SE_adLIF" else root


def dataset_protocol_name(dataset):
    """SSC's explicit mean-loss protocol leaves SHD/ECG metadata unchanged."""
    name = EVALUATION['name']
    if dataset == 'SSC' and name.startswith(str(_BASELINE.version)):
        return name.replace(str(_BASELINE.version), 'ssc_mean_loss_v1', 1)
    return name


def matches_paper_baseline(previous, resolved, dataset):
    """Accept legacy metadata only after comparing all effective settings."""
    baseline_keys = ('paper_baseline', 'paper_baseline_version',
                     'paper_baseline_dataset', 'paper_baseline_deviations')
    if (dataset not in ('SHD', 'SSC', 'ECG')
            or not re.fullmatch(r'controlled_v2(?:_custom_[0-9a-f]{12})?',
                                str(previous.get('evaluation_protocol', '')))
            or any(key in previous for key in baseline_keys)
            or resolved.get('evaluation_protocol') != resolved.get('paper_baseline_version')
            or resolved.get('paper_baseline_dataset') != dataset):
        return False
    old, current = copy.deepcopy(previous), copy.deepcopy(resolved)
    for key in (*baseline_keys, 'evaluation_protocol'):
        old.pop(key, None)
        current.pop(key, None)
    # SHDLDM's omitted value has always defaulted to False.
    if dataset == 'SHD':
        old.setdefault('dataset', {}).setdefault('additional_test_set_validation', False)
    return old == current


def build_plan(model, adaptation_update_interval=2, seeds=SEEDS, datasets=None, resume=1):
    if resume not in (0, 1):
        raise ValueError('resume must be 0 or 1')
    seeds = tuple(dict.fromkeys(seeds))
    if not seeds or any(isinstance(seed, bool) or seed not in SEEDS for seed in seeds):
        raise ValueError(f"Select one or more canonical seeds: {SEEDS}")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", model):
        raise ValueError("Model must be an experiment suffix, for example MT_SE_adLIF")
    if (isinstance(adaptation_update_interval, bool)
            or not isinstance(adaptation_update_interval, int) or adaptation_update_interval < 1):
        raise ValueError("adaptation_update_interval must be a positive integer")
    plan = []
    selected = tuple(dict.fromkeys(datasets if datasets is not None else PROTOCOL))
    if not selected or any(dataset not in PROTOCOL for dataset in selected):
        raise ValueError(f'Select datasets from {tuple(PROTOCOL)}')
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        for dataset in selected:
            experiment = f"{dataset}_{model}" + ("_2layer" if dataset == "ECG" else "")
            if not (ROOT / "config" / "experiment" / f"{experiment}.yaml").is_file():
                raise ValueError(f"Missing experiment config: config/experiment/{experiment}.yaml")
            for seed in seeds:
                directory = result_root(model, adaptation_update_interval) / dataset / f"seed_{seed}"
                if RUN_LOGDIR == 'results/valid' and model != 'SE_adLIF':
                    retained = ROOT / 'results/not_improved' / directory.relative_to(ROOT / RUN_LOGDIR)
                    if retained.exists():
                        raise ValueError(f'Previous experiment retained at {retained}; use an explicit separate --logdir for an intentional new experiment')
                # Leave active historical-layout jobs in place until they finish.
                if RUN_LOGDIR == 'results/valid' and not directory.exists():
                    legacy = ROOT / 'results/full_runs/protocol_v2' / model
                    if model == 'MR_SE_adLIF':
                        legacy /= f'K_{adaptation_update_interval}'
                    legacy = legacy / dataset / f'seed_{seed}'
                    if legacy.exists():
                        directory = legacy
                launcher_root = directory.parents[2] if model != 'MR_SE_adLIF' else directory.parents[3]
                launcher_logdir = launcher_root.relative_to(ROOT).as_posix()
                if resume == 0 and directory.exists() and any(directory.iterdir()):
                    raise ValueError(f'--resume 0 requires an empty output directory; use --logdir for a fresh run: {directory}')
                overrides = [
                    f"experiment={experiment}", f"random_seed={seed}",
                    f"+comparison={dataset}",
                    f"logdir={json.dumps(portable_path(directory, ROOT))}",
                    f"hydra.run.dir={json.dumps(directory.as_posix())}",
                    "++launcher_script=runner.py", f"++model_variant={model}",
                    f"++run_mode={'reference' if model == 'SE_adLIF' else 'ours'}",
                    f"++launcher_logdir={launcher_logdir}",
                    f"++evaluation_protocol={dataset_protocol_name(dataset)}",
                    f"++paper_baseline_deviations={json.dumps([] if model == 'SE_adLIF' else [f'model_variant={model}'])}",
                ]
                overrides += protocol_overrides(EVALUATION, dataset)
                if dataset == 'SSC':
                    # Enforce the runner's SSC default in the child command too,
                    # rather than depending only on Hydra defaults ordering.
                    ssc_loss = OmegaConf.load(ROOT / 'config/comparison/SSC.yaml').loss_agg
                    if ssc_loss != 'mean_membrane_potentials':
                        raise ValueError('SSC runner configuration must use mean_membrane_potentials')
                    overrides.append(f'++loss_agg={ssc_loss}')
                    deviations = ([] if model == 'SE_adLIF' else [f'model_variant={model}'])
                    deviations.append('loss_agg=mean_membrane_potentials (released SSC implementation)')
                    overrides.append(f'++paper_baseline_deviations={json.dumps(deviations)}')
                if model == "MR_SE_adLIF":
                    overrides.append(f"adaptation_update_interval={adaptation_update_interval}")
                cfg = compose(config_name="main", overrides=overrides)
                resolved = portable_config(OmegaConf.to_container(cfg, resolve=True), ROOT)
                # Validate model registry, architecture and temporal aggregation
                # before launching any expensive training subprocess.
                from models.pl_module import layer_map
                for layer in (cfg.l1, cfg.l2):
                    if layer.cell not in layer_map:
                        raise ValueError(f"Unregistered cell: {layer.cell}")
                expected_loss = OmegaConf.load(ROOT / 'config' / 'comparison' / f'{dataset}.yaml').loss_agg
                if cfg.loss_agg != expected_loss or not cfg.two_layers:
                    raise ValueError(f"{experiment} must use two layers and {expected_loss}")
                manifest = directory / "FULL_RUN_CONFIG.json"
                compatibility = None
                if manifest.exists():
                    original_config = json.loads(manifest.read_text(encoding="utf-8"))
                    previous = portable_config(copy.deepcopy(original_config), ROOT)
                    # Directory-only relocation; preserve the original saved manifest.
                    if RUN_LOGDIR == 'results/valid':
                        for key in ('logdir', 'launcher_logdir'):
                            value = previous.get(key)
                            if isinstance(value, str):
                                previous[key] = value.replace('results/valid/protocol_v2', 'results/valid')
                    # These newly explicit defaults preserve older runner behavior.
                    for key, value in {
                        'init_ckpt_path': None, 'run_test': True, 'scheduler': 'plateau',
                        'scheduler_min_lr': 0.0, 'gradient_clip_val': 1.5,
                    }.items():
                        previous.setdefault(key, value)
                    # The public launcher rename does not change training settings.
                    if previous.get('launcher_script') == 'full_runner.py':
                        previous['launcher_script'] = 'runner.py'
                    if (RUN_LOGDIR == 'results/valid'
                            and previous.get('launcher_logdir') == 'results/full_runs/protocol_v2'
                            and str(previous.get('logdir', '')).replace('\\', '/') ==
                            portable_path(directory, ROOT).replace('results/valid/', 'results/full_runs/protocol_v2/')):
                        previous['logdir'] = resolved['logdir']
                        previous['launcher_logdir'] = resolved['launcher_logdir']
                    if previous != resolved:
                        if not matches_paper_baseline(previous, resolved, dataset):
                            raise ValueError(f"Configuration changed; refusing to reuse {directory}")
                        compatibility = {
                            'original_config': original_config,
                            'resolved_config': resolved,
                            'paper_baseline_version': resolved['paper_baseline_version'],
                            'paper_baseline_dataset': dataset,
                            'note': 'Effective settings match; original protocol metadata retained in FULL_RUN_CONFIG.json.',
                        }
                    elif original_config != resolved:
                        compatibility = {
                            'original_config': original_config,
                            'resolved_config': resolved,
                            'paper_baseline_version': resolved['paper_baseline_version'],
                            'paper_baseline_dataset': dataset,
                            'note': 'Equivalent relocated paths or explicit defaults; original manifest retained.',
                        }
                elif directory.exists() and any(directory.iterdir()):
                    raise ValueError(f"Unrecognized existing results; refusing to overwrite {directory}")
                if (directory / "completed.txt").exists():
                    status = "complete"
                elif (directory / "ckpt" / "last.ckpt").exists():
                    status = "resume"
                    overrides.append(f"ckpt_path={json.dumps(portable_path(directory / 'ckpt' / 'last.ckpt', ROOT))}")
                elif (directory / "ckpt").exists() and any((directory / "ckpt").iterdir()):
                    raise ValueError(f"Checkpoints exist without last.ckpt: {directory}")
                else:
                    if directory.exists() and any(
                        path.name != "FULL_RUN_CONFIG.json" for path in directory.iterdir()
                    ):
                        raise ValueError(f"Existing artifacts have no resumable checkpoint: {directory}")
                    status = "new"
                plan.append(dict(dataset=dataset, seed=seed, directory=directory,
                                 config=resolved, overrides=overrides, status=status,
                                 compatibility=compatibility))
    return plan


def collect_costs(directory, dataset):
    """A measurement failure must not make completed training resumable again."""
    try:
        result = subprocess.run([sys.executable, str(ROOT / 'measure_run_costs.py'),
                                 '--run-dir', portable_path(directory, ROOT), '--dataset', dataset], cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f'Cost evaluator exited with status {result.returncode}')
        if not (directory / 'COST_RESULTS.json').exists():
            raise RuntimeError('Cost evaluator produced no completion record')
    except (OSError, RuntimeError) as error:
        (directory / 'COST_ERROR.json').write_text(json.dumps(dict(
            status='failed', created_utc=datetime.now(timezone.utc).isoformat(), error=str(error),
            note='Training/test completed. Retry with --measure-existing-costs; no retraining needed.'), indent=2) + '\n', encoding='utf-8')
        print(f'Warning: test-cost measurement pending for {directory}: {error}', file=sys.stderr)


def cost_aggregate_lines(root):
    """Aggregate only completed, independently seeded cost records."""
    activity = ['| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |',
                '|---|---:|---:|---:|---:|---|---|']
    compute = ['| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |',
               '|---|---:|---:|---:|---:|---:|---:|']
    keys = ('total_parameters', 'trainable_parameters', 'spikes_total_per_sample',
            'spikes_l1_per_sample', 'spikes_l2_per_sample', 'firing_l1_percent', 'firing_l2_percent',
            'synops_per_sample', 'dense_gflops_test_padded', 'arithmetic_gflops_test_padded',
            'logical_gops_test_padded', 'forward_ms_per_sample')
    for dataset in PROTOCOL:
        records = []
        for seed in SEEDS:
            directory = root / dataset / f'seed_{seed}'
            path = directory / 'COST_RESULTS.json'
            if not (directory / 'completed.txt').exists() or not path.exists():
                continue
            record = normalized_record(json.loads(path.read_text(encoding='utf-8')))
            if (record.get('version') not in COST_VERSIONS or record.get('status') != 'complete'
                    or record.get('dataset') != dataset or record.get('seed') != seed
                    or any(not isinstance(record.get('measurements', {}).get(k), (int, float))
                           or not math.isfinite(record['measurements'][k]) for k in keys)):
                raise ValueError(f'Invalid cost record: {path}')
            records.append(record)
        label = 'SHD*' if dataset == 'SHD' else dataset
        if not records:
            activity.append(f'| {label} | 0 | pending | pending | pending | pending | pending |')
            compute.append(f'| {label} | 0 | pending | pending | pending | pending | pending |')
            continue
        if len({r['data_sha256'] for r in records}) != 1:
            raise ValueError(f'Cost records use different test data/batching: {dataset}')
        def cell(key, digits=2, scale=1):
            values = [r['measurements'][key] / scale for r in records]
            mean = f'{statistics.mean(values):,.{digits}f}'
            return mean + (f' ± {statistics.stdev(values):,.{digits}f}' if len(values) > 1 else ' (n=1)')
        pair = lambda a, b, digits: cell(a, digits) + ' / ' + cell(b, digits)
        activity.append(f'| {label} | {len(records)} | ' + ' | '.join([
            cell('total_parameters', 0), cell('trainable_parameters', 0), cell('spikes_total_per_sample', 0),
            pair('spikes_l1_per_sample', 'spikes_l2_per_sample', 0),
            pair('firing_l1_percent', 'firing_l2_percent', 2)]) + ' |')
        hardware = {tuple(r['environment'].get(k) for k in ('gpu', 'torch', 'matmul_precision', 'device'))
                    for r in records}
        compute.append(f'| {label} | {len(records)} | ' + ' | '.join([
            cell('synops_per_sample', 2, 1e6), cell('dense_gflops_test_padded', 6),
            cell('arithmetic_gflops_test_padded', 6), cell('logical_gops_test_padded', 6),
            cell('forward_ms_per_sample', 3) if len(hardware) == 1 else 'mixed hardware; see per-seed records']) + ' |')
    return ['<!-- runner-costs:start -->', '## Automatically saved test-set activity and cost', '',
            'Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.', '',
            *activity, '', *compute, '',
            'Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. '
            'Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. '
            'ECG SynOps exclude analog input, whose dense MACs are stored separately. '
            'Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. '
            'Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. '
            'Future measurement does not change training settings or validate baseline compatibility.',
            '<!-- runner-costs:end -->', '']


def write_aggregate(model, adaptation_update_interval=2):
    root = result_root(model, adaptation_update_interval)
    settings_by_dataset = {}
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        for dataset in PROTOCOL:
            cfg = compose(config_name='main', overrides=[f'+comparison={dataset}'] +
                          protocol_overrides(EVALUATION, dataset))
            keys = ('loss_agg', 'n_epochs', 'batch_size', 'lr', 'early_stopping',
                    'early_stopping_patience', 'early_stopping_min_delta', 'patience', 'factor')
            settings_by_dataset[dataset] = {key: OmegaConf.select(cfg, key) for key in keys}
            for key in ('validate_on', 'valid_fraction', 'random_seed'):
                settings_by_dataset[dataset][f'dataset.{key}'] = OmegaConf.select(cfg, f'dataset.{key}')
    lines = [f"# {model}: full evaluation", "", "Statistics use final test accuracy from completed runs only.",
             f"Protocol: {EVALUATION['name']}. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.",
             'Dataset losses/protocols: ' + '; '.join(
                 f"{d}: {s['loss_agg']} ({dataset_protocol_name(d)})" for d, s in settings_by_dataset.items()),
             f"SHD* training-derived validation: {settings_by_dataset['SHD']['dataset.validate_on']:.0%}; "
             f"ECG: {settings_by_dataset['ECG']['dataset.valid_fraction']:.0%}; "
             f"split seeds: SHD* {settings_by_dataset['SHD']['dataset.random_seed']}, ECG {settings_by_dataset['ECG']['dataset.random_seed']}.",
             "SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.", "",
             'Epochs/batch: ' + ', '.join(f"{d} {s['n_epochs']}/{s['batch_size']}" for d, s in settings_by_dataset.items()),
             'Training settings: ' + '; '.join(
                 f"{d}: LR={s['lr']}, early stopping={s['early_stopping']}, "
                 f"patience={s['early_stopping_patience']}, delta={s['early_stopping_min_delta']}, "
                 f"scheduler patience={s['patience']}, factor={s['factor']}"
                 for d, s in settings_by_dataset.items()), "",
             '| Dataset | ' + ' | '.join(f'Seed {s}' for s in SEEDS) + ' | n | Mean ± sample SD |',
             '|---|' + '---:|' * (len(SEEDS) + 2)]
    if model == "MR_SE_adLIF":
        lines.insert(2, f"Adaptation update interval K={adaptation_update_interval}.")
    for dataset in PROTOCOL:
        values = []
        cells = []
        for seed in SEEDS:
            directory = root / dataset / f"seed_{seed}"
            value = None
            metrics = directory / "logs" / "mlp_snn" / "version_0" / "metrics.csv"
            if (directory / "completed.txt").exists() and metrics.exists():
                with metrics.open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("test_acc"):
                            value = 100 * float(row["test_acc"])
                if value is None or not math.isfinite(value):
                    raise ValueError(f"Missing finite test accuracy for completed run: {directory}")
            if value is not None:
                values.append(value)
            cells.append(f"{value:.2f}%" if value is not None else "pending")
        aggregate = (f"{statistics.mean(values):.2f}% \u00b1 {statistics.stdev(values):.2f}%"
                     if len(values) > 1 else f"{values[0]:.2f}% (n=1; SD unavailable)" if values else "pending")
        label = 'SHD*' if dataset == 'SHD' else dataset
        lines.append(f"| {label} | {' | '.join(cells)} | {len(values)} | {aggregate} |")
    root.mkdir(parents=True, exist_ok=True)
    # Preserve hand-maintained provenance/history sections while regenerating
    # the runner's accuracy table and its explicitly delimited cost section.
    existing = root / 'RESULTS.md'
    tail = ''
    if existing.exists():
        previous = existing.read_text(encoding='utf-8')
        previous = re.sub(r'<!-- runner-costs:start -->.*?<!-- runner-costs:end -->\s*', '', previous, flags=re.S)
        if '\n## ' in previous:
            tail = '\n## ' + previous.split('\n## ', 1)[1]
    lines += ['', *cost_aggregate_lines(root)]
    if tail:
        lines.append(tail.rstrip())
    temporary = root / "RESULTS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(root / "RESULTS.md")


def run(model, adaptation_update_interval=2, seeds=SEEDS, datasets=None, resume=1,
        measure_costs=True, measure_existing_costs=False):
    plan = build_plan(model, adaptation_update_interval, seeds, datasets, resume)  # Preflight selected runs before writes.
    for item in plan:
        directory = item["directory"]
        print(f"{item['dataset']} seed {item['seed']}: {item['status']} -> {directory}", flush=True)
        if item.get('compatibility'):
            (directory / 'PAPER_BASELINE_COMPATIBILITY.json').write_text(
                json.dumps(item['compatibility'], indent=2, sort_keys=True) + '\n',
                encoding='utf-8')
        if item["status"] == "complete":
            if measure_costs and measure_existing_costs:
                collect_costs(directory, item['dataset'])
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if not item.get('compatibility'):
            (directory / "FULL_RUN_CONFIG.json").write_text(
                json.dumps(item["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), *item["overrides"]], cwd=ROOT)
        if result.returncode:
            write_aggregate(model, adaptation_update_interval)
            raise RuntimeError(f"Run failed ({result.returncode}). Rerun the same command to resume: {directory}")
        (directory / "completed.txt").write_text("Training and final test evaluation completed.\n", encoding="utf-8")
        if measure_costs:
            collect_costs(directory, item['dataset'])
        write_aggregate(model, adaptation_update_interval)
    write_aggregate(model, adaptation_update_interval)
    print(f"Finished. Report: {result_root(model, adaptation_update_interval) / 'RESULTS.md'}")


def main(argv=None):
    global EVALUATION, PROTOCOL, RUN_LOGDIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True,
                        help="SE_adLIF, DTH_SE_adLIF, LG_DTH_SE_adLIF (including _slow/_transient), or configured future models")
    parser.add_argument("--adaptation-update-interval", type=int, default=2,
                        help="Fixed K for MR_SE_adLIF only (default: 2)")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS,
                        help="Training seeds to run (default: 42 123 456)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and display plans without training or writes")
    parser.add_argument('--no-measure-costs', action='store_true',
                        help='Skip automatic post-test spike/compute/timing measurement')
    parser.add_argument('--measure-existing-costs', action='store_true',
                        help='Also collect missing costs for selected completed runs; reuse matching saved costs')
    parser.add_argument('--resume', type=int, choices=(0, 1), default=1,
                        help='1: resume/skip existing runs (default); 0: fresh runs in empty directories')
    parser.add_argument('--datasets', nargs='+', choices=tuple(PROTOCOL), default=None,
                        help='Select datasets; defaults to SHD SSC ECG')
    parser.add_argument('--epochs', type=int, help='Override epoch limit for selected datasets')
    parser.add_argument('--batch-size', type=int, help='Override both model and dataset batch size')
    parser.add_argument('--lr', type=float, help='Override initial learning rate')
    parser.add_argument('--early-stopping', choices=('true', 'false'),
                        help='Override whether early stopping is enabled')
    parser.add_argument('--early-stopping-patience', type=int)
    parser.add_argument('--early-stopping-min-delta', type=float)
    parser.add_argument('--lr-scheduler-patience', type=int)
    parser.add_argument('--lr-scheduler-factor', type=float)
    parser.add_argument('--logdir', help='Results root; overrides default to a separate experiments folder')
    parser.add_argument('--ssc-mean-loss', action='store_true',
                        help='Compatibility flag: SSC mean loss is now automatic; requires --datasets SSC')
    parser.add_argument('--tta', action='store_true', help='MT, DTH or MT-DTH test-time augmentation only; no training')
    parser.add_argument('--tta-split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--tta-output', help='Separate evaluation output; defaults to <model>_tta_v1 experiments')
    args = parser.parse_args(argv)
    if args.no_measure_costs and args.measure_existing_costs:
        parser.error('--no-measure-costs and --measure-existing-costs cannot be combined')
    original = EVALUATION, PROTOCOL, RUN_LOGDIR
    try:
        if args.tta:
            selected_models = list(dict.fromkeys(args.models))
            if len(selected_models) != 1 or selected_models[0] not in ('MT_SE_adLIF', 'DTH_SE_adLIF', 'MT_DTH_SE_adLIF'):
                raise ValueError('--tta supports one model at a time: MT_SE_adLIF, DTH_SE_adLIF or MT_DTH_SE_adLIF')
            if (args.resume == 0 or args.logdir or args.ssc_mean_loss or args.no_measure_costs
                    or args.measure_existing_costs or any(getattr(args, k) is not None for k in
                    ('epochs','batch_size','lr','early_stopping','early_stopping_patience',
                     'early_stopping_min_delta','lr_scheduler_patience','lr_scheduler_factor'))):
                raise ValueError('TTA reuses saved checkpoints; do not supply training/cost overrides')
            from evaluate_tta import main as tta_main
            model = selected_models[0]
            tta_main(['--model', model, '--datasets', *(args.datasets or tuple(PROTOCOL)), '--seeds',
                      *(str(s) for s in args.seeds), '--split', args.tta_split,
                      '--output', args.tta_output or f'results/experiments/{model.removesuffix("_SE_adLIF").lower()}_tta_v1']
                     + (['--dry-run'] if args.dry_run else []))
            return
        if args.ssc_mean_loss:
            if args.datasets != ['SSC']:
                raise ValueError('--ssc-mean-loss requires --datasets SSC; other datasets are unchanged')
        overrides = {}
        for argument, key in (
            ('epochs', 'n_epochs'), ('batch_size', 'batch_size'), ('lr', 'lr'),
            ('early_stopping_patience', 'early_stopping_patience'),
            ('early_stopping_min_delta', 'early_stopping_min_delta'),
            ('lr_scheduler_patience', 'patience'), ('lr_scheduler_factor', 'factor')):
            value = getattr(args, argument)
            if value is not None:
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'--{argument.replace("_", "-")} must be finite and nonnegative')
                if key in ('n_epochs', 'batch_size', 'lr') and value == 0:
                    raise ValueError(f'--{argument.replace("_", "-")} must be positive')
                if key == 'factor' and not 0 < value < 1:
                    raise ValueError('--lr-scheduler-factor must be between 0 and 1')
                overrides[key] = value
        if args.batch_size is not None:
            overrides['dataset.batch_size'] = args.batch_size
        if args.early_stopping is not None:
            overrides['early_stopping'] = args.early_stopping == 'true'
        if overrides:
            EVALUATION = copy.deepcopy(EVALUATION)
            fingerprint = hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:12]
            EVALUATION['name'] += f'_custom_{fingerprint}'
            for settings in EVALUATION['datasets'].values():
                settings.update(overrides)
            PROTOCOL = EVALUATION['datasets']
            RUN_LOGDIR = f'results/experiments/{EVALUATION["name"]}'
            print('Training overrides: ' + json.dumps(overrides, sort_keys=True))
        if args.logdir:
            RUN_LOGDIR = portable_path(args.logdir, ROOT)
        # Preserve order and avoid scheduling duplicate model suites.
        models = list(dict.fromkeys(args.models))
        plans = [(model, build_plan(model, args.adaptation_update_interval, args.seeds, args.datasets, args.resume)) for model in models]
        if args.dry_run:
            for model, plan in plans:
                for item in plan:
                    print(f"{model} {item['dataset']} seed {item['seed']}: {item['status']} "
                          f"(loss={item['config']['loss_agg']}) -> {item['directory']}")
            print(f"Validated {sum(len(plan) for _, plan in plans)} runs; no training or writes.")
        else:
            for model in models:
                run(model, args.adaptation_update_interval, args.seeds, args.datasets, args.resume,
                    measure_costs=not args.no_measure_costs, measure_existing_costs=args.measure_existing_costs)
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        EVALUATION, PROTOCOL, RUN_LOGDIR = original


if __name__ == '__main__':
    main()

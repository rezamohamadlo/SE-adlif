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

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from evaluation_protocol import load_protocol, protocol_overrides, protocol_settings


ROOT = Path(__file__).resolve().parent
EVALUATION = load_protocol()
SEEDS = tuple(EVALUATION['seeds'])
RUN_LOGDIR = "results/valid/protocol_v2"
# One explicit protocol shared by reference and proposed models.
PROTOCOL = EVALUATION['datasets']


def result_root(model, adaptation_update_interval=2):
    root = ROOT / RUN_LOGDIR / model
    return root / f"K_{adaptation_update_interval}" if model == "MR_SE_adLIF" else root


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
                # Leave active historical-layout jobs in place until they finish.
                if RUN_LOGDIR == 'results/valid/protocol_v2' and not directory.exists():
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
                    f"logdir={json.dumps(directory.as_posix())}",
                    f"hydra.run.dir={json.dumps(directory.as_posix())}",
                    "++launcher_script=runner.py", f"++model_variant={model}",
                    f"++run_mode={'reference' if model == 'SE_adLIF' else 'ours'}",
                    f"++launcher_logdir={launcher_logdir}",
                    f"++evaluation_protocol={EVALUATION['name']}",
                ]
                overrides += protocol_overrides(EVALUATION, dataset)
                if model == "MR_SE_adLIF":
                    overrides.append(f"adaptation_update_interval={adaptation_update_interval}")
                cfg = compose(config_name="main", overrides=overrides)
                resolved = OmegaConf.to_container(cfg, resolve=True)
                # Validate model registry, architecture and temporal aggregation
                # before launching any expensive training subprocess.
                from models.pl_module import layer_map
                for layer in (cfg.l1, cfg.l2):
                    if layer.cell not in layer_map:
                        raise ValueError(f"Unregistered cell: {layer.cell}")
                expected_loss = ("sum_softmax_over_time" if dataset == "SHD"
                                 else "summed_membrane_potentials")
                if cfg.loss_agg != expected_loss or not cfg.two_layers:
                    raise ValueError(f"{experiment} must use two layers and {expected_loss}")
                manifest = directory / "FULL_RUN_CONFIG.json"
                if manifest.exists():
                    previous = json.loads(manifest.read_text(encoding="utf-8"))
                    # These newly explicit defaults preserve older runner behavior.
                    for key, value in {
                        'init_ckpt_path': None, 'run_test': True, 'scheduler': 'plateau',
                        'scheduler_min_lr': 0.0, 'gradient_clip_val': 1.5,
                    }.items():
                        previous.setdefault(key, value)
                    # The public launcher rename does not change training settings.
                    if previous.get('launcher_script') == 'full_runner.py':
                        previous['launcher_script'] = 'runner.py'
                    if (RUN_LOGDIR == 'results/valid/protocol_v2'
                            and previous.get('launcher_logdir') == 'results/full_runs/protocol_v2'
                            and str(previous.get('logdir', '')).replace('\\', '/') ==
                            directory.as_posix().replace('/results/valid/protocol_v2/', '/results/full_runs/protocol_v2/')):
                        previous['logdir'] = resolved['logdir']
                        previous['launcher_logdir'] = resolved['launcher_logdir']
                    if previous != resolved:
                        raise ValueError(f"Configuration changed; refusing to reuse {directory}")
                elif directory.exists() and any(directory.iterdir()):
                    raise ValueError(f"Unrecognized existing results; refusing to overwrite {directory}")
                if (directory / "completed.txt").exists():
                    status = "complete"
                elif (directory / "ckpt" / "last.ckpt").exists():
                    status = "resume"
                    overrides.append(f"ckpt_path={json.dumps((directory / 'ckpt' / 'last.ckpt').as_posix())}")
                elif (directory / "ckpt").exists() and any((directory / "ckpt").iterdir()):
                    raise ValueError(f"Checkpoints exist without last.ckpt: {directory}")
                else:
                    if directory.exists() and any(
                        path.name != "FULL_RUN_CONFIG.json" for path in directory.iterdir()
                    ):
                        raise ValueError(f"Existing artifacts have no resumable checkpoint: {directory}")
                    status = "new"
                plan.append(dict(dataset=dataset, seed=seed, directory=directory,
                                 config=resolved, overrides=overrides, status=status))
    return plan


def write_aggregate(model, adaptation_update_interval=2):
    root = result_root(model, adaptation_update_interval)
    lines = [f"# {model}: full evaluation", "", "Statistics use final test accuracy from completed runs only.",
             f"Protocol: {EVALUATION['name']}. Settings: evaluation.yaml.",
             f"SHD training-derived validation: {PROTOCOL['SHD']['dataset.validate_on']:.0%}; "
             f"ECG: {PROTOCOL['ECG']['dataset.valid_fraction']:.0%}; "
             f"split seeds: SHD {PROTOCOL['SHD']['dataset.random_seed']}, ECG {PROTOCOL['ECG']['dataset.random_seed']}.",
             "SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.", "",
             'Epochs/batch: ' + ', '.join(f"{d} {s['n_epochs']}/{s['batch_size']}" for d, s in PROTOCOL.items()),
             'Training settings: ' + '; '.join(
                 f"{d}: LR={s['lr']}, early stopping={s['early_stopping']}, "
                 f"patience={s['early_stopping_patience']}, delta={s['early_stopping_min_delta']}, "
                 f"scheduler patience={s['patience']}, factor={s['factor']}"
                 for d in PROTOCOL for s in [protocol_settings(EVALUATION, d)]), "",
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
        aggregate = (f"{statistics.mean(values):.2f}% ± {statistics.stdev(values):.2f}%"
                     if len(values) > 1 else "pending")
        lines.append(f"| {dataset} | {' | '.join(cells)} | {len(values)} | {aggregate} |")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "RESULTS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(root / "RESULTS.md")


def run(model, adaptation_update_interval=2, seeds=SEEDS, datasets=None, resume=1):
    plan = build_plan(model, adaptation_update_interval, seeds, datasets, resume)  # Preflight selected runs before writes.
    for item in plan:
        directory = item["directory"]
        print(f"{item['dataset']} seed {item['seed']}: {item['status']} -> {directory}", flush=True)
        if item["status"] == "complete":
            continue
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "FULL_RUN_CONFIG.json").write_text(
            json.dumps(item["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), *item["overrides"]], cwd=ROOT)
        if result.returncode:
            write_aggregate(model, adaptation_update_interval)
            raise RuntimeError(f"Run failed ({result.returncode}). Rerun the same command to resume: {directory}")
        (directory / "completed.txt").write_text("Training and final test evaluation completed.\n", encoding="utf-8")
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
    args = parser.parse_args(argv)
    original = EVALUATION, PROTOCOL, RUN_LOGDIR
    try:
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
            RUN_LOGDIR = args.logdir
        # Preserve order and avoid scheduling duplicate model suites.
        models = list(dict.fromkeys(args.models))
        plans = [(model, build_plan(model, args.adaptation_update_interval, args.seeds, args.datasets, args.resume)) for model in models]
        if args.dry_run:
            for model, plan in plans:
                for item in plan:
                    print(f"{model} {item['dataset']} seed {item['seed']}: {item['status']} -> {item['directory']}")
            print(f"Validated {sum(len(plan) for _, plan in plans)} runs; no training or writes.")
        else:
            for model in models:
                run(model, args.adaptation_update_interval, args.seeds, args.datasets, args.resume)
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        EVALUATION, PROTOCOL, RUN_LOGDIR = original


if __name__ == '__main__':
    main()

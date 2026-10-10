"""Sequential three-dataset, three-seed evaluation: python full_runner.py MODEL [MODEL ...]."""

import argparse
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


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
RUN_LOGDIR = "results/full_runs/protocol_v2"
SPLIT_SEED = 42
# One explicit protocol shared by reference and proposed models.
PROTOCOL = {
    "SHD": {"epochs": 300, "batch_size": 512, "early_stopping": True},
    "SSC": {"epochs": 40, "batch_size": 256, "early_stopping": False},
    "ECG": {"epochs": 400, "batch_size": 64, "early_stopping": True},
}


def result_root(model, adaptation_update_interval=2):
    root = ROOT / RUN_LOGDIR / model
    return root / f"K_{adaptation_update_interval}" if model == "MR_SE_adLIF" else root


def build_plan(model, adaptation_update_interval=2):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", model):
        raise ValueError("Model must be an experiment suffix, for example MT_SE_adLIF")
    if (isinstance(adaptation_update_interval, bool)
            or not isinstance(adaptation_update_interval, int) or adaptation_update_interval < 1):
        raise ValueError("adaptation_update_interval must be a positive integer")
    plan = []
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        for dataset, settings in PROTOCOL.items():
            experiment = f"{dataset}_{model}" + ("_2layer" if dataset == "ECG" else "")
            if not (ROOT / "config" / "experiment" / f"{experiment}.yaml").is_file():
                raise ValueError(f"Missing experiment config: config/experiment/{experiment}.yaml")
            for seed in SEEDS:
                directory = result_root(model, adaptation_update_interval) / dataset / f"seed_{seed}"
                overrides = [
                    f"experiment={experiment}", f"random_seed={seed}",
                    f"n_epochs={settings['epochs']}",
                    f"batch_size={settings['batch_size']}",
                    f"dataset.batch_size={settings['batch_size']}",
                    "dataset.num_workers=0", "patience=9999", "factor=0.9",
                    f"early_stopping={str(settings['early_stopping']).lower()}",
                    "early_stopping_patience=50", "early_stopping_min_delta=0.001",
                    f"logdir={json.dumps(directory.as_posix())}",
                    f"hydra.run.dir={json.dumps(directory.as_posix())}",
                    "++launcher_script=full_runner.py", f"++model_variant={model}",
                    f"++run_mode={'reference' if model == 'SE_adLIF' else 'ours'}",
                    f"++launcher_logdir={RUN_LOGDIR}",
                    "++evaluation_protocol=controlled_v2",
                ]
                if dataset == "SHD":
                    overrides += ["dataset.validate_on=0.2", f"++dataset.random_seed={SPLIT_SEED}",
                                  "++dataset.cache_namespace=controlled_v2"]
                elif dataset == "ECG":
                    overrides += ["dataset.valid_fraction=0.05", f"dataset.random_seed={SPLIT_SEED}"]
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
                    if json.loads(manifest.read_text(encoding="utf-8")) != resolved:
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
             "Controlled v2: SHD uses 20% training-derived validation; ECG uses 5%; split seed is 42.",
             "SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.", "",
             "Protocol: SHD 300/512, SSC 40/256, ECG 400/64 (epochs/batch).",
             "LR scheduler patience 9999; SHD/ECG early stopping patience 50, min_delta 0.001; SSC fixed 40 epochs.", "",
             "| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |",
             "|---|---:|---:|---:|---:|---:|"]
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


def run(model, adaptation_update_interval=2):
    plan = build_plan(model, adaptation_update_interval)  # Preflight all nine before writes.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="SE_adLIF, DTH_SE_adLIF, MT_SE_adLIF, or configured future models")
    parser.add_argument("--adaptation-update-interval", type=int, default=2,
                        help="Fixed K for MR_SE_adLIF only (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and display plans without training or writes")
    args = parser.parse_args()
    try:
        # Preserve order and avoid scheduling duplicate model suites.
        models = list(dict.fromkeys(args.models))
        plans = [(model, build_plan(model, args.adaptation_update_interval)) for model in models]
        if args.dry_run:
            for model, plan in plans:
                for item in plan:
                    print(f"{model} {item['dataset']} seed {item['seed']}: {item['status']} -> {item['directory']}")
            print(f"Validated {sum(len(plan) for _, plan in plans)} runs; no training or writes.")
        else:
            for model in models:
                run(model, args.adaptation_update_interval)
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

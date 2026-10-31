"""SSC-only VS-MT experiment using the official runner lifecycle."""
import argparse
from contextlib import contextmanager, redirect_stdout
import copy
import csv
import io
import json
import math
import statistics

import runner
from project_paths import local_path, portable_config, portable_path

MODEL = 'VS_MT_SE_adLIF'
OUTPUT = 'results/experiments/vs_mt_ssc_v1'


@contextmanager
def output_root(value):
    previous = runner.RUN_LOGDIR
    runner.RUN_LOGDIR = value
    try:
        yield
    finally:
        runner.RUN_LOGDIR = previous


def plans(seeds, resume=1):
    with output_root(OUTPUT):
        variant = runner.build_plan(MODEL, seeds=seeds, datasets=['SSC'], resume=resume)
    with output_root('results/valid'):
        # MT is read-only comparison evidence, never a training/resume target.
        # Git transfers its records but not ignored checkpoints/completion flags.
        baseline = runner.build_plan('MT_SE_adLIF', seeds=seeds, datasets=['SSC'], inspect_existing=False)
    for new, old in zip(variant, baseline):
        a, b = copy.deepcopy(new['config']), copy.deepcopy(old['config'])
        for cfg in (a, b):
            for key in ('exp_name', 'logdir', 'launcher_logdir', 'model_variant', 'paper_baseline_deviations'):
                cfg.pop(key, None)
            for layer in ('l1', 'l2'):
                cfg[layer]['cell'] = 'mt_se_adlif'
        if a != b:
            raise ValueError('VS-MT and existing MT have incompatible effective configurations')
        read_baseline(old)
    return variant, baseline


def read_baseline(item):
    """Use compatible verified MT records even without transferred weights."""
    directory = item['directory']
    item['evidence_only'] = True
    item['status'] = 'unavailable'
    manifest = directory / 'FULL_RUN_CONFIG.json'
    result = directory / 'RESULT.json'
    if not manifest.is_file() or not result.is_file():
        item['evidence_note'] = 'MT resolved config or verified RESULT.json is missing; baseline training is never scheduled.'
        return
    previous = portable_config(json.loads(manifest.read_text()), runner.ROOT)
    if previous != item['config']:
        item['status'] = 'incompatible'
        item['evidence_note'] = 'MT saved config differs from the effective comparison config; excluded.'
        return
    record = json.loads(result.read_text())
    if (record.get('seed') != item['seed'] or record.get('dataset') != 'SSC'
            or record.get('comparison_protocol') != 'ssc_mean_loss_v1'
            or record.get('paper_baseline_version') != item['config']['paper_baseline_version']
            or record.get('paper_baseline_dataset') != 'SSC'):
        raise ValueError(f'MT result identity/protocol mismatch: {result}')
    score = record.get('test_acc')
    digest = record.get('checkpoint_sha256', '')
    if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not 0 <= score <= 1 or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError(f'Invalid verified MT test/checkpoint record: {result}')
    checkpoint = local_path(record['checkpoint'], root=runner.ROOT)
    if not checkpoint.is_relative_to(directory.resolve() / 'ckpt'):
        raise ValueError(f'MT result checkpoint is outside its seed directory: {result}')
    metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
    if metrics.is_file():
        measured = test_accuracy(dict(item, status='complete', evidence_only=False))
        if abs(measured - 100 * score) > 1e-5:
            raise ValueError(f'MT RESULT.json and metrics disagree: {directory}')
    test_result = directory / 'TEST_RESULTS.json'
    if test_result.is_file() and abs(json.loads(test_result.read_text())['metrics']['test_acc'] - score) > 1e-7:
        raise ValueError(f'MT RESULT.json and test JSON disagree: {directory}')
    item.update(status='complete', test_accuracy_percent=100 * score,
                evidence_checkpoint=checkpoint, evidence_sha256=digest,
                evidence_note=('Verified RESULT.json and resolved config; CSV cross-checked where locally available. '
                               'Baseline weights are optional for reporting, never used to initialize VS-MT.'))


def test_accuracy(item):
    if item['status'] != 'complete':
        return None
    if item.get('evidence_only'):
        return item['test_accuracy_percent']
    directory = item['directory']
    with (directory / 'logs/mlp_snn/version_0/metrics.csv').open(newline='') as stream:
        values = [float(row['test_acc']) for row in csv.DictReader(stream) if row.get('test_acc')]
    if len(values) != 1 or not math.isfinite(values[0]):
        raise ValueError(f'Expected one finite final-test score: {directory}')
    result = directory / 'TEST_RESULTS.json'
    if result.exists():
        recorded = json.loads(result.read_text())['metrics']['test_acc']
        if abs(recorded - values[0]) > 1e-7:
            raise ValueError(f'Test JSON and metrics disagree: {directory}')
    return values[0] * 100


def parameter_record(model):
    """Aligned per-neuron arrays from the validation-selected checkpoint."""
    result = {}
    for name in ('l1', 'l2'):
        cell = getattr(model, name)
        mix = cell.mix_logits.sigmoid()
        a_slow = getattr(cell, 'a_slow', cell.a)
        tensors = dict(a_fast=cell.a, a_slow=a_slow, b_shared=cell.b, mix_fast=mix,
                       mix_slow=1-mix, weighted_a_fast=mix * cell.a,
                       weighted_a_slow=(1-mix) * a_slow,
                       tau_u=cell.tau_u_trainer.get_tau(),
                       tau_fast=cell.tau_w_trainer.get_tau(),
                       tau_slow=cell.tau_w_slow_trainer.get_tau())
        result[name] = dict(q=cell.q, dt=cell.dt,
                            **{k: v.detach().cpu().tolist() for k, v in tensors.items()})
    return result


def write_comparison(seeds):
    from omegaconf import OmegaConf
    import torch
    from models.pl_module import MLPSNN
    from measure_shd_costs import selected_checkpoint, sha256
    variant, baseline = plans(seeds)
    rows = []
    for label, items in [('MT', baseline), ('VS-MT', variant)]:
        for item in items:
            directory = item['directory']
            score = test_accuracy(item)
            row = dict(model=label, seed=item['seed'], directory=portable_path(directory),
                       status=item['status'], test_accuracy_percent=score, costs=None)
            if item.get('evidence_note'):
                row['evidence_note'] = item['evidence_note']
            if score is not None:
                checkpoint_path = item.get('evidence_checkpoint') or selected_checkpoint(directory)
                digest = item.get('evidence_sha256')
                if checkpoint_path.is_file():
                    actual_digest = sha256(checkpoint_path)
                    if digest is not None and actual_digest != digest:
                        raise ValueError(f'MT checkpoint hash differs from its result: {checkpoint_path}')
                    digest = actual_digest
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    cfg = OmegaConf.create(portable_config(OmegaConf.to_container(
                        OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True)))
                    expected = item['config']
                    saved = portable_config(OmegaConf.to_container(cfg, resolve=True))
                    for key in ('random_seed', 'l1', 'l2', 'dataset', 'loss_agg', 'lr', 'batch_size', 'n_epochs'):
                        if saved.get(key) != expected.get(key):
                            raise ValueError(f'Checkpoint config differs for {key}: {checkpoint_path}')
                    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
                        model = MLPSNN(cfg).eval()
                    model.load_state_dict(checkpoint['state_dict'], strict=True)
                    record = dict(checkpoint=portable_path(checkpoint_path), checkpoint_sha256=digest,
                                  seed=item['seed'], layers=parameter_record(model))
                    if label == 'VS-MT':
                        (directory / 'NEURON_PARAMETERS.json').write_text(json.dumps(record, indent=2) + '\n')
                    row['parameters'] = record
                elif label == 'MT':
                    row['parameters_note'] = 'MT checkpoint not local; parameter arrays unavailable. No retraining needed.'
                else:
                    raise FileNotFoundError(checkpoint_path)
                cost_path = directory / 'COST_RESULTS.json'
                if cost_path.exists():
                    from cost_records import normalized_record
                    cost = normalized_record(json.loads(cost_path.read_text()))
                    if (cost.get('status') != 'complete' or cost.get('checkpoint_sha256') != digest
                            or cost.get('dataset') != 'SSC' or cost.get('seed') != item['seed']):
                        raise ValueError(f'Cost identity mismatch: {cost_path}')
                    if abs(cost['measurements']['reproduced_test_acc'] * 100 - score) > 1e-4:
                        raise ValueError(f'Cost test accuracy differs from the saved test result: {cost_path}')
                    row['costs'] = cost['measurements']
                error_path = directory / 'COST_ERROR.json'
                if row['costs'] is None and error_path.exists():
                    row['cost_error'] = json.loads(error_path.read_text()).get('error')
            rows.append(row)
    def metric(values):
        if not values:
            return 'pending'
        return (f'{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}' if len(values) > 1
                else f'{values[0]:.2f} (n=1; SD unavailable)')
    lines = ['# VS-MT versus MT on SSC', '',
             'Ordinary inference without TTA; validation-selected final-test checkpoints.',
             'Matched baseline configuration; the intended deviation is one independent slow voltage coefficient per hidden neuron.',
             'Previous test results have been observed. This is an experimental hypothesis; no guaranteed improvement or novelty claim.', '',
             '| Model | Seed | Test accuracy (%) | Forward ms/sample | Hidden spikes/sample | Arithmetic GFLOPs/sample |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        costs = row['costs'] or {}
        score = f"{row['test_accuracy_percent']:.2f}" if row['test_accuracy_percent'] is not None else 'pending'
        values = [str(costs.get(k, 'unavailable')) for k in ('forward_ms_per_sample', 'spikes_total_per_sample',
                                                          'arithmetic_gflops_test_padded')]
        lines.append(f"| {row['model']} | {row['seed']} | {score} | " + ' | '.join(values) + ' |')
    lines += ['', '| Model | Completed n | Test mean +/- sample SD (%) |', '|---|---:|---:|']
    summaries = {}
    for label in ('MT', 'VS-MT'):
        values = [r['test_accuracy_percent'] for r in rows if r['model'] == label and r['test_accuracy_percent'] is not None]
        lines.append(f'| {label} | {len(values)} | {metric(values)} |')
        summaries[label] = dict(n=len(values), mean_percent=statistics.mean(values) if values else None,
                                sample_sd_pp=statistics.stdev(values) if len(values) > 1 else None)
    diffs = [test_accuracy(a) - test_accuracy(b) for a, b in zip(variant, baseline)
             if a['status'] == b['status'] == 'complete']
    lines += ['', f'Paired VS-MT minus MT (pp), n={len(diffs)}: {metric(diffs)}.', '',
              'Costs are available only where recorded; unavailable/failed costs never imply failed training or require retraining.',
              'MT uses compatible verified RESULT.json records and cross-checks CSV metrics where available. Missing MT weights only prevent parameter export; they never schedule baseline training or block new VS-MT runs.',
              'Timing was collected in different sessions and may use different environments; it is descriptive, not a controlled speedup or energy claim.',
              'Per-neuron voltage sensitivities, shared spike coefficient, mixture weights, weighted voltage strengths, q and time constants are retained in COMPARISON.json; VS-MT also saves NEURON_PARAMETERS.json beside each completed seed.']
    root = local_path(OUTPUT)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'COMPARISON.json').write_text(json.dumps(dict(dataset='SSC', seeds=list(seeds),
        paper_baseline_version=variant[0]['config']['paper_baseline_version'], protocol='ssc_mean_loss_v1',
        rows=rows, summaries=summaries, paired_differences_pp=diffs), indent=2) + '\n')
    (root / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['SSC'], default=['SSC'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=runner.SEEDS, default=list(runner.SEEDS))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-only', action='store_true', help='Read saved results and parameters; no training or test inference')
    parser.add_argument('--resume', type=int, choices=[0, 1], default=1)
    costs = parser.add_mutually_exclusive_group()
    costs.add_argument('--no-measure-costs', action='store_true')
    costs.add_argument('--measure-existing-costs', action='store_true')
    args = parser.parse_args(argv)
    seeds = list(dict.fromkeys(args.seeds))
    if args.report_only and (args.no_measure_costs or args.measure_existing_costs):
        parser.error('--report-only does not perform cost measurement')
    variant, _ = plans(seeds, args.resume)
    if not args.report_only:
        data_root = local_path(variant[0]['config']['dataset']['data_path']) / 'SSCWrapper'
        for split in ('train', 'valid', 'test'):
            if not (data_root / f'ssc_{split}.h5').is_file():
                raise FileNotFoundError(f'Missing local SSC dataset: {data_root / f"ssc_{split}.h5"}')
    commands = ['--models', MODEL, '--datasets', 'SSC', '--seeds', *map(str, seeds),
                '--logdir', OUTPUT, '--resume', str(args.resume)]
    if args.dry_run:
        runner.main(commands + ['--dry-run'])
        return
    from extended_seed_study import study_lock
    with study_lock(local_path(OUTPUT)):
        plans(seeds, args.resume)
        if not args.report_only:
            if args.no_measure_costs:
                commands.append('--no-measure-costs')
            if args.measure_existing_costs:
                commands.append('--measure-existing-costs')
            runner.main(commands)
        write_comparison(seeds)


if __name__ == '__main__':
    main()

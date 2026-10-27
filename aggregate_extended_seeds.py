"""Aggregate imported validation-only extension evidence, never test reports."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t

from extended_seed_study import cohort_definition, read_completed
from project_paths import local_path, portable_config


def paired_statistics(values):
    n = len(values)
    mean = statistics.mean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n) if n > 1 else None
    return dict(n=n, mean_pp=mean, sample_sd_pp=sd,
                ci95_pp=[mean - half, mean + half] if half is not None else None)


def normalized_settings(config):
    config = portable_config(copy.deepcopy(config))
    for key in ('random_seed', 'logdir', 'launcher_logdir', 'ckpt_path'):
        config.pop(key, None)
    return config


def check_frozen_config(config):
    """Recompose from versioned configs without touching imported/checkpoint files."""
    import runner
    original = runner.SEED_COHORT, runner.VALIDATION_ONLY, runner.RUN_LOGDIR
    try:
        runner.SEED_COHORT, runner.VALIDATION_ONLY = 'extension_v1', True
        runner.RUN_LOGDIR = config['launcher_logdir']
        if (Path(runner.RUN_LOGDIR).is_absolute() or ':' in runner.RUN_LOGDIR
                or '..' in Path(runner.RUN_LOGDIR).parts
                or not runner.RUN_LOGDIR.startswith('results/experiments/')):
            raise ValueError('Imported launcher root is not a portable experiment path')
        expected = runner.build_plan(config['model_variant'], seeds=(config['random_seed'],),
                                     datasets=(config['paper_baseline_dataset'],),
                                     inspect_existing=False)[0]['config']
        if portable_config(config) != expected:
            raise ValueError('Imported configuration differs from the frozen study configuration')
    finally:
        runner.SEED_COHORT, runner.VALIDATION_ONLY, runner.RUN_LOGDIR = original


def aggregate(roots, include_canonical=False):
    cohort = cohort_definition()
    seeds = cohort['ordered_seeds'] if include_canonical else cohort['additional_seeds']
    expected = {(model, assignment['dataset'], seed)
                for assignment in cohort['assignments'].values()
                for model in assignment['models'] for seed in seeds}
    found, settings, provenance = {}, {}, {}
    for root in roots:
        root = local_path(root, must_exist=True)
        for path in sorted(root.glob('*/*/seed_*')):
            if not path.is_dir():
                continue
            model, dataset = path.parent.parent.name, path.parent.name
            try:
                seed = int(path.name.removeprefix('seed_'))
            except ValueError as error:
                raise ValueError(f'Unrecognized seed directory: {path}') from error
            key = model, dataset, seed
            if key not in expected:
                raise ValueError(f'Unexpected study run (or use --include-canonical): {path}')
            if key in found:
                raise ValueError(f'Duplicate model/dataset/seed imported: {key}')
            found[key] = dict(model=model, dataset=dataset, seed=seed, status='incomplete')
            if not (path / 'validation_completed.txt').exists():
                if (path / 'completed.txt').exists():
                    raise ValueError(f'Test evidence cannot enter validation aggregation: {path}')
                continue
            record, config = read_completed(path, require_checkpoints=False)
            check_frozen_config(config)
            if (record['model'], record['dataset'], record['seed']) != key:
                raise ValueError(f'Directory identity mismatch: {path}')
            if record['seed_cohort'] != 'extension_v1' or record['seed_cohort_order'] != cohort['ordered_seeds']:
                raise ValueError(f'Cohort mismatch: {path}')
            signature = normalized_settings(config)
            group = model, dataset
            if group in settings and settings[group] != signature:
                raise ValueError(f'Incompatible effective configurations: {path}')
            settings[group] = signature
            environments = json.loads((path / 'LAUNCH_ENVIRONMENTS.json').read_text(encoding='utf-8'))
            if not environments or any(not e.get('git_commit') for e in environments):
                raise ValueError(f'Missing source/environment provenance: {path}')
            identity = {(e['git_commit'], e['tracked_diff_sha256']) for e in environments}
            if len(identity) != 1 or (dataset in provenance and provenance[dataset] != identity):
                raise ValueError(f'Mixed source revisions for paired study: {path}')
            provenance[dataset] = identity
            found[key].update(record)
    rows = [found.get(key, dict(model=key[0], dataset=key[1], seed=key[2], status='missing'))
            for key in sorted(expected)]
    summaries, comparisons = [], []
    for assignment in cohort['assignments'].values():
        dataset = assignment['dataset']
        models = assignment['models']
        for model in models:
            values = [100 * r['best_validation_accuracy'] for r in rows
                      if r['model'] == model and r['dataset'] == dataset and r['status'] == 'complete']
            summaries.append(dict(dataset=dataset, model=model, n=len(values),
                                  mean_validation_percent=statistics.mean(values) if values else None,
                                  sample_sd_percent=statistics.stdev(values) if len(values) > 1 else None))
        positive, negative = ('SA_MT_SE_adLIF', 'MT_SE_adLIF') if dataset == 'ECG' else ('MT_SE_adLIF', 'DTH_SE_adLIF')
        pairs = []
        for seed in seeds:
            a, b = found.get((positive, dataset, seed), {}), found.get((negative, dataset, seed), {})
            if a.get('status') == b.get('status') == 'complete':
                pairs.append(dict(seed=seed, difference_pp=100 * (a['best_validation_accuracy'] - b['best_validation_accuracy'])))
        comparisons.append(dict(dataset=dataset, difference=f'{positive} minus {negative}', pairs=pairs,
                                **paired_statistics([p['difference_pp'] for p in pairs])))
    return dict(metric='validation_accuracy', cohort='extension_v1', expected_seeds=seeds,
                runs=rows, summaries=summaries, paired_comparisons=comparisons)


def render(record):
    lines = ['# Extension validation-only results', '',
             'Accuracy is validation accuracy only. SHD is reported as SHD*. Epochs are zero based.', '',
             '95% paired Student t interval: mean(d) +/- t(0.975, n-1) * sample_SD(d) / sqrt(n), in percentage points. '
             'Assumes independent training seeds and approximately normal paired differences. '
             'Shared fixed splits and small n limit generalization; intervals are descriptive, unadjusted for multiple comparisons. '
             'n < 2 has no interval or sample SD. No test accuracy or ten-seed test mean is computed.', '',
             '| Dataset | Model | n | Mean validation (%) | Sample SD (%) |', '|---|---|---:|---:|---:|']
    for s in record['summaries']:
        lines.append(f"| {'SHD*' if s['dataset'] == 'SHD' else s['dataset']} | {s['model']} | {s['n']} | {s['mean_validation_percent']} | {s['sample_sd_percent']} |")
    for c in record['paired_comparisons']:
        lines += ['', f"## {'SHD*' if c['dataset'] == 'SHD' else c['dataset']}: {c['difference']}", '',
                  f"n={c['n']}; mean difference={c['mean_pp']} pp; 95% CI={c['ci95_pp']} pp.", '',
                  '| Seed | Paired difference (pp) |', '|---:|---:|']
        lines += [f"| {p['seed']} | {p['difference_pp']:+.6f} |" for p in c['pairs']]
    lines += ['', '## Completion and selected epochs', '',
              '| Dataset | Model | Seed | Status | Best validation (%) | Selected epoch |', '|---|---|---:|---|---:|---:|']
    for r in record['runs']:
        accuracy = 100 * r['best_validation_accuracy'] if r['status'] == 'complete' else 'pending'
        lines.append(f"| {'SHD*' if r['dataset'] == 'SHD' else r['dataset']} | {r['model']} | {r['seed']} | {r['status']} | {accuracy} | {r.get('selected_epoch', 'pending')} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True, help='PC output roots, imported with model/dataset/seed layout')
    parser.add_argument('--output', required=True, help='New aggregate directory; existing outputs are never overwritten')
    parser.add_argument('--include-canonical', action='store_true', help='Require validation-only evidence also for seeds 42, 123, 456')
    args = parser.parse_args(argv)
    output = local_path(args.output)
    if output.exists():
        parser.error('Choose a new aggregate directory; output already exists')
    record = aggregate(args.roots, args.include_canonical)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'VALIDATION_AGGREGATE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    (output / 'VALIDATION_AGGREGATE.md').write_text(render(record), encoding='utf-8')


if __name__ == '__main__':
    main()

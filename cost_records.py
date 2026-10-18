"""Canonical per-seed measurement paths and read-only legacy schema support."""
import copy
import json
from pathlib import Path

ARITHMETIC = {'add', 'sub', 'rsub', 'mul', 'div', 'neg', 'reciprocal',
              'sum', 'mean', 'bias_add', 'square'}
VERSIONS = {'runner_test_costs_v1', 'shd_costs_v1', 'reference_test_costs_v1'}


def seed_directory(root, model, dataset, seed):
    root = Path(root)
    if model == 'SE_adLIF':
        return root / 'results/valid/reference' / dataset / f'seed_{seed}'
    path = root / 'results/valid' / model
    if model == 'MR_SE_adLIF':
        path /= 'K_2'
    directory = path / dataset / f'seed_{seed}'
    retained = root / 'results/not_improved' / directory.relative_to(root / 'results/valid')
    return retained if retained.exists() and not directory.exists() else directory


def record_path(root, model, dataset, seed, output=None):
    if output is not None:
        # Explicit custom benchmark outputs retain their existing layout.
        group = model if dataset == 'SHD' else dataset
        return Path(output) / group / f'seed_{seed}.json'
    return seed_directory(root, model, dataset, seed) / 'COST_RESULTS.json'


def load_measurement(root, model, dataset, seed, output=None):
    return json.loads(record_path(root, model, dataset, seed, output).read_text(encoding='utf-8'))


def normalized_record(record):
    """Adapt historical records in memory; never rewrite their measured values.

    Older SHD records did not store arithmetic/logical totals. Derive those
    columns from their retained scalar-operation breakdown using the same
    convention as the later reference/runner evaluators.
    """
    r = copy.deepcopy(record)
    if r.get('version') not in VERSIONS:
        raise ValueError('Unknown measurement record version')
    if r['version'] == 'shd_costs_v1':
        r.setdefault('dataset', 'SHD')
    if r['version'] != 'runner_test_costs_v1':
        if not isinstance(r.get('measurements', {}).get('reproduced_test_acc'), (int, float)):
            raise ValueError('Historical measurement lacks its test-accuracy check')
        r.setdefault('status', 'complete')
    metrics = r['measurements']
    if 'arithmetic_gflops_test_padded' not in metrics:
        n = metrics['totals']['samples']
        if n <= 0:
            raise ValueError('Measurement has no evaluated samples')
        arithmetic = sum(v for k, v in metrics['nonmatrix_scalar_operations_by_kind'].items() if k in ARITHMETIC)
        metrics['arithmetic_gflops_test_padded'] = metrics['dense_gflops_test_padded'] + arithmetic / n / 1e9
    if 'logical_gops_test_padded' not in metrics:
        metrics['logical_gops_test_padded'] = metrics['dense_gflops_test_padded'] + metrics['nonmatrix_ops_per_sample'] / 1e9
    return r

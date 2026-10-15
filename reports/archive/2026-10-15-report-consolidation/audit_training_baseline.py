"""Read saved classification records; write a snapshot audit without loading weights."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'config/paper_baseline.yaml').exists())
BASELINE = OmegaConf.to_container(OmegaConf.load(ROOT / 'config/paper_baseline.yaml'), resolve=True)['paper_baseline']


def get(cfg, key, default=None):
    value = cfg
    for part in key.split('.'):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def load(path):
    if path.suffix == '.json':
        obj = json.loads(path.read_text(encoding='utf-8-sig'))
        return obj.get('config', obj), obj
    obj = OmegaConf.load(path)
    cfg = obj.get('cfg', obj)
    return OmegaConf.to_container(cfg, resolve=True), {}


def expectations(dataset):
    common, policy, data = BASELINE['common'], BASELINE['reproduction_policy'], BASELINE['datasets'][dataset]
    expected = {k: common[k] for k in ('dropout', 'alpha', 'dt', 'gradient_clip_val', 'tracking_metric', 'tracking_mode')}
    expected.update({k: data[k] for k in ('lr', 'batch_size', 'n_epochs')})
    expected.update({k: policy[k] for k in ('early_stopping', 'scheduler', 'patience', 'factor')})
    expected.update({'dataset.batch_size': data['batch_size'], 'dataset.num_workers': policy['num_workers'],
                     'two_layers': True, 'scheduler_min_lr': 0.0, 'run_test': True,
                     'loss_agg': 'sum_softmax_over_time' if dataset == 'SHD' else 'summed_membrane_potentials'})
    for key in ('spatial_factor', 'time_factor', 'window_size', 'pad_to_min_size',
                'ignore_first_timesteps', 'validate_on', 'valid_fraction'):
        if key in data:
            expected['dataset.' + key] = data[key]
    if dataset in ('SHD', 'ECG'):
        expected['dataset.random_seed'] = policy['split_seed']
    if dataset == 'SHD':
        expected['dataset.cache_namespace'] = policy['cache_namespace']
        expected['dataset.additional_test_set_validation'] = False
    arch = data['reference_architecture']
    expected['c'] = arch['c']
    for layer in ('l1', 'l2'):
        for key in ('q', 'c', 'tau_u_range', 'tau_w_range'):
            expected[f'{layer}.{key}'] = arch[key]
        expected[f'{layer}.n_neurons'] = arch['neurons_per_layer']
        for key in ('alpha', 'dt'):
            expected[f'{layer}.{key}'] = common[key]
    expected['l_out.tau_u_range'] = [arch['output_tau']] * 2
    return expected


def audit(path):
    cfg, obj = load(path)
    dataset = str(cfg.get('exp_name', '')).split('_')[0].upper()
    if dataset not in BASELINE['datasets']:
        raise ValueError(f'Unknown dataset in {path}')
    # Locate the actual run, rather than its nested logger directory.
    directory = path.parent
    if path.name == 'hparams.yaml':
        directory = path.parents[3]
    elif path.name == 'config.yaml' and path.parent.name == '.hydra':
        directory = path.parents[1]
    relative = directory.relative_to(ROOT).as_posix()
    differences, defaults = [], {}
    effective_defaults = {'scheduler': 'plateau', 'scheduler_min_lr': 0.0,
                          'gradient_clip_val': 1.5, 'run_test': True,
                          'dataset.additional_test_set_validation': False}
    for key, expected in expectations(dataset).items():
        actual = get(cfg, key, '__missing__')
        if actual == '__missing__' and key in effective_defaults:
            actual = effective_defaults[key]
            defaults[key] = actual
        if actual != expected:
            differences.append({'key': key, 'actual': actual, 'baseline': expected})
    if cfg.get('init_ckpt_path'):
        differences.append({'key': 'initialization', 'actual': 'weights-only continuation', 'baseline': 'fresh training'})
    if obj.get('reference') and '/full_train/' not in relative:
        if not any(d['key'] == 'initialization' for d in differences):
            differences.append({'key': 'initialization', 'actual': 'reference checkpoint screening', 'baseline': 'fresh training'})
    metrics_path = directory / 'logs/mlp_snn/version_0/metrics.csv'
    max_epoch, test_values, issues = None, [], []
    if metrics_path.exists():
        last_epoch = None
        for row in csv.DictReader(metrics_path.read_text(encoding='utf-8-sig').splitlines()):
            if row.get('epoch'):
                epoch = int(float(row['epoch']))
                if last_epoch is not None and epoch < last_epoch:
                    issues.append('epoch order decreases; possible overlapping/resumed histories')
                last_epoch = epoch
                max_epoch = max(epoch, max_epoch if max_epoch is not None else epoch)
            if row.get('test_acc'):
                test_values.append(float(row['test_acc']))
    test_records = []
    for name in ('TEST_RESULTS.json', 'RESULT.json'):
        candidate = directory / name
        if candidate.exists():
            record = json.loads(candidate.read_text(encoding='utf-8-sig'))
            test_records.append({'source': name, 'record': record})
            if 'overlap' in str(record.get('note', '')).lower():
                issues.append('result record flags overlapping training logs')
    if path.name == 'REFERENCE_RESULT.json':
        test_records.append({'source': path.name, 'record': {k: v for k, v in obj.items() if k != 'config'}})
    completed_marker = (directory / 'completed.txt').exists()
    if completed_marker and not (test_values or test_records):
        issues.append('completion marker exists without local test evidence')
    summary = directory / 'TRAINING_SUMMARY.md'
    summary_text = summary.read_text(encoding='utf-8-sig') if summary.exists() else ''
    if 'overlap' in summary_text.lower() or 'mixed histor' in summary_text.lower():
        issues.append('summary mentions overlapping/mixed histories; inspect provenance')
    counterpart_conflicts = []
    check_keys = set(expectations(dataset)) | {'random_seed', 'init_ckpt_path', 'l1', 'l2', 'l_out'}
    for counterpart in (directory / '.hydra/config.yaml', directory / 'logs/mlp_snn/version_0/hparams.yaml'):
        if not counterpart.exists() or counterpart == path:
            continue
        other, _ = load(counterpart)
        for key in check_keys:
            actual = get(cfg, key, effective_defaults.get(key))
            alternate = get(other, key, effective_defaults.get(key))
            if actual != alternate:
                counterpart_conflicts.append({'source': counterpart.relative_to(ROOT).as_posix(),
                                              'key': key, 'primary': actual, 'other': alternate})
    if counterpart_conflicts:
        issues.append('saved config sources disagree; see JSON conflicts')
    status = ('completed reference record' if path.name == 'REFERENCE_RESULT.json' else
              'completed marker + test evidence' if completed_marker and (test_values or test_records) else
              'test evidence, completion not independently established' if test_values or test_records else
              'partial snapshot; no final test evidence' if max_epoch is not None else
              'config only; completion unverified')
    return {'run': relative, 'source': path.relative_to(ROOT).as_posix(), 'dataset': dataset,
            'model': cfg.get('model_variant', get(cfg, 'l1.cell')), 'seed': cfg.get('random_seed'),
            'protocol': cfg.get('evaluation_protocol'), 'scope': 'tuning' if '/experiments/' in relative else
            'historical SHD reference' if dataset == 'SHD' and path.name == 'hparams.yaml' else 'comparison',
            'matched': not differences, 'differences': differences, 'assumed_defaults': defaults,
            'status': status, 'last_metric_epoch': max_epoch, 'test_metrics': test_values,
            'test_records': test_records, 'issues': sorted(set(issues)),
            'config_conflicts': counterpart_conflicts,
            'metrics_source': metrics_path.relative_to(ROOT).as_posix() if metrics_path.exists() else None}


def main():
    primary = sorted(p for name in ('FULL_RUN_CONFIG.json', 'REFERENCE_RESULT.json', 'SWEEP_CONFIG.json')
                     for p in (ROOT / 'results').rglob(name))
    directories = {p.parent for p in primary}
    sources = primary + [p for p in (ROOT / 'results').rglob('hparams.yaml') if p.parents[3] not in directories]
    directories.update(p.parents[3] for p in sources if p.name == 'hparams.yaml')
    sources += [p for p in (ROOT / 'results').rglob('config.yaml')
                if p.parent.name == '.hydra' and p.parents[1] not in directories]
    rows, errors = [], []
    for path in sorted(sources):
        try:
            rows.append(audit(path))
        except Exception as exc:
            errors.append({'source': path.relative_to(ROOT).as_posix(), 'error': str(exc)})
    payload = {'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'baseline_version': BASELINE['version'],
               'baseline_source': 'config/paper_baseline.yaml', 'runs': rows, 'errors': errors}
    (Path(__file__).with_name('training-baseline-audit.json')).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    lines = ['# Saved training configuration audit', '', f"Baseline: `{BASELINE['version']}`; snapshot UTC: {payload['snapshot_utc']}.", '',
             'This compares saved settings with each dataset section of `config/paper_baseline.yaml`.',
             'Model-specific neuron parameters are allowed; shared layer sizes and dynamics are checked.',
             'Matching settings do not prove historical code/environment equivalence or independent test selection.',
             'Active metrics are snapshots; missing tests remain pending. Epoch numbers below are zero-based.',
             'Optimizer is Adam in the current shared implementation; historical optimizer versions/state were not loaded.',
             'Missing optional keys use the current documented defaults, listed in the JSON audit.', '',
             '| Run | Match | Status | Last epoch | Differences (actual → baseline) |', '|---|---|---|---:|---|']
    for row in rows:
        diff = '; '.join(f"{d['key']}: {d['actual']} → {d['baseline']}" for d in row['differences']) or 'none'
        issues = '; '.join(row['issues'])
        lines.append(f"| `{row['run']}` | {'yes' if row['matched'] else 'no'} | {row['status']}{'; ' + issues if issues else ''} | {row['last_metric_epoch'] if row['last_metric_epoch'] is not None else '—'} | {diff} |")
    if errors:
        lines += ['', 'Unreadable records: ' + json.dumps(errors)]
    (Path(__file__).with_name('training-baseline-audit.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'records': len(rows), 'errors': errors,
                      'by_scope_match': dict(Counter(f"{r['scope']}: {'match' if r['matched'] else 'mismatch'}" for r in rows))}, indent=2))
    for row in rows:
        if row['scope'] != 'tuning':
            print(row['run'], '|', row['status'], '|', '; '.join(f"{d['key']}={d['actual']} (baseline {d['baseline']})" for d in row['differences']) or 'MATCH', '|', row['issues'])


if __name__ == '__main__':
    main()

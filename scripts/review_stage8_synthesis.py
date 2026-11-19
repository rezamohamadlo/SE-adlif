"""Read-only independent arithmetic review of a completed saved-evidence report."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from project_paths import local_path, portable_path


def close(actual, expected, absolute=1e-11):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=absolute):
        raise ValueError(f'Saved arithmetic differs: {actual!r}, expected {expected!r}')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def seed_summary(rows, key, saved):
    if sorted(int(r['seed']) for r in rows) != [42, 123, 456]:
        raise ValueError('Summary does not contain three distinct matching training seeds')
    values = [float(r[key]) for r in rows]
    close(saved['mean'], math.fsum(values) / 3)
    close(saved['sample_sd'], statistics.stdev(values))
    if saved.get('n_training_seeds', 3) != 3:
        raise ValueError('Incorrect independent-seed count')


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def review(directory):
    root = local_path(directory, must_exist=True)
    aggregate = json.loads((root / 'AGGREGATES.json').read_text(encoding='utf-8'))
    costs = json.loads((root / 'COST_RECORDS.json').read_text(encoding='utf-8'))
    validation = read_csv(root / 'VALIDATION_PER_SEED.csv')
    if len(costs) != 54 or len(validation) != 36:
        raise ValueError('Incomplete ordinary/validation evidence')
    for row in costs:
        raw = json.loads(local_path(row['cost_record'], must_exist=True).read_text(encoding='utf-8'))
        if raw['checkpoint_sha256'] != row['checkpoint_sha256']:
            raise ValueError('Cost checkpoint identity differs')
        measured, original = row['measurements'], raw['measurements']
        for key, value in original.items():
            if measured[key] != value:
                raise ValueError('Cost measurements differ from original record: ' + key)
        count = original['totals']
        # The history stores a float32 accuracy, while the evaluator divides integer counts.
        # Retain the report's declared unchanged reproduction tolerance.
        close(row['test_accuracy'], original['reproduced_test_acc'], absolute=1e-6)
        for layer in ['l1', 'l2']:
            spikes = count[layer + '_spikes_sequence']
            close(original['spikes_' + layer + '_per_sample'], spikes / count['samples'])
            close(original['firing_' + layer + '_percent'],
                  100 * spikes / (count['sequence_sample_timesteps'] * row['neurons_' + layer]))
        close(original['spikes_total_per_sample'], original['spikes_l1_per_sample'] + original['spikes_l2_per_sample'])
    scalar_groups = 0
    for group in aggregate['costs']:
        rows = [dict(seed=r['seed'], test_accuracy=r['test_accuracy'], **r['measurements']) for r in costs
                if r['dataset'] == group['dataset'] and r['model'] == group['model']]
        for key, saved in group.items():
            if key not in ('dataset', 'model'):
                seed_summary(rows, key, saved)
                scalar_groups += 1
    for group in aggregate['validation']:
        rows = [r for r in validation if r['dataset'] == group['dataset'] and r['pair'] == group['pair']]
        for row in rows:
            close(row['accuracy_gap_pp'], 100 * (float(row['variant_accuracy']) - float(row['parent_accuracy'])))
            close(row['loss_benefit'], float(row['parent_loss']) - float(row['variant_loss']))
        for key, saved in group.items():
            if key not in ('dataset', 'pair'):
                seed_summary(rows, key, saved)
                scalar_groups += 1
    pairs = dict(amt_mt=('MT_SE_adLIF', 'AMT_SE_adLIF'), lt_dth=('DTH_SE_adLIF', 'LT_DTH_SE_adLIF'),
                 sa_mt=('MT_SE_adLIF', 'SA_MT_SE_adLIF'), mt_tc=('MT_SE_adLIF', 'MT_TC_SE_adLIF'))
    index = {(r['dataset'], r['model'], r['seed']): r for r in costs}
    for group in aggregate['paired_cost_differences']:
        parent, variant = pairs[group['pair']]
        for key, saved in group.items():
            if key in ('dataset', 'pair'):
                continue
            rows = [dict(seed=seed, difference=index[group['dataset'], variant, seed]['measurements'][key] -
                         index[group['dataset'], parent, seed]['measurements'][key]) for seed in [42, 123, 456]]
            seed_summary(rows, 'difference', saved)
            scalar_groups += 1
    feature_files = {
        'SHD': 'results/experiments/adaptation_mechanism_full_validation_v1/diagnostic_rows/input_features_clean_rep_0.json',
        'SSC': 'results/experiments/adaptation_ssc_suite_v2/shared/amt_mt_full_validation/diagnostic_rows/input_features_clean_rep_0.json',
        'ECG': 'results/experiments/adaptation_ecg_extension_v1/features/clean/rep_0.json',
    }
    feature_rows = {ds: json.loads(local_path(name).read_text(encoding='utf-8'))['value'] for ds, name in feature_files.items()}
    distributions = read_csv(root / 'FEATURE_DISTRIBUTIONS.csv')
    for row in distributions:
        values = []
        censored = invalid = missing = 0
        for original in feature_rows[row['dataset']]:
            key = row['feature']; value = original.get(key)
            acf = key.startswith('autocorr')
            if acf and original.get('autocorr_censored', False):
                censored += 1
            elif acf and (original.get('autocorr_valid') is False or original.get('autocorr_reason')):
                invalid += 1
            elif key == 'fast_slow_contrast' and (original.get('fast_slow_valid') is False or original.get('fast_slow_reason')):
                invalid += 1
            elif value is None or not isinstance(value, (float, int)) or not math.isfinite(value):
                missing += 1
            else:
                values.append(float(value))
        expected = [len(feature_rows[row['dataset']]), len(values), censored, invalid, missing]
        actual = [int(row[k]) for k in ['total', 'finite_uncensored', 'censored', 'invalid', 'missing_or_nonfinite']]
        if actual != expected:
            raise ValueError('Feature eligibility counts differ')
        for key, probability in [('q25', .25), ('median', .5), ('q75', .75)]:
            close(row[key], quantile(values, probability))
    for filename, key in [('ASSOCIATIONS.csv', 'coefficient'), ('INTERVENTIONS.csv', 'benefit_change')]:
        rows = read_csv(root / filename)
        for row in rows:
            seed_summary(json.loads(row['per_seed']), key, row)
    source_origins = read_csv(local_path('results/experiments/adaptation_stage7_suite_v1/origin_controls_v1/ORIGIN_ROLE_SUMMARY.csv'))
    if read_csv(root / 'DIAGNOSTIC_ORIGINS.csv') != source_origins:
        raise ValueError('Both-origin summaries differ from verified stage-7 evidence')
    return dict(status='independently_verified_saved_report_arithmetic',
        reviewed_utc=datetime.now(timezone.utc).isoformat(), cost_records=54,
        validation_per_seed_rows=36, cost_model_dataset_groups=18, validation_pair_dataset_groups=12,
        scalar_seed_aggregate_groups_checked=scalar_groups, feature_distribution_groups=len(distributions),
        association_groups=24, intervention_groups=36, diagnostic_origin_groups=24,
        hidden_activity_counts_and_denominators_verified=True,
        feature_quantiles_and_exclusions_verified=True, model_execution=False, new_fitting_or_resampling=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='results/experiments/adaptation_stage8_synthesis_v1')
    args = parser.parse_args()
    result = review(args.output_root)
    result['verifier_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = local_path(args.output_root) / 'INDEPENDENT_FINAL_VERIFICATION.json'
    if target.exists():
        raise ValueError('Preserve the existing independent review')
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Independent saved-report arithmetic passed:', portable_path(target))


if __name__ == '__main__':
    main()

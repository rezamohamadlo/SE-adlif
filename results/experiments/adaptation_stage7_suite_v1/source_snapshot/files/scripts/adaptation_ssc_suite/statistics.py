"""SSC-specific exploratory statistics. Execution is always user-launched."""
import contextlib
import importlib.metadata
import platform
import time

import numpy as np

from project_paths import portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_statistics.methods import make_design, within_speaker, cluster_analysis
from scripts.adaptation_statistics.analyze import summarize as summarize_existing_method
from scripts.adaptation_ssc_suite.common import (
    require, guard, verify_completion, source_features, predictions, seal,
)


def regression_cases(ids, features, feature_names):
    keep, excluded = [], []
    for index, sid in enumerate(ids):
        row = features[sid]
        reasons = []
        for name in feature_names + ['duration_ms', 'mean_rate_hz']:
            if row.get(name) is None or not np.isfinite(row[name]):
                reasons.append('undefined_' + name)
        if not row['fast_slow_valid']:
            reasons.append('invalid_fast_slow')
        if row['autocorr_censored']:
            reasons.append('right_censored_ACF_not_imputed')
        elif not row['autocorr_valid']:
            reasons.append('invalid_ACF')
        if ((row.get('duration_ms') is not None and row['duration_ms'] <= 0) or
                (row.get('mean_rate_hz') is not None and row['mean_rate_hz'] < 0)):
            reasons.append('invalid_duration_or_rate')
        if reasons:
            excluded.append(dict(sample_id=sid, speaker=row['speaker'], target=row['target'], reasons=reasons,
                                 autocorr_reason=row['autocorr_reason'], autocorr_horizon_ms=row['autocorr_horizon_ms']))
        else:
            keep.append(index)
    return keep, excluded


def load_inputs(prepared, pair, directory, manifest, source, source_manifest, verify=True):
    spec = prepared['spec']
    method = dict(spec['statistics'], training_seeds=spec['seeds'])
    if verify:
        verify_completion(directory, prepared['root'])
    ids = manifest['sample_ids']
    require(ids == list(range(9981)) and manifest['seeds'] == spec['seeds'] and
            manifest['comparison'] == pair and manifest['comparison_loss'] == spec['loss_agg'],
            'Statistics requires the frozen full SSC supervised comparison')
    clean_condition = next(c for c in spec['conditions'] if c[0] == 'clean')
    features = {r['sample_id']: r for r in source_features(source, source_manifest, clean_condition)}
    benefit = {}
    for condition in spec['conditions']:
        name, rep, _, _ = condition
        transformed = {r['sample_id']: r for r in source_features(source, source_manifest, condition)}
        outcomes = []
        for seed in spec['seeds']:
            both = [predictions(directory, manifest, model, seed, condition, source, source_manifest)
                    for model in (pair['parent'], pair['variant'])]
            for rows in both:
                require(all(all(r[k] == value for k, value in transformed[r['sample_id']].items()) for r in rows),
                        'Statistics input predictions do not match shared feature metadata')
            pairs = pilot.paired_rows(*both)
            require([r['sample_id'] for r in pairs] == ids, 'Statistics pairing/order differs')
            require(all(r['speaker'] == features[r['sample_id']]['speaker'] and
                        r['target'] == features[r['sample_id']]['target'] for r in pairs), 'Class/speaker metadata differs')
            outcomes.append([r['benefit'] for r in pairs])
            del both, pairs
        benefit[name, rep] = np.asarray(outcomes, dtype=np.float64).T
    clean = benefit['clean', 0]
    changes = []
    for name in method['secondary_conditions']:
        reps = method['deletion_replicates'] if name.startswith('delete_') else [0]
        changes.append(np.mean([benefit[name, rep] for rep in reps], axis=0) - clean)
    effects = np.stack(changes, axis=1)
    require(np.isfinite(clean).all() and np.isfinite(effects).all(), 'Nonfinite analysis outcome')
    keep, excluded = regression_cases(ids, features, method['features'])
    require(bool(keep), 'No complete cases')
    speaker_ids = sorted({str(features[sid]['speaker']) for sid in ids})
    require(len(speaker_ids) == spec['expected_speaker_groups'], 'SSC speaker cohort differs')
    speaker_codes = {speaker: index for index, speaker in enumerate(speaker_ids)}
    all_speakers = np.asarray([speaker_codes[str(features[sid]['speaker'])] for sid in ids], dtype=np.int64)
    speakers = all_speakers[keep]
    require(len(np.unique(speakers)) >= method['bootstrap']['minimum_speakers'], 'Too few complete-case speakers')
    rows = [features[ids[i]] for i in keep]
    classes = np.asarray([r['target'] for r in rows])
    require(set(np.unique(classes).tolist()) == set(range(35)), 'Complete cases do not cover all 35 SSC classes')
    design, scaling = make_design(np.asarray([[r[f] for f in method['features']] for r in rows]),
        np.asarray([r['duration_ms'] for r in rows]), np.asarray([r['mean_rate_hz'] for r in rows]), classes, speakers)
    # Design-only eligibility check; no coefficients or fitted scientific result.
    eigenvalues = np.linalg.eigvalsh(design.T @ design)
    require(eigenvalues[-1] > 0 and eigenvalues[0] > method['bootstrap']['rank_relative_tolerance'] * eigenvalues[-1],
            'Complete-case design is rank deficient or ill conditioned; preserve evidence and inspect')
    return dict(ids=ids, features=features, clean=clean, effects=effects, keep=keep, excluded=excluded,
        design=design, centered_outcome=within_speaker(clean[keep], speakers), scaling=scaling,
        speakers=speakers, all_speakers=all_speakers, speaker_ids=speaker_ids, method=method)


def make_manifest(prepared, pair, directory, evaluation_manifest, inputs):
    return dict(version=prepared['spec']['statistics']['version'], dataset='SSC official validation',
        comparison=pair, specification=inputs['method'], source_evaluation=portable_path(directory),
        source_manifest_sha256=pilot.digest(evaluation_manifest),
        source_completion_sha256=pilot.sha256(directory / 'COMPLETE.json'), sample_ids=inputs['ids'],
        complete_case_ids=[inputs['ids'][i] for i in inputs['keep']], excluded=inputs['excluded'],
        speaker_code_to_original_id=inputs['speaker_ids'], scaling=inputs['scaling'],
        complete_case_speaker_count=len(np.unique(inputs['speakers'])),
        source_sha256=prepared['manifest']['source_sha256'], source_hash_encoding='CRLF_normalized_to_LF',
        comparison_loss='supervised_mean_membrane_only_no_auxiliary_JS',
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         matplotlib=importlib.metadata.version('matplotlib')),
        paper_baseline_version=prepared['spec']['paper_baseline_version'], paper_baseline_dataset='SSC',
        training=False, new_inference=False, official_test_opened=False,
        reuse_interpretation='Shared MT outputs are the same observations across comparisons; intervals are not independent replications.')


def summarize(result, prepared, pair, inputs):
    summary = summarize_existing_method(result, inputs['method'], inputs)
    summary['n_complete_case_speakers'] = len(np.unique(inputs['speakers']))
    summary['comparison'] = pair
    summary['dataset'] = 'SSC official validation'
    summary['comparison_loss'] = 'supervised_mean_membrane_only_no_auxiliary_JS'
    summary['interval_families'] = dict(primary_features_across_four_pairs=8, secondary_conditions_across_four_pairs=12,
        alpha=.05, note='Separate approximate Bonferroni families; no combined primary/secondary guarantee or formal significance claim.')
    for record in summary['leave_one_speaker_out']:
        record['original_speaker_id'] = inputs['speaker_ids'][record['omitted_speaker']]
    summary['limitations'] = [
        'Exploratory SSC validation analysis defined after observing AMT descriptive results; not preregistration or independent confirmation.',
        'Intervals condition on three observed trained model pairs; bootstrap speakers and repeated predictions are not new training seeds.',
        'Speaker-pairs percentile intervals are approximate; 2000 draws provide limited precision for multiplicity-adjusted tail quantiles.',
        'Eight feature/pair coefficients and twelve intervention/pair effects form separate Bonferroni families; no combined guarantee.',
        'Per-seed intervals and leave-one-speaker-out estimates are diagnostics, not an additional simultaneous testing family.',
        'Class and speaker fixed effects plus linear log-duration/activity adjustment do not remove all difficulty confounding.',
        'Joint regression excludes explicitly recorded invalid/censored features without imputation; intervention changes retain all recordings.',
        'Shared MT predictions across comparisons are identical observations and must not be counted as independent replication.',
        'MT-TC uses a distinct training objective; evaluation benefit excludes its auxiliary JS. Input features do not directly measure output-window stability.',
        'LT-DTH trace descriptors are model/layer constants, not independent per-record measurements; no trace-parameter mediation regression.',
        'No official-test access, new neural inference, training or causal confirmation in this statistical job.']
    return summary


def write_outputs(destination, summary, result, inputs, pair):
    raw = destination / 'diagnostic_rows'
    raw.mkdir(exist_ok=True)
    pilot.write_json(destination / 'SUMMARY.json', summary)
    pilot.write_json(destination / 'EXCLUSIONS.json', inputs['excluded'])
    rows = []
    for index, sid in enumerate(inputs['ids']):
        f = inputs['features'][sid]
        row = dict(sample_id=sid, speaker=f['speaker'], target=f['target'], duration_ms=f['duration_ms'],
                   mean_rate_hz=f['mean_rate_hz'], **{name: f[name] for name in inputs['method']['features']})
        for k, seed in enumerate(inputs['method']['training_seeds']):
            row[f'clean_benefit_seed_{seed}'] = float(inputs['clean'][index, k])
            for j, name in enumerate(inputs['method']['secondary_conditions']):
                row[f'{name}_benefit_change_seed_{seed}'] = float(inputs['effects'][index, j, k])
        rows.append(row)
    pilot.write_csv(raw / 'sample_analysis.csv', rows)
    np.savez_compressed(raw / 'bootstrap_estimates.npz', primary=result['bootstrap'], secondary=result['effect_bootstrap'],
                        speaker_weights=result['cluster_weights'])
    lines = [f"# SSC exploratory statistics: {pair['variant']} versus {pair['parent']}", '',
        f"Records: {summary['n_samples']}; joint-regression complete cases: {summary['n_complete_cases']}; "
        f"speakers: {summary['n_speakers']}; fixed training seeds: 3.", '',
        f"Outcome = loss({pair['parent']}) - loss({pair['variant']}); positive favors the variant. "
        'Both standardized clean-input features enter jointly, controlling class, speaker, log1p(duration) and log1p(rate). '
        'Speaker effects use within-speaker demeaning. Scales remain fixed during resampling.', '',
        f"Hypothesis: `{pair['hypothesis']}`. Inspect per-seed signs alongside every interval. "
        'Negative and inconclusive outcomes are retained. These exploratory intervals are conditional on the existing trained models; '
        'they do not establish formal significance, causal effects or robustness across new training seeds.', '',
        '## Adjusted clean-input associations', '',
        '| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 99.375% interval | Conditional direction |',
        '|---|---:|---:|---:|---:|---:|---|']
    for record in summary['primary']:
        values = ' | '.join(f"{x['coefficient']:+.6f}" for x in record['per_seed'])
        lines.append(f"| {record['feature']} | {values} | {record['mean']:+.6f} +/- {record['sample_sd']:.6f} | "
                     f"[{record['interval'][0]:+.6f}, {record['interval'][1]:+.6f}] | {record['direction']} |")
    lines += ['', 'Primary intervals use the eight-coefficient family across four pairs. '
              'Per-seed intervals in JSON are diagnostic only. SD is variation across the three training seeds, not a confidence interval.', '',
              '## Changes from clean input', '',
              'Deletion realizations are averaged within recording and seed first. All 9,981 records enter these effects, '
              'including the two censored ACF records. Approximate 99.5833% individual intervals use the separate twelve-effect family.', '',
              '| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |',
              '|---|---:|---:|---:|---:|---:|']
    for record in summary['secondary']:
        values = ' | '.join(f"{x['benefit_change']:+.6f}" for x in record['per_seed'])
        lines.append(f"| {record['condition']} | {values} | {record['mean']:+.6f} +/- {record['sample_sd']:.6f} | "
                     f"[{record['interval'][0]:+.6f}, {record['interval'][1]:+.6f}] |")
    lines += ['', 'Positive change means the variant improves its relative loss gap against clean input; '
              'absolute loss/accuracy need not improve.', '', '## Limits', '']
    lines += ['- ' + text for text in summary['limitations']]
    lines += ['', 'Original speaker strings and code mappings are retained in the manifest. '
              'Exclusions describe regression eligibility only; no diagnostic predictions were silently removed.', '']
    (destination / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, groups, title in zip(axes, [summary['primary'], summary['secondary']],
                                   ['Adjusted associations per feature SD', 'Change in benefit from clean']):
        for position, record in enumerate(groups):
            axis.plot(record['interval'], [position, position], color='tab:blue')
            axis.scatter([record['mean']], [position], color='tab:blue', marker='D')
            values = [r.get('coefficient', r.get('benefit_change')) for r in record['per_seed']]
            axis.scatter(values, [position]*3, color='grey', marker='x', alpha=.6)
        axis.set_yticks(range(len(groups)), [r.get('feature', r.get('condition')) for r in groups])
        axis.axvline(0, color='black', linewidth=.8)
        axis.set_title(title)
        axis.set_xlabel('Supervised loss-benefit units')
        axis.grid(alpha=.2)
    figure.suptitle(f"SSC {pair['id']}: {summary['n_speakers']} speaker clusters; 3 fixed training seeds")
    figure.tight_layout()
    try:
        for extension in ('png', 'svg'):
            figure.savefig(destination / ('adjusted_effects.' + extension), dpi=160)
    finally:
        plt.close(figure)
    return [destination/'SUMMARY.json', destination/'SUMMARY.md', destination/'EXCLUSIONS.json',
            raw/'sample_analysis.csv', raw/'bootstrap_estimates.npz', destination/'adjusted_effects.png', destination/'adjusted_effects.svg']


def execute(prepared, pair, directory, evaluation_manifest, source, source_manifest):
    destination = prepared['root'] / 'pairs' / pair['id'] / 'statistics_v1'
    inputs = load_inputs(prepared, pair, directory, evaluation_manifest, source, source_manifest)
    manifest = make_manifest(prepared, pair, directory, evaluation_manifest, inputs)
    guard(destination, manifest)
    with pilot.study_lock(destination / '.analysis.lock'):
        guard(destination, manifest)
        if (destination / 'COMPLETE.json').exists():
            verify_completion(destination, prepared['root'])
            print(pair['id'] + ': complete statistics verified; no fitting/resampling repeated.', flush=True)
            return destination
        pilot.write_json(destination / 'MANIFEST.json', manifest)
        started = time.monotonic()
        pilot.write_json(destination / 'PROGRESS.json', dict(status='running', phase='speaker_cluster_statistics',
            bootstrap_draws=inputs['method']['bootstrap']['draws'], samples=len(inputs['ids']), complete_cases=len(inputs['keep'])))
        print(f"{pair['id']}: SSC statistics, {len(inputs['keep'])} complete cases, {len(inputs['speaker_ids'])} speakers, "
              f"{inputs['method']['bootstrap']['draws']} speaker resamples.", flush=True)
        try:
            from threadpoolctl import threadpool_limits
            pool = threadpool_limits(limits=1, user_api='blas')
        except ImportError:
            pool = contextlib.nullcontext()
        with pool:
            result = cluster_analysis(inputs['design'], inputs['centered_outcome'], inputs['speakers'],
                                      inputs['effects'], inputs['all_speakers'], inputs['method']['bootstrap'])
        summary = summarize(result, prepared, pair, inputs)
        outputs = write_outputs(destination, summary, result, inputs, pair)
        seal(destination, manifest, outputs, elapsed_seconds=time.monotonic()-started,
             new_inference=False, bootstrap_valid=summary['bootstrap_valid'], bootstrap_rejected=summary['bootstrap_rejected'])
        pilot.write_json(destination / 'PROGRESS.json', dict(status='complete', elapsed_seconds=time.monotonic()-started))
    return destination

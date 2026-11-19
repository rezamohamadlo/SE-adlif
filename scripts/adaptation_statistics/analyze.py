"""Exploratory Stage 5 statistics from authenticated saved validation outputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from project_paths import local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory, verify_completion, verify_package
from scripts.adaptation_statistics.methods import make_design, within_speaker, cluster_analysis, interval, direction

SPEC = 'config/studies/adaptation_statistics_v1.yaml'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_inputs(spec):
    require(spec['features'] == ['fast_slow_contrast','autocorr_tau_ms'], 'Unsupported feature specification')
    require(spec['controls'] == ['class_fixed_effects','speaker_fixed_effects','log1p_duration_ms','log1p_mean_rate_hz'],
            'Unsupported control specification')
    require(spec['bootstrap']['unit'] == 'speaker' and not spec['training'] and not spec['new_inference'] and
            not spec['official_test_access'], 'Unsupported execution specification')
    require(spec['uncertainty']['primary_family_size'] == len(spec['features']) and
            spec['uncertainty']['secondary_family_size'] == len(spec['secondary_conditions']), 'Multiplicity family differs')
    source = experiment_directory(spec['source'])
    verify_package(source)
    manifest = pilot.read_json(source / 'MANIFEST.json')
    require(pilot.digest(manifest) == spec['source_manifest_sha256'], 'Source manifest differs from specification')
    require(pilot.sha256(source / 'COMPLETE.json') == spec['source_completion_sha256'], 'Source completion differs')
    require(manifest['scope'] == 'full_validation' and manifest['ordinary_inference'] and
            not manifest['training'] and not manifest['official_test_opened'], 'Unexpected source protocol')
    require(manifest['seeds'] == spec['training_seeds'] and len(set(manifest['seeds'])) == 3, 'Training seed cohort differs')
    ids = manifest['sample_ids']
    require(len(ids) == len(set(ids)) == spec['expected_samples'], 'Sample cohort differs')
    raw = source / 'diagnostic_rows'
    feature_cache = pilot.read_json(raw / 'input_features_clean_rep_0.json')
    require(feature_cache['value_sha256'] == pilot.digest(feature_cache['value']), 'Feature cache changed')
    features = {r['sample_id']: r for r in feature_cache['value']}
    require(len(feature_cache['value']) == len(features) and set(features) == set(ids), 'Feature sample IDs differ')
    seeds = spec['training_seeds']
    benefit = {}
    for condition, rep, _, _ in manifest['condition_definitions']:
        outcomes = []
        for seed in seeds:
            caches = []
            for model in (manifest['resolved_study']['parent'], manifest['resolved_study']['variant']):
                cache = pilot.read_json(raw / f'{model}_seed_{seed}_{condition}_rep_{rep}.json')
                run = next(r for r in manifest['runs'] if r['seed'] == seed and r['model'] == model)
                expected = dict(manifest_sha256=pilot.digest(manifest), checkpoint_sha256=run['checkpoint_sha256'],
                                condition=condition, replicate=rep)
                require(cache['identity'] == expected and cache['value_sha256'] == pilot.digest(cache['value']), 'Prediction cache identity differs')
                require(len(cache['value']) == len(ids), 'Prediction coverage differs')
                require(all(r['model'] == model and r['seed'] == seed for r in cache['value']), 'Prediction model/seed differs')
                caches.append(cache['value'])
            pairs = {r['sample_id']: r for r in pilot.paired_rows(*caches)}
            require(set(pairs) == set(ids), 'Paired sample IDs differ')
            require(all(pairs[i]['target'] == features[i]['target'] and pairs[i]['speaker'] == features[i]['speaker'] for i in ids), 'Labels/speakers differ')
            if condition == 'clean':
                require(all(pairs[i]['input_sha256'] == features[i]['input_sha256'] for i in ids), 'Clean feature/input identity differs')
            outcomes.append([pairs[i]['benefit'] for i in ids])
        benefit[condition, rep] = np.asarray(outcomes, dtype=float).T
    clean = benefit['clean', 0]
    changes = []
    for condition in spec['secondary_conditions']:
        reps = spec['deletion_replicates'] if condition.startswith('delete_') else [0]
        changes.append(np.mean([benefit[condition, rep] for rep in reps], axis=0) - clean)
    effects = np.stack(changes, axis=1)
    require(np.isfinite(effects).all() and np.isfinite(clean).all(), 'Nonfinite outcomes')
    excluded, keep = [], []
    for index, sid in enumerate(ids):
        row = features[sid]
        reasons = []
        for field in spec['features'] + ['duration_ms', 'mean_rate_hz']:
            value = row.get(field)
            if value is None or not np.isfinite(value):
                reasons.append('undefined_' + field)
        if not row['fast_slow_valid']: reasons.append('invalid_fast_slow')
        if not row['autocorr_valid'] or row['autocorr_censored']: reasons.append('invalid_or_censored_autocorrelation')
        if row.get('duration_ms') is not None and row['duration_ms'] <= 0: reasons.append('nonpositive_duration')
        if row.get('mean_rate_hz') is not None and row['mean_rate_hz'] < 0: reasons.append('negative_rate')
        if reasons: excluded.append(dict(sample_id=sid, reasons=reasons))
        else: keep.append(index)
    require(bool(keep), 'No complete cases')
    all_speakers = np.asarray([features[i]['speaker'] for i in ids], dtype=int)
    rows = [features[ids[i]] for i in keep]
    speakers = all_speakers[keep]
    require(len(np.unique(speakers)) >= spec['bootstrap']['minimum_speakers'], 'Too few complete-case speakers')
    design, scaling = make_design(np.asarray([[r[f] for f in spec['features']] for r in rows]),
        np.asarray([r['duration_ms'] for r in rows]), np.asarray([r['mean_rate_hz'] for r in rows]),
        np.asarray([r['target'] for r in rows]), speakers)
    return dict(source_manifest=manifest, ids=ids, features=features, clean=clean, effects=effects,
                keep=keep, speakers=speakers, all_speakers=all_speakers, excluded=excluded,
                design=design, centered_outcome=within_speaker(clean[keep], speakers), scaling=scaling)


def make_manifest(spec, inputs):
    paths = list((ROOT / 'scripts/adaptation_statistics').glob('*.py'))
    paths += [local_path(SPEC), local_path('scripts/adaptation_analysis/pilot.py'),
              local_path('scripts/adaptation_analysis/records.py'), local_path('project_paths.py')]
    return dict(version=spec['version'], specification=spec, source=spec['source'],
        dataset='SHD* validation', paper_baseline_version=inputs['source_manifest']['paper_baseline_version'],
        paper_baseline_dataset=inputs['source_manifest']['paper_baseline_dataset'],
        source_manifest_sha256=spec['source_manifest_sha256'], source_completion_sha256=spec['source_completion_sha256'],
        sample_ids=inputs['ids'], complete_case_ids=[inputs['ids'][i] for i in inputs['keep']],
        excluded=inputs['excluded'], scaling=inputs['scaling'],
        source_sha256={portable_path(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)},
        source_hash_encoding='CRLF_normalized_to_LF',
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         matplotlib=importlib.metadata.version('matplotlib'), PyYAML=importlib.metadata.version('PyYAML')),
        training=False, new_inference=False, official_test_opened=False)


def summarize(result, spec, inputs):
    alpha = spec['uncertainty']['alpha']
    estimates = result['beta'][:2]
    draws = result['bootstrap']
    bands = interval(draws.mean(axis=2), alpha, spec['uncertainty']['primary_family_size'])
    per_seed_bands = interval(draws, alpha, spec['uncertainty']['primary_family_size'])
    primary = []
    for j, feature in enumerate(spec['features']):
        values = estimates[j]
        low, high = bands[:, j]
        loo = [np.mean(x['feature_coefficients'][j]) for x in result['loo'] if x['feature_coefficients'] is not None]
        primary.append(dict(feature=feature, mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
            interval=[float(low), float(high)], direction=direction(low, high),
            per_seed=[dict(seed=seed, coefficient=float(values[k]), interval=per_seed_bands[:,j,k].tolist())
                      for k, seed in enumerate(spec['training_seeds'])],
            seed_direction_consistent=bool(np.all(values > 0) or np.all(values < 0)),
            leave_one_speaker_out_mean_range=[float(min(loo)), float(max(loo))] if loo else None,
            leave_one_speaker_out_all_estimable=len(loo)==len(result['loo'])))
    bands = interval(result['effect_bootstrap'].mean(axis=2), alpha, spec['uncertainty']['secondary_family_size'])
    secondary = []
    for j, name in enumerate(spec['secondary_conditions']):
        values = result['effects'][j]
        low, high = bands[:, j]
        secondary.append(dict(condition=name, mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
            interval=[float(low), float(high)], direction=direction(low, high),
            per_seed=[dict(seed=seed, benefit_change=float(values[k])) for k, seed in enumerate(spec['training_seeds'])]))
    diagnostics = sorted(zip(inputs['keep'], result['leverage']),key=lambda x:-x[1])[:10]
    return dict(version=spec['version'], scope='exploratory_validation_fixed_seed_cohort',
        n_samples=len(inputs['ids']), n_complete_cases=len(inputs['keep']), excluded=inputs['excluded'],
        n_speakers=len(np.unique(inputs['all_speakers'])), n_training_seeds=3,
        bootstrap_valid=len(draws), bootstrap_rejected=result['rejected'],
        primary=primary, secondary=secondary, design_condition_number=result['design_condition_number'],
        within_speaker_r_squared_per_seed=[float(x) if np.isfinite(x) else None for x in result['r_squared']],
        highest_leverage_samples=[dict(sample_id=inputs['ids'][i],leverage=float(h)) for i,h in diagnostics],
        leave_one_speaker_out=result['loo'], limitations=[
            'Only ten speaker clusters: percentile intervals may have poor coverage; no formal significance or causal confirmation.',
            'Intervals condition on three observed trained model pairs and do not generalize over training seeds.',
            'Three-seed sample SD is not a confidence interval; samples/replicates are not independent training runs.',
            'Primary two-feature and secondary three-intervention Bonferroni families are separate; no combined familywise guarantee.',
            'Seed-specific intervals are diagnostic, not a simultaneous six-coefficient testing family.',
            'Exploratory specification written after descriptive validation results were observed; not preregistration.',
            'Speaker/class/linear duration/activity adjustment does not remove all sample-difficulty confounding.',
            'No official-test access, new neural-model inference or training.'])


def write_outputs(destination, summary, result, spec, inputs):
    raw = destination / 'diagnostic_rows'; raw.mkdir(exist_ok=True)
    pilot.write_json(destination / 'SUMMARY.json', summary)
    pilot.write_json(destination / 'EXCLUSIONS.json', inputs['excluded'])
    rows = []
    for index, sid in enumerate(inputs['ids']):
        feature = inputs['features'][sid]
        row = dict(sample_id=sid, speaker=feature['speaker'], target=feature['target'],
                   duration_ms=feature['duration_ms'], mean_rate_hz=feature['mean_rate_hz'],
                   **{f:feature[f] for f in spec['features']})
        for k, seed in enumerate(spec['training_seeds']):
            row[f'clean_benefit_seed_{seed}'] = float(inputs['clean'][index,k])
            for j, name in enumerate(spec['secondary_conditions']):
                row[f'{name}_benefit_change_seed_{seed}'] = float(inputs['effects'][index,j,k])
        rows.append(row)
    pilot.write_csv(raw / 'sample_analysis.csv', rows)
    np.savez_compressed(raw / 'bootstrap_estimates.npz', primary=result['bootstrap'],
                        secondary=result['effect_bootstrap'], speaker_weights=result['cluster_weights'])
    lines = ['# Stage 5: exploratory adjusted associations', '',
        f"Samples: {summary['n_samples']}; complete cases: {summary['n_complete_cases']}; speakers: {summary['n_speakers']}; training seeds: 3.", '',
        'Outcome = loss(MT) - loss(AMT), positive favors AMT. Both features enter jointly. Controls: class, speaker, log1p(duration), log1p(activity). Feature coefficients use one original-cohort SD; bootstrap scales are fixed.', '',
        'Intervals are approximate speaker-cluster percentile sensitivity intervals conditional on three trained model pairs. There are only ten speakers; coverage may be poor. No p-values, formal significance claim, causal confirmation or independent test claim is made.', '',
        '## Adjusted clean-input associations', '',
        '| Feature | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate 97.5% interval | Conditional direction |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in summary['primary']:
        values = [f"{x['coefficient']:+.6f}" for x in r['per_seed']]
        lines.append('| '+r['feature']+' | '+' | '.join(values)+f" | {r['mean']:+.6f} +/- {r['sample_sd']:.6f} | [{r['interval'][0]:+.6f}, {r['interval'][1]:+.6f}] | {r['direction']} |")
    lines += ['', 'Two-feature Bonferroni family, nominal 95% family level subject to unreliable small-cluster bootstrap coverage. See JSON for seed-specific intervals and leave-one-speaker-out sensitivity.', '',
        '## Changes from clean input', '', 'Deletion realizations are averaged within sample and seed first. Three interventions form a separate Bonferroni family (nominal 98.3333% individual intervals).', '',
        '| Condition | Seed 42 | Seed 123 | Seed 456 | Mean +/- seed SD | Approximate interval |', '|---|---:|---:|---:|---:|---:|']
    for r in summary['secondary']:
        values=[f"{x['benefit_change']:+.6f}" for x in r['per_seed']]
        lines.append('| '+r['condition']+' | '+' | '.join(values)+f" | {r['mean']:+.6f} +/- {r['sample_sd']:.6f} | [{r['interval'][0]:+.6f}, {r['interval'][1]:+.6f}] |")
    lines += ['', '## Limits and interpretation', ''] + ['- '+x for x in summary['limitations']]
    lines += ['', 'The fast/slow hypothesis predicts a positive adjusted coefficient. Inspect both seed consistency and leave-one-speaker-out ranges; a conditional interval direction is not evidence of population-wide training-seed robustness. Autocorrelation is exploratory, with no directional AMT mechanism specified. Negative or inconclusive outcomes are retained.', '']
    (destination / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, groups, title in zip(axes, [summary['primary'], summary['secondary']],
                                ['Adjusted associations (per feature SD)', 'Change in benefit from clean']):
        for pos, r in enumerate(groups):
            low, high = r['interval']
            ax.plot([low, high], [pos, pos], color='tab:blue')
            ax.scatter([r['mean']], [pos], color='tab:blue', marker='D')
            values=[x.get('coefficient',x.get('benefit_change')) for x in r['per_seed']]
            ax.scatter(values, [pos]*3, color='grey', alpha=.6, marker='x')
        ax.set_yticks(range(len(groups)), [r.get('feature',r.get('condition')) for r in groups])
        ax.axvline(0,color='black',linewidth=.8)
        ax.set_title(title); ax.set_xlabel('Loss-benefit units'); ax.grid(alpha=.2)
    fig.suptitle('Exploratory speaker-cluster intervals; 10 speakers, 3 fixed training seeds')
    fig.tight_layout()
    try:
        for ext in ('png','svg'): fig.savefig(destination / ('adjusted_effects.'+ext), dpi=160)
    finally: plt.close(fig)
    return [destination/'SUMMARY.json', destination/'SUMMARY.md', destination/'EXCLUSIONS.json',
            raw/'sample_analysis.csv',raw/'bootstrap_estimates.npz',destination/'adjusted_effects.png',destination/'adjusted_effects.svg']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root')
    flags=parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only',action='store_true')
    flags.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    spec=yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    destination=experiment_directory(args.output_root or spec['output_root'])
    require(destination != experiment_directory(spec['source']), 'Statistics cannot overwrite evaluation outputs')
    if args.verify_only:
        print('Verified statistics outputs:',len(verify_completion(destination,filename='COMPLETE.json')))
        return 0
    inputs=load_inputs(spec)
    manifest=make_manifest(spec,inputs)
    path=destination/'MANIFEST.json'
    if path.exists(): pilot.validate_manifest(pilot.read_json(path),manifest)
    elif destination.exists() and any(p.name != '.analysis.lock' for p in destination.iterdir()):
        raise ValueError('Nonempty statistics destination has no matching manifest')
    if args.check_only:
        print(f"Statistics preflight passed: {len(inputs['ids'])} samples, {len(inputs['keep'])} complete cases, {len(np.unique(inputs['all_speakers']))} speakers; no output writes or adjusted fits.")
        return 0
    with pilot.study_lock(destination/'.analysis.lock'):
        if path.exists(): pilot.validate_manifest(pilot.read_json(path),manifest)
        else: pilot.write_json(path,manifest)
        if (destination/'COMPLETE.json').exists():
            verify_completion(destination,filename='COMPLETE.json'); print('Already complete and verified.'); return 0
        started=time.monotonic()
        result=cluster_analysis(inputs['design'], inputs['centered_outcome'], inputs['speakers'],
                                inputs['effects'], inputs['all_speakers'], spec['bootstrap'])
        summary=summarize(result,spec,inputs)
        outputs=[path]+write_outputs(destination,summary,result,spec,inputs)
        pilot.write_json(destination/'COMPLETE.json',dict(status='complete',manifest_sha256=pilot.digest(manifest),
            completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            outputs={portable_path(p):pilot.sha256(p) for p in outputs}))
        print('Statistics completed:',portable_path(destination))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr,flush=True);raise SystemExit(1)

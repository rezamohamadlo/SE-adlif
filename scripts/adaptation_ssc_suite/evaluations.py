"""Suite pair evaluations, reusing authenticated SSC inputs and MT observations."""
import time

import torch

from project_paths import portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.evaluate import check_anchor
from scripts.adaptation_ssc.inputs import InputSequence
from scripts.adaptation_ssc.runtime import load_model, check_rows
from scripts.adaptation_ssc_full.runtime import write_paired_csv
from scripts.adaptation_ssc_suite.common import (
    require, guard, source_features, predictions, validate_source, seal, verify_completion,
)
from scripts.adaptation_ssc_suite.reporting import write_reports


def pair_manifest(prepared, pair):
    spec, root, base = prepared['spec'], prepared['root'], prepared['manifest']
    models = [pair['parent'], pair['variant']]
    reused = models if pair['id'] == 'amt_mt' else (['MT_SE_adLIF'] if pair['parent'] == 'MT_SE_adLIF' else [])
    return dict(version=spec['version'] + '_' + pair['id'], scope='ssc_suite_full_validation',
        comparison=pair, sample_ids=base['sample_ids'], seeds=spec['seeds'], data=base['data'],
        runs=[r for r in base['runs'] if r['model'] in models], features=spec['features'],
        condition_definitions=spec['conditions'], shared_source=portable_path(root / 'shared/amt_mt_full_validation'),
        shared_source_manifest_sha256=spec['source_evaluation']['manifest_sha256'], reuse_models=reused,
        source_sha256=base['source_sha256'], environment=base['environment'],
        paper_baseline_version=spec['paper_baseline_version'], paper_baseline_dataset='SSC',
        comparison_loss=spec['loss_agg'], training=False, official_test_opened=False,
        ordinary_inference=True, auxiliary_JS_in_comparison_loss=False,
        reuse_interpretation='Shared MT predictions and input rows are the same observations, not new independent replications.',
        objective_protocols={r['model']: r['config']['evaluation_protocol'] for r in base['runs'] if r['model'] in models})


def pair_directory(prepared, pair):
    return prepared['root'] / 'pairs' / pair['id'] / 'evaluation'


def evaluate(prepared, pair, source):
    spec = prepared['spec']
    destination = pair_directory(prepared, pair)
    manifest = pair_manifest(prepared, pair)
    source_manifest, _, mapping = validate_source(spec, source)
    guard(destination, manifest)
    with pilot.study_lock(destination / '.analysis.lock'):
        guard(destination, manifest)
        if (destination / 'COMPLETE.json').exists():
            verify_completion(destination, prepared['root'])
            print(pair['id'] + ': complete evaluation verified; no inference repeated.', flush=True)
            return destination, manifest
        pilot.write_json(destination / 'MANIFEST.json', manifest)
        started = time.monotonic()
        # AMT/MT is already executed. Index its exact snapshot, without creating
        # an artificial rerun or changing any historical manifest/cache identity.
        if pair['id'] == 'amt_mt':
            views = {name: portable_path(source / 'analyses' / name / 'SUMMARY.json')
                     for name in ('clean', 'time_scaling', 'spike_deletion')}
            pilot.write_json(destination / 'REPORT_INDEX.json', dict(reused=True, reports=views,
                paired_csv=portable_path(source / 'diagnostic_rows/paired_samples.csv')))
            pilot.write_json(destination / 'VALIDATION_ANCHORS.json', pilot.read_json(source / 'VALIDATION_ANCHORS.json'))
            outputs = [source / 'COMPLETE.json', source / 'PATH_INDEX.json',
                       destination / 'REPORT_INDEX.json', destination / 'VALIDATION_ANCHORS.json']
            from project_paths import local_path
            outputs += [local_path(path) for path in mapping.values()]
            seal(destination, manifest, outputs, elapsed_seconds=time.monotonic()-started,
                 new_inference=False, paired_records=179658, complete_training_seed_cohort=True)
            return destination, manifest
        device = torch.device(manifest['environment']['device'])
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = False
        raw = destination / 'diagnostic_rows'
        raw.mkdir(exist_ok=True)
        ids = manifest['sample_ids']
        outputs = [source / 'MANIFEST.json', source / 'COMPLETE.json', source / 'PATH_INDEX.json']
        anchors, timings = [], []
        clean = [c for c in spec['conditions'] if c[0] == 'clean']
        require(len(clean) == 1, 'Exactly one clean condition is required')
        conditions = clean + [c for c in spec['conditions'] if c[0] != 'clean']
        for condition in conditions:
            name, rep, _, _ = condition
            phase_started = time.monotonic()
            features = source_features(source, source_manifest, condition)
            feature_cache = {r['sample_id']: r for r in features}
            outputs.append(source / 'diagnostic_rows' / f'input_features_{name}_rep_{rep}.json')
            for run in manifest['runs']:
                pilot.write_json(destination / 'PROGRESS.json', dict(status='running', pair=pair['id'],
                    model=run['model'], seed=run['seed'], condition=name, replicate=rep,
                    samples=len(ids), elapsed_seconds=time.monotonic()-started))
                reused = run['model'] in manifest['reuse_models']
                print(f"{pair['id']}: {run['model']} seed {run['seed']}, {name}/{rep}" +
                      (' (verified shared observations)' if reused else ''), flush=True)
                path = raw / f"{run['model']}_seed_{run['seed']}_{name}_rep_{rep}.json"
                if reused:
                    rows = predictions(destination, manifest, run['model'], run['seed'], condition, source, source_manifest)
                    outputs.append(source / 'diagnostic_rows' / path.name)
                else:
                    def produce():
                        model = load_model(run, device)
                        try:
                            if name == 'clean' and pair['id'] == 'lt_dth':
                                from scripts.adaptation_pairs.parameters import extract
                                parameter_path = destination / 'parameters' / f"{run['model']}_seed_{run['seed']}.json"
                                parameter_path.parent.mkdir(exist_ok=True)
                                pilot.write_json(parameter_path, extract(model, run, manifest['data']['bin_ms']))
                            return pilot.infer(model, run['config'], prepared['data'],
                                InputSequence(prepared['data'], ids, condition, spec, feature_cache), device,
                                f"SSC {pair['id']} {run['model']} {run['seed']}: {name}/{rep}")
                        finally:
                            del model
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                    rows = pilot.cached_json(path, dict(manifest_sha256=pilot.digest(manifest),
                        checkpoint_sha256=run['checkpoint_sha256'], condition=name, replicate=rep,
                        kind='ssc_suite_predictions'), produce)
                    outputs.append(path)
                check_rows(rows, ids, run, condition)
                require(all(all(row[k] == value for k, value in feature_cache[row['sample_id']].items()) for row in rows),
                        'Prediction/shared input metadata differs')
                if name == 'clean':
                    anchors.append(check_anchor(rows, run, spec))
                del rows
            if name == 'clean':
                require(len(anchors) == 6, 'All six pair anchors must pass before perturbation inference')
                pilot.write_json(destination / 'VALIDATION_ANCHORS.json', anchors)
                outputs.append(destination / 'VALIDATION_ANCHORS.json')
                print(pair['id'] + ': all six clean anchors passed.', flush=True)
            timings.append(dict(condition=name, replicate=rep, elapsed_seconds=time.monotonic()-phase_started))
            del features, feature_cache

        def pairs():
            for seed in spec['seeds']:
                for condition in spec['conditions']:
                    both = [predictions(destination, manifest, model, seed, condition, source, source_manifest)
                            for model in (pair['parent'], pair['variant'])]
                    yield from pilot.paired_rows(*both)

        paired_path = raw / 'paired_samples.csv'
        rows = write_paired_csv(paired_path, pairs())
        require(len(rows) == 179658, 'Pair coverage differs from all recordings/seeds/conditions')
        outputs.append(paired_path)
        metadata = dict(analyses=['clean', 'time_scaling', 'spike_deletion'], seeds=spec['seeds'],
            executed_seeds=spec['seeds'], samples=len(ids), full_clean_anchor_samples=len(ids),
            parent=pair['parent'], variant=pair['variant'], scope='ssc_full_validation',
            split='official_validation', training=False, official_test_opened=False,
            paper_baseline_version=spec['paper_baseline_version'], paper_baseline_dataset='SSC',
            loss_agg=spec['loss_agg'], comparison_loss='supervised_mean_membrane_only_no_auxiliary_JS',
            objective_protocols=manifest['objective_protocols'], reuse_models=manifest['reuse_models'])
        outputs += write_reports(destination, rows, metadata)
        if pair['id'] == 'lt_dth':
            parameter_paths = sorted((destination / 'parameters').glob('*.json'))
            require(len(parameter_paths) == 6, 'Missing learned/fixed trace descriptors; preserve evidence and inspect')
            outputs += parameter_paths
        seal(destination, manifest, outputs, elapsed_seconds=time.monotonic()-started,
             new_inference=True, phase_timings=timings, paired_records=len(rows), complete_training_seed_cohort=True)
        pilot.write_json(destination / 'PROGRESS.json', dict(status='complete', elapsed_seconds=time.monotonic()-started))
    return destination, manifest

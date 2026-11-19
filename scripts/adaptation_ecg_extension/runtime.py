"""User-launched features, 36 new mapped predictions and descriptive reports."""
import contextlib
from datetime import datetime, timezone
import io
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ecg.data import require
from scripts.adaptation_ecg.preflight import guard
from scripts.adaptation_ecg.runtime import aggregate, check_anchor, score_outputs
from scripts.adaptation_ecg_suite.reporting import describe
from scripts.adaptation_ecg_extension.analysis import associations, pair_records
from scripts.adaptation_ecg_extension.inputs import signed_features
from scripts.adaptation_ecg_extension.preflight import source_directory, source_file, verify_source
from scripts.ssc_resume_io import AtomicJSONWriter


def snapshot(root, spec, write):
    source = source_directory(root, spec)
    manifest, complete = verify_source(source, spec)
    destination = root/'shared/ecg_pairs_v2'
    destination.mkdir(parents=True, exist_ok=True)
    files = dict(complete['outputs'])
    files[spec['source']['root']+'/COMPLETE.json'] = spec['source']['completion_sha256']
    outputs, index = [], {}
    for name, checksum in files.items():
        target = source_file(destination, name, spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copyfile(source_file(source, name, spec), target)
        require(pilot.sha256(target) == checksum, 'Snapshot differs from frozen ECG evidence: '+name)
        outputs.append(target)
        index[name] = portable_path(target)
    path = destination/'PATH_INDEX.json'
    record = dict(path_index=index, source=spec['source'], mode='byte-for-byte preservation; original manifests unchanged')
    if path.exists(): pilot.validate_manifest(pilot.read_json(path), record)
    else: write(path, record)
    return destination, outputs+[path], manifest


def infer(run, data, condition, replicate, device):
    from models.pl_module import MLPSNN
    require(torch.get_float32_matmul_precision() == 'high', 'Matrix precision drifted before mapped ECG inference')
    torch.manual_seed(run['seed'])
    checkpoint = torch.load(local_path(run['checkpoint'], must_exist=True), weights_only=False, map_location='cpu')
    with contextlib.redirect_stdout(io.StringIO()):
        model = MLPSNN(OmegaConf.create(run['config']))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.requires_grad_(False).eval().to(device)
    del checkpoint
    rows = []
    try:
        with torch.inference_mode():
            for start in range(0, len(data.ids), run['config']['dataset']['batch_size']):
                ids = data.ids[start:start+run['config']['dataset']['batch_size']]
                x, labels, blocks = data.batch(ids, condition, replicate)
                scores = score_outputs(model(x.to(device)), labels.to(device), blocks.to(device))
                for sample_id, score in zip(ids, scores):
                    rows.append(dict(data.item(sample_id, condition, replicate)[-1],
                                     model=run['model'], seed=run['seed'], **score))
    finally:
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
    return rows


def reused_rows(record, data, run, condition, replicate, source_digest):
    require(record['identity']['manifest_sha256'] == source_digest and
            record['identity']['checkpoint_sha256'] == run['checkpoint_sha256'] and
            record['identity']['condition'] == condition and record['identity']['replicate'] == replicate and
            pilot.digest(record['value']) == record['value_sha256'], 'Reused prediction identity/content differs')
    result = []
    for row in record['value']:
        identity = data.item(row['sample_id'], condition, replicate)[-1]
        require(all(row[k] == v for k, v in identity.items() if k in row), 'Reused input/target/block identity differs')
        result.append(dict(row, **identity))
    data.validate_rows(result, run, condition, replicate)
    return result


def verify(directory):
    root = experiment_directory(directory)
    manifest, complete = [pilot.read_json(root/n) for n in ('MANIFEST.json', 'COMPLETE.json')]
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest) and
            complete['outputs'].get(portable_path(root/'MANIFEST.json')) == pilot.sha256(root/'MANIFEST.json'),
            'ECG extension completion/manifest identity differs')
    for name, checksum in complete['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path) == checksum, 'Extension output missing/changed: '+name)
    verify_source(root/'shared/ecg_pairs_v2', manifest['specification'])
    anchors = pilot.read_json(root/'REUSED_CLEAN_ANCHORS.json')
    require(len(anchors) == 18 and all(r['status'] == 'passed' for r in anchors), 'Reused anchors failed')
    tests = pilot.read_json(root/'CONTRACT_TESTS.json')
    require(tests['status'] == 'passed' and tests['returncode'] == 0, 'Extension contract tests did not pass')
    summary = pilot.read_json(root/'SUMMARY.json')
    require(summary['feature_rows'] == 180 and summary['new_prediction_chunks'] == 36 and
            summary['reused_prediction_chunks'] == 72 and summary['paired_rows'] == 2160 and
            len(summary['pairs']) == 4 and not summary['population_inference_performed'] and
            not summary['stage6_complete'], 'Extension scope/coverage differs')
    return complete


def execute(prepared):
    root, spec, manifest, data = (prepared[k] for k in ('root', 'spec', 'manifest', 'data'))
    started = time.monotonic()
    with pilot.study_lock(root/'.analysis.lock'):
        guard(root, manifest)
        if (root/'COMPLETE.json').exists():
            print('Extension already complete; verified outputs:', len(verify(root)['outputs']))
            return
        write = AtomicJSONWriter(root, pilot.write_json)
        if not (root/'MANIFEST.json').exists(): write(root/'MANIFEST.json', manifest)
        identity = pilot.digest(manifest)
        outputs, predictions, features = [root/'MANIFEST.json'], {}, {}

        def progress(phase, **details):
            write(root/'PROGRESS.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **details))

        def cached(path, contract, produce):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                record = pilot.read_json(path)
                pilot.validate_manifest(record['identity'], contract)
                require(pilot.digest(record['value']) == record['value_sha256'], 'Extension cached payload damaged')
                value = record['value']
            else:
                value = produce()
                write(path, dict(identity=contract, value=value, value_sha256=pilot.digest(value)))
            outputs.append(path)
            return value

        try:
            progress('contract_tests')
            modules = ['tests.test_adaptation_ecg', 'tests.test_adaptation_ecg_suite', 'tests.test_adaptation_ecg_extension']
            tests = subprocess.run([sys.executable, '-m', 'unittest', *modules], cwd=ROOT, capture_output=True,
                                   text=True, encoding='utf-8', errors='replace')
            (root/'CONTRACT_TESTS.txt').write_text(tests.stdout+tests.stderr, encoding='utf-8')
            write(root/'CONTRACT_TESTS.json', dict(status='passed' if tests.returncode == 0 else 'failed',
                returncode=tests.returncode, modules=modules, log_sha256=pilot.sha256(root/'CONTRACT_TESTS.txt')))
            outputs += [root/'CONTRACT_TESTS.txt', root/'CONTRACT_TESTS.json']
            require(tests.returncode == 0, 'Extension contract tests failed; inspect CONTRACT_TESTS.txt')
            progress('snapshot_completed_evidence')
            snapshot_root, copied, source = snapshot(root, spec, write)
            outputs += copied
            prediction_index = []
            for entry in pilot.read_json(snapshot_root/'PREDICTION_INDEX.json'):
                run = next(r for r in manifest['runs'] if r['model'] == entry['model'] and r['seed'] == entry['seed'])
                condition, rep = entry['condition'], entry['replicate']
                key = run['model'], run['seed'], condition, rep
                require(key not in predictions, 'Duplicate reused prediction chunk')
                path = source_file(snapshot_root, entry['path'], spec)
                require(pilot.sha256(path) == entry['sha256'], 'Reused prediction index hash differs')
                predictions[key] = reused_rows(pilot.read_json(path), data, run, condition, rep, pilot.digest(source))
                prediction_index.append(dict(entry, path=portable_path(path), kind='reused_completed_evidence'))
            expected_native = {(r['model'], r['seed'], c, p) for r in manifest['runs'] for c, p in spec['conditions'][:4]}
            require(set(predictions) == expected_native and len(predictions) == 72, 'Reused prediction cohort differs')
            anchors = [check_anchor(predictions[r['model'], r['seed'], 'clean', 0], r, spec) for r in manifest['runs']]
            write(root/'REUSED_CLEAN_ANCHORS.json', anchors)
            outputs.append(root/'REUSED_CLEAN_ANCHORS.json')
            print('All 18 reused clean anchors passed; no clean/deletion inference is repeated.', flush=True)

            for condition, rep in spec['conditions']:
                progress('input_features', condition=condition, replicate=rep)
                print(f'ECG input features: {condition}/{rep}, 30 records', flush=True)

                def produce_features():
                    rows = []
                    for sample_id in data.ids:
                        x, labels, mask, _blocks, row_identity = data.item(sample_id, condition, rep)
                        source_count = int(np.abs(data.base.x[sample_id]).sum(dtype=np.int64))
                        f = signed_features(x, labels, mask, spec['features'])
                        rows.append(dict(row_identity, **f, source_total_events=source_count,
                                         event_count_change=f['total_events']-source_count))
                    return rows

                path = root/'features'/condition/f'rep_{rep}.json'
                values = cached(path, dict(manifest_sha256=identity, condition=condition, replicate=rep,
                                          kind='model_independent_signed_input_features'), produce_features)
                require([r['sample_id'] for r in values] == data.ids, 'Feature coverage/order differs')
                for row in values:
                    require(all(row[k] == v for k, v in data.item(row['sample_id'], condition, rep)[-1].items()),
                            'Feature input/target/map identity differs')
                features[condition, rep] = {r['sample_id']: r for r in values}

            for condition, rep in spec['conditions'][4:]:
                for number, run in enumerate(manifest['runs'], 1):
                    progress('mapped_inference', condition=condition, model=run['model'], seed=run['seed'],
                             model_number=number, total_models=18)
                    print(f"ECG {condition} ({number}/18): {run['model']} seed {run['seed']}", flush=True)
                    path = root/'predictions'/run['model']/f"seed_{run['seed']}"/condition/f'rep_{rep}.json'
                    rows = cached(path, dict(manifest_sha256=identity, checkpoint_sha256=run['checkpoint_sha256'],
                        condition=condition, replicate=rep, float32_matmul_precision='high'),
                        lambda: infer(run, data, condition, rep, torch.device(manifest['environment']['device'])))
                    data.validate_rows(rows, run, condition, rep)
                    predictions[run['model'], run['seed'], condition, rep] = rows
                    prediction_index.append(dict(model=run['model'], seed=run['seed'], condition=condition, replicate=rep,
                        path=portable_path(path), sha256=pilot.sha256(path), kind='new_aligned_resampling_prediction'))
            require(len(predictions) == 108 and len(prediction_index) == 108, 'Effective prediction coverage differs')
            write(root/'PREDICTION_INDEX.json', prediction_index)
            outputs.append(root/'PREDICTION_INDEX.json')
            clean_features = [features['clean', 0][i] for i in data.ids]
            pair_summaries = []
            for pair in spec['pairs']:
                progress('paired_analysis', pair=pair['id'])
                directory = root/'pairs'/pair['id']
                directory.mkdir(parents=True, exist_ok=True)
                paired, metrics = [], []
                for seed in spec['seeds']:
                    cp, cv = [predictions[m, seed, 'clean', 0] for m in (pair['parent'], pair['variant'])]
                    for condition, rep in spec['conditions']:
                        p, v = [predictions[m, seed, condition, rep] for m in (pair['parent'], pair['variant'])]
                        paired.extend(pair_records(p, v, cp, cv, features))
                        metrics.extend(dict(model=m, seed=seed, condition=condition, replicate=rep, **aggregate(rows))
                                       for m, rows in ((pair['parent'], p), (pair['variant'], v)))
                require(len(paired) == 540, 'ECG pair expected 540 paired records')
                pilot.write_csv(directory/'paired_records.csv', paired)
                outputs.append(directory/'paired_records.csv')
                analysis = associations(clean_features, paired, spec['seeds'], spec['analysis'])
                write(directory/'ASSOCIATIONS.json', analysis)
                outputs.append(directory/'ASSOCIATIONS.json')
                summary = dict(pair=pair, paired_rows=540, n_training_seeds=3, validation_records=30,
                    descriptive=describe(paired, spec['seeds']), metrics=metrics, association_status=analysis['status'],
                    population_inference=False, resampling='joint event/label resampling sensitivity, not a count-preserving intervention')
                write(directory/'SUMMARY.json', summary)
                outputs.append(directory/'SUMMARY.json')
                text = [f"# ECG extension: {pair['variant']} / {pair['parent']}", '',
                    'All 30 validation rows; seeds 42/123/456; clean, three 10% deletion realizations and 0.9/1.1 time resampling.',
                    'Positive benefit means lower variant supervised CE. Repeated realizations are averaged within training seed.',
                    'Input, labels and masks share one map; event counts/class proportions can change. No causal interpretation.', '',
                    '| Condition | Accuracy gap mean ± sample SD (pp) | CE benefit mean ± sample SD | Change from clean mean ± sample SD |',
                    '|---|---:|---:|---:|']
                for group in summary['descriptive']:
                    a = group['aggregate']
                    text.append(f"| {group['condition']} | {a['accuracy_gap_pp']['mean']:+.6f} ± {a['accuracy_gap_pp']['sample_sd']:.6f} | "
                        f"{a['loss_benefit']['mean']:+.8f} ± {a['loss_benefit']['sample_sd']:.8f} | "
                        f"{a['benefit_change_from_clean']['mean']:+.8f} ± {a['benefit_change_from_clean']['sample_sd']:.8f} |")
                text += ['', '## Descriptive feature associations', '',
                    'Clean-cohort predictors: fast/slow activity contrast and ACF lag in model steps. Controls: log event rate, '
                    'duration if nonconstant, and label-class proportions with class 0 omitted.',
                    'Status: '+analysis['status']+'. No patient intervals, p-values or population/causal claims.',
                    'Complete-case IDs, explicit exclusions, dropped controls, rank/df, per-seed coefficients and row influence: '
                    '[ASSOCIATIONS.json](ASSOCIATIONS.json).',
                    'The feature windows are 20/200 model steps; they are not established physical milliseconds.', '']
                if analysis['status'] == 'not_estimable': text += ['Reason: '+analysis['design']['reason'], '']
                else:
                    text += ['| Clean feature | Coefficient mean ± sample SD across 3 seeds |', '|---|---:|']
                    for key, value in analysis['feature_summary'].items():
                        text.append(f"| {key} | {value['mean']:+.8f} ± {value['sample_sd_across_training_seeds']:.8f} |")
                    text += ['', 'Coefficients are CE benefit per population-SD change in that predictor within the fixed cohort.', '']
                (directory/'SUMMARY.md').write_text('\n'.join(text), encoding='utf-8')
                outputs.append(directory/'SUMMARY.md')
                pair_summaries.append(summary)

            feature_overview = []
            for condition, rep in spec['conditions']:
                rows = list(features[condition, rep].values())
                feature_overview.append(dict(condition=condition, replicate=rep, records=len(rows),
                    total_events=sum(r['total_events'] for r in rows), source_total_events=sum(r['source_total_events'] for r in rows),
                    event_count_change=sum(r['event_count_change'] for r in rows),
                    undefined_fast_slow=[r['sample_id'] for r in rows if r['fast_slow_contrast'] is None],
                    undefined_acf=[r['sample_id'] for r in rows if r['autocorr_tau_steps'] is None],
                    censored_acf=[r['sample_id'] for r in rows if r['autocorr_censored']]))
            summary = dict(status='extension_executed', feature_rows=180, feature_conditions=feature_overview,
                new_prediction_chunks=36, reused_prediction_chunks=72, effective_prediction_chunks=108,
                paired_rows=2160, pairs=pair_summaries, population_inference_performed=False,
                stage6_complete=False, limitations=manifest['limitations'], remaining_stage6_items=manifest['remaining_stage6_items'],
                training=False, official_test_opened=False)
            write(root/'SUMMARY.json', summary)
            outputs.append(root/'SUMMARY.json')
            lines = ['# ECG stage-6 extension', '',
                'Executed: 180 input-feature records; 72 preserved and 36 new prediction chunks; 2160 paired records.',
                'No training, official-test access or population inference. All-zero labels retain the original class-0 argmax.',
                'Feature time units are model steps. Time resampling changes input events and label proportions jointly.', '',
                '| Pair | Paired rows | Descriptive association status | Evidence |', '|---|---:|---|---|']
            for p in pair_summaries:
                lines.append(f"| {p['pair']['variant']} / {p['pair']['parent']} | 540 | {p['association_status']} | "
                             f"[report](pairs/{p['pair']['id']}/SUMMARY.md) |")
            lines += ['', 'Stage 6 remains open for evidence/scope review and the other listed diagnostics.',
                      'No patient identities were inferred. Non-estimable designs are reported without replacement models.', '']
            (root/'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
            outputs.append(root/'SUMMARY.md')
            write(root/'COMPLETE.json', dict(status='complete', manifest_sha256=identity,
                completed_utc=datetime.now(timezone.utc).isoformat(), attempt_elapsed_seconds=time.monotonic()-started,
                stage6_complete=False, population_inference_performed=False,
                outputs={portable_path(p): pilot.sha256(p) for p in dict.fromkeys(outputs)}))
            progress('complete', registered_outputs=len(verify(root)['outputs']))
            print('ECG extension complete:', portable_path(root/'SUMMARY.md'), flush=True)
        except Exception as error:
            progress('failed', error_type=type(error).__name__, error=str(error))
            raise

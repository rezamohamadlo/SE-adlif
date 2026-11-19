"""User-launched clean inference with resumable original-batch chunks."""
import contextlib
import csv
from datetime import datetime, timezone
import io
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ssc_suite.common import require, guard
from scripts.adaptation_ssc.inputs import InputSequence
from scripts.adaptation_ssc_windows.metrics import METRICS, windows, anchor_row, validate_windows
from scripts.adaptation_ssc_windows.preflight import read_cached, source_directory, source_file, verify_source
from scripts.adaptation_ssc_windows.reporting import summarize, markdown
from scripts.ssc_resume_io import AtomicJSONWriter

IDENTITY_FIELDS = ('sample_id', 'target', 'speaker', 'input_sha256', 'original_support_frames', 'stored_frames')
CSV_FIELDS = ['model', 'seed', *IDENTITY_FIELDS, 'prediction', 'loss', 'window_frames',
              'valid_windows', 'valid_support_frames', 'dropped_tail_frames', 'undefined_reason', *METRICS]


def snapshot(root, spec, write):
    source = source_directory(root, spec)
    verify_source(source, spec)
    target = root/'shared/ssc_clean_evidence'
    target.mkdir(parents=True, exist_ok=True)
    outputs, index = [], {}
    for suffix, checksum in spec['source']['files'].items():
        path = source_file(target, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): shutil.copyfile(source_file(source, suffix), path)
        require(pilot.sha256(path) == checksum, 'Selected source snapshot differs')
        outputs.append(path)
        index[spec['source']['root']+'/'+suffix] = portable_path(path)
    p = target/'PATH_INDEX.json'
    record = dict(path_index=index, source=spec['source']['root'],
        scope='authenticated clean MT/MT-TC subset only; omitted suite outputs remain outside this snapshot',
        original_manifests_rewritten=False)
    if p.exists(): pilot.validate_manifest(pilot.read_json(p), record)
    else: write(p, record)
    return target, outputs+[p]


def load_model(run, device):
    from models.pl_module import MLPSNN
    require(torch.get_float32_matmul_precision() == 'high', 'SSC matrix precision drifted')
    torch.manual_seed(run['seed'])
    checkpoint = torch.load(local_path(run['checkpoint'], must_exist=True), weights_only=False, map_location='cpu')
    with contextlib.redirect_stdout(io.StringIO()):
        model = MLPSNN(OmegaConf.create(run['config']))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.requires_grad_(False).eval().to(device)


def infer_batch(model, run, data, batch, device, spec):
    from datasets.utils.pad_tensors import PadTensors
    from models.pl_module import aggregate_temporal_outputs
    x, targets, blocks = PadTensors()([data.model_item(sid, values) for sid, values, _row in batch])
    x, targets, blocks = x.to(device), targets.to(device), blocks.to(device)
    with torch.inference_mode():
        output = model(x)
        logits = aggregate_temporal_outputs(output, blocks, targets.size(1), run['config']['loss_agg'])[:, 1, :]
        losses = F.cross_entropy(logits, targets[:, 1].long(), reduction='none')
        require(torch.isfinite(output).all().item() and torch.isfinite(losses).all().item(), 'Nonfinite SSC outputs/loss')
        probabilities = output.float().softmax(-1).cpu().numpy()
        raw = output.float().cpu().numpy()
        predictions = logits.argmax(-1).cpu().tolist()
        loss_values = losses.cpu().tolist()
    rows = []
    for i, (_sid, _counts, feature) in enumerate(batch):
        row = {key: feature[key] for key in IDENTITY_FIELDS}
        row.update(model=run['model'], seed=run['seed'], condition='clean', replicate=0,
            prediction=predictions[i], correct=int(predictions[i] == feature['target']), loss=loss_values[i],
            minimum_padding_frames=feature['stored_frames']-feature['original_support_frames'],
            batch_padding_frames=raw.shape[1]-feature['stored_frames'],
            windows=windows(raw[i], probabilities[i], feature['target'], feature['original_support_frames'],
                            spec['window_frames'], spec['burn_in_frames']))
        rows.append(row)
    return rows


def validate_chunk(rows, ids, run, references, spec):
    require([r['sample_id'] for r in rows] == ids, 'Window chunk coverage/order differs')
    differences = []
    for row in rows:
        require(row['model'] == run['model'] and row['seed'] == run['seed'] and
                row['condition'] == 'clean' and row['replicate'] == 0, 'Window chunk model/seed/condition differs')
        differences.append(anchor_row(row, references[row['sample_id']]))
        validate_windows(row['windows'], spec, row['target'])
        available = max(0, row['original_support_frames']-spec['burn_in_frames'])
        for descriptor in row['windows']:
            width = descriptor['window_frames']
            require(descriptor['valid_windows'] == available//width and descriptor['dropped_tail_frames'] == available%width and
                    descriptor['valid_support_frames'] == available, 'Window masks/support/tail differ')
        require(row['minimum_padding_frames'] == row['stored_frames']-row['original_support_frames'] >= 0 and
                row['batch_padding_frames'] >= 0, 'Synthetic padding accounting differs')
    return differences


def verify(directory):
    root = experiment_directory(directory)
    manifest, complete = [pilot.read_json(root/name) for name in ('MANIFEST.json', 'COMPLETE.json')]
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest) and
            complete['outputs'].get(portable_path(root/'MANIFEST.json')) == pilot.sha256(root/'MANIFEST.json'),
            'SSC window completion/manifest identity differs')
    for name, checksum in complete['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path) == checksum, 'SSC window output changed: '+name)
    verify_source(root/'shared/ssc_clean_evidence', manifest['specification'])
    anchors = pilot.read_json(root/'ANCHORS.json')
    require(len(anchors) == 6 and len({(r['model'], r['seed']) for r in anchors}) == 6 and
            all(r['status'] == 'passed' and r['records'] == 9981 and r['prediction_mismatches'] == 0 for r in anchors),
            'All six full clean anchors must pass')
    summary = pilot.read_json(root/'SUMMARY.json')
    require(summary['prediction_records'] == 59886 and summary['descriptor_records'] == 179658 and
            summary['paired_records'] == 89829 and summary['chunks'] == 234, 'Window diagnostic coverage differs')
    require(pilot.read_json(root/'CONTRACT_TESTS.json')['status'] == 'passed', 'No passing user contract evidence')
    return complete


def execute(prepared):
    root, spec, manifest, data = (prepared[k] for k in ('root', 'spec', 'manifest', 'data'))
    started = time.monotonic()
    with pilot.study_lock(root/'.analysis.lock'):
        guard(root, manifest)
        if (root/'COMPLETE.json').exists():
            print('SSC windows already complete; verified outputs:', len(verify(root)['outputs']))
            return
        write = AtomicJSONWriter(root, pilot.write_json)
        if not (root/'MANIFEST.json').exists(): write(root/'MANIFEST.json', manifest)
        identity, outputs = pilot.digest(manifest), [root/'MANIFEST.json']

        def progress(phase, **details):
            write(root/'PROGRESS.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **details))

        try:
            progress('contract_tests')
            tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_adaptation_ssc_windows'],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            (root/'CONTRACT_TESTS.txt').write_text(tests.stdout+tests.stderr, encoding='utf-8')
            write(root/'CONTRACT_TESTS.json', dict(status='passed' if tests.returncode == 0 else 'failed',
                returncode=tests.returncode, log_sha256=pilot.sha256(root/'CONTRACT_TESTS.txt')))
            outputs += [root/'CONTRACT_TESTS.txt', root/'CONTRACT_TESTS.json']
            require(tests.returncode == 0, 'SSC window contract tests failed; inspect CONTRACT_TESTS.txt')
            evidence, copied = snapshot(root, spec, write)
            outputs += copied
            features = read_cached(source_file(evidence, spec['clean_features_file']))['value']
            feature_cache = {r['sample_id']: r for r in features}
            ids, seeds = manifest['sample_ids'], spec['seeds']
            id_index = {sid: i for i, sid in enumerate(ids)}
            clean = ('clean', 0, 'identity', 1.)
            inputs = InputSequence(data, ids, clean, spec, feature_cache)
            device = torch.device(manifest['environment']['device'])
            values, anchors, chunks, descriptors = {}, [], [], 0
            temporary = root/'SAMPLE_WINDOWS.csv.tmp'
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for run in manifest['runs']:
                    reference = read_cached(source_file(evidence, run['reference_file']))['value']
                    references = {r['sample_id']: r for r in reference}
                    matrix = np.full((len(spec['window_frames']), len(ids), len(METRICS)), np.nan)
                    drift, count, max_drift, total_loss, total_correct, model = 0., 0, 0., 0., 0, None
                    batch_size = run['config']['dataset']['batch_size']
                    try:
                        for start in range(0, len(ids), batch_size):
                            batch_ids = ids[start:start+batch_size]
                            path = root/'chunks'/run['model']/f"seed_{run['seed']}"/f'batch_{start//batch_size:04d}.json'
                            path.parent.mkdir(parents=True, exist_ok=True)
                            contract = dict(manifest_sha256=identity, checkpoint_sha256=run['checkpoint_sha256'],
                                batch_start=start, sample_ids=batch_ids, batch_size=batch_size, float32_matmul_precision='high')
                            progress('clean_forward_and_windows', model=run['model'], seed=run['seed'],
                                     records_completed=start, records_total=len(ids), chunk=start//batch_size+1)
                            if path.exists():
                                record = read_cached(path)
                                pilot.validate_manifest(record['identity'], contract)
                                rows = record['value']
                            else:
                                if model is None: model = load_model(run, device)
                                rows = infer_batch(model, run, data, inputs[start:start+batch_size], device, spec)
                                write(path, dict(identity=contract, value=rows, value_sha256=pilot.digest(rows)))
                            delta = validate_chunk(rows, batch_ids, run, references, spec)
                            drift += sum(delta)
                            max_drift = max(max_drift, max(abs(v) for v in delta))
                            count += len(rows)
                            total_loss += sum(r['loss'] for r in rows)
                            total_correct += sum(r['correct'] for r in rows)
                            outputs.append(path)
                            chunks.append(dict(model=run['model'], seed=run['seed'], batch_start=start,
                                samples=len(rows), path=portable_path(path), sha256=pilot.sha256(path)))
                            for row in rows:
                                for w, descriptor in enumerate(row['windows']):
                                    for k, metric in enumerate(METRICS):
                                        value = descriptor[metric]
                                        if value is not None: matrix[w, id_index[row['sample_id']], k] = value
                                    output = {key: row[key] for key in ['model', 'seed', *IDENTITY_FIELDS, 'prediction', 'loss']}
                                    output.update({key: descriptor[key] for key in CSV_FIELDS if key in descriptor})
                                    writer.writerow(output)
                                    descriptors += 1
                            print(f"SSC windows {run['model']} seed {run['seed']}: {start+len(rows)}/{len(ids)}", flush=True)
                    finally:
                        if model is not None: del model
                        if device.type == 'cuda': torch.cuda.empty_cache()
                    require(count == 9981 and abs(drift/count) <= spec['loss_tolerance'],
                            'SSC full clean mean-loss anchor drift exceeded the unchanged source tolerance')
                    loss_delta = total_loss/count-run['validation']['loss']
                    accuracy_delta = total_correct/count-run['validation']['accuracy']
                    require(abs(loss_delta) <= spec['loss_tolerance'] and abs(accuracy_delta) <= spec['accuracy_tolerance'],
                            'SSC ordinary clean anchor differs from the original validation-selected metrics')
                    anchors.append(dict(status='passed', model=run['model'], seed=run['seed'], records=count,
                        prediction_mismatches=0, mean_loss_difference=drift/count, max_absolute_record_loss_difference=max_drift,
                        validation_accuracy_difference=accuracy_delta, validation_loss_difference=loss_delta,
                        selected_epoch=run['selected_epoch']))
                    write(root/'ANCHORS.json', anchors)
                    values[run['model'], run['seed']] = matrix
            require(len(anchors) == 6 and len(chunks) == 234 and descriptors == 179658, 'Incomplete clean-window diagnostic cohort')
            os.replace(temporary, root/'SAMPLE_WINDOWS.csv')
            outputs += [root/'SAMPLE_WINDOWS.csv', root/'ANCHORS.json']
            progress('paired_reports')
            paired_temp = root/'PAIRED_WINDOWS.csv.tmp'
            paired_fields = ['sample_id', 'target', 'speaker', 'input_sha256', 'seed', 'window_frames',
                             *[k for metric in METRICS for k in (metric+'_parent', metric+'_variant', metric+'_difference')]]
            paired_count = 0
            with paired_temp.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=paired_fields)
                writer.writeheader()
                for seed in seeds:
                    p, v = values['MT_SE_adLIF', seed], values['MT_TC_SE_adLIF', seed]
                    for w, width in enumerate(spec['window_frames']):
                        for i, feature in enumerate(features):
                            row = {k: feature[k] for k in ('sample_id', 'target', 'speaker', 'input_sha256')}
                            row.update(seed=seed, window_frames=width)
                            for k, metric in enumerate(METRICS):
                                a, b = p[w, i, k], v[w, i, k]
                                row[metric+'_parent'] = float(a) if np.isfinite(a) else None
                                row[metric+'_variant'] = float(b) if np.isfinite(b) else None
                                row[metric+'_difference'] = float(b-a) if np.isfinite(a) and np.isfinite(b) else None
                            writer.writerow(row)
                            paired_count += 1
            require(paired_count == 89829, 'Paired window CSV coverage differs')
            os.replace(paired_temp, root/'PAIRED_WINDOWS.csv')
            outputs.append(root/'PAIRED_WINDOWS.csv')
            write(root/'CHUNK_INDEX.json', chunks)
            outputs.append(root/'CHUNK_INDEX.json')
            summary = summarize(values, seeds, spec['window_frames'])
            summary.update(version=spec['version'], dataset='SSC official clean validation', validation_records=9981,
                prediction_records=59886, descriptor_records=descriptors, paired_records=paired_count, chunks=len(chunks),
                window_frames=spec['window_frames'], physical_bin_ms=4., burn_in_frames=10,
                anchors_passed=6, training=False, official_test_opened=False, stage6_complete=False,
                remaining='Review this evidence and remaining study/code scope before closing stage 6; stage 7 is separate')
            write(root/'SUMMARY.json', summary)
            (root/'SUMMARY.md').write_text(markdown(summary), encoding='utf-8')
            outputs += [root/'SUMMARY.json', root/'SUMMARY.md']
            write(root/'COMPLETE.json', dict(status='complete', manifest_sha256=identity,
                completed_utc=datetime.now(timezone.utc).isoformat(), attempt_elapsed_seconds=time.monotonic()-started,
                stage6_complete=False, atomic_json_retries=len(write.retries),
                outputs={portable_path(p): pilot.sha256(p) for p in dict.fromkeys(outputs)}))
            progress('complete', registered_outputs=len(verify(root)['outputs']))
            print('SSC direct-window diagnostics complete:', portable_path(root/'SUMMARY.md'), flush=True)
        except Exception as error:
            progress('failed', error_type=type(error).__name__, error=str(error))
            raise

"""User-launched ECG clean anchors and signed-event deletion, without fitting."""
import contextlib
from datetime import datetime, timezone
import io
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_ecg.data import array_hash, require
from scripts.adaptation_ecg.preflight import guard
from scripts.ssc_resume_io import AtomicJSONWriter


def score_outputs(outputs, targets, blocks):
    """Match the original temporal aggregation; exclude dummy and padded targets."""
    from models.pl_module import aggregate_temporal_outputs
    require(outputs.shape[:2] == blocks.shape and targets.dtype == torch.long, 'ECG output/label shape or dtype differs')
    require(torch.isfinite(outputs).all().item(), 'Nonfinite ECG model outputs')
    logits = aggregate_temporal_outputs(outputs, blocks, targets.size(1), 'summed_membrane_potentials')
    valid = targets != -1
    support = torch.zeros_like(targets).scatter_add_(1, blocks, torch.ones_like(blocks))
    require(not valid[:, 0].any().item() and (support[valid] == 1).all().item(),
            'ECG requires exactly one model step for each valid temporal target')
    require((valid.sum(1) > 0).all().item(), 'ECG record has no valid targets')
    losses = F.cross_entropy(logits.transpose(1, 2).float(), targets, ignore_index=-1, reduction='none')
    predictions = logits.argmax(-1)
    require(torch.isfinite(losses[valid]).all().item(), 'Nonfinite ECG comparison losses')
    rows = []
    for loss, pred, target, mask in zip(losses, predictions, targets, valid):
        n = int(mask.sum().item())
        total = float(loss[mask].double().sum().item())
        correct = int((pred[mask] == target[mask]).sum().item())
        rows.append(dict(valid_steps=n, loss_sum=total, loss=total/n, correct_steps=correct,
            accuracy=correct/n, predictions=pred[mask].cpu().tolist()))
    return rows


def identities(data, spec, ids, condition, replicate):
    return [dict(sample_id=i, input_sha256=array_hash(data.input(i, condition, replicate, spec)),
                 targets_sha256=array_hash(data.y[i]),
                 blocks_sha256=array_hash(np.arange(1, len(data.y[i])+1, dtype=np.int64))) for i in ids]


def infer(run, data, spec, condition, replicate, device):
    from models.pl_module import MLPSNN
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
                x, targets, blocks = data.batch(ids, condition, replicate, spec)
                scores = score_outputs(model(x.to(device)), targets.to(device), blocks.to(device))
                for identity, score in zip(identities(data, spec, ids, condition, replicate), scores):
                    rows.append(dict(identity, model=run['model'], seed=run['seed'],
                                     condition=condition, replicate=replicate, **score))
    finally:
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
    return rows


def validate_rows(rows, data, spec, run, condition, replicate):
    expected = identities(data, spec, data.ids, condition, replicate)
    require(len(rows) == len(expected), 'ECG prediction coverage differs')
    for row, identity in zip(rows, expected):
        require(all(row[k] == v for k, v in identity.items()) and row['model'] == run['model'] and
                row['seed'] == run['seed'] and row['condition'] == condition and row['replicate'] == replicate,
                'ECG cached input/target/block/model/condition identity differs')
        require(row['valid_steps'] == len(data.y[row['sample_id']]) and
                len(row['predictions']) == row['valid_steps'] and np.isfinite(row['loss_sum']) and
                row['loss_sum'] >= 0 and 0 <= row['correct_steps'] <= row['valid_steps'], 'Invalid ECG prediction record')
        require(all(0 <= p < 6 for p in row['predictions']) and
                row['correct_steps'] == sum(p == int(y) for p, y in zip(row['predictions'], data.y[row['sample_id']])),
                'ECG prediction/correct count differs')
        require(row['loss'] == row['loss_sum']/row['valid_steps'] and
                row['accuracy'] == row['correct_steps']/row['valid_steps'], 'ECG metric reduction differs')


def aggregate(rows):
    n = sum(r['valid_steps'] for r in rows)
    return dict(records=len(rows), valid_steps=n,
        step_weighted_loss=sum(r['loss_sum'] for r in rows)/n,
        step_weighted_accuracy=sum(r['correct_steps'] for r in rows)/n,
        record_mean_loss=sum(r['loss'] for r in rows)/len(rows),
        record_mean_accuracy=sum(r['accuracy'] for r in rows)/len(rows))


def check_anchor(rows, run, spec):
    metrics = aggregate(rows)
    accuracy_delta = metrics['step_weighted_accuracy']-run['validation']['accuracy']
    loss_delta = metrics['step_weighted_loss']-run['validation']['loss']
    require(abs(accuracy_delta) <= spec['accuracy_tolerance'] and abs(loss_delta) <= spec['loss_tolerance'],
            f"ECG clean anchor failed for {run['model']} seed {run['seed']}: "
            f"accuracy delta={accuracy_delta}, loss delta={loss_delta}. Deletion remains blocked.")
    return dict(model=run['model'], seed=run['seed'], selected_epoch=run['selected_epoch'],
        saved_validation=run['validation'], measured=metrics, accuracy_delta=accuracy_delta, loss_delta=loss_delta,
        status='passed')


def verify(root):
    manifest = pilot.read_json(root/'MANIFEST.json')
    complete = pilot.read_json(root/'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest),
            'ECG completion identity differs')
    require(complete['outputs'].get(portable_path(root/'MANIFEST.json')) == pilot.sha256(root/'MANIFEST.json'),
            'ECG completion must register its own manifest')
    for name, expected in complete['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path) == expected, 'ECG registered output changed: '+name)
    return complete


def execute(spec, root, manifest, data):
    started = time.monotonic()
    with pilot.study_lock(root/'.analysis.lock'):
        guard(root, manifest)
        if (root/'COMPLETE.json').exists():
            print('ECG pilot already complete; verified outputs:', len(verify(root)['outputs']))
            return
        write = AtomicJSONWriter(root, pilot.write_json)
        if not (root/'MANIFEST.json').exists(): write(root/'MANIFEST.json', manifest)
        outputs = [root/'MANIFEST.json']
        raw = root/'raw'
        raw.mkdir(exist_ok=True)
        identity = pilot.digest(manifest)

        def progress(phase, **details):
            write(root/'PROGRESS.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **details))

        def predictions(run, condition, replicate):
            path = raw/f"{run['model']}_seed_{run['seed']}_{condition}_rep_{replicate}.json"
            contract = dict(manifest_sha256=identity, checkpoint_sha256=run['checkpoint_sha256'],
                            condition=condition, replicate=replicate)
            if path.exists():
                cached = pilot.read_json(path)
                pilot.validate_manifest(cached['identity'], contract)
                require(pilot.digest(cached['value']) == cached['value_sha256'], 'ECG prediction cache damaged')
                rows = cached['value']
            else:
                rows = infer(run, data, spec, condition, replicate, torch.device(manifest['environment']['device']))
                write(path, dict(identity=contract, value=rows, value_sha256=pilot.digest(rows)))
            validate_rows(rows, data, spec, run, condition, replicate)
            outputs.append(path)
            return rows

        try:
            progress('contract_tests')
            tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_adaptation_ecg'],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            test_log = root/'CONTRACT_TESTS.txt'
            test_log.write_text(tests.stdout + tests.stderr, encoding='utf-8')
            outputs.append(test_log)
            require(tests.returncode == 0, 'ECG contract tests failed; inspect '+portable_path(test_log))
            cache, anchors = {}, []
            for run in manifest['runs']:
                progress('clean_anchor', model=run['model'], seed=run['seed'])
                print(f"ECG clean anchor: {run['model']} seed {run['seed']}", flush=True)
                rows = predictions(run, 'clean', 0)
                anchors.append(check_anchor(rows, run, spec))
                cache[run['model'], run['seed'], 'clean', 0] = rows
            write(root/'VALIDATION_ANCHORS.json', anchors)
            outputs.append(root/'VALIDATION_ANCHORS.json')
            print('All six full-validation ECG anchors passed; starting signed-event deletion.', flush=True)
            for condition, replicate in spec['conditions'][1:]:
                for run in manifest['runs']:
                    progress('signed_event_deletion', model=run['model'], seed=run['seed'], replicate=replicate)
                    print(f"ECG {condition}/{replicate}: {run['model']} seed {run['seed']}", flush=True)
                    cache[run['model'], run['seed'], condition, replicate] = predictions(run, condition, replicate)
            paired = []
            for seed in spec['seeds']:
                for condition, rep in spec['conditions']:
                    parent = cache[spec['parent'], seed, condition, rep]
                    variant = cache[spec['variant'], seed, condition, rep]
                    clean_parent = cache[spec['parent'], seed, 'clean', 0]
                    clean_variant = cache[spec['variant'], seed, 'clean', 0]
                    for p, v, cp, cv in zip(parent, variant, clean_parent, clean_variant):
                        require(all(p[k] == v[k] for k in ('sample_id', 'input_sha256', 'targets_sha256',
                                                          'blocks_sha256', 'valid_steps')), 'Unpaired ECG predictions')
                        clean_benefit = cp['loss']-cv['loss']
                        benefit = p['loss']-v['loss']
                        paired.append(dict(sample_id=p['sample_id'], seed=seed, condition=condition, replicate=rep,
                            input_sha256=p['input_sha256'], targets_sha256=p['targets_sha256'], blocks_sha256=p['blocks_sha256'],
                            valid_steps=p['valid_steps'], parent_loss=p['loss'], variant_loss=v['loss'],
                            parent_accuracy=p['accuracy'], variant_accuracy=v['accuracy'],
                            loss_benefit=benefit, clean_loss_benefit=clean_benefit,
                            benefit_change_from_clean=benefit-clean_benefit))
            pilot.write_csv(root/'paired_records.csv', paired)
            outputs.append(root/'paired_records.csv')
            summaries = [dict(model=m, seed=s, condition=c, replicate=r, **aggregate(rows))
                         for (m, s, c, r), rows in cache.items()]
            summary = dict(status='pilot_complete', scope=manifest['scope'], records=30,
                paired_rows=len(paired), anchors_passed=len(anchors), prediction_chunks=len(cache),
                full_mechanism_study_complete=False, training=False, official_test_opened=False,
                results=summaries, limitations=manifest['limitations'],
                next_step='Review clean anchors, signed deletion and temporal losses before features/time scaling or other ECG pairs')
            write(root/'SUMMARY.json', summary)
            outputs.append(root/'SUMMARY.json')
            text = ['# ECG validation alignment pilot', '',
                'Pilot only: six clean anchors and signed-event deletion for AMT/MT, seeds 42/123/456.',
                'All 30 validation records; no training or official-test access. Positive benefit means lower variant CE.',
                'Step-weighted metrics use all valid target positions; record means weight each record equally.',
                'All records currently contain 1301 scored steps under the saved loader, so these weighting schemes agree.',
                'Original argmax behavior is preserved: all-zero class vectors are scored as class 0. No new exclusion mask is inferred.',
                'Replicates and time steps are not independent training seeds or patients.', '',
                '| Model | Seed | Condition | Replicate | Validation accuracy (%) | Step-weighted CE | Record-mean CE |',
                '|---|---:|---|---:|---:|---:|---:|']
            for row in summaries:
                text.append(f"| {row['model']} | {row['seed']} | {row['condition']} | {row['replicate']} | "
                    f"{row['step_weighted_accuracy']*100:.6f} | {row['step_weighted_loss']:.8f} | {row['record_mean_loss']:.8f} |")
            text += ['', 'Features, time warping, statistical inference and the other ECG pairs remain pending.', '']
            (root/'SUMMARY.md').write_text('\n'.join(text), encoding='utf-8')
            outputs.append(root/'SUMMARY.md')
            write(root/'COMPLETE.json', dict(status='complete', manifest_sha256=identity,
                completed_utc=datetime.now(timezone.utc).isoformat(), attempt_elapsed_seconds=time.monotonic()-started,
                outputs={portable_path(p): pilot.sha256(p) for p in outputs}))
            progress('complete', verified_outputs=len(verify(root)['outputs']))
            print('ECG pilot complete:', portable_path(root/'SUMMARY.md'), flush=True)
        except Exception as error:
            progress('failed', error_type=type(error).__name__, error=str(error))
            raise

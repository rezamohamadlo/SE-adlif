"""User-launched forward-only matrix; no precision winner or evidence rewriting."""
from datetime import datetime, timezone
import subprocess
import sys
import time

import torch

from datasets.utils.pad_tensors import PadTensors
from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_batch_precision.contracts import (
    MODES, batch_identity, compare_score, precision_mode, repeated_target,
    reversed_batch, singleton_from_original, require_original_anchor,
)
from scripts.adaptation_gradient_pilot.engine import load_model, scores, supervised
from scripts.adaptation_gradient_pilot.methods import atomic_copy, count_hash, require, require_frozen
from scripts.adaptation_gradient_pilot.preflight import guard, read_cache
from scripts.ssc_resume_io import AtomicJSONWriter


def evaluate(model, batch, cfg, device, slots):
    with torch.no_grad():
        x, target, blocks = [v.to(device) for v in batch]
        losses, logits = supervised(model, x, target, blocks, cfg)
        result = scores(losses, logits, target)
        raw = logits.detach().cpu().tolist()
    return [dict(result[slot], logits=raw[slot]) for slot in slots]


def compare_rows(rows):
    """Descriptive paired comparisons, no resampling or precision selection."""
    index = {(r['model'], r['mode'], r['layout'], r['sample_id']): r for r in rows}
    comparisons = []
    def add(row, ref, contrast):
        comparisons.append(dict(model=row['model'], seed=42, sample_id=row['sample_id'],
            mode=row['mode'], layout=row['layout'], reference_mode=ref['mode'], reference_layout=ref['layout'],
            contrast=contrast, **compare_score(row, ref),
            max_absolute_logit_difference=max(abs(a-b) for a,b in zip(row['logits'], ref['logits']))))
    for r in rows:
        key = (r['model'], r['mode'], 'original_batch', r['sample_id'])
        if r['layout'] != 'original_batch': add(r, index[key], 'versus_same_mode_original_batch')
        if r['layout'] == 'singleton_original_padding':
            add(r, index[r['model'], r['mode'], 'singleton_natural', r['sample_id']], 'padding_at_batch_one')
        if r['mode'] != MODES[0]:
            add(r, index[r['model'], MODES[0], r['layout'], r['sample_id']], 'precision_at_fixed_layout')
    return comparisons


def verify(directory):
    root = experiment_directory(directory)
    manifest, complete = [pilot.read_json(root/n) for n in ('MANIFEST.json', 'COMPLETE.json')]
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest) and
            complete['outputs'].get(portable_path(root/'MANIFEST.json')) == pilot.sha256(root/'MANIFEST.json'),
            'Follow-up completion identity differs')
    for name, checksum in complete['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path) == checksum, 'Follow-up output changed: '+name)
    summary = pilot.read_json(root/'SUMMARY.json')
    require(summary['layout_rows'] == 720 and summary['focus_control_rows'] == 24 and
            summary['full_validation_score_rows'] == 9786 and summary['matrix_jobs'] == 546 and
            summary['source_high_original_anchors_passed'] and not summary['stage7_complete'], 'Coverage/gates differ')
    require(pilot.read_json(root/'CONTRACT_TESTS.json')['status'] == 'passed', 'No passing user contracts')
    return complete


def execute(prepared):
    root, spec, m, data, old_root = [prepared[k] for k in ('root', 'spec', 'manifest', 'data', 'pilot_root')]
    started = time.monotonic()
    device = torch.device(m['environment']['device'])
    with pilot.study_lock(root/'.analysis.lock'):
        guard(root, m)
        if (root/'COMPLETE.json').exists():
            print('Already complete; verified outputs:', len(verify(root)['outputs']));return
        write = AtomicJSONWriter(root, pilot.write_json)
        if not (root/'MANIFEST.json').exists(): write(root/'MANIFEST.json', m)
        identity = pilot.digest(m)
        outputs = [root/'MANIFEST.json']
        models = {}
        def progress(phase, **kw):
            write(root/'PROGRESS.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **kw))
        try:
            progress('user_contract_tests')
            tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_adaptation_batch_precision'],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            log = root/'CONTRACT_TESTS.txt';log.write_text(tests.stdout+tests.stderr, encoding='utf-8')
            write(root/'CONTRACT_TESTS.json', dict(status='passed' if tests.returncode == 0 else 'failed',
                returncode=tests.returncode, log_sha256=pilot.sha256(log)))
            outputs += [log, root/'CONTRACT_TESTS.json']
            require(tests.returncode == 0, 'User contracts failed; inspect CONTRACT_TESTS.txt')
            names = ['MANIFEST.json', 'COMPLETE.json', 'FULL_VALIDATION_ANCHORS.json', 'FROZEN_STATE_CHECKS.json',
                'GRADIENT_SUMMARIES.json', 'SUMMARY.json', 'CONTRACT_TESTS.json', 'CONTRACT_TESTS.txt']
            names += ['anchors/'+r['model']+'_seed_42.json' for r in m['runs']]
            ledger = pilot.read_json(old_root/'COMPLETE.json')['outputs'];path_index = {}
            for suffix in names:
                source = old_root/suffix;target = root/'shared/pilot'/suffix;target.parent.mkdir(parents=True, exist_ok=True)
                expected = spec['pilot_complete_sha256'] if suffix == 'COMPLETE.json' else ledger[portable_path(source)]
                require(pilot.sha256(source) == expected, 'Pilot snapshot source changed')
                if not target.exists():atomic_copy(source, target)
                require(pilot.sha256(target) == expected, 'Pilot snapshot differs')
                outputs.append(target);path_index[portable_path(source)] = portable_path(target)
            write(root/'shared/PATH_INDEX.json', path_index);outputs.append(root/'shared/PATH_INDEX.json')
            original = {r['model']: {v['sample_id']: v for v in
                read_cache(root/'shared/pilot/anchors'/f"{r['model']}_seed_42.json")['value']} for r in m['runs']}
            prior_single = {(r['model'], r['sample_id']): r['clean'] for r in
                pilot.read_json(root/'shared/pilot/GRADIENT_SUMMARIES.json')}
            rows, full_anchors, jobs = [], [], []
            cohort = set(m['pilot_ids']);focus = spec['focus_sample_id']
            source_reference = original[m['runs'][0]['model']]
            # Reconstruct original batches once, authenticate counts/labels/speakers,
            # and retain collated CPU tensors only. No dataset mutation or padding rewrite.
            batches = []
            for ids in m['original_batch_ids']:
                items = []
                for sid in ids:
                    item = data.load(sid);old = source_reference[sid]
                    require(count_hash(item['counts']) == old['input_sha256'] and item['target'] == old['target'] and
                            item['speaker'] == old['speaker'] and item['support_frames'] == old['original_support_frames'],
                            'Reconstructed input/target/support identity differs')
                    items.append(data.model_item(sid, item['counts']))
                batches.append(PadTensors()(items))
            for run in m['runs']: models[run['model']] = load_model(run, device)

            def job(run, mode, backend, layout, key, batch, slots, ids):
                contract = dict(manifest_sha256=identity, checkpoint_sha256=run['checkpoint_sha256'],
                    mode=mode, backend=backend, layout=layout, sample_ids=ids, slots=slots, batch=batch_identity(batch))
                path = root/'jobs'/run['model']/mode/(key+'.json');path.parent.mkdir(parents=True, exist_ok=True)
                progress('forward_matrix', completed_jobs=len(jobs), total_jobs=m['expected_forward_calls'],
                         model=run['model'], mode=mode, layout=layout)
                print(f"Numerical check {len(jobs)+1}/{m['expected_forward_calls']}: {run['model']} {mode} {key}", flush=True)
                if path.exists():
                    saved = read_cache(path);pilot.validate_manifest(saved['identity'], contract);value = saved['value']
                else:
                    value = evaluate(models[run['model']], batch, run['config'], device, slots)
                    write(path, dict(identity=contract, value=value, value_sha256=pilot.digest(value)))
                require(len(value) == len(ids), 'Forward cache coverage differs')
                require_frozen(models[run['model']], run['model_state_sha256'])
                result = []
                for sid, score in zip(ids, value):
                    require(len(score['logits']) == 20 and all(torch.isfinite(torch.tensor(score['logits']))) and
                            0 <= score['prediction'] < 20 and 0 <= score['true_probability'] <= 1 and
                            torch.isfinite(torch.tensor(score['loss'])).item() and score['loss'] >= 0 and
                            score['correct'] == int(score['prediction'] == source_reference[sid]['target']), 'Invalid cached score')
                    result.append(dict(model=run['model'], seed=42, mode=mode, layout=layout, sample_id=sid,
                        target=source_reference[sid]['target'], input_sha256=source_reference[sid]['input_sha256'], **score))
                outputs.append(path);jobs.append(portable_path(path))
                return result

            for mode in spec['precision_modes']:
                with precision_mode(mode, m['source_backend']) as backend:
                    for run in m['runs']:
                        full = []
                        for b, ids in enumerate(m['original_batch_ids']):
                            result = job(run, mode, backend, 'original_batch', f'original_batch_{b}', batches[b], list(range(len(ids))), ids)
                            full += result;rows += [r for r in result if r['sample_id'] in cohort]
                        accuracy = sum(r['correct'] for r in full)/len(full);loss = sum(r['loss'] for r in full)/len(full)
                        source = original[run['model']]
                        mismatches = sum(r['prediction'] != source[r['sample_id']]['prediction'] for r in full)
                        anchor = dict(model=run['model'], mode=mode, samples=len(full), accuracy=accuracy, loss=loss,
                            source_prediction_mismatches=mismatches, accuracy_difference=accuracy-run['validation']['accuracy'],
                            loss_difference=loss-run['validation']['loss'], backend=backend,
                            role='required_original_contract' if mode == MODES[0] else 'diagnostic_only_not_a_replacement_anchor')
                        if mode == MODES[0]:
                            old_spec = pilot.read_json(root/'shared/pilot/MANIFEST.json')['specification']
                            require_original_anchor(anchor, old_spec)
                        full_anchors.append(anchor)
                        for b, ids in enumerate(m['original_batch_ids']):
                            for slot, sid in enumerate(ids):
                                if sid not in cohort:continue
                                for layout in spec['layouts'][1:]:
                                    batch = singleton_from_original(batches[b], slot, source[sid]['stored_frames'], layout.endswith('original_padding'))
                                    rows += job(run, mode, backend, layout, f'{layout}_{sid}', batch, [0], [sid])
                        b = m['focus_original_batch_index'];slot = m['focus_original_slot'];batch = batches[b]
                        controls = [
                            ('original_repeat', batch, slot),
                            ('singleton_repeat', singleton_from_original(batch, slot, source[focus]['stored_frames'], False), 0),
                            ('replicated_original_padding', repeated_target(batch, slot), slot),
                            ('reversed_original', reversed_batch(batch), len(batch[0])-1-slot),
                        ]
                        for layout, control, selected_slot in controls:
                            rows += job(run, mode, backend, layout, layout, control, [selected_slot], [focus])
            require(len(jobs) == 546 and len(rows) == 744, 'Matrix coverage differs')
            comparisons = compare_rows(rows)
            flat_rows = [{k:v for k,v in r.items() if k != 'logits'} for r in rows]
            for name, value in [('SCORES.json', rows), ('FULL_VALIDATION_CHECKS.json', full_anchors), ('JOB_INDEX.json', jobs),
                                ('FROZEN_STATE_CHECKS.json', [dict(model=r['model'], status='unchanged',
                                    model_state_sha256=r['model_state_sha256'], all_parameter_gradients_none=True) for r in m['runs']])]:
                write(root/name, value);outputs.append(root/name)
            pilot.write_csv(root/'SCORES.csv', flat_rows);outputs.append(root/'SCORES.csv')
            pilot.write_csv(root/'COMPARISONS.csv', comparisons);outputs.append(root/'COMPARISONS.csv')
            reproduction = []
            for run in m['runs']:
                selected = [r for r in rows if r['model'] == run['model'] and r['mode'] == MODES[0] and r['layout'] == 'singleton_natural']
                reproduction.append(dict(model=run['model'], records=len(selected),
                    prediction_mismatches=sum(r['prediction'] != prior_single[r['model'], r['sample_id']]['prediction'] for r in selected),
                    max_absolute_loss_difference=max(abs(r['loss']-prior_single[r['model'], r['sample_id']]['loss']) for r in selected)))
            summary = dict(status='diagnostic_executed_requires_review', samples=40, training_seeds=[42],
                layout_rows=720, focus_control_rows=24, full_validation_score_rows=9786, matrix_jobs=len(jobs),
                source_high_original_anchors_passed=True, singleton_pilot_reproduction=reproduction,
                group_comparisons=[dict(model=model, mode=mode, layout=layout, records=len(group),
                    prediction_changes=sum(r['prediction_changed'] for r in group),
                    max_absolute_loss_difference=max(abs(r['loss_difference']) for r in group))
                    for model in spec['models'] for mode in spec['precision_modes'] for layout in spec['layouts'][1:]
                    for group in [[r for r in comparisons if r['model']==model and r['mode']==mode and r['layout']==layout and
                        r['contrast']=='versus_same_mode_original_batch']]],
                focus_comparisons=[r for r in comparisons if r['sample_id']==focus],
                backend_modes=[dict(mode=a['mode'], backend=a['backend']) for a in full_anchors[::2]],
                precision_winner_selected=False, cause_established=False, training=False, backward=False,
                stage7_complete=False, input_optimization=False,
                next_step='Read-only review of batching, padding, repeat/order and precision contrasts before any new gradient study.')
            write(root/'SUMMARY.json', summary);outputs.append(root/'SUMMARY.json')
            text = ['# SHD* batching/precision follow-up', '',
                'User-launched forward-only diagnostic: MT/AMT, seed 42, 40 pilot records; original-batch checks cover all 1,631 validation records.',
                'Source-high original anchors passed unchanged tolerances. Stricter precision results are diagnostic and do not replace saved evidence.',
                'Three precision modes, original / natural singleton / original-padded singleton layouts, plus four controls for recording 3896.',
                'All state hashes unchanged. No gradients, training, official test, inference statistics or automatic precision selection.', '',
                '| Model | Mode | Layout | Prediction changes vs same-mode original | Max absolute CE drift |',
                '|---|---|---|---:|---:|']
            for r in summary['group_comparisons']:
                text.append(f"| {r['model']} | {r['mode']} | {r['layout']} | {r['prediction_changes']} / {r['records']} | {r['max_absolute_loss_difference']:.9g} |")
            text += ['', 'Inspect SUMMARY.json focus comparisons and COMPARISONS.csv for fixed-layout precision and padding contrasts.',
                'Repeat/order controls are limited to recording 3896. Highest precision is not exact arithmetic; this matrix alone does not prove a unique cause.',
                'No gradient maps were regenerated. Stage 7 remains incomplete; review before expanding or changing the gradient protocol.', '']
            (root/'SUMMARY.md').write_text('\n'.join(text), encoding='utf-8');outputs.append(root/'SUMMARY.md')
            write(root/'COMPLETE.json', dict(status='complete', manifest_sha256=identity,
                completed_utc=datetime.now(timezone.utc).isoformat(), attempt_elapsed_seconds=time.monotonic()-started,
                stage7_complete=False, outputs={portable_path(p):pilot.sha256(p) for p in dict.fromkeys(outputs)}))
            progress('complete', registered_outputs=len(verify(root)['outputs']))
            print('Numerical follow-up complete:', portable_path(root/'SUMMARY.md'), flush=True)
        except Exception as error:
            progress('failed', error_type=type(error).__name__, error=str(error));raise
        finally:
            models.clear()
            if torch.cuda.is_available():torch.cuda.empty_cache()

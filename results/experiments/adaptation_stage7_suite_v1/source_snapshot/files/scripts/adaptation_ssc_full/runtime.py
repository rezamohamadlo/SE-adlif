"""User-launched full SSC diagnostics with streamed inputs and paired CSV output."""
import csv
from datetime import datetime, timezone
import time

import torch

from project_paths import portable_path
from scripts.adaptation_analysis import pilot, reporting
from scripts.adaptation_analysis.evaluate import check_anchor
from scripts.adaptation_analysis.records import verify_package
from scripts.adaptation_ssc.inputs import InputSequence, input_rows
from scripts.adaptation_ssc.runtime import check_rows, load_model
from scripts.adaptation_ssc_full.reports import write_reports


def feature_rows(data, ids, condition, study, batch_size):
    """Retain metadata only; release count arrays after each preprocessing batch."""
    result = []
    for start in range(0, len(ids), batch_size):
        batch = input_rows(data, ids[start:start+batch_size], condition, study)
        result.extend(row for _, _, row in batch)
        del batch
    return result


def paired_records(raw, study, seeds):
    """Load two prediction chunks at a time; enforce strict pairing before CSV."""
    for seed in seeds:
        for name, rep, _, _ in study['conditions']:
            both = [pilot.read_json(raw / f'{model}_seed_{seed}_{name}_rep_{rep}.json')['value']
                    for model in (study['parent'], study['variant'])]
            yield from pilot.paired_rows(*both)


def write_paired_csv(path, rows):
    """Write complete evidence while keeping only fields needed by the reports."""
    temporary = path.with_suffix('.csv.tmp')
    retained = []
    fields = {'seed', 'condition', 'replicate', 'sample_id', 'parent_loss', 'variant_loss',
              'benefit', 'parent_correct', 'variant_correct', 'correctness_difference'}
    fields.update(feature for feature, _ in reporting.FEATURES)
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            retained.append({key: row[key] for key in fields})
        if writer is None:
            raise ValueError('Cannot complete full validation without paired records')
    temporary.replace(path)
    return retained


def execute(study, destination, manifest, data):
    device = torch.device(manifest['environment']['device'])
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    with pilot.study_lock(destination / '.analysis.lock'):
        manifest_path = destination / 'MANIFEST.json'
        if manifest_path.exists():
            pilot.validate_manifest(pilot.read_json(manifest_path), manifest)
        elif any(p.name != '.analysis.lock' for p in destination.iterdir()):
            raise ValueError('Destination changed since preflight; preserve it and choose a new root')
        else:
            pilot.write_json(manifest_path, manifest)
        if (destination / 'COMPLETE.json').exists():
            verify_package(destination)
            print('SSC full validation already complete; saved outputs verified.', flush=True)
            return
        started = time.monotonic()
        raw = destination / 'diagnostic_rows'
        raw.mkdir(exist_ok=True)
        ids = manifest['sample_ids']
        if ids != manifest['data']['validation_indices']:
            raise ValueError('Full-validation execution requires every official validation ID in order')
        outputs = [manifest_path]
        identity = pilot.digest(manifest)
        clean = [c for c in study['conditions'] if c[0] == 'clean']
        if len(clean) != 1:
            raise ValueError('Exactly one clean condition is required')
        conditions = clean + [c for c in study['conditions'] if c[0] != 'clean']
        anchors = []
        phase_timings = []

        def progress(phase, **fields):
            pilot.write_json(destination / 'PROGRESS.json', dict(
                status='running', scope='ssc_full_validation', phase=phase, samples=len(ids),
                elapsed_seconds=time.monotonic()-started, completed_phases=phase_timings, **fields))

        # Clean inference is both the first full diagnostic condition and the
        # compatibility gate. Perturbations cannot start until every seed passes.
        for condition in conditions:
            name, rep, _, _ = condition
            condition_started = time.monotonic()
            progress('full_validation_features', condition=name, replicate=rep)
            print(f'SSC full validation: input features for {name}, replicate {rep}, {len(ids)} records', flush=True)
            feature_path = raw / f'input_features_{name}_rep_{rep}.json'
            features = pilot.cached_json(feature_path, dict(manifest_sha256=identity,
                condition=name, replicate=rep, kind='model_independent_full_validation_features'),
                lambda: feature_rows(data, ids, condition, study,
                                     manifest['runs'][0]['config']['dataset']['batch_size']))
            if [r['sample_id'] for r in features] != ids:
                raise ValueError('Feature cache must cover the complete validation split in order')
            feature_cache = {row['sample_id']: row for row in features}
            outputs.append(feature_path)

            for run in manifest['runs']:
                progress('full_validation_inference', model=run['model'], seed=run['seed'],
                         condition=name, replicate=rep)
                print(f"SSC full validation: {run['model']} seed {run['seed']}, {name}, replicate {rep}", flush=True)
                path = raw / f"{run['model']}_seed_{run['seed']}_{name}_rep_{rep}.json"

                def produce():
                    model = load_model(run, device)
                    try:
                        return pilot.infer(model, run['config'], data,
                            InputSequence(data, ids, condition, study, feature_cache), device,
                            f"SSC {run['model']} seed {run['seed']}: {name}/{rep}")
                    finally:
                        del model
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

                rows = pilot.cached_json(path, dict(manifest_sha256=identity,
                    checkpoint_sha256=run['checkpoint_sha256'], condition=name, replicate=rep,
                    kind='full_validation_predictions'), produce)
                check_rows(rows, ids, run, condition)
                if name == 'clean':
                    anchors.append(check_anchor(rows, run, study))
                outputs.append(path)
                del rows
            if name == 'clean':
                anchor_path = destination / 'VALIDATION_ANCHORS.json'
                pilot.write_json(anchor_path, anchors)
                outputs.append(anchor_path)
                print('All full-validation clean anchors passed; perturbation diagnostics may proceed.', flush=True)
            phase_timings.append(dict(condition=name, replicate=rep,
                elapsed_seconds=time.monotonic()-condition_started))
            del features, feature_cache

        progress('full_validation_reports')
        paired_path = raw / 'paired_samples.csv'
        paired = write_paired_csv(paired_path, paired_records(raw, study, manifest['seeds']))
        expected_pairs = len(ids)*len(manifest['seeds'])*len(conditions)
        if len(paired) != expected_pairs:
            raise ValueError('Paired records do not cover every sample, seed and condition')
        outputs.append(paired_path)
        outputs += write_reports(destination, paired, dict(
            analyses=study['analyses'], seeds=study['seeds'], executed_seeds=manifest['seeds'],
            samples=len(ids), full_clean_anchor_samples=len(ids),
            parent=study['parent'], variant=study['variant'], scope='ssc_full_validation',
            selection='all official validation records, original file order and class frequencies',
            split='official_validation', training=False, official_test_opened=False,
            paper_baseline_version=study['paper_baseline_version'], paper_baseline_dataset='SSC',
            loss_agg=study['loss_agg'], evaluation_protocol=study['evaluation_protocol']))
        elapsed = time.monotonic()-started
        pilot.write_json(destination / 'COMPLETE.json', dict(status='complete', scope='ssc_full_validation',
            complete_training_seed_cohort=manifest['seeds'] == study['seeds'],
            manifest_sha256=identity, completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=elapsed, phase_timings=phase_timings, paired_records=expected_pairs,
            outputs={portable_path(p): pilot.sha256(p) for p in outputs}))
        pilot.write_json(destination / 'PROGRESS.json', dict(status='complete', scope='ssc_full_validation',
            elapsed_seconds=elapsed, completed_phases=phase_timings))
        print(f'SSC full-validation diagnostics finished: {portable_path(destination)}', flush=True)

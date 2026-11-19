"""User-launched frozen SSC anchors and bounded paired diagnostics; no training."""
import contextlib
from datetime import datetime, timezone
import io
import time

import torch
from omegaconf import OmegaConf

from project_paths import local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.evaluate import check_anchor
from scripts.adaptation_analysis.records import verify_package
from scripts.adaptation_ssc.inputs import InputSequence, input_rows
from scripts.adaptation_ssc.reports import write_reports


def check_rows(rows, ids, run, condition):
    """Require original order and exact identities, including cached predictions."""
    if [row['sample_id'] for row in rows] != list(ids):
        raise ValueError('SSC prediction coverage/order differs from frozen inputs')
    for row in rows:
        if (row['model'] != run['model'] or row['seed'] != run['seed'] or
                row['condition'] != condition[0] or row['replicate'] != condition[1]):
            raise ValueError('SSC cached prediction identity differs')


def load_model(run, device):
    from models.pl_module import MLPSNN
    torch.manual_seed(run['seed'])
    checkpoint = torch.load(local_path(run['checkpoint'], must_exist=True),
                            weights_only=False, map_location='cpu')
    with contextlib.redirect_stdout(io.StringIO()):
        model = MLPSNN(OmegaConf.create(run['config']))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model


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
            print('SSC pilot already complete; saved outputs verified.', flush=True)
            return
        started = time.monotonic()
        raw = destination / 'diagnostic_rows'
        raw.mkdir(exist_ok=True)
        outputs = [manifest_path]
        identity = pilot.digest(manifest)
        all_ids = manifest['data']['validation_indices']
        pilot_ids = manifest['sample_ids']
        clean = next(c for c in study['conditions'] if c[0] == 'clean')

        def progress(phase, **fields):
            pilot.write_json(destination / 'PROGRESS.json', dict(
                status='running', phase=phase, elapsed_seconds=time.monotonic()-started, **fields))

        # Every full clean anchor must pass before generating pilot diagnostics.
        # Streaming preserves the original full-validation batch boundaries.
        anchors = []
        for run in manifest['runs']:
            print(f"Full SSC clean anchor: {run['model']} seed {run['seed']} ({len(all_ids)} records)", flush=True)
            progress('full_clean_anchor', model=run['model'], seed=run['seed'], samples=len(all_ids))
            path = raw / f"{run['model']}_seed_{run['seed']}_full_validation.json"

            def full_clean():
                model = load_model(run, device)
                try:
                    return pilot.infer(model, run['config'], data,
                        InputSequence(data, all_ids, clean, study), device, 'SSC full clean anchor')
                finally:
                    del model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

            rows = pilot.cached_json(path, dict(manifest_sha256=identity,
                checkpoint_sha256=run['checkpoint_sha256'], kind='full_clean_validation_anchor'), full_clean)
            check_rows(rows, all_ids, run, clean)
            anchors.append(check_anchor(rows, run, study))
            outputs.append(path)
        anchor_path = destination / 'VALIDATION_ANCHORS.json'
        pilot.write_json(anchor_path, anchors)
        outputs.append(anchor_path)
        print('All full-validation clean anchors passed. Starting the fixed pilot subset.', flush=True)

        features = {}
        for condition in study['conditions']:
            name, rep, _, _ = condition
            progress('pilot_features', condition=name, replicate=rep, samples=len(pilot_ids))
            path = raw / f'input_features_{name}_rep_{rep}.json'
            rows = pilot.cached_json(path, dict(manifest_sha256=identity, condition=name, replicate=rep,
                kind='model_independent_pilot_features'),
                lambda: [row for _, _, row in input_rows(data, pilot_ids, condition, study)])
            if [r['sample_id'] for r in rows] != pilot_ids:
                raise ValueError('SSC feature cache coverage/order differs')
            features[name, rep] = {row['sample_id']: row for row in rows}
            outputs.append(path)

        # Pilot batches differ from full-validation batches and are identified
        # separately. Never treat pilot accuracy as a full-validation anchor.
        for run in manifest['runs']:
            model = load_model(run, device)
            try:
                for condition in study['conditions']:
                    name, rep, _, _ = condition
                    print(f"SSC pilot: {run['model']} seed {run['seed']}, {name}, replicate {rep}", flush=True)
                    progress('pilot_inference', model=run['model'], seed=run['seed'],
                             condition=name, replicate=rep, samples=len(pilot_ids))
                    path = raw / f"{run['model']}_seed_{run['seed']}_{name}_rep_{rep}.json"
                    rows = pilot.cached_json(path, dict(manifest_sha256=identity,
                        checkpoint_sha256=run['checkpoint_sha256'], condition=name, replicate=rep,
                        kind='pilot_predictions'),
                        lambda: pilot.infer(model, run['config'], data, InputSequence(
                            data, pilot_ids, condition, study, features[name, rep]), device, 'SSC '+name))
                    check_rows(rows, pilot_ids, run, condition)
                    outputs.append(path)
            finally:
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        progress('pilot_reports', samples=len(pilot_ids))
        paired = []
        for seed in manifest['seeds']:
            for name, rep, _, _ in study['conditions']:
                rows = [pilot.read_json(raw / f'{model}_seed_{seed}_{name}_rep_{rep}.json')['value']
                        for model in (study['parent'], study['variant'])]
                paired.extend(pilot.paired_rows(*rows))
        path = raw / 'paired_samples.csv'
        pilot.write_csv(path, paired)
        outputs.append(path)
        outputs += write_reports(destination, paired, dict(
            analyses=study['analyses'], seeds=study['seeds'], executed_seeds=manifest['seeds'],
            samples=len(pilot_ids), full_clean_anchor_samples=len(all_ids),
            parent=study['parent'], variant=study['variant'], scope='ssc_validation_pilot',
            selection='fixed class-balanced round-robin subset, independent of predictions; original file order',
            split='official_validation', training=False, official_test_opened=False,
            paper_baseline_version=study['paper_baseline_version'], paper_baseline_dataset='SSC',
            loss_agg=study['loss_agg'], evaluation_protocol=study['evaluation_protocol']))
        elapsed = time.monotonic()-started
        pilot.write_json(destination / 'COMPLETE.json', dict(status='complete', scope='ssc_validation_pilot',
            complete_training_seed_cohort=manifest['seeds'] == study['seeds'],
            manifest_sha256=identity, completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=elapsed, outputs={portable_path(p): pilot.sha256(p) for p in outputs}))
        pilot.write_json(destination / 'PROGRESS.json', dict(status='complete', elapsed_seconds=elapsed))
        print(f'SSC pilot finished: {portable_path(destination)}', flush=True)

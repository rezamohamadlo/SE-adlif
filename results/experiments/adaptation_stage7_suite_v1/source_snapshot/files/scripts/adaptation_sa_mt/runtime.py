"""Versioned SA-MT runtime: shared features and reused parent predictions.

Based on the unchanged full-validation runtime. Only SA-MT inference is new.
"""
import contextlib
import io
import time
from datetime import datetime,timezone
from project_paths import local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.evaluate import all_conditions,check_anchor
from scripts.adaptation_analysis.records import verify_package

def evaluate(study, destination, manifest):
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    from scripts.adaptation_analysis.shd_data import SHDValidationData
    from scripts.adaptation_sa_mt.reports import write_reports
    from scripts.adaptation_sa_mt.reuse import import_reuse,reuse_parent
    device = torch.device(manifest['environment']['device'])
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    with pilot.study_lock(destination / '.analysis.lock'):
        manifest_path = destination / 'MANIFEST.json'
        if manifest_path.exists():
            pilot.validate_manifest(pilot.read_json(manifest_path), manifest)
        else:
            pilot.write_json(manifest_path, manifest)
        if (destination / 'COMPLETE.json').exists():
            verify_package(destination)
            print('Already complete and verified.', flush=True)
            return
        raw = destination / 'diagnostic_rows'
        raw.mkdir(exist_ok=True)
        started = time.monotonic()
        data = SHDValidationData(manifest['runs'][0]['config'], sample_ids=manifest['sample_ids'], verify_hash=False)
        anchors = []
        outputs = [manifest_path, import_reuse(destination,manifest)]
        identity = pilot.digest(manifest)
        feature_cache = {}
        for condition in all_conditions(study):
            name, rep, _, _ = condition
            path = raw / f'input_features_{name}_rep_{rep}.json'
            rows = pilot.cached_json(path, dict(manifest_sha256=identity, condition=name, replicate=rep,
                kind='model_independent_input_features'),
                lambda: [row for _, _, row in pilot.input_rows(data, data.sample_ids, condition, study)])
            if len(rows) != len(data) or {r['sample_id'] for r in rows} != set(data.sample_ids):
                raise ValueError('Input features do not cover full validation')
            feature_cache[name, rep] = {row['sample_id']: row for row in rows}
            outputs.append(path)
        # Retain one condition's input tensors at a time. Completed prediction
        # chunks are validated and reused after interruption; weights never train.
        for run in manifest['runs']:
            if run['model']==study['parent']:
                reuse_parent(run,study,manifest,raw,anchors,outputs)
                continue
            torch.manual_seed(run['seed'])
            checkpoint = torch.load(local_path(run['checkpoint']), weights_only=False, map_location='cpu')
            with contextlib.redirect_stdout(io.StringIO()):
                model = MLPSNN(OmegaConf.create(run['config']))
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            del checkpoint
            model.requires_grad_(False).eval().to(device)
            for condition in all_conditions(study):
                name, replicate, _, _ = condition
                print(f"Frozen {run['model']} seed {run['seed']}: {name}, replicate {replicate}", flush=True)
                pilot.write_json(destination / 'PROGRESS.json', dict(status='running', model=run['model'],
                    seed=run['seed'], condition=name, replicate=replicate, samples=len(data), elapsed_seconds=time.monotonic()-started))
                path = raw / f"{run['model']}_seed_{run['seed']}_{name}_rep_{replicate}.json"
                def produce():
                    inputs = pilot.input_rows(data, data.sample_ids, condition, study, features=False)
                    enriched = []
                    for sid, counts, row in inputs:
                        saved = feature_cache[name, replicate][sid]
                        if any(saved[k] != value for k, value in row.items()):
                            raise ValueError('Reconstructed input differs from shared feature cache')
                        enriched.append((sid, counts, saved))
                    return pilot.infer(model, run['config'], data, enriched, device, name)
                rows = pilot.cached_json(path, dict(manifest_sha256=identity, checkpoint_sha256=run['checkpoint_sha256'],
                                                   condition=name, replicate=replicate), produce)
                if len(rows) != len(data) or {r['sample_id'] for r in rows} != set(data.sample_ids):
                    raise ValueError('Cached rows do not cover the full validation split')
                if name == 'clean':
                    anchors.append(check_anchor(rows, run, study))
                outputs.append(path)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        paired = []
        for seed in manifest['seeds']:
            for name, rep, _, _ in all_conditions(study):
                rows = [pilot.read_json(raw / f'{model}_seed_{seed}_{name}_rep_{rep}.json')['value']
                        for model in (study['parent'], study['variant'])]
                paired.extend(pilot.paired_rows(*rows))
        pilot.write_json(destination / 'VALIDATION_ANCHORS.json', anchors)
        pilot.write_csv(raw / 'paired_samples.csv', paired)
        outputs += [destination / 'VALIDATION_ANCHORS.json', raw / 'paired_samples.csv']
        outputs += write_reports(destination, paired, dict(analyses=study['analyses'], seeds=manifest['seeds'],
                    samples=len(data), parent=study['parent'], variant=study['variant'], scope='full_validation',
                    paper_baseline_version=study['baseline_version'], paper_baseline_dataset=study['dataset']))
        pilot.write_json(destination / 'COMPLETE.json', dict(status='complete', manifest_sha256=identity,
            completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            outputs={portable_path(p): pilot.sha256(p) for p in outputs}))
        pilot.write_json(destination / 'PROGRESS.json', dict(status='complete', elapsed_seconds=time.monotonic()-started))
        print(f'Full-validation analysis finished: {portable_path(destination)}', flush=True)

"""Machine-independent full SHD* validation diagnostics. No training or official test access."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from project_paths import local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory, verify_package

STUDY = 'config/studies/adaptation_mechanism_full_validation_v1.yaml'
ANALYSES = ('clean', 'time_scaling', 'spike_deletion')


def load_study():
    definition = yaml.safe_load(local_path(STUDY).read_text(encoding='utf-8'))
    base = yaml.safe_load(local_path(definition['inherits']).read_text(encoding='utf-8'))
    study = dict(base, **definition)
    if study['scope'] != 'full_validation' or study['training'] or study['test_set_access']:
        raise ValueError('Only frozen full-validation diagnostics are supported')
    if study['analyses'] != list(ANALYSES):
        raise ValueError('Analysis definition differs from the registered full-validation study')
    study['conditions'] = [list(c) for c in pilot.conditions('pc2', base) + pilot.conditions('pc3', base)[1:]]
    del study['arms']
    study['samples'] = study['expected_validation_samples']
    study['max_samples'] = study['expected_validation_samples']
    return study


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true', help='Validate prerequisites and manifest; no inference or output writes')
    flags.add_argument('--verify-only', action='store_true', help='Verify completed package; no checkpoints, data or inference required')
    return parser


def full_sample_ids(data, study):
    ids = data['validation_indices']
    if len(ids) != study['expected_validation_samples'] or len(set(ids)) != len(ids):
        raise ValueError('Full validation membership/count differs from the frozen definition')
    return list(ids)


def all_conditions(study):
    # Preserve the pilot condition names and perturbation seed derivation.
    return study['conditions']


def source_identity(study):
    paths = set(pilot.source_identity()) | {STUDY, study['inherits']}
    # LF-normalized source identities survive Git's Windows checkout conversion.
    # Historical pilot records keep their original raw-byte hashes unchanged.
    import hashlib
    return {p: hashlib.sha256(local_path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(paths)}


def prepare(args):
    import torch
    study = load_study()
    destination = experiment_directory(args.output_root or study['output_root'])
    # Reuse the authenticated checkpoint/config/metrics and data checks, while
    # discarding its subset selection. Full membership is selected explicitly.
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        runs, data, _ = pilot.preflight(SimpleNamespace(samples=study['samples'], seeds=study['seeds'], device=args.device), study)
    for line in capture.getvalue().splitlines():
        if not line.startswith('SHD*:'):
            print(line, flush=True)
    ids = full_sample_ids(data, study)
    manifest = dict(version=study['version'], scope='full_validation', analyses=study['analyses'],
                    study_file=STUDY, resolved_study=study, sample_ids=ids, samples=len(ids),
                    seeds=study['seeds'], data=data, runs=runs,
                    paper_baseline_version=study['baseline_version'], paper_baseline_dataset=study['dataset'],
                    source_sha256=source_identity(study), source_hash_encoding='bytes_with_CRLF_normalized_to_LF',
                    environment=pilot.environment(torch.device(args.device)),
                    training=False, official_test_opened=False, ordinary_inference=True,
                    condition_definitions=all_conditions(study))
    if (destination / 'ARTIFACT_INDEX.json').exists() or (destination / 'PILOT_COMPLETE.json').exists():
        raise ValueError('Destination contains pilot evidence; choose a new full-validation root')
    existing = destination / 'MANIFEST.json'
    if existing.exists():
        pilot.validate_manifest(pilot.read_json(existing), manifest)
    elif destination.exists() and any(p.name != '.analysis.lock' for p in destination.iterdir()):
        raise ValueError('Destination is not empty and has no matching manifest')
    print(f'SHD*: ALL {len(ids)} validation samples, {len(study["seeds"])} training seeds, {len(all_conditions(study))} conditions; no official test access.', flush=True)
    return study, destination, manifest


def check_anchor(rows, run, study):
    accuracy = float(np.mean([r['correct'] for r in rows]))
    loss = float(np.mean([r['loss'] for r in rows]))
    expected = run['validation']
    if abs(accuracy - expected['accuracy']) > study['accuracy_tolerance'] or abs(loss - expected['loss']) > study['loss_tolerance']:
        raise ValueError('Full clean validation anchor differs from saved validation evidence')
    return dict(model=run['model'], seed=run['seed'], n=len(rows), accuracy=accuracy, loss=loss,
                expected=expected, accuracy_difference=accuracy-expected['accuracy'], loss_difference=loss-expected['loss'])


def evaluate(study, destination, manifest):
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    from scripts.adaptation_analysis.shd_data import SHDValidationData
    from scripts.adaptation_analysis.analysis_reporting import write_reports
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
        outputs = [manifest_path]
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify_only:
        paths = verify_package(args.output_root or load_study()['output_root'])
        print('Verified:', {k: len(v) for k, v in paths.items()})
        return 0
    study, destination, manifest = prepare(args)
    if args.check_only:
        print(f'Preflight passed. No inference or output writes. Destination: {portable_path(destination)}')
        return 0
    evaluate(study, destination, manifest)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)

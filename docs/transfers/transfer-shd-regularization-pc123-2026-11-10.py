"""Non-training PC1/PC3 import and pre-import PC2 export; immutable hash checks."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
import torch
import extended_seed_study as evidence
from project_paths import portable_config, portable_path
import shd_speaker_holdout as speaker
import shd_speaker_regularization as study

PC1 = Path('G:/reza/pc1')
PC3 = Path('G:/reza/SE-adlif-2026-11-10')
PC2 = Path('G:/reza/pc2')
PROOF = 'results/provenance/pc3-import-pc1-export-2026-11-10.json'
MANIFEST = 'results/provenance/shd-regularization-pc123-transfer-2026-11-10.json'
SNAPSHOT = 'results/provenance/shd-regularization-pc123-2026-11-10'
STUDY = 'results/experiments/shd_speaker_regularization_v1'
TESTS = {
    'pc1': 'results/experiments/shd_regularization_test_seed42_v1',
    'pc2': 'results/experiments/shd_speaker_regularization_tests_v1',
    'pc3': 'results/experiments/shd_speaker_regularization_test_v1',
}
JOBS = {
    'pc1': [(p, m, 42) for p in study.PROFILES for m in speaker.MODELS],
    'pc2': [(study.PROFILES[0], m, seed) for seed in (123, 456) for m in speaker.MODELS]
           + [(study.PROFILES[1], m, 123) for m in speaker.MODELS],
    'pc3': [(study.PROFILES[1], m, 456) for m in speaker.MODELS]
           + [(study.PROFILES[2], m, seed) for seed in (123, 456) for m in speaker.MODELS],
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Artifact escaped explicit root: {relative}')
    return path


def external(root, recorded):
    relative = Path(portable_path(recorded))
    if relative.is_absolute():
        raise ValueError(f'Unmapped external artifact: {recorded}')
    return inside(root, relative)


def entry(root, path):
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def payload(root, prefix):
    return [entry(root, path) for path in sorted(inside(root, prefix).rglob('*'))
            if path.is_file() and path.suffix not in ('.lock', '.pyc', '.tmp', '.temp')
            and '__pycache__' not in path.parts]


def check_file(root, item):
    path = inside(root, item['path'])
    if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
        raise ValueError(f'File size/hash differs: {root} / {item["path"]}')


def checkpoint_config(state):
    return portable_config(OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']), resolve=True))


def audit_run(root, owner, job, frozen_split):
    profile, model, seed = job
    relative = f'{STUDY}/{profile}/{model}/SHD/seed_{seed}'
    directory = inside(root, relative)
    config = portable_config(read(directory / 'FULL_RUN_CONFIG.json'))
    record = read(directory / 'VALIDATION_RESULTS.json')
    evidence.validate_config(config)
    evidence.check_record(directory, record, config)
    evidence.read_provenance(directory)
    if (record['model'] != model or record['seed'] != seed
            or config['shd_regularization_study']['profile'] != profile
            or read(directory / 'SPLIT_MANIFEST.json') != frozen_split
            or not (directory / 'validation_completed.txt').is_file()):
        raise ValueError(f'Wrong run identity/split/completion: {relative}')
    selected = external(root, record['best_checkpoint'])
    last_path = external(root, record['last_checkpoint'])
    states = [torch.load(path, map_location='cpu', weights_only=False) for path in (selected, last_path)]
    for state in states:
        saved, expected = checkpoint_config(state), copy.deepcopy(config)
        saved.pop('ckpt_path', None)
        expected.pop('ckpt_path', None)
        if saved != expected:
            raise ValueError(f'Checkpoint settings differ from saved run: {relative}')
    best_state, last = states
    callbacks = [s for s in last.get('callbacks', {}).values() if isinstance(s, dict) and s.get('monitor') == 'val_acc']
    if (best_state['epoch'] != record['selected_epoch'] or last['epoch'] + 1 != record['completed_epochs']
            or last['epoch'] + 1 != 300 or not last.get('optimizer_states') or not last.get('lr_schedulers')
            or len(callbacks) != 1
            or external(root, callbacks[0]['best_model_path']) != selected
            or not math.isclose(float(callbacks[0]['best_model_score']), record['best_validation_accuracy'], abs_tol=1e-7)):
        raise ValueError(f'Checkpoint selection/budget/resume state differs: {relative}')
    test_dir = inside(root, f'{TESTS[owner]}/{profile}/{model}/SHD/seed_{seed}')
    test = read(test_dir / 'TEST_RESULTS.json')
    test_checkpoint = external(root, test['checkpoint'])
    selected_hash = sha(selected)
    if sha(test_checkpoint) != selected_hash:
        raise ValueError(f'Test checkpoint differs from validation selection: {relative}')
    if test.get('checkpoint_sha256') and test['checkpoint_sha256'] != selected_hash:
        raise ValueError(f'Test checkpoint hash disagrees: {relative}')
    for key in ('test_acc', 'test_loss'):
        if not math.isfinite(float(test['metrics'][key])):
            raise ValueError(f'Nonfinite test result: {relative}')
    if not 0 <= test['metrics']['test_acc'] <= 1:
        raise ValueError(f'Invalid test accuracy: {relative}')
    if owner == 'pc1' and read(test_dir / 'SOURCE_VALIDATION_RESULTS.json') != record:
        raise ValueError(f'PC1 copied validation selection differs: {relative}')
    return dict(owner=owner, profile=profile, model=model, seed=seed,
                validation_record=f'{relative}/VALIDATION_RESULTS.json',
                selected_checkpoint=portable_path(record['best_checkpoint']),
                checkpoint_sha256=selected_hash, selected_epoch=record['selected_epoch'],
                completed_epochs=record['completed_epochs'],
                best_validation_accuracy=record['best_validation_accuracy'],
                test_record=f'{TESTS[owner]}/{profile}/{model}/SHD/seed_{seed}/TEST_RESULTS.json',
                test_accuracy=test['metrics']['test_acc'])


def preflight_copy(source, destination, files, mapping=None):
    for item in files:
        check_file(source, item)
        target = inside(destination, (mapping or {}).get(item['path'], item['path']))
        if target.exists() and (not target.is_file() or sha(target) != item['sha256']):
            raise ValueError(f'Different existing destination retained: {target}')


def copy_verified(source, destination, files, mapping=None):
    preflight_copy(source, destination, files, mapping)
    copied = matched = 0
    for item in files:
        relative = (mapping or {}).get(item['path'], item['path'])
        target = inside(destination, relative)
        if target.exists():
            matched += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inside(source, item['path']), target)
            copied += 1
        check_file(destination, dict(item, path=relative))
    return dict(copied=copied, matched=matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    frozen_split = read(ROOT / 'results/experiments/shd_speaker_augmented_v1/SPLIT_MANIFEST.json')
    proof = read(PC1 / PROOF)
    incoming = {'pc1': proof['pc1_export_files'], 'pc3': proof['pc3_import_files']}
    for owner, source in (('pc1', PC1), ('pc3', PC3)):
        for item in incoming[owner]:
            check_file(source, item)
    audited = []
    for owner, source in (('pc1', PC1), ('pc2', ROOT), ('pc3', PC3)):
        for job in JOBS[owner]:
            audit = audit_run(source, owner, job, frozen_split)
            audited.append(audit)
            print(f'Verified {owner}: {job[0]} / {job[1]} / seed {job[2]}', flush=True)
    # Freeze PC2's own six runs before any incoming results enter the checkout.
    own = payload(ROOT, STUDY) + payload(ROOT, TESTS['pc2'])
    own_runs = {p.parent.relative_to(ROOT).as_posix() for p in (ROOT / STUDY).rglob('VALIDATION_RESULTS.json')}
    expected = {f'{STUDY}/{p}/{m}/SHD/seed_{s}' for p, m, s in JOBS['pc2']}
    if own_runs != expected:
        raise ValueError('PC2 export must be taken before import and contain exactly its six assigned runs')
    mappings = {'pc1': {}, 'pc3': {f'{STUDY}/VALIDATION_COMPARISON.md': f'{SNAPSHOT}/pc3/VALIDATION_COMPARISON.md'}}
    preflight_copy(ROOT, PC2, own)
    for owner, source in (('pc1', PC1), ('pc3', PC3)):
        preflight_copy(source, ROOT, incoming[owner], mappings[owner])
    preflight_copy(PC1, ROOT, [entry(PC1, PC1 / PROOF)])
    print(json.dumps(dict(audited_runs=len(audited), pc2_export_files=len(own),
                          pc1_import_files=len(incoming['pc1']), pc3_import_files=len(incoming['pc3']))), flush=True)
    if args.dry_run:
        print('No files copied; no training or inference performed.')
        return
    with evidence.study_lock(ROOT / STUDY):
        exported = copy_verified(ROOT, PC2, own)
        imported = {}
        for owner, source in (('pc1', PC1), ('pc3', PC3)):
            imported[owner] = copy_verified(source, ROOT, incoming[owner], mappings[owner])
        copy_verified(PC1, ROOT, [entry(PC1, PC1 / PROOF)])
        # Inspect all imported checkpoints again using normal repository path resolution.
        for owner in ('pc1', 'pc2', 'pc3'):
            for profile, model, seed in JOBS[owner]:
                evidence.read_completed(ROOT / STUDY / profile / model / 'SHD' / f'seed_{seed}')
        study.write_summary(frozen_split)
        transfer = dict(schema='shd_regularization_pc123_transfer_v1',
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        sources=dict(pc1=PC1.as_posix(), pc3=PC3.as_posix()),
                        pc2_export=PC2.as_posix(), originals_retained=True,
                        results_verified=audited, pc2_export_files=own,
                        incoming_files=incoming, destination_mappings=mappings,
                        copied=imported, exported=exported,
                        notes=['All payload sizes and SHA-256 hashes verified.',
                               'Training and exploratory test records retain their original bytes and paths.',
                               'PC3 partial aggregate preserved as a provenance snapshot.',
                               'PC2 bundle captured before import; it contains PC2 jobs only.',
                               'Combined validation aggregate uses validation scores only; tests remain exploratory.'])
        path = ROOT / MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError('Transfer receipt already exists; preserving the previous receipt')
        evidence.atomic_json(path, transfer)
        copy_verified(ROOT, PC2, [entry(ROOT, path)])
        for source, files in ((PC1, incoming['pc1']), (PC3, incoming['pc3'])):
            for item in files:
                check_file(source, item)
        for item in own:
            check_file(PC2, item)
        print(json.dumps(dict(imported=imported, pc2_export=exported,
                              receipt=MANIFEST, all_runs_local=18)), flush=True)


if __name__ == '__main__':
    main()

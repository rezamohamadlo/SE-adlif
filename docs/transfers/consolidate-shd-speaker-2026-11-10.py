"""Stage all speaker artifacts in the combined folder; deletion is separately guarded."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import portable_path

COMBINED = ROOT / 'results/experiments/shd_speaker_combined_v1'
NAMES = ('shd_speaker_augmented_v1', 'shd_speaker_regularization_v1',
         'shd_regularization_test_seed42_v1', 'shd_speaker_regularization_tests_v1',
         'shd_speaker_regularization_test_v1', 'shd_speaker_augmented_tta_v1')
RECEIPT = ROOT / 'results/provenance/shd-speaker-consolidation-2026-11-10.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def copy_checked(source, target):
    if target.exists():
        if sha(target) != sha(source):
            raise ValueError(f'Different target retained: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if source.stat().st_size != target.stat().st_size or sha(source) != sha(target):
        raise ValueError('Consolidation copy failed verification')


def main():
    if RECEIPT.exists():
        raise ValueError('Existing consolidation receipt retained')
    inventory, disposable, aliases, history = [], [], {}, []
    for name in NAMES:
        source = (ROOT / 'results/experiments' / name).resolve()
        target = (COMBINED / 'artifacts' / name).resolve()
        if source.parent != ROOT / 'results/experiments' or not target.is_relative_to(COMBINED):
            raise ValueError('Consolidation target escaped workspace')
        if not source.is_dir():
            raise FileNotFoundError(source)
        aliases[source.relative_to(ROOT).as_posix()] = target.relative_to(ROOT).as_posix()
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            if path.suffix in ('.lock', '.pyc', '.tmp', '.temp') or '__pycache__' in path.parts:
                disposable.append(path.relative_to(ROOT).as_posix())
                continue
            destination = target / path.relative_to(source)
            inventory.append(dict(source=path.relative_to(ROOT).as_posix(),
                                  destination=destination.relative_to(ROOT).as_posix(),
                                  bytes=path.stat().st_size, sha256=sha(path)))
    # Preserve the original derived view before it is reindexed to canonical paths.
    old_view = [p for p in COMBINED.iterdir() if p.is_file()]
    old_view += list(COMBINED.glob('*/*/SHD/seed_*/RESULT.json'))
    for source in old_view:
        target = COMBINED / 'provenance/pre_consolidation' / source.relative_to(COMBINED)
        copy_checked(source, target)
        history.append(dict(source=source.relative_to(ROOT).as_posix(),
                            archived=target.relative_to(ROOT).as_posix(), sha256=sha(source)))
    for item in inventory:
        copy_checked(ROOT / item['source'], ROOT / item['destination'])
    shuffled = []
    external = Path('G:/reza/SE-adlif-2026-11-10')
    for model in ('DTH_SE_adLIF', 'LG_DTH_SE_adLIF'):
        for seed in (42, 123, 456):
            relative = Path('results/valid' if model == 'DTH_SE_adLIF' else 'results/not_improved') / model / 'SHD' / f'seed_{seed}'
            origin = external / relative
            cost, local_cost = read(origin / 'COST_RESULTS.json'), read(ROOT / relative / 'COST_RESULTS.json')
            if cost['checkpoint_sha256'] != local_cost['checkpoint_sha256'] or cost['best_epoch'] != local_cost['best_epoch']:
                raise ValueError('Shuffled source differs from the recorded reference checkpoint')
            with (origin / 'logs/mlp_snn/version_0/metrics.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            val = [r for r in rows if r.get('val_acc')]
            if [int(float(r['epoch'])) for r in val] != list(range(300)):
                raise ValueError('Shuffled validation history is incomplete/mixed')
            selected = [r for r in val if int(float(r['epoch'])) == cost['best_epoch']]
            if len(selected) != 1 or float(selected[0]['val_acc']) != max(float(r['val_acc']) for r in val):
                raise ValueError('Shuffled selection does not match its validation history')
            tests = {float(r['test_acc']) for r in rows if r.get('test_acc')}
            if len(tests) != 1 or abs(next(iter(tests)) - local_cost['measurements']['reproduced_test_acc']) > 1e-6:
                raise ValueError('Shuffled test history disagrees with the accepted result')
            destination = COMBINED / 'comparison_evidence/shuffled' / model / f'seed_{seed}'
            for source, filename in ((origin / 'logs/mlp_snn/version_0/metrics.csv', 'metrics.csv'),
                                     (origin / 'FULL_RUN_CONFIG.json', 'FULL_RUN_CONFIG.json'),
                                     (origin / 'COST_RESULTS.json', 'COST_RESULTS.json')):
                target = destination / filename
                copy_checked(source, target)
                shuffled.append(dict(external_source=source.as_posix(), destination=target.relative_to(ROOT).as_posix(),
                                     bytes=source.stat().st_size, sha256=sha(source)))
    record = dict(schema='shd_speaker_consolidation_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  aliases=aliases, files=inventory, excluded_disposable_files=disposable,
                  pre_consolidation_view=history, shuffled_evidence=shuffled,
                  original_folders_removed=False,
                  note='Original checkpoint and result bytes retained; aliases preserve legacy identities.')
    RECEIPT.write_bytes((json.dumps(record, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps(dict(staged_files=len(inventory), archived_view_files=len(history),
                          shuffled_evidence_files=len(shuffled), aliases=aliases)))


if __name__ == '__main__':
    main()

"""Audit a results backup, then import absent files only; never train or overwrite.

Use --source <external-results-root> to record the plan. Review its manifest,
then use --apply with the same source. Recorded paths use project_paths aliases.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import local_path, portable_config, portable_path

MANIFEST = ROOT / 'results/provenance/reza-full-results-sync-2026-11-10.json'
LOCAL = ROOT / 'results'
HASHES = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    stamp = path.stat()
    key = (str(path), stamp.st_size, stamp.st_mtime_ns)
    if key not in HASHES:
        with path.open('rb') as stream:
            HASHES[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return HASHES[key]


def disposable(path):
    return (path.suffix.lower() in ('.lock', '.pyc', '.tmp', '.temp')
            or '__pycache__' in path.parts)


def inventory(root):
    found = []
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in dirs + files:
            if (Path(directory) / name).lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f'Reparse point requires separate review: {directory}/{name}')
        for name in files:
            path = Path(directory) / name
            if path == MANIFEST:
                continue
            stamp = path.stat()
            rel = path.relative_to(root).as_posix()
            canonical = portable_path('results/' + rel)
            target = local_path(canonical)
            if not target.is_relative_to(LOCAL.resolve()):
                raise ValueError(f'Alias escaped results: {rel}')
            found.append(dict(path=rel, canonical=canonical, bytes=stamp.st_size,
                              mtime_ns=stamp.st_mtime_ns, disposable=disposable(path)))
    return sorted(found, key=lambda item: item['path'])


def seed_root(canonical):
    parts = Path(canonical).parts
    indices = [i for i, part in enumerate(parts) if re.fullmatch(r'seed_\d+', part)]
    return Path(*parts[:indices[-1] + 1]).as_posix() if indices else None


def equivalence(a, b):
    if sha(a) == sha(b):
        return 'identical'
    if a.suffix.lower() in ('.ckpt', '.pt', '.pth'):
        return 'different'
    first, second = a.read_bytes(), b.read_bytes()
    if first.replace(b'\r\n', b'\n') == second.replace(b'\r\n', b'\n'):
        return 'newline_only'
    if a.suffix.lower() == '.json':
        try:
            one = portable_config(json.loads(first.decode('utf-8-sig')))
            two = portable_config(json.loads(second.decode('utf-8-sig')))
            if one == two:
                return 'json_format_or_registered_paths_only'
        except (ValueError, UnicodeError):
            pass
    return 'different'


def save(data):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def checkpoint_requirements(local_inventory):
    expected = defaultdict(set)
    record_names = {'COST_RESULTS.json', 'REFERENCE_RESULT.json', 'VALIDATION_RESULTS.json',
                    'TEST_RESULTS.json', 'RESULT.json'}
    def visit(value, record):
        if isinstance(value, dict):
            digest = value.get('checkpoint_sha256') or value.get('best_checkpoint_sha256')
            name = value.get('checkpoint') or value.get('best_checkpoint')
            if isinstance(name, str) and isinstance(digest, str) and re.fullmatch('[0-9a-fA-F]{64}', digest):
                if name.replace('\\', '/').startswith(('ckpt/', './ckpt/')) or '/' not in name:
                    target = record.parent / name
                else:
                    target = local_path(name)
                if target.resolve().is_relative_to(LOCAL.resolve()):
                    expected[portable_path(target)].add(digest.lower())
            for child in value.values():
                visit(child, record)
        elif isinstance(value, list):
            for child in value:
                visit(child, record)
    for item in local_inventory:
        p = LOCAL / item['path']
        if (p.name in record_names and seed_root(item['canonical'])
                and 'measurement_history' not in p.parts and 'provenance' not in p.parts):
            visit(json.loads(p.read_text(encoding='utf-8-sig')), p)
    return expected


def plan(source):
    if MANIFEST.exists():
        raise ValueError('Existing transfer receipt retained; inspect it instead of replacing it')
    local, external = inventory(LOCAL), inventory(source)
    local_by, external_by = defaultdict(list), defaultdict(list)
    for item in local:
        local_by[item['canonical']].append(item)
    for item in external:
        external_by[item['canonical']].append(item)
    local_only = []
    for item in local:
        if item['canonical'] not in external_by:
            local_only.append(dict(**item, sha256=sha(LOCAL / item['path'])))
    shared, missing, collisions, skipped = [], [], [], []
    for canonical, candidates in sorted(external_by.items()):
        item = candidates[0]
        if len(candidates) > 1 and len({sha(source / c['path']) for c in candidates}) != 1:
            collisions.append(dict(canonical=canonical, sources=candidates))
            continue
        if item['disposable']:
            skipped.append(item)
            continue
        local_candidates = local_by.get(canonical, [])
        if local_candidates:
            for current in local_candidates:
                external_path, local_file = source / item['path'], LOCAL / current['path']
                shared.append(dict(source=item['path'], local=current['path'], canonical=canonical,
                                   status=equivalence(external_path, local_file),
                                   external_sha256=sha(external_path), local_sha256=sha(local_file)))
        else:
            missing.append(item)
    identity_names = {'FULL_RUN_CONFIG.json', 'REFERENCE_RESULT.json', 'VALIDATION_RESULTS.json',
                      'SWEEP_CONFIG.json', 'RUN_MANIFEST.json', 'metrics.csv'}
    blocked = defaultdict(list)
    for item in shared:
        if item['status'] != 'different':
            continue
        seed = seed_root(item['canonical'])
        if seed and (Path(item['canonical']).name in identity_names
                     or Path(item['canonical']).suffix.lower() == '.ckpt'):
            blocked[seed].append(item['canonical'])
    for item in missing:
        seed = seed_root(item['canonical'])
        item['blocked_by_run_conflict'] = blocked.get(seed, [])
    data = dict(schema='missing_results_import_v1', created_utc=now(), status='planned',
                external_root=source.as_posix(), local_root='results', training_launched=False,
                source_modified=False, overwrite_existing=False,
                local_inventory=local, external_inventory=external,
                local_only=local_only,
                raw_local_only=[i['path'] for i in local if i['path'] not in {e['path'] for e in external}],
                shared=shared, missing=missing, source_collisions=collisions,
                skipped_disposable=skipped, blocked_runs=dict(blocked), copied=[])
    save(data)
    print(json.dumps(dict(local_files=len(local), external_files=len(external),
                          local_only=len(local_only), local_only_checkpoints=sum(i['path'].endswith('.ckpt') for i in local_only),
                          shared_statuses=dict(Counter(i['status'] for i in shared)),
                          missing_files=len(missing), missing_checkpoints=sum(i['path'].endswith('.ckpt') for i in missing),
                          missing_bytes=sum(i['bytes'] for i in missing),
                          blocked_runs=dict(blocked), source_collisions=collisions,
                          manifest=portable_path(MANIFEST)), indent=2), flush=True)


def apply(source):
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if data['external_root'] != source.as_posix() or data['status'] != 'planned':
        raise ValueError('Source/transfer state changed; refuse to replay or overwrite the receipt')
    for collision in data['source_collisions']:
        direct = [i for i in collision['sources'] if 'results/' + i['path'] == collision['canonical']]
        if len(direct) != 1 or local_path(collision['canonical']).exists():
            continue
        # A canonical source and an older alias are distinct historical versions.
        # Import the canonical file and retain the alias version as provenance.
        data['missing'].append(dict(**direct[0], blocked_by_run_conflict=[]))
        for item in collision['sources']:
            if item is direct[0]:
                continue
            archived = dict(item)
            archived['canonical'] = ('results/provenance/reza-full-results-sync-2026-11-10/'
                                     'source_variants/' + item['path'])
            archived['blocked_by_run_conflict'] = []
            data['missing'].append(archived)
        collision['resolution'] = 'canonical file imported; distinct legacy version preserved under provenance'
    expected = checkpoint_requirements(data['local_inventory'])
    checks = []
    for item in data['missing']:
        if item['blocked_by_run_conflict'] or not item['path'].lower().endswith('.ckpt'):
            continue
        if item['canonical'] in expected:
            digest = sha(source / item['path'])
            accepted = expected[item['canonical']]
            matches = len(accepted) == 1 and digest in accepted
            checks.append(dict(checkpoint=item['canonical'], sha256=digest,
                               expected_sha256=sorted(accepted), matches=matches))
            if not matches:
                key = seed_root(item['canonical']) or item['canonical']
                data['blocked_runs'].setdefault(key, []).append('checkpoint SHA-256 disagrees with local evidence')
        if item['canonical'] in expected and len(checks) % 25 == 0:
            print(f'Checked saved identities for {len(checks)} missing checkpoints', flush=True)
    for item in data['missing']:
        key = seed_root(item['canonical']) or item['canonical']
        item['blocked_by_run_conflict'] = data['blocked_runs'].get(key, item['blocked_by_run_conflict'])
    data['checkpoint_identity_checks'] = checks
    eligible = [i for i in data['missing'] if not i['blocked_by_run_conflict']]
    if shutil.disk_usage(LOCAL).free < sum(i['bytes'] for i in eligible) + 1024**3:
        raise ValueError('Insufficient free disk space')
    for current in data['local_inventory']:
        p = LOCAL / current['path']
        s = p.stat()
        if (s.st_size, s.st_mtime_ns) != (current['bytes'], current['mtime_ns']):
            raise ValueError(f'Local file changed after audit: {p}')
    for item in eligible:
        p = source / item['path']
        s = p.stat()
        if (s.st_size, s.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError(f'Backup file changed after audit: {p}')
        target = local_path(item['canonical'])
        if target.exists():
            raise FileExistsError(f'Target appeared after audit: {target}')
        if not target.is_relative_to(LOCAL.resolve()):
            raise ValueError('Target escaped results')
    data['status'] = 'copying'
    data['started_utc'] = now()
    save(data)
    try:
        for index, item in enumerate(eligible, 1):
            p = source / item['path']
            target = local_path(item['canonical'])
            target.parent.mkdir(parents=True, exist_ok=True)
            before = p.stat()
            # Exclusive creation prevents replacement of an existing result.
            with p.open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Source changed during copy: {p}')
            source_hash, destination_hash = sha(p), sha(target)
            if source_hash != destination_hash or after.st_size != target.stat().st_size:
                raise ValueError(f'Copied bytes failed verification: {target}')
            shutil.copystat(p, target)
            data['copied'].append(dict(source=item['path'], destination=item['canonical'],
                                       bytes=after.st_size, sha256=source_hash))
            if index % 100 == 0 or index == len(eligible):
                save(data)
                print(f'Copied and SHA-256 verified {index}/{len(eligible)} files', flush=True)
        for item in data['shared']:
            if sha(LOCAL / item['local']) != item['local_sha256']:
                raise ValueError(f'Existing local evidence changed: {item["local"]}')
        for item in data['local_only']:
            if sha(LOCAL / item['path']) != item['sha256']:
                raise ValueError(f'Local-only evidence changed: {item["path"]}')
        unresolved = [c for c in data['source_collisions'] if not c.get('resolution')]
        data['status'] = 'completed_with_conflicts_preserved' if data['blocked_runs'] or unresolved else 'completed'
        data['completed_utc'] = now()
        data['summary'] = dict(copied_files=len(data['copied']),
                               copied_checkpoints=sum(i['destination'].endswith('.ckpt') for i in data['copied']),
                               copied_bytes=sum(i['bytes'] for i in data['copied']),
                               local_only_preserved=len(data['local_only']),
                               differing_existing_preserved=sum(i['status']=='different' for i in data['shared']),
                               blocked_missing_files=sum(bool(i['blocked_by_run_conflict']) for i in data['missing']))
        save(data)
        print(json.dumps(data['summary'], indent=2), flush=True)
    except BaseException as error:
        data['status'] = 'interrupted_or_failed'
        data['error'] = str(error)
        save(data)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    backup = args.source.resolve(strict=True)
    if backup == LOCAL.resolve() or backup.is_relative_to(LOCAL.resolve()) or LOCAL.resolve().is_relative_to(backup):
        raise ValueError('External source must be separate from local results')
    (apply if args.apply else plan)(backup)

"""User-run, CPU-only finalization of authenticated temporal-permutation caches.

No model imports, inference, checkpoint loading, input transforms or training.
The original failed run is read-only; repaired reports use a separate root.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_temporal_permutation.run import digest, read, require, sha, source_identity
from scripts.adaptation_temporal_permutation.source import FrozenSources

DEFAULT_SOURCE = 'results/experiments/adaptation_temporal_permutation_full_v1'
DEFAULT_OUTPUT = 'results/experiments/adaptation_temporal_permutation_finalized_v1'
TEST = 'tests/test_finalize_temporal_permutation.py'
SIDES = ('clean_parent', 'clean_variant', 'permuted_parent', 'permuted_variant')
FIELDS = ['delta_benefit'] + [side + '_' + metric for side in SIDES for metric in ('loss', 'accuracy')]


def replace_file(temporary, destination):
    for attempt in range(12):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 11:
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.))


def atomic_write(path, produce):
    """Bounded-memory UTF-8 output; retain failed temp, never truncate destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
        result = produce(stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(name, path)
    return result


def write_json(path, value):
    def produce(stream):
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    atomic_write(path, produce)


def write_rows(path, rows):
    def produce(stream):
        count = 0
        for row in rows:
            json.dump(row, stream, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
            count += 1
        return count
    return atomic_write(path, produce)


@contextmanager
def held_lock(path):
    path = Path(path)
    with path.open('r+b' if path.exists() else 'w+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def scoped(path):
    p = local_path(path)
    require(p.parent == ROOT / 'results/experiments' and
            p.name.startswith('adaptation_temporal_permutation_'), 'Unscoped temporal-permutation root')
    return p


def original_ids(m, run):
    return m['dataset_metadata'][run['dataset']]['validation_indices']


def batch_info(m, run, number):
    ids = original_ids(m, run)
    size = run['config']['dataset']['batch_size']
    batch = ids[number * size:(number + 1) * size]
    selected = set(m['cohorts'][run['dataset']])
    return batch, [sid for sid in batch if sid in selected]


def score_path(source, run, condition, number):
    return source / 'scores' / run['dataset'] / run['model'] / f"seed_{run['seed']}" / condition[0] / f'batch_{number:04d}.json'


def cache_rows(source, m, run, condition, number, expected_sha=None):
    p = score_path(source, run, condition, number)
    require(p.is_file(), 'Missing saved inference chunk: ' + portable_path(p))
    raw = p.read_bytes()
    if expected_sha is not None:
        require(hashlib.sha256(raw).hexdigest() == expected_sha, 'Source chunk changed: ' + portable_path(p))
    record = json.loads(raw)
    batch, wanted = batch_info(m, run, number)
    contract = dict(manifest_sha256=digest(m), checkpoint=run['checkpoint_sha256'],
                    original_batch_ids=batch, selected_ids=wanted, condition=condition)
    require(record['identity'] == contract, 'Cache identity mismatch: ' + portable_path(p))
    rows = record['value']
    require(record['value_sha256'] == digest(rows), 'Cache payload checksum mismatch: ' + portable_path(p))
    require([r['sample_id'] for r in rows] == wanted, 'Cache coverage/order mismatch')
    for row in rows:
        require(math.isfinite(row['loss']) and math.isfinite(row['accuracy']) and row['loss'] >= 0 and
                row['valid_steps'] == len(row['predictions']) > 0 and
                0 <= row['correct_steps'] <= row['valid_steps'] and
                row['accuracy'] == row['correct_steps'] / row['valid_steps'], 'Invalid saved score')
        require(isinstance(row['transformation'], dict), 'Missing saved transformation audit')
    return rows


def conditions(spec):
    return [['clean', 0, 0]] + [[f'width_{w}_rep_{r}', w, r]
        for w in spec['window_widths'] for r in range(spec['permutation_replicates'])]


def descriptors(m):
    for run in m['runs']:
        for condition in conditions(m['specification']):
            size = run['config']['dataset']['batch_size']
            for number in range(math.ceil(len(original_ids(m, run)) / size)):
                if batch_info(m, run, number)[1]:
                    yield run, condition, number


def own_sources():
    return {p: sha(local_path(p, must_exist=True)) for p in
            ('scripts/finalize_temporal_permutation.py', TEST)}


def inspect_source(source, output):
    """Read-only artifact/identity checks. No means, paired results or output writes."""
    source, output = scoped(source), scoped(output)
    require(source != output, 'Finalization must not overwrite the source run')
    m = read(source / 'MANIFEST.json')
    require(m['phase'] == 'full' and not m['training'] and not m['official_test_opened'], 'Wrong source scope')
    require(local_path(m['output_root']) == source, 'Source output identity differs')
    spec = m['specification']
    require(spec['window_widths'] == [5, 10, 20] and spec['permutation_replicates'] == 3 and
            spec['training_seeds'] == [42, 123, 456], 'Registered statistical contract differs')
    require(source_identity(spec) == m['extension_source_sha256'], 'Frozen extension sources changed')
    FrozenSources(m['scientific_source_revision']).authenticate(
        {p: v['expected'] for p, v in m['scientific_source_audit'].items()})
    expected_runs = {(ds, model, seed) for ds in m['cohorts'] for seed in spec['training_seeds']
        for model in {p[k] for p in spec['pairs'] for k in ('parent', 'variant')}}
    require({(r['dataset'], r['model'], r['seed']) for r in m['runs']} == expected_runs and
            len(m['runs']) == len(expected_runs), 'Run coverage differs')
    for ds, ids in m['cohorts'].items():
        require(ids == m['dataset_metadata'][ds]['validation_indices'] and len(ids) == spec['validation_sizes'][ds],
                'Finalizer requires the exact full-validation cohorts')
    hashes = {}
    for name in ['MANIFEST.json', 'ANCHORS.json', 'CONTRACT_TESTS.txt', 'RUNTIME.json', 'FAILED.json']:
        p = source / name
        require(p.is_file(), 'Required failed-run evidence missing: ' + name)
        hashes[portable_path(p)] = sha(p)
    anchors = read(source / 'ANCHORS.json')
    require(len(anchors) == len(m['runs']) and all(a['passed'] for a in anchors), 'Clean anchor gate incomplete')
    amap = {(a['dataset'], a['model'], a['seed']): a for a in anchors}
    require(set(amap) == expected_runs, 'Anchor membership differs')
    for run in m['runs']:
        a = amap[run['dataset'], run['model'], run['seed']]
        require(a['expected'] == run['validation'] and
                abs(a['accuracy'] - a['expected']['accuracy']) <= spec['accuracy_tolerance'] and
                abs(a['loss'] - a['expected']['loss']) <= spec['loss_tolerance'], 'Saved anchor failed')
    for name, checksum in m['historical_packages'].items():
        require(sha(local_path(name, must_exist=True)) == checksum, 'Historical package changed')
        hashes[name] = checksum
    count = 0
    expected_paths = set()
    for run, condition, number in descriptors(m):
        p = score_path(source, run, condition, number)
        raw = sha(p)
        cache_rows(source, m, run, condition, number, raw)
        hashes[portable_path(p)] = raw
        expected_paths.add(p)
        count += 1
        if count % 1000 == 0:
            print('Authenticated saved chunks:', count, flush=True)
    require(expected_paths == set((source / 'scores').rglob('*.json')), 'Unexpected or missing source chunks')
    contract = dict(version='temporal_permutation_stream_finalize_v1', source_root=portable_path(source),
        output_root=portable_path(output), source_manifest_sha256=digest(m), source_files_sha256=hashes,
        finalizer_source_sha256=own_sources(), specification=spec,
        datasets=m['datasets'], blocked=m['blocked'], inference=False, training=False,
        paired_format='UTF-8 JSONL shards; per-shard completion sidecars and index; original row schema')
    if (output / 'MANIFEST.json').exists():
        require(read(output / 'MANIFEST.json') == contract, 'Finalization resume identity differs')
    else:
        require(not output.exists() or all(p.name == '.analysis.lock' or
                (p.name.startswith('.MANIFEST.json-') and p.name.endswith('.tmp')) for p in output.iterdir()),
                'Nonempty output without manifest')
    return source, output, m, contract


def pair_rows(ds, seed, pair, width, clean_parent, clean_variant, parent_reps, variant_reps):
    require(len(parent_reps) == len(variant_reps) == 3, 'Need exactly three replicates')
    ids = [r['sample_id'] for r in clean_parent]
    require(all([r['sample_id'] for r in rows] == ids for rows in
        [clean_variant, *parent_reps, *variant_reps]), 'Paired sample alignment differs')
    for i, sid in enumerate(ids):
        cp, cv = clean_parent[i], clean_variant[i]
        for rep in range(3):
            parent, variant = parent_reps[rep][i], variant_reps[rep][i]
            require(parent['transformation'] == variant['transformation'], 'Paired transformed inputs differ')
            audit = parent['transformation']
            require(audit['original_sha256'] == cp['transformation']['original_sha256'] ==
                    cv['transformation']['original_sha256'] and audit['width'] == width and
                    audit['channel_counts'] == cp['transformation']['channel_counts'] ==
                    cv['transformation']['channel_counts'] and audit['event_count'] == cp['transformation']['event_count'],
                    'Clean/permuted input identity or counts differ')
            benefit = cp['loss'] - cv['loss']
            changed = parent['loss'] - variant['loss']
            yield dict(dataset=ds, seed=seed, pair=pair, sample_id=sid, width=width, replicate=rep,
                clean_parent=cp, clean_variant=cv, permuted_parent=parent, permuted_variant=variant,
                benefit_clean=benefit, benefit_permuted=changed, delta_benefit=changed - benefit)


def sample_metrics(rows):
    """At most one sample's three rows in memory; preserve original mean order."""
    bucket = []
    for row in rows:
        if bucket:
            require(row['sample_id'] == bucket[0]['sample_id'], 'Sample replicate alignment differs')
        require(row['replicate'] == len(bucket), 'Replicates out of order or duplicated')
        bucket.append(row)
        if len(bucket) == 3:
            values = [statistics.mean(r['delta_benefit'] for r in bucket)]
            values += [statistics.mean(r[side][metric] for r in bucket)
                       for side in SIDES for metric in ('loss', 'accuracy')]
            yield dict(zip(FIELDS, values))
            bucket = []
    require(not bucket, 'Missing permutation replicate')


def ensure_shard(path, identity, produce):
    sidecar = path.with_suffix('.meta.json')
    if sidecar.exists():
        record = read(sidecar)
        require(record['identity'] == identity and path.is_file() and sha(path) == record['sha256'],
                'Missing, corrupt or incompatible paired shard; refusing replacement')
        return record
    # A shard without its sidecar is an unpublished interrupted write, never a
    # completed shard; deterministic reconstruction is safe under the output lock.
    count = write_rows(path, produce())
    record = dict(identity=identity, sha256=sha(path), rows=count)
    write_json(sidecar, record)
    return record


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def authenticate_files(hashes):
    for name, checksum in hashes.items():
        require(sha(local_path(name, must_exist=True)) == checksum, 'Bound artifact changed: ' + name)


def validate_clean(source, m, contract):
    """User-run only: recompute saved clean means, never neural outputs."""
    anchors = {(a['dataset'], a['model'], a['seed']): a for a in read(source / 'ANCHORS.json')}
    for run in m['runs']:
        values = []
        correct = steps = 0
        size = run['config']['dataset']['batch_size']
        for number in range(math.ceil(len(original_ids(m, run)) / size)):
            p = score_path(source, run, ['clean', 0, 0], number)
            rows = cache_rows(source, m, run, ['clean', 0, 0], number,
                              contract['source_files_sha256'][portable_path(p)])
            values.extend(r['loss'] for r in rows)
            correct += sum(r['correct_steps'] for r in rows)
            steps += sum(r['valid_steps'] for r in rows)
        a = anchors[run['dataset'], run['model'], run['seed']]
        require(statistics.mean(values) == a['loss'] and correct / steps == a['accuracy'],
                'Cached clean scores do not reproduce recorded anchors')


def finalize(source, output, m, contract):
    spec = m['specification']
    validate_clean(source, m, contract)
    index, groups = [], {}
    for ds in m['cohorts']:
        for pair in spec['pairs']:
            for width in spec['window_widths']:
                per_seed = {}
                for seed in spec['training_seeds']:
                    runmap = {r['model']: r for r in m['runs'] if r['dataset'] == ds and r['seed'] == seed}
                    parent, variant = runmap[pair['parent']], runmap[pair['variant']]
                    require(parent['config']['dataset']['batch_size'] == variant['config']['dataset']['batch_size'],
                            'Paired batching differs')
                    vectors = {name: [] for name in FIELDS}
                    size = parent['config']['dataset']['batch_size']
                    for number in range(math.ceil(len(original_ids(m, parent)) / size)):
                        dependencies = {}
                        def load(run, condition):
                            p = score_path(source, run, condition, number)
                            checksum = contract['source_files_sha256'][portable_path(p)]
                            dependencies[portable_path(p)] = checksum
                            return cache_rows(source, m, run, condition, number, checksum)
                        cp, cv = load(parent, ['clean', 0, 0]), load(variant, ['clean', 0, 0])
                        pp, vp = [], []
                        for rep in range(3):
                            cond = [f'width_{width}_rep_{rep}', width, rep]
                            pp.append(load(parent, cond)); vp.append(load(variant, cond))
                        path = output / 'paired' / ds / pair['id'] / f'seed_{seed}' / f'width_{width}' / f'batch_{number:04d}.jsonl'
                        identity = dict(manifest_sha256=digest(contract), dataset=ds, pair=pair['id'], width=width,
                                        seed=seed, batch=number, dependencies=dependencies)
                        def produce():
                            return pair_rows(ds, seed, pair['id'], width, cp, cv, pp, vp)
                        record = ensure_shard(path, identity, produce)
                        require(record['rows'] == len(cp) * 3, 'Paired shard coverage differs')
                        metrics = list(sample_metrics(read_rows(path)))
                        require(len(metrics) == len(cp), 'Sample aggregation coverage differs')
                        for name in FIELDS:
                            vectors[name].extend(row[name] for row in metrics)
                        index.append(dict(path=portable_path(path), sidecar=portable_path(path.with_suffix('.meta.json')),
                                          sha256=record['sha256'], rows=record['rows']))
                    require(all(len(v) == len(m['cohorts'][ds]) for v in vectors.values()), 'Full cohort coverage differs')
                    per_seed[str(seed)] = {k: statistics.mean(v) for k, v in vectors.items()}
                groups[ds, pair['id'], width] = per_seed
                print('Finalized saved scores:', ds, pair['id'], 'width', width, flush=True)
    results = []
    for (ds, pair, width), values in sorted(groups.items()):
        means = {seed: v['delta_benefit'] for seed, v in values.items()}
        metrics = {}
        for name in FIELDS[1:]:
            seeds = {seed: v[name] for seed, v in values.items()}
            metrics[name] = dict(per_seed=seeds, mean=statistics.mean(seeds.values()), sample_sd=statistics.stdev(seeds.values()))
        results.append(dict(dataset=ds, pair=pair, width=width, per_seed=means, metrics=metrics,
            mean_delta_benefit=statistics.mean(means.values()), sample_sd=statistics.stdev(means.values()), n_training_seeds=3))
    expected = sum(len(ids) for ids in m['cohorts'].values()) * len(spec['pairs']) * 3 * 3 * 3
    require(sum(r['rows'] for r in index) == expected, 'Final paired-row count differs')
    write_json(output / 'PAIRED_RESULTS_INDEX.json', dict(format='jsonl_shards', rows=expected, shards=index))
    write_json(output / 'SUMMARY.json', dict(results=results, blocked=m['blocked'],
        interpretation='Exploratory validation, three training seeds; no independent test or causal proof'))


def verify(output):
    output = scoped(output)
    m, c = read(output / 'MANIFEST.json'), read(output / 'COMPLETE.json')
    require(c['status'] == 'complete' and c['manifest_sha256'] == digest(m), 'Finalization completion identity differs')
    require(c['outputs'].get(portable_path(output / 'MANIFEST.json')) == sha(output / 'MANIFEST.json'), 'Unbound manifest')
    require(own_sources() == m['finalizer_source_sha256'], 'Finalizer sources changed')
    for name in c['outputs']:
        require(local_path(name).is_relative_to(output), 'Completion output escapes the finalization root')
    authenticate_files(c['outputs'])
    authenticate_files(m['source_files_sha256'])
    source_manifest = read(local_path(m['source_root']) / 'MANIFEST.json')
    require(digest(source_manifest) == m['source_manifest_sha256'] and
            source_identity(source_manifest['specification']) == source_manifest['extension_source_sha256'],
            'Original source manifest/extension identity changed')
    FrozenSources(source_manifest['scientific_source_revision']).authenticate(
        {p: v['expected'] for p, v in source_manifest['scientific_source_audit'].items()})
    index = read(output / 'PAIRED_RESULTS_INDEX.json')
    require(sum(r['rows'] for r in index['shards']) == index['rows'], 'Index row coverage differs')
    for shard in index['shards']:
        require(shard['sha256'] == c['outputs'][shard['path']] and
                local_path(shard['path']).is_relative_to(output), 'Shard index identity differs')
    return c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if args.verify_only:
        c = verify(args.output_root)
        print('Saved finalization verified; rows:', c['paired_rows'], 'no inference or analysis rerun.')
        return 0
    source, output, m, contract = inspect_source(args.source_root, args.output_root)
    if args.check_only:
        print('Read-only finalization preflight passed. No tests, inference, aggregation or output writes.', flush=True)
        return 0
    # Source lock is the same OS lock used by the original scientific launcher.
    with held_lock(source / '.analysis.lock'):
        output.mkdir(parents=True, exist_ok=True)
        with held_lock(output / '.analysis.lock'):
            if (output / 'COMPLETE.json').exists():
                verify(output)
                print('Previously completed finalization verified.')
                return 0
            if (output / 'MANIFEST.json').exists():
                require(read(output / 'MANIFEST.json') == contract, 'Finalization identity changed')
            else:
                write_json(output / 'MANIFEST.json', contract)
            test = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_finalize_temporal_permutation'],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            (output / 'CONTRACT_TESTS.txt').write_text(test.stdout + test.stderr, encoding='utf-8')
            require(test.returncode == 0, 'Finalization contract tests failed')
            finalize(source, output, m, contract)
            authenticate_files(contract['source_files_sha256'])
            require(own_sources() == contract['finalizer_source_sha256'], 'Finalizer changed during execution')
            outputs = {portable_path(p): sha(p) for p in output.rglob('*') if p.is_file() and
                p.name not in ('COMPLETE.json', '.analysis.lock') and not p.name.endswith('.tmp')}
            rows = read(output / 'PAIRED_RESULTS_INDEX.json')['rows']
            write_json(output / 'COMPLETE.json', dict(status='complete', manifest_sha256=digest(contract), outputs=outputs,
                paired_rows=rows, completed_utc=datetime.now(timezone.utc).isoformat(),
                registered_scope_complete=not bool(m['blocked']), datasets=m['datasets'], blocked=m['blocked'],
                inference=False, training=False, scientific_source_revision=m['scientific_source_revision']))
            print('Memory-safe finalization completed:', portable_path(output), 'rows:', rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

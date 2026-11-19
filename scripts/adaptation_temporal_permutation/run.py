"""User-launched temporal permutation; --check-only never runs models or tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

import yaml

from .source import ROOT, FrozenSources
from .transform import derive_seed, permute, identity

SPEC = 'config/studies/adaptation_temporal_permutation_v1.yaml'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def output_path(value):
    from project_paths import local_path
    path = local_path(value)
    parent = ROOT / 'results/experiments'
    require(path.parent == parent and path.name.startswith('adaptation_temporal_permutation_'),
            'Output must be results/experiments/adaptation_temporal_permutation_*/')
    return path


def source_identity(spec):
    paths = [ROOT / SPEC, *sorted(Path(__file__).parent.glob('*.py')),
             ROOT / 'tests/test_adaptation_temporal_permutation.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def prepare(args, spec, frozen):
    import torch
    from omegaconf import OmegaConf
    from project_paths import local_path, portable_path, portable_config
    from scripts.adaptation_stage7.data import Adapter
    manifest_path = local_path(spec['source_manifest'], must_exist=True)
    require(sha(manifest_path) == spec['source_manifest_sha256'], 'Registered source manifest changed')
    saved = read(manifest_path)
    audit = frozen.authenticate(saved['source_sha256'])
    require(spec['window_widths'] == [5, 10, 20] and spec['permutation_replicates'] == 3 and
            spec['training_seeds'] == [42, 123, 456] and spec['precision'] == 'high' and
            not spec['training'] and not spec['official_test_access'], 'Registered contract changed')
    datasets = args.datasets or (['SHD'] if args.phase == 'pilot' else ['SHD', 'SSC', 'ECG'])
    require(len(set(datasets)) == len(datasets), 'Duplicate dataset assignment')
    require(args.phase != 'pilot' or datasets == ['SHD'], 'Pilot is exactly the SHD* 40-record cohort')
    root = output_path(args.output_root or spec['output_roots'][args.phase])
    other = output_path(spec['output_roots']['full' if args.phase == 'pilot' else 'pilot'])
    require(root != other, 'Pilot and full roots must differ')
    require(not root.exists() or not any(root.iterdir()) or (root / 'MANIFEST.json').is_file(),
            'Nonempty output without a manifest; choose a new root')
    blocked = {'ECG': 'Temporal block IDs score each saved timestep; a defensible independent label-region '
               'and boundary policy has not been authenticated. No ECG inference is permitted.'} if 'ECG' in datasets else {}
    adapters, runs, sources = {}, [], {}
    models = {p[k] for p in spec['pairs'] for k in ('parent', 'variant')}
    baseline = yaml.safe_load(frozen.read(spec['inherits']))['paper_baseline']
    for ds in datasets:
        if ds in blocked:
            continue
        entries = saved['specification']['sources'][ds]
        packages = []
        for entry in entries:
            directory = local_path(entry['root'], must_exist=True)
            mp, cp = directory / 'MANIFEST.json', directory / 'COMPLETE.json'
            require(sha(mp) == entry['manifest_sha256'] and sha(cp) == entry['complete_sha256'],
                    'Historical package identities differ: ' + ds)
            m, c = read(mp), read(cp)
            from scripts.adaptation_analysis import pilot
            require(c['status'] == 'complete' and c['manifest_sha256'] == pilot.digest(m),
                    'Historical completion guard differs')
            frozen.authenticate(m['source_sha256'])
            # Clean references are verified against metrics and checkpoint identities below;
            # no historical prediction cache is substituted for fresh clean evaluation.
            packages.append(m)
            sources[portable_path(mp)] = sha(mp)
            sources[portable_path(cp)] = sha(cp)
        ds_runs = [r for r in saved['runs'] if r['dataset'] == ds]
        require({(r['model'], r['seed']) for r in ds_runs} ==
                {(m, s) for m in models for s in spec['training_seeds']}, 'Missing registered checkpoints')
        for run in ds_runs:
            checkpoint = local_path(run['checkpoint'], must_exist=True)
            require(sha(checkpoint) == run['checkpoint_sha256'], 'Checkpoint changed')
            metric = local_path(run['validation']['path'], must_exist=True)
            require(sha(metric) == run['validation']['sha256'], 'Saved metrics changed')
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
            cfg = portable_config(OmegaConf.to_container(OmegaConf.create(ckpt['hyper_parameters']['cfg']), resolve=True))
            require(cfg == run['config'] and ckpt['epoch'] == run['selected_epoch'], 'Checkpoint configuration differs')
            require(cfg['paper_baseline'] == baseline, 'Dataset-specific baseline identity differs')
            del ckpt
            runs.append(run)
        adapter = Adapter(ds, packages[0], ds_runs[0]['config'], baseline,
                          dict(samples={'SHD': 40, 'SSC': 70, 'ECG': 30}, cohort_seed=spec['pilot_selection_seed'],
                               repeat_records=8, diagnostic_records=4))
        require(len(adapter.ids) == spec['validation_sizes'][ds], 'Validation membership count differs')
        adapters[ds] = adapter
    cohorts = {ds: list(a.selected if args.phase == 'pilot' else a.ids) for ds, a in adapters.items()}
    if args.phase == 'pilot':
        require(len(cohorts['SHD']) == 40, 'Pilot must contain 40 records')
        require(cohorts['SHD'] == spec['pilot_sample_ids'], 'Registered pilot cohort IDs differ')
        labels = adapters['SHD'].native.labels[cohorts['SHD']]
        from collections import Counter
        require(set(Counter(labels.tolist()).values()) == {2}, 'Pilot must have two records per class')
    require(torch.get_default_dtype() == torch.float32, 'Expected ordinary float32')
    manifest = dict(specification=spec, phase=args.phase, datasets=datasets, cohorts=cohorts,
        output_root=portable_path(root), runs=runs, blocked=blocked, historical_packages=sources,
        scientific_source_revision=frozen.revision, scientific_source_audit=audit,
        extension_source_sha256=source_identity(spec), integrated_revision=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
        dataset_metadata={ds: a.meta for ds, a in adapters.items()},
        device=args.device, training=False, official_test_opened=False,
        protocol='Exploratory validation; ordinary original batches; all clean anchors before any permutations')
    if (root / 'MANIFEST.json').exists():
        previous = dict(read(root / 'MANIFEST.json'))
        previous.pop('pilot_review', None)
        require(previous == manifest, 'Resume identity differs; use original arguments and sources')
    return root, manifest, adapters


def verify(root):
    from project_paths import local_path
    root = output_path(root)
    m, c = read(root / 'MANIFEST.json'), read(root / 'COMPLETE.json')
    require(c['manifest_sha256'] == digest(m) and c['status'] == 'complete', 'Completion identity differs')
    for name, checksum in c['outputs'].items():
        p = local_path(name, must_exist=True)
        require(p.is_relative_to(root) and sha(p) == checksum, 'Output missing/changed: ' + name)
    require(c['outputs'].get((root / 'MANIFEST.json').relative_to(ROOT).as_posix()) == sha(root / 'MANIFEST.json'),
            'Manifest is not authenticated by completion')
    require(source_identity(m['specification']) == m['extension_source_sha256'], 'Extension source identity changed')
    frozen = FrozenSources(m['scientific_source_revision'])
    frozen.authenticate({p: v['expected'] for p, v in m['scientific_source_audit'].items()})
    for name, checksum in m['historical_packages'].items():
        require(sha(local_path(name, must_exist=True)) == checksum, 'Historical source package changed')
    return c


class Store:
    def __init__(self, root, manifest):
        from scripts.ssc_resume_io import AtomicJSONWriter
        from scripts.adaptation_analysis.pilot import write_json
        self.root, self.manifest = root, manifest
        self.write = AtomicJSONWriter(root, write_json)
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'MANIFEST.json').exists():
            require(read(root / 'MANIFEST.json') == manifest, 'Existing manifest differs')
        else:
            self.write(root / 'MANIFEST.json', manifest)

    def cached(self, suffix, identity, produce):
        p = self.root / suffix
        contract = dict(manifest_sha256=digest(self.manifest), **identity)
        if p.exists():
            record = read(p)
            require(record['identity'] == contract and record['value_sha256'] == digest(record['value']),
                    'Corrupt or incompatible cache: ' + suffix)
            return record['value']
        value = produce()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.write(p, dict(identity=contract, value=value, value_sha256=digest(value)))
        return value


def evaluate(store, run, adapter, ids, condition, device, frozen):
    import torch
    from scripts.adaptation_gradient_pilot.engine import load_model
    from scripts.adaptation_gradient_pilot.methods import require_frozen
    from scripts.adaptation_stage7.methods import forward
    model = load_model(run, device)
    rows = []
    try:
        size = run['config']['dataset']['batch_size']
        selected = set(ids)
        for number, start in enumerate(range(0, len(adapter.ids), size)):
            original_ids = adapter.ids[start:start + size]
            wanted = [sid for sid in original_ids if sid in selected]
            if not wanted:
                continue
            suffix = f"scores/{run['dataset']}/{run['model']}/seed_{run['seed']}/{condition[0]}/batch_{number:04d}.json"
            def produce():
                physical = {sid: adapter.input(sid) for sid in original_ids}
                overrides, audits = {}, {}
                for sid in original_ids:
                    if sid not in selected:
                        continue
                    x = physical[sid]
                    if condition[0] == 'clean':
                        audits[sid] = dict(status='clean', original_sha256=identity(x),
                            transformed_sha256=identity(x), event_count=int(x.sum(dtype=object)),
                            channel_counts=x.sum(0, dtype=object).tolist(), support=len(x), burn=adapter.burn)
                        continue
                    seed = derive_seed(store.manifest['specification']['permutation_seed'], run['dataset'],
                                       sid, condition[1], condition[2])
                    overrides[sid], audits[sid] = permute(x, condition[1], seed, burn=adapter.burn)
                # Preserve original-batch membership, shape, padding and clean context.
                batch = adapter.batch(original_ids, {sid: overrides.get(sid, physical[sid]) for sid in original_ids})
                values = forward(model, batch, run['config'], device)
                require_frozen(model, run['model_state_sha256'])
                return [dict(sample_id=sid, **value, transformation=audits.get(sid))
                        for sid, value in zip(original_ids, values) if sid in selected]
            chunk = store.cached(suffix, dict(checkpoint=run['checkpoint_sha256'], original_batch_ids=original_ids,
                selected_ids=wanted, condition=condition), produce)
            require([r['sample_id'] for r in chunk] == wanted, 'Cached membership differs')
            rows.extend(chunk)
        require_frozen(model, run['model_state_sha256'])
    finally:
        del model
        torch.cuda.empty_cache()
    return rows


def summary(paired, spec):
    groups = {}
    for row in paired:
        key = (row['dataset'], row['pair'], row['width'], row['seed'], row['sample_id'])
        groups.setdefault(key, []).append(row['delta_benefit'])
    per_seed = {}
    for (ds, pair, width, seed, sid), values in groups.items():
        require(len(values) == spec['permutation_replicates'], 'Replicate coverage differs')
        per_seed.setdefault((ds, pair, width, seed), []).append(statistics.mean(values))
    final = {}
    for (ds, pair, width, seed), values in per_seed.items():
        final.setdefault((ds, pair, width), {})[str(seed)] = statistics.mean(values)
    result = []
    for (ds, pair, width), seeds in sorted(final.items()):
        require(set(seeds) == set(map(str, spec['training_seeds'])), 'Training-seed coverage differs')
        metrics = {}
        for side in ('clean_parent', 'clean_variant', 'permuted_parent', 'permuted_variant'):
            for metric in ('loss', 'accuracy'):
                sample_values = {}
                for row in paired:
                    if (row['dataset'], row['pair'], row['width']) == (ds, pair, width):
                        sample_values.setdefault((row['seed'], row['sample_id']), []).append(row[side][metric])
                values_by_seed = {str(seed): statistics.mean(statistics.mean(v) for (s, sid), v in sample_values.items()
                    if s == seed) for seed in spec['training_seeds']}
                metrics[side + '_' + metric] = dict(per_seed=values_by_seed,
                    mean=statistics.mean(values_by_seed.values()), sample_sd=statistics.stdev(values_by_seed.values()))
        result.append(dict(dataset=ds, pair=pair, width=width, per_seed=seeds, metrics=metrics,
            mean_delta_benefit=statistics.mean(seeds.values()), sample_sd=statistics.stdev(seeds.values()), n_training_seeds=3))
    return result


def execute(root, manifest, adapters, args, frozen):
    import torch
    from scripts.adaptation_analysis import pilot
    spec = manifest['specification']
    root.mkdir(parents=True, exist_ok=True)
    with pilot.study_lock(root / '.analysis.lock'):
        if (root / 'COMPLETE.json').exists():
            return verify(root)
        store = Store(root, manifest)
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_adaptation_temporal_permutation'],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (root / 'CONTRACT_TESTS.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
        require(tests.returncode == 0, 'User-run contract tests failed; inspect CONTRACT_TESTS.txt')
        require(torch.cuda.is_available() and args.device.startswith('cuda:'), 'Ordinary CUDA protocol required')
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        # high selects the original TF32 matmul policy; do not substitute highest.
        runtime = dict(environment=pilot.environment(torch.device(args.device)), precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark, torch_version=torch.__version__)
        rp = root / 'RUNTIME.json'
        if rp.exists():
            require(read(rp) == runtime, 'Resume backend/environment changed')
        else:
            store.write(rp, runtime)
        try:
            clean = {}
            anchors = []
            # All selected dataset/model/seed full-validation anchors gate all perturbations.
            for run in manifest['runs']:
                a = adapters[run['dataset']]
                values = evaluate(store, run, a, a.ids, ['clean', 0, 0], torch.device(args.device), frozen)
                accuracy = sum(r['correct_steps'] for r in values) / sum(r['valid_steps'] for r in values)
                loss = statistics.mean(r['loss'] for r in values)
                passed = (abs(accuracy - run['validation']['accuracy']) <= spec['accuracy_tolerance'] and
                          abs(loss - run['validation']['loss']) <= spec['loss_tolerance'])
                anchors.append(dict(dataset=run['dataset'], model=run['model'], seed=run['seed'],
                    accuracy=accuracy, loss=loss, expected=run['validation'], passed=passed))
                store.write(root / 'ANCHORS.json', anchors)
                clean[run['dataset'], run['model'], run['seed']] = {r['sample_id']: r for r in values}
            require(all(a['passed'] for a in anchors), 'Clean anchors failed. No permutation evaluation allowed.')
            paired = []
            for ds, a in adapters.items():
                ids = manifest['cohorts'][ds]
                for seed in spec['training_seeds']:
                    runs = {r['model']: r for r in manifest['runs'] if r['dataset'] == ds and r['seed'] == seed}
                    for width in spec['window_widths']:
                        for rep in range(spec['permutation_replicates']):
                            condition = [f'width_{width}_rep_{rep}', width, rep]
                            scores = {name: {r['sample_id']: r for r in evaluate(store, run, a, ids, condition,
                                torch.device(args.device), frozen)} for name, run in runs.items()}
                            for pair in spec['pairs']:
                                for sid in ids:
                                    parent, variant = [scores[pair[k]][sid] for k in ('parent', 'variant')]
                                    cp, cv = [clean[ds, pair[k], seed][sid] for k in ('parent', 'variant')]
                                    require(parent['transformation'] == variant['transformation'], 'Paired inputs differ')
                                    benefit = cp['loss'] - cv['loss']
                                    changed = parent['loss'] - variant['loss']
                                    paired.append(dict(dataset=ds, seed=seed, pair=pair['id'], sample_id=sid,
                                        width=width, replicate=rep, clean_parent=cp, clean_variant=cv,
                                        permuted_parent=parent, permuted_variant=variant, benefit_clean=benefit,
                                        benefit_permuted=changed, delta_benefit=changed - benefit))
            store.write(root / 'PAIRED_RESULTS.json', paired)
            store.write(root / 'SUMMARY.json', dict(results=summary(paired, spec), blocked=manifest['blocked'],
                interpretation='Exploratory validation, three training seeds; no independent test or causal proof'))
            require(source_identity(spec) == manifest['extension_source_sha256'], 'Extension changed during execution')
            require(all(checksum == frozen.loaded.get(path, checksum) for path, checksum in
                        ((p, row['expected']) for p, row in manifest['scientific_source_audit'].items())),
                    'Loaded scientific source differs')
            store.write(root / 'LOADED_SOURCES.json', frozen.loaded)
            outputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()
                and p.name not in ('COMPLETE.json', '.analysis.lock', 'FAILED.json') and not p.name.endswith('.tmp')}
            complete = dict(status='complete', manifest_sha256=digest(manifest), outputs=outputs,
                completed_utc=datetime.now(timezone.utc).isoformat(), registered_scope_complete=not bool(manifest['blocked']),
                blocked=manifest['blocked'], training=False, official_test_opened=False)
            store.write(root / 'COMPLETE.json', complete)
            return complete
        except Exception:
            store.write(root / 'FAILED.json', dict(status='failed', traceback=traceback.format_exc(),
                recorded_utc=datetime.now(timezone.utc).isoformat()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--datasets', nargs='+', choices=['SHD', 'SSC', 'ECG'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    spec = yaml.safe_load((ROOT / SPEC).read_text(encoding='utf-8'))
    frozen = FrozenSources(spec['scientific_source_revision'])
    frozen.install()
    if args.verify_only:
        c = verify(args.output_root or spec['output_roots'][args.phase])
        print('Verified completed executable scope; registered scope complete:', c['registered_scope_complete'],
              'blocked:', c['blocked'])
        return 0
    root, manifest, adapters = prepare(args, spec, frozen)
    if args.check_only:
        print('Read-only preflight passed; no tests, model construction, inference or output writes.')
        print('Cohorts:', {ds: len(ids) for ds, ids in manifest['cohorts'].items()}, 'Blocked:', manifest['blocked'])
        if args.phase == 'full':
            print('Full execution additionally requires a verified pilot and checksum-bound human PILOT_REVIEW.json.')
        return 0
    if args.phase == 'full':
        pilot_root = output_path(spec['output_roots']['pilot'])
        verify(pilot_root)
        review = read(pilot_root / 'PILOT_REVIEW.json')
        pilot_manifest = read(pilot_root / 'MANIFEST.json')
        require(pilot_manifest['extension_source_sha256'] == manifest['extension_source_sha256'] and
                pilot_manifest['specification'] == manifest['specification'], 'Reviewed pilot uses another implementation')
        require(review.get('approved_for_full') is True and bool(review.get('reviewer')) and
                review.get('pilot_complete_sha256') == sha(pilot_root / 'COMPLETE.json'),
                'Full execution requires a human pilot review bound to its completion checksum')
        manifest['pilot_review'] = dict(review=review, review_sha256=sha(pilot_root / 'PILOT_REVIEW.json'),
                                      complete_sha256=sha(pilot_root / 'COMPLETE.json'))
    complete = execute(root, manifest, adapters, args, frozen)
    print('Executable scope completed; blocked datasets:', complete['blocked'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Frozen SSC contracts, portable saved-evidence access and preservation guards."""
import hashlib
from pathlib import Path, PurePosixPath
import shutil

import yaml

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory

SPEC = 'config/studies/adaptation_ssc_suite_v1.yaml'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_spec():
    from scripts.adaptation_ssc_full.evaluate import load_study
    spec = yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    baseline = yaml.safe_load(local_path(spec['inherits']).read_text(encoding='utf-8'))['paper_baseline']
    definitions, _ = load_study()
    require(spec['dataset_section'] == 'SSC' and not spec['training'] and not spec['new_test_access'],
            'Only frozen SSC validation evaluation/statistics are supported')
    require(spec['seeds'] == [42, 123, 456] and spec['expected_validation_samples'] == 9981,
            'Suite requires the complete fixed SSC cohort')
    expected = [('amt_mt', 'MT_SE_adLIF', 'AMT_SE_adLIF'), ('lt_dth', 'DTH_SE_adLIF', 'LT_DTH_SE_adLIF'),
                ('sa_mt', 'MT_SE_adLIF', 'SA_MT_SE_adLIF'), ('mt_tc', 'MT_SE_adLIF', 'MT_TC_SE_adLIF')]
    require([(p['id'], p['parent'], p['variant']) for p in spec['pairs']] == expected, 'Frozen pair list differs')
    statistics = spec['statistics']
    require(statistics['features'] == ['fast_slow_contrast', 'autocorr_tau_ms'] and
            statistics['controls'] == ['class_fixed_effects', 'speaker_fixed_effects',
                                      'log1p_duration_ms', 'log1p_mean_rate_hz'], 'Unsupported statistical design')
    require(statistics['bootstrap']['unit'] == 'speaker' and statistics['bootstrap']['draws'] >= 100 and
            statistics['uncertainty']['primary_family_size'] == 8 and
            statistics['uncertainty']['secondary_family_size'] == 12, 'SSC suite interval family differs')
    spec.update(features=definitions['features'], conditions=definitions['conditions'],
                minimum_frames=baseline['datasets']['SSC']['pad_to_min_size'],
                paper_baseline_version=baseline['version'], paper_baseline_dataset='SSC')
    return spec, baseline


def source_identity():
    paths = set(local_path(name) for name in pilot.source_identity())
    paths.update((ROOT / 'scripts/adaptation_ssc').glob('*.py'))
    paths.update((ROOT / 'scripts/adaptation_ssc_full').glob('*.py'))
    paths.update((ROOT / 'scripts/adaptation_ssc_suite').glob('*.py'))
    paths.update((ROOT / 'scripts/adaptation_statistics').glob('*.py'))
    paths.update([local_path(SPEC), local_path('config/studies/adaptation_ssc_amt_full_validation_v1.yaml'),
                  local_path('config/studies/adaptation_ssc_amt_pilot_v1.yaml'), local_path('config/path_aliases.yaml'),
                  local_path('datasets/ssc.py'), local_path('datasets/utils/transforms.py'),
                  local_path('scripts/adaptation_pairs/parameters.py')])
    paths.update(local_path('tests/' + name + '.py') for name in
                 ('test_adaptation_ssc', 'test_adaptation_ssc_full', 'test_adaptation_ssc_suite'))
    return {portable_path(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(paths)}


def guard(directory, manifest):
    path = directory / 'MANIFEST.json'
    if path.exists():
        pilot.validate_manifest(pilot.read_json(path), manifest)
    elif directory.exists() and any(p.name != '.analysis.lock' for p in directory.iterdir()):
        raise ValueError('Nonempty destination has no matching manifest; preserve it and choose a new suite root')


def verify_completion(directory, boundary=None):
    """Check a current step without depending on ignored data or weight files."""
    directory = experiment_directory(directory)
    boundary = local_path(boundary or directory).resolve()
    record = pilot.read_json(directory / 'COMPLETE.json')
    manifest = pilot.read_json(directory / 'MANIFEST.json')
    require(record['status'] == 'complete' and record['manifest_sha256'] == pilot.digest(manifest),
            'Completion/manifest identity differs')
    require(record['outputs'].get(portable_path(directory / 'MANIFEST.json')) == pilot.sha256(directory / 'MANIFEST.json'),
            'Completion must include its own manifest')
    for name, checksum in record['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(boundary), 'Registered output escapes its result package')
        require(pilot.sha256(path) == checksum, 'Completed output changed: ' + name)
    return record, manifest


def validate_source(spec, directory=None):
    """An exact historical snapshot uses an explicit index, never rewritten manifests."""
    source = local_path(directory or spec['source_evaluation']['root'], must_exist=True)
    manifest = pilot.read_json(source / 'MANIFEST.json')
    complete = pilot.read_json(source / 'COMPLETE.json')
    frozen = spec['source_evaluation']
    require(pilot.digest(manifest) == frozen['manifest_sha256'] and
            pilot.sha256(source / 'COMPLETE.json') == frozen['completion_sha256'], 'Frozen AMT/MT source differs')
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest),
            'Source is not complete')
    require(manifest['scope'] == 'ssc_full_validation' and not manifest['training'] and
            not manifest['official_test_opened'] and manifest['ordinary_inference'], 'Source protocol differs')
    require(manifest['sample_ids'] == list(range(9981)) and manifest['seeds'] == spec['seeds'] and
            manifest['condition_definitions'] == spec['conditions'], 'Source coverage/conditions differ')
    require(manifest['data']['sha256'] == spec['dataset_sha256'] and
            manifest['resolved_study']['features'] == spec['features'], 'Source input definitions differ')
    require((manifest['resolved_study']['parent'], manifest['resolved_study']['variant']) ==
            ('MT_SE_adLIF', 'AMT_SE_adLIF'), 'Source comparison differs')
    prefix = PurePosixPath(frozen['root'])
    mapping = {}
    for name, checksum in complete['outputs'].items():
        suffix = PurePosixPath(name).relative_to(prefix)
        require('..' not in suffix.parts, 'Source path escapes its recorded prefix')
        actual = (source / Path(suffix.as_posix())).resolve()
        require(actual.is_relative_to(source.resolve()), 'Source relocation escapes its package')
        require(actual.is_file() and pilot.sha256(actual) == checksum, 'Source output missing or damaged: ' + name)
        mapping[name] = portable_path(actual)
    if directory is not None:
        index = pilot.read_json(source / 'PATH_INDEX.json')
        require(index['relocations'] == mapping and index['source_completion_sha256'] == frozen['completion_sha256'],
                'Snapshot path index differs')
    return manifest, complete, mapping


def snapshot_directory(suite_root):
    """Use one physical spelling for both the path index and its validator."""
    root = local_path(suite_root)
    destination = (root / 'shared/amt_mt_full_validation').resolve()
    require(destination.is_relative_to(root), 'Snapshot directory escapes the suite package')
    return destination


def snapshot_source(spec, suite_root):
    destination = snapshot_directory(suite_root)
    if (destination / 'PATH_INDEX.json').exists():
        validate_source(spec, destination)
        return destination
    original = local_path(spec['source_evaluation']['root'], must_exist=True)
    manifest, complete, _ = validate_source(spec)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = PurePosixPath(spec['source_evaluation']['root'])
    mapping = {}
    items = [(local_path(name), destination / PurePosixPath(name).relative_to(prefix).as_posix(), checksum)
             for name, checksum in complete['outputs'].items()]
    items.append((original / 'COMPLETE.json', destination / 'COMPLETE.json',
                  spec['source_evaluation']['completion_sha256']))
    for source, target, checksum in items:
        require(target.resolve().is_relative_to(destination.resolve()), 'Snapshot target escapes its package')
        if target.exists():
            require(pilot.sha256(target) == checksum, 'Existing snapshot changed; preserve it')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.copy.tmp')
            shutil.copy2(source, temporary)
            require(pilot.sha256(temporary) == checksum, 'Snapshot copy checksum differs')
            temporary.replace(target)
    for name in complete['outputs']:
        mapping[name] = portable_path(destination / PurePosixPath(name).relative_to(prefix).as_posix())
    index = dict(source_root=spec['source_evaluation']['root'], source_manifest_sha256=pilot.digest(manifest),
                 source_completion_sha256=spec['source_evaluation']['completion_sha256'], relocations=mapping,
                 note='Historical records and cache identities are copied byte-for-byte; paths resolve through this index.')
    path = destination / 'PATH_INDEX.json'
    if path.exists():
        pilot.validate_manifest(pilot.read_json(path), index)
    else:
        pilot.write_json(path, index)
    validate_source(spec, destination)
    return destination


def cache(path, identity, ids):
    record = pilot.read_json(path)
    require(record['identity'] == identity and record['value_sha256'] == pilot.digest(record['value']),
            'Cached value/identity differs: ' + portable_path(path))
    require([r['sample_id'] for r in record['value']] == ids, 'Cached IDs/order differ')
    return record['value']


def source_features(source, source_manifest, condition):
    name, rep, _, _ = condition
    return cache(source / 'diagnostic_rows' / f'input_features_{name}_rep_{rep}.json',
                 dict(manifest_sha256=pilot.digest(source_manifest), condition=name, replicate=rep,
                      kind='model_independent_full_validation_features'), source_manifest['sample_ids'])


def predictions(directory, manifest, model, seed, condition, source, source_manifest):
    name, rep, _, _ = condition
    reused = model in manifest['reuse_models']
    origin = source_manifest if reused else manifest
    run = next(r for r in origin['runs'] if r['model'] == model and r['seed'] == seed)
    root = source if reused else directory
    rows = cache(root / 'diagnostic_rows' / f'{model}_seed_{seed}_{name}_rep_{rep}.json',
                 dict(manifest_sha256=pilot.digest(origin), checkpoint_sha256=run['checkpoint_sha256'],
                      condition=name, replicate=rep,
                      kind='full_validation_predictions' if reused else 'ssc_suite_predictions'), manifest['sample_ids'])
    require(all(r['model'] == model and r['seed'] == seed and r['condition'] == name and
                r['replicate'] == rep for r in rows), 'Prediction identity differs')
    return rows


def seal(directory, manifest, outputs, **fields):
    from datetime import datetime, timezone
    outputs = list(dict.fromkeys([directory / 'MANIFEST.json', *outputs]))
    record = dict(status='complete', manifest_sha256=pilot.digest(manifest),
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  outputs={portable_path(p): pilot.sha256(p) for p in outputs}, **fields)
    pilot.write_json(directory / 'COMPLETE.json', record)
    return record

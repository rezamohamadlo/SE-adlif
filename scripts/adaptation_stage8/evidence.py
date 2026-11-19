"""Authenticate saved records without loading models, data sets or fitting models."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from cost_records import normalized_record
from project_paths import ROOT, local_path, portable_config, portable_path

SPEC = 'config/studies/adaptation_stage8_synthesis_v1.json'
SEEDS = [42, 123, 456]
DATASETS = ['SHD', 'SSC', 'ECG']
PAIRS = {
    'amt_mt': ('MT_SE_adLIF', 'AMT_SE_adLIF'),
    'lt_dth': ('DTH_SE_adLIF', 'LT_DTH_SE_adLIF'),
    'sa_mt': ('MT_SE_adLIF', 'SA_MT_SE_adLIF'),
    'mt_tc': ('MT_SE_adLIF', 'MT_TC_SE_adLIF'),
}
SHD = {
    'amt_mt': 'adaptation_mechanism_full_validation_v1',
    'lt_dth': 'adaptation_lt_dth_full_validation_v1',
    'sa_mt': 'adaptation_sa_mt_full_validation_v1',
    'mt_tc': 'adaptation_mt_tc_full_validation_v1',
}
EXPERIMENTS = 'results/experiments/'
STAGE7 = EXPERIMENTS + 'adaptation_stage7_suite_v1'
SSC = EXPERIMENTS + 'adaptation_ssc_suite_v2'
ECG = EXPERIMENTS + 'adaptation_ecg_extension_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}
        self.registered = {}
        self.packages = []
        self.paths = {}
        self.keys = {}

    def path(self, name):
        token = str(name).replace('\\', '/')
        if token in self.paths:
            return self.paths[token]
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(ROOT), 'Evidence must remain inside the checkout: ' + str(name))
        key = portable_path(path)
        self.paths[token] = self.paths[str(path).replace('\\', '/')] = self.paths[key] = path
        self.keys[path] = key
        return path

    def bind(self, name, expected=None):
        path = self.path(name)
        key = self.keys[path]
        actual = self.files.get(key)
        if actual is None:
            actual = sha(path)
            self.files[key] = actual
            if len(self.files) % 5000 == 0:
                print('Authenticated source files:', len(self.files), flush=True)
        require(expected is None or actual == expected, 'Evidence hash differs: ' + key)
        return path

    def read(self, name, registered=False):
        path = self.bind(name)
        if registered:
            require(self.keys[path] in self.registered, 'Unsealed scientific record: ' + str(name))
        return json.loads(path.read_text(encoding='utf-8'))

    def seal(self, name, expected=None, allowed_roots=(), indexed=False):
        root = self.path(name)
        allowed = [self.path(p) for p in allowed_roots]
        relocations = {}
        if indexed:
            index = self.read(root / 'PATH_INDEX.json', registered=True)
            relocations = index['relocations']
            require(expected is None or expected == index['source_completion_sha256'], 'Snapshot identity differs')
            expected = index['source_completion_sha256']
            manifest = self.read(root / 'MANIFEST.json', registered=True)
            legacy_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
            require(legacy_digest == index['source_manifest_sha256'], 'Historical snapshot manifest identity differs')
            oldroot = local_path(index['source_root'])
            for old, new in relocations.items():
                require(local_path(old).is_relative_to(oldroot) and self.path(new).is_relative_to(root),
                        'Snapshot path index escapes its explicit roots')
        self.bind(root / 'COMPLETE.json', expected)
        completion = self.read(root / 'COMPLETE.json')
        require(completion['status'] == 'complete', 'Incomplete package: ' + str(name))
        require(isinstance(completion['outputs'], dict), 'Missing output ledger: ' + str(name))
        for filename, checksum in completion['outputs'].items():
            require(not indexed or filename in relocations, 'Historical snapshot output missing from path index')
            path = self.path(relocations.get(filename, filename))
            require(path.is_relative_to(root) or any(path.is_relative_to(p) for p in allowed),
                    'Package output escapes its declared roots: ' + filename)
            key = self.keys[path]
            require(key not in self.registered or self.registered[key] == checksum,
                    'Conflicting source ledgers: ' + key)
            self.bind(path, checksum)
            self.registered[key] = checksum
        self.packages.append(portable_path(root))
        print('Verified saved package:', portable_path(root), flush=True)


def test_accuracy(rows, reproduced):
    values = [float(row['test_acc']) for row in rows if row.get('test_acc')]
    require(len(values) == 1 and math.isfinite(values[0]),
            'Expected exactly one final-test metric; overlapping histories require review')
    require(math.isfinite(reproduced) and abs(values[0] - reproduced) <= 1e-6,
            'Saved final-test and cost-evaluator accuracies disagree')
    return values[0]


def clean_groups(summary):
    groups = [g for g in summary['groups'] if g['condition'] == 'clean' and g['replicate'] == 0]
    require(sorted(g['seed'] for g in groups) == SEEDS, 'Clean seed cohort differs')
    return [dict(seed=g['seed'], parent_accuracy=g['parent_accuracy'],
                 variant_accuracy=g['variant_accuracy'],
                 parent_loss=g['parent_mean_loss'], variant_loss=g['variant_mean_loss'],
                 accuracy_gap_pp=g['accuracy_difference_pp'], loss_benefit=g['mean_loss_benefit'])
            for g in sorted(groups, key=lambda g: g['seed'])]


def check_portability(names):
    """Preflight-only path checks under a changed cwd and simulated checkout root."""
    expected = {name: local_path(name, must_exist=True) for name in names}
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='stage8_path_preflight_') as folder:
        other = Path(folder).resolve()
        try:
            os.chdir(other)
            for name, path in expected.items():
                require(local_path(name, must_exist=True) == path, 'Path depends on working directory')
                require(local_path(name, root=other) == other / name, 'Relocated checkout changes artifact suffix')
                require(portable_path(other / name, root=other) == name, 'Relocated path is not portable')
        finally:
            os.chdir(previous)


def prepare(output=None):
    e = Evidence()
    spec = e.read(SPEC)
    require(spec['version'] == 'adaptation_stage8_synthesis_v1' and spec['inherits'] == 'config/paper_baseline.yaml' and
            spec['training'] is False and spec['new_inference'] is False and
            spec['new_fitting_or_resampling'] is False, 'Synthesis scope differs')
    destination = local_path(output or spec['output_root'])
    base = local_path(EXPERIMENTS)
    require(destination.parent == base,
            'Use a separate study output directory inside results/experiments')
    require(not any(destination.is_relative_to(local_path(root)) for root in
                    [STAGE7, SSC, ECG] + [EXPERIMENTS + x for x in SHD.values()]),
            'Do not write inside a frozen source package')
    for name, checksum in spec['pinned_files'].items():
        e.bind(name, checksum)
    closeout = e.read(EXPERIMENTS + 'adaptation_mechanism_study_v1/STAGE6_CLOSEOUT_AUDIT.json')
    require(closeout['stage6_scoped_complete'], 'Stage 6 has not passed scope review')
    e.bind(closeout['audit_path'], closeout['audit_sha256'])
    e.bind(closeout['evidence_check_path'], closeout['evidence_check_sha256'])
    review = e.read(STAGE7 + '/POST_RUN_REVIEW.json')
    require(review['stage7_scoped_complete'] and review['model_states_unchanged'],
            'Stage 7 has not passed review')
    e.seal(STAGE7, review['main_complete_sha256'])
    e.seal(STAGE7 + '/presentation_v1', review['presentation_complete_sha256'])
    e.seal(STAGE7 + '/origin_controls_v1', review['origin_controls_complete_sha256'])
    stage7 = e.read(STAGE7 + '/MANIFEST.json', registered=True)
    for name, checksum in stage7['source_sha256'].items():
        path = e.bind(name)
        require(hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == checksum,
                'Frozen scientific implementation differs: ' + name)
    for source in stage7['sources']:
        e.bind(source['root'] + '/MANIFEST.json', source['manifest_sha256'])
        e.seal(source['root'], source['complete_sha256'])
    e.seal(ECG)
    e.seal(EXPERIMENTS + 'adaptation_ssc_windows_v1')
    e.seal(EXPERIMENTS + SHD['mt_tc'] + '/window_evidence_v1')
    validations, associations, interventions, traces = [], [], [], []
    reports = {}
    for ds in DATASETS:
        for pair, (parent, variant) in PAIRS.items():
            if ds == 'SHD':
                root = EXPERIMENTS + SHD[pair]
                summary = e.read(root + '/analyses/clean/SUMMARY.json', registered=True)
                groups = clean_groups(summary)
                statroot = root + '/statistics_v1'
                e.seal(statroot)
                stat = e.read(statroot + '/SUMMARY.json', registered=True)
                reports[ds, pair] = root + '/analyses/clean/SUMMARY.md'
            elif ds == 'SSC':
                root = SSC + '/pairs/' + pair
                e.seal(root + '/evaluation', allowed_roots=[SSC + '/shared/amt_mt_full_validation'])
                if pair == 'amt_mt':
                    index = e.read(root + '/evaluation/REPORT_INDEX.json', registered=True)
                    e.seal(SSC + '/shared/amt_mt_full_validation', indexed=True)
                    clean_path = index['reports']['clean']
                else:
                    clean_path = root + '/evaluation/analyses/clean/SUMMARY.json'
                groups = clean_groups(e.read(clean_path, registered=True))
                statroot = root + '/statistics_v1'
                e.seal(statroot)
                stat = e.read(statroot + '/SUMMARY.json', registered=True)
                reports[ds, pair] = statroot + '/SUMMARY.md'
            else:
                root = ECG + '/pairs/' + pair
                summary = e.read(root + '/SUMMARY.json', registered=True)
                groups = next(g['per_seed'] for g in summary['descriptive'] if g['condition'] == 'clean')
                require(sorted(g['seed'] for g in groups) == SEEDS, 'ECG clean seeds differ')
                stat = e.read(root + '/ASSOCIATIONS.json', registered=True)
                require(stat['status'] == 'descriptive_analysis_complete' and not stat['inference'],
                        'ECG must remain descriptive')
                reports[ds, pair] = root + '/SUMMARY.md'
            validations.extend(dict(dataset=ds, pair=pair, parent=parent, variant=variant, **g) for g in groups)
            if ds in ('SHD', 'SSC'):
                for feature in stat['primary']:
                    associations.append(dict(dataset=ds, pair=pair, **feature,
                        interval_protocol='SHD two-feature per-pair family' if ds == 'SHD' else 'SSC eight-feature/pair family',
                        complete_cases=stat['n_complete_cases'], speakers=stat['n_speakers']))
                interventions.extend(dict(dataset=ds, pair=pair, **row,
                    interval_protocol='SHD three-intervention per-pair family' if ds == 'SHD' else 'SSC twelve-intervention/pair family')
                    for row in stat['secondary'])
                tracepath = statroot + '/TRACE_SUMMARY.json'
                if pair == 'lt_dth' and local_path(tracepath).exists():
                    traces.extend(dict(dataset=ds, **row) for row in e.read(tracepath, registered=True)['groups'])
            else:
                for feature, row in stat['feature_summary'].items():
                    associations.append(dict(dataset=ds, pair=pair, feature=feature,
                        mean=row['mean'], sample_sd=row['sample_sd_across_training_seeds'], interval=None,
                        direction='descriptive_only', seed_direction_consistent=None,
                        per_seed=[dict(seed=r['seed'], coefficient=r['coefficients'][feature]) for r in stat['per_seed']],
                        interval_protocol='none; no verified patient clusters',
                        complete_cases=stat['design']['n_records'], speakers=None))
                for row in summary['descriptive']:
                    if row['condition'] == 'clean':
                        continue
                    val = row['aggregate']['benefit_change_from_clean']
                    interventions.append(dict(dataset=ds, pair=pair, condition=row['condition'],
                        mean=val['mean'], sample_sd=val['sample_sd'], interval=None,
                        direction='descriptive_only', interval_protocol='none; no verified patient clusters',
                        per_seed=[dict(seed=r['seed'], benefit_change=r['benefit_change_from_clean']) for r in row['per_seed']]))
    features = {}
    feature_sources = {
        'SHD': EXPERIMENTS + SHD['amt_mt'] + '/diagnostic_rows/input_features_clean_rep_0.json',
        'SSC': SSC + '/shared/amt_mt_full_validation/diagnostic_rows/input_features_clean_rep_0.json',
        'ECG': ECG + '/features/clean/rep_0.json',
    }
    for ds, filename in feature_sources.items():
        cache = e.read(filename, registered=True)
        # The enclosing output ledger authenticates the complete JSON, including its cache identity.
        rows = cache['value']
        require(len(rows) == {'SHD': 1631, 'SSC': 9981, 'ECG': 30}[ds], 'Feature cohort size differs')
        require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate feature records')
        require(all(r['condition'] == 'clean' and r['replicate'] == 0 for r in rows), 'Mixed feature conditions')
        features[ds] = rows
    costs = []
    require(len(stage7['runs']) == 54, 'Expected 54 study checkpoints')
    for run in stage7['runs']:
        checkpoint = e.bind(run['checkpoint'], run['checkpoint_sha256'])
        metrics = e.bind(run['validation']['path'], run['validation']['sha256'])
        name = checkpoint.parent.parent / 'COST_RESULTS.json'
        raw = e.read(name)
        record = normalized_record(raw)
        require(record['status'] == 'complete' and record['seed'] == run['seed'] and
                record['dataset'] == run['dataset'] and record['checkpoint_sha256'] == run['checkpoint_sha256'] and
                record['best_epoch'] == run['selected_epoch'], 'Cost checkpoint identity differs: ' + str(name))
        require(portable_config(record['saved_config']) == run['config'], 'Cost saved config differs: ' + str(name))
        require(local_path(record['checkpoint']) == checkpoint, 'Cost path does not identify the same checkpoint')
        require(record['environment']['matmul_precision'] == 'high', 'Unexpected ordinary-inference precision')
        with metrics.open(newline='', encoding='utf-8-sig') as stream:
            test = test_accuracy(list(csv.DictReader(stream)), record['measurements']['reproduced_test_acc'])
        costs.append(dict(dataset=run['dataset'], model=run['model'], seed=run['seed'],
            checkpoint=portable_path(checkpoint), checkpoint_sha256=run['checkpoint_sha256'],
            cost_record=portable_path(name), cost_version=raw['version'], test_accuracy=test,
            measurements=record['measurements'], environment=record['environment'],
            scopes=record.get('scopes', dict(activity=record['environment'].get('sequence_scope'),
                timing=record['environment'].get('timing_scope'))), batch_size=run['config']['batch_size'],
            neurons_l1=run['config']['l1']['n_neurons'], neurons_l2=run['config']['l2']['n_neurons']))
    for ds in DATASETS:
        for model in stage7['specification']['models']:
            require(sorted(r['seed'] for r in costs if r['dataset'] == ds and r['model'] == model) == SEEDS,
                    'Incomplete cost/model seed cohort')
    require(not any(destination.is_relative_to(local_path(p)) for p in e.packages),
            'Report destination is inside an authenticated source package')
    check_portability([SPEC, stage7['runs'][0]['checkpoint'], stage7['runs'][-1]['validation']['path']])
    for path in (ROOT / 'scripts/adaptation_stage8').glob('*.py'):
        e.bind(path)
    for path in ['cost_records.py', 'project_paths.py', 'tests/test_adaptation_stage8.py',
                 'config/path_aliases.yaml', 'config/paper_baseline.yaml']:
        e.bind(path)
    require(stage7['paper_baseline_version'] == 'paper_classification_v1', 'Unexpected source baseline')
    manifest = dict(version=spec['version'], specification=spec, output_root=portable_path(destination),
        source_files_sha256=e.files, authenticated_packages=sorted(set(e.packages)),
        datasets=DATASETS, training_seeds=SEEDS, checkpoints=54, comparison_pairs=PAIRS,
        source_baseline_version=stage7['paper_baseline_version'],
        scope='saved-evidence presentation and descriptive seed arithmetic only',
        training=False, new_inference=False, official_test_opened=False, new_fitting_or_resampling=False)
    if destination.exists():
        if (destination / 'MANIFEST.json').exists():
            previous = json.loads((destination / 'MANIFEST.json').read_text(encoding='utf-8'))
            require(previous == manifest, 'Synthesis manifest differs; preserve evidence and use a new --output-root')
        else:
            require(not any(destination.iterdir()), 'Nonempty output without synthesis manifest')
    return dict(root=destination, manifest=manifest, validations=validations,
        associations=associations, interventions=interventions, traces=traces,
        features=features, costs=costs, stage7_review=review, reports=reports,
        models=stage7['specification']['models'])

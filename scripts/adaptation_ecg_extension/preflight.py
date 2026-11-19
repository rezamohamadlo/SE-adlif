"""Read-only artifact, checkpoint and time-map contract checks."""
import contextlib
import hashlib
import io

import torch
import yaml
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_config, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ecg.data import array_hash, load_validation, require
from scripts.adaptation_ecg.preflight import guard, selected_metrics, validate_config, compare_configs
from scripts.adaptation_ecg_suite.preflight import numerical_contract, PAIRS
from scripts.adaptation_ecg_extension.inputs import ExtensionData, time_map

SPEC = 'config/studies/adaptation_ecg_extension_v1.yaml'


def load_spec():
    spec = yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    baseline = yaml.safe_load(local_path(spec['inherits']).read_text(encoding='utf-8'))['paper_baseline']
    require(spec['dataset_section'] == 'ECG' and not spec['training'] and not spec['official_test_access'],
            'Only frozen ECG validation extensions are supported')
    require(spec['conditions'] == [['clean', 0], ['delete_0.1', 0], ['delete_0.1', 1], ['delete_0.1', 2],
                                  ['time_scale_0.9', 0], ['time_scale_1.1', 0]] and
            spec['time_scaling'] == {'time_scale_0.9': .9, 'time_scale_1.1': 1.1} and
            spec['features'] == {'short_steps': 20, 'long_steps': 200}, 'Frozen ECG definitions differ')
    require(spec['analysis']['rank_relative_tolerance'] == 1e-10 and
            spec['analysis']['minimum_residual_df'] == 5 and spec['analysis']['inference'] is False,
            'Unsupported descriptive analysis policy')
    require(spec['analysis']['controls'] == ['log1p_events_per_step', 'log1p_stored_steps', 'class_fractions_1_to_5'],
            'ECG descriptive controls differ from the frozen design')
    return spec, baseline


def source_file(directory, name, spec):
    prefix = spec['source']['root'].rstrip('/')+'/'
    normalized = portable_path(name)
    require(normalized.startswith(prefix), 'Source output is outside its frozen study root')
    path = (directory/normalized[len(prefix):]).resolve()
    require(path.is_relative_to(directory.resolve()), 'Source path traversal is forbidden')
    return path


def source_directory(root, spec):
    snapshot = root/'shared/ecg_pairs_v2'
    return snapshot if (snapshot/'PATH_INDEX.json').exists() else local_path(spec['source']['root'])


def verify_source(directory, spec):
    for name, field in [('MANIFEST.json', 'manifest_sha256'), ('COMPLETE.json', 'completion_sha256')]:
        path = directory/name
        require(path.is_file() and pilot.sha256(path) == spec['source'][field], 'Frozen ECG source identity differs: '+name)
    manifest, complete = [pilot.read_json(directory/n) for n in ('MANIFEST.json', 'COMPLETE.json')]
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest) and
            complete['completed_pair_evaluations'] == [i for i, _p, _v in PAIRS] and len(complete['outputs']) == 96,
            'Source is not the completed four-pair ECG evaluation')
    for name, checksum in complete['outputs'].items():
        path = source_file(directory, name, spec)
        require(path.is_file() and pilot.sha256(path) == checksum, 'Source output missing or changed: '+name)
    anchors = pilot.read_json(directory/'CLEAN_ANCHOR_REVIEW.json')
    require(len(anchors) == 18 and len({(r['model'], r['seed']) for r in anchors}) == 18 and
            all(r['status'] == 'passed' for r in anchors), 'Source clean anchors are not complete/passing')
    tests = pilot.read_json(directory/'CONTRACT_TESTS.json')
    require(tests['status'] == 'passed' and tests['returncode'] == 0, 'Source contracts did not pass')
    return manifest, complete


def source_identity(source):
    paths = {local_path(p) for p in source['source_sha256']}
    paths.update((ROOT/'scripts/adaptation_ecg_extension').glob('*.py'))
    paths.update(local_path(p) for p in (SPEC, 'tests/test_adaptation_ecg_extension.py'))
    return {portable_path(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)}


def prepare(args):
    from models.pl_module import MLPSNN
    spec, baseline = load_spec()
    root = experiment_directory(args.output_root or spec['output_root'])
    source, _complete = verify_source(source_directory(root, spec), spec)
    for name, checksum in source['source_sha256'].items():
        require(hashlib.sha256(local_path(name, must_exist=True).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == checksum,
                'Completed ECG source dependency changed; preserve its frozen implementation: '+name)
    require([(p['id'], p['parent'], p['variant']) for p in source['specification']['pairs']] == PAIRS,
            'ECG source pair registry differs')
    resolved = dict(source['specification'], **spec)
    require(resolved['seeds'] == [42, 123, 456], 'ECG extension requires all three frozen seeds')
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or
            (torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count())), 'Requested device unavailable')
    env = dict(pilot.environment(device), numerical_contract=numerical_contract())
    require({k: v for k, v in env.items() if k != 'device'} ==
            {k: v for k, v in source['environment'].items() if k != 'device'},
            'Extension software/GPU/numerical environment differs from reused clean/deletion evidence')
    runs = source['runs']
    require(len(runs) == 18, 'Incomplete frozen ECG checkpoint registry')
    for run in runs:
        for path, checksum in [(run['checkpoint'], run['checkpoint_sha256']),
                (run['cost_record'], run['cost_record_sha256']), (run['validation']['path'], run['validation']['sha256'])]:
            require(pilot.sha256(local_path(path, must_exist=True)) == checksum, 'Recorded ECG artifact changed: '+path)
        checkpoint = torch.load(local_path(run['checkpoint']), weights_only=False, map_location='cpu')
        cfg = portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True))
        require(cfg == run['config'] and checkpoint['epoch'] == run['selected_epoch'], 'Frozen checkpoint config/epoch differs')
        validate_config(cfg, run, baseline)
        require(selected_metrics(run, baseline['datasets']['ECG']['n_epochs']) == run['validation'],
                'Frozen validation selection differs')
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del model, checkpoint
        print(f"ECG extension prerequisite OK: {run['model']} seed {run['seed']}", flush=True)
    for pair in resolved['pairs']:
        for seed in resolved['seeds']:
            configs = [next(r['config'] for r in runs if r['model'] == pair[k] and r['seed'] == seed)
                       for k in ('parent', 'variant')]
            compare_configs(*configs, pair['id'])
    base, metadata = load_validation(resolved, baseline, runs[0]['config'])
    require(metadata == source['data'], 'ECG loader/split/mask contract differs from completed evidence')
    maps = {}
    for condition, scale in spec['time_scaling'].items():
        mapping = time_map(metadata['steps_per_record'], scale)
        require(mapping.min() >= 0 and mapping.max() < metadata['steps_per_record'] and
                (mapping[1:] >= mapping[:-1]).all(), 'Invalid aligned time map')
        maps[condition] = dict(scale=scale, source_steps=metadata['steps_per_record'], target_steps=len(mapping),
            map_sha256=array_hash(mapping), mapping='floor((2*j+1)*T/(2*M)), M=floor(T*scale+0.5)')
    manifest = dict(version=spec['version'], specification=resolved, output_root=portable_path(root),
        scope='ecg_validation_features_aligned_resampling_descriptive_associations', source=spec['source'],
        data=metadata, time_maps=maps, runs=runs, source_sha256=source_identity(source),
        source_hash_encoding='CRLF_normalized_to_LF', environment=env,
        paper_baseline_version=baseline['version'], paper_baseline_dataset='ECG',
        expected_feature_rows=180, reused_prediction_chunks=72, new_prediction_chunks=36,
        expected_paired_rows=2160, training=False, official_test_opened=False, population_inference=False,
        limitations=['30 stored rows are not verified independent patients',
            'all-zero label vectors remain class 0 under the saved argmax contract',
            'resampling can change event counts and class proportions; it is a joint transformation',
            'feature windows/ACF use model steps, not verified physical milliseconds',
            'fixed-cohort descriptive coefficients and row influence have no population confidence intervals',
            'feature/control/rank/df failures are explicit, with no post-hoc model search'],
        remaining_stage6_items=['review ECG evidence and scientific scope', 'direct SSC window diagnostics',
                               'complete broader study/code audit if still outstanding'])
    guard(root, manifest)
    print('Extension preflight passed: 96 source hashes, 18 checkpoints, 30 validation records, 1171/1431-step maps.', flush=True)
    print('No feature extraction, mapped inference, tests, regression or scientific results executed.', flush=True)
    return dict(spec=resolved, root=root, manifest=manifest, data=ExtensionData(base, resolved))

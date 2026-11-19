"""Read-only frozen SSC source, six checkpoint and loader-contract verification."""
import contextlib
import hashlib
import io

import torch
import yaml
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_config, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ssc_suite.common import require, guard
from scripts.adaptation_ssc.data import metadata, SSCValidationData, check_legacy_contract
from scripts.adaptation_ssc.evaluate import selected_metrics
from scripts.adaptation_ssc_suite.preflight import validate_config_and_objective, compare_pair_configs

SPEC = 'config/studies/adaptation_ssc_windows_v1.yaml'
MODELS = ('MT_SE_adLIF', 'MT_TC_SE_adLIF')


def load_spec():
    spec = yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    baseline = yaml.safe_load(local_path(spec['inherits']).read_text(encoding='utf-8'))['paper_baseline']
    require(spec['dataset_section'] == 'SSC' and not spec['training'] and not spec['official_test_access'] and
            spec['window_frames'] == [5, 10, 20] and spec['stride'] == 'equal_to_window_width' and
            spec['burn_in_frames'] == 10 and spec['minimum_windows'] == 2 and spec['physical_bin_ms'] == 4.,
            'Frozen SSC window definition differs')
    require(all(type(w) is int for w in spec['window_frames']) and spec['models'] == list(MODELS),
            'SSC model/window types differ')
    require(len(spec['references']) == 6 and not spec['auxiliary_JS_in_prediction_loss'] and spec['no_benefit_regression'],
            'SSC diagnostic scope/loss definition differs')
    return spec, baseline


def source_directory(root, spec):
    directory = root/'shared/ssc_clean_evidence'
    return directory if (directory/'PATH_INDEX.json').exists() else local_path(spec['source']['root'])


def source_file(directory, suffix):
    require('..' not in suffix.replace('\\', '/').split('/'), 'Source traversal is forbidden')
    path = (directory/suffix).resolve()
    require(path.is_relative_to(directory.resolve()), 'Source path escapes the selected evidence package')
    return path


def verify_source(directory, spec):
    source = spec['source']
    for suffix, checksum in source['files'].items():
        path = source_file(directory, suffix)
        require(path.is_file() and pilot.sha256(path) == checksum, 'Selected source evidence missing/changed: '+suffix)
    manifest, complete = [pilot.read_json(directory/n) for n in ('MANIFEST.json', 'COMPLETE.json')]
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest) and
            len(complete['outputs']) == 230, 'Frozen SSC suite completion differs')
    for suffix, checksum in source['files'].items():
        if suffix != 'COMPLETE.json':
            require(complete['outputs'].get(source['root']+'/'+suffix) == checksum,
                    'Selected evidence is not authenticated by the source completion')
    return manifest


def read_cached(path):
    record = pilot.read_json(path)
    require(pilot.digest(record['value']) == record['value_sha256'], 'Selected clean cache content is damaged')
    return record


def source_identity(source):
    paths = {local_path(p) for p in source['source_sha256']}
    paths.update((ROOT/'scripts/adaptation_ssc_windows').glob('*.py'))
    paths.update(local_path(p) for p in (SPEC, 'scripts/adaptation_tc_windows/metrics.py',
                  'scripts/ssc_resume_io.py', 'tests/test_adaptation_ssc_windows.py', 'run.py'))
    return {portable_path(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)}


def prepare(args):
    from models.pl_module import MLPSNN
    spec, baseline = load_spec()
    root = experiment_directory(args.output_root or spec['output_root'])
    source_dir = source_directory(root, spec)
    source = verify_source(source_dir, spec)
    for path, checksum in source['source_sha256'].items():
        require(hashlib.sha256(local_path(path, must_exist=True).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == checksum,
                'Completed SSC dependency changed; preserve frozen sources: '+path)
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or
            (torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count())), 'Requested device unavailable')
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    env = pilot.environment(device)
    require({k: v for k, v in env.items() if k != 'device'} ==
            {k: v for k, v in source['environment'].items() if k != 'device'},
            'SSC software/GPU environment differs from the reference clean predictions')
    resolved = dict(source['specification'], **spec)
    require(resolved['seeds'] == [42, 123, 456] and resolved['loss_agg'] == 'mean_membrane_potentials',
            'Saved SSC seed/loss cohort differs')
    runs = [dict(r) for r in source['runs'] if r['model'] in MODELS]
    require(len(runs) == 6 and {(r['model'], r['seed']) for r in runs} ==
            {(m, s) for m in MODELS for s in resolved['seeds']}, 'Need six unique frozen MT/MT-TC checkpoints')
    features = read_cached(source_file(source_dir, spec['clean_features_file']))['value']
    ids = source['sample_ids']
    require(len(ids) == 9981 and [f['sample_id'] for f in features] == ids, 'Clean validation features/order differ')
    for run in runs:
        checkpoint_path = local_path(run['checkpoint'], must_exist=True)
        require(pilot.sha256(checkpoint_path) == run['checkpoint_sha256'], 'Selected checkpoint checksum differs')
        cost = pilot.read_json(local_path(run['cost_record'], must_exist=True))
        require(cost['status'] == 'complete' and cost['dataset'] == 'SSC' and cost['seed'] == run['seed'] and
                pilot.cost_model_identity(cost) == run['model'] and
                portable_path(cost['checkpoint']) == run['checkpoint'] and
                cost['checkpoint_sha256'] == run['checkpoint_sha256'] and cost['best_epoch'] == run['selected_epoch'] and
                cost['environment']['matmul_precision'] == 'high', 'Checkpoint/cost/precision identity differs')
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        cfg = portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True))
        require(cfg == run['config'] and checkpoint['epoch'] == run['selected_epoch'] and
                pilot.effective_config(cfg) == pilot.effective_config(portable_config(cost['saved_config'])),
                'Saved checkpoint config/epoch differs')
        validate_config_and_objective(cfg, run, resolved, baseline)
        require(cfg['paper_baseline'] == baseline, 'Saved dataset-specific baseline differs')
        require(selected_metrics(run, baseline['datasets']['SSC']['n_epochs']) == run['validation'],
                'Recorded validation selection differs')
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del model, checkpoint
        reference = next(r for r in spec['references'] if r['model'] == run['model'] and r['seed'] == run['seed'])
        cached = read_cached(source_file(source_dir, reference['file']))
        require(cached['identity']['checkpoint_sha256'] == run['checkpoint_sha256'] and
                cached['identity']['condition'] == 'clean' and cached['identity']['replicate'] == 0 and
                [r['sample_id'] for r in cached['value']] == ids, 'Reference clean prediction identities differ')
        for row, feature in zip(cached['value'], features):
            require(row['model'] == run['model'] and row['seed'] == run['seed'] and
                    all(row[k] == feature[k] for k in ('sample_id', 'target', 'speaker', 'input_sha256',
                        'original_support_frames', 'stored_frames')), 'Reference clean inputs do not pair')
        run.update(reference_file=reference['file'], cost_record_sha256=pilot.sha256(local_path(run['cost_record'])))
        print(f"SSC window prerequisite OK: {run['model']} seed {run['seed']}", flush=True)
    for seed in resolved['seeds']:
        compare_pair_configs(*[next(r['config'] for r in runs if r['model'] == model and r['seed'] == seed)
                               for model in MODELS], dict(id='mt_tc'))
    meta, labels, speakers = metadata(runs[0]['config'], resolved, baseline)
    require(meta == source['data'] and meta['bin_ms'] == spec['physical_bin_ms'] and
            meta['ignore_first_timesteps'] == spec['burn_in_frames'], 'SSC data/support/burn-in contract differs')
    data = SSCValidationData(runs[0]['config'], meta, labels, speakers, ids)
    contract = check_legacy_contract(data)
    manifest = dict(version=spec['version'], specification=resolved, output_root=portable_path(root),
        sample_ids=ids, runs=runs, data=meta, environment=dict(env, float32_matmul_precision='high', cudnn_benchmark=False),
        source_sha256=source_identity(source), source_hash_encoding='CRLF_normalized_to_LF', loader_preflight=contract,
        scope='ssc_clean_validation_direct_output_windows', source_scope='authenticated clean MT/MT-TC subset; not a full suite snapshot',
        paper_baseline_version=baseline['version'], paper_baseline_dataset='SSC', ordinary_loss=resolved['loss_agg'],
        expected_prediction_records=59886, expected_descriptor_records=179658, expected_paired_records=89829,
        original_batch_size=baseline['datasets']['SSC']['batch_size'],
        training=False, official_test_opened=False, auxiliary_JS_in_comparison_loss=False,
        interpretation='descriptive clean validation; no independent-window/patient inference, causal claim or confidence-benefit regression')
    guard(root, manifest)
    print('SSC windows preflight: 9981 clean records, six checkpoints, 20/40/80-ms windows; no forward or metrics executed.', flush=True)
    return dict(spec=resolved, root=root, manifest=manifest, data=data)

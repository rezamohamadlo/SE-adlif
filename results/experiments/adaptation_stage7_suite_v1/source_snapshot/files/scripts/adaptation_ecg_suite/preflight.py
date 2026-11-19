"""Numerical-contract and artifact checks; no inference, tests or analysis."""
import contextlib
import hashlib
import io

import torch
import yaml
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_config, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ecg.data import load_validation, require
from scripts.adaptation_ecg.preflight import (
    CELLS, compare_configs, guard, selected_metrics, source_identity as pilot_sources, validate_config,
)

SPEC = 'config/studies/adaptation_ecg_pairs_v2.yaml'
PAIRS = [('amt_mt', 'MT_SE_adLIF', 'AMT_SE_adLIF'), ('lt_dth', 'DTH_SE_adLIF', 'LT_DTH_SE_adLIF'),
         ('sa_mt', 'MT_SE_adLIF', 'SA_MT_SE_adLIF'), ('mt_tc', 'MT_SE_adLIF', 'MT_TC_SE_adLIF')]


def numerical_contract():
    """Match run.py; setting backend precision does not execute any model."""
    torch.set_float32_matmul_precision('high')
    require(torch.get_float32_matmul_precision() == 'high', 'Cannot establish the training precision contract')
    return dict(float32_matmul_precision='high', default_dtype=str(torch.get_default_dtype()),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                source='run.py main: torch.set_float32_matmul_precision("high")',
                amp=False, tolerances_relaxed=False)


def load_spec():
    spec = yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    registry = yaml.safe_load(local_path(spec['checkpoint_registry']).read_text(encoding='utf-8'))
    baseline = yaml.safe_load(local_path(spec['inherits']).read_text(encoding='utf-8'))['paper_baseline']
    require(spec['dataset_section'] == 'ECG' and not spec['training'] and not spec['official_test_access'],
            'Only validation-only saved-checkpoint ECG evaluations are supported')
    require([(p['id'], p['parent'], p['variant']) for p in registry['pairs']] == PAIRS,
            'Expected exactly the four registered ECG pairs')
    resolved = dict(registry, **spec)
    require(resolved['conditions'] == [['clean', 0], ['delete_0.1', 0], ['delete_0.1', 1], ['delete_0.1', 2]] and
            resolved['seeds'] == [42, 123, 456] and resolved['deletion_probability'] == .1,
            'Frozen ECG cohort/conditions differ')
    return resolved, baseline


def source_identity():
    paths = {local_path(p) for p in pilot_sources()}
    paths.update((ROOT/'scripts/adaptation_ecg_suite').glob('*.py'))
    paths.update(local_path(p) for p in (SPEC, 'run.py', 'tests/test_adaptation_ecg_suite.py'))
    return {portable_path(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)}


def failure_source(root, spec):
    snapshot = root/'reviews/failed_alignment_pilot_v1'
    source = snapshot if (snapshot/'PATH_INDEX.json').exists() else local_path(spec['failed_pilot']['root'])
    for suffix, checksum in spec['failed_pilot']['files'].items():
        path = source/suffix
        require(path.is_file() and pilot.sha256(path) == checksum,
                'Failed-pilot evidence missing or changed; transfer the exact recorded file: '+portable_path(path))
    return source


def prepare(args):
    from models.pl_module import MLPSNN
    numeric = numerical_contract()
    spec, baseline = load_spec()
    root = experiment_directory(args.output_root or spec['output_root'])
    failed = failure_source(root, spec)
    old_manifest = pilot.read_json(failed/'MANIFEST.json')
    for name, checksum in old_manifest['source_sha256'].items():
        current = hashlib.sha256(local_path(name, must_exist=True).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        require(current == checksum, 'Historical pilot dependency changed; preserve the frozen source: '+name)
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or
            (torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count())), 'Requested device unavailable')
    runs = []
    for frozen in spec['runs']:
        run = dict(frozen)
        cost = pilot.read_json(local_path(run['cost_record'], must_exist=True))
        require(cost['status'] == 'complete' and cost['dataset'] == 'ECG' and cost['seed'] == run['seed'] and
                pilot.cost_model_identity(cost) == run['model'] and
                portable_path(cost['checkpoint']) == run['checkpoint'] and
                cost['checkpoint_sha256'] == run['checkpoint_sha256'] and cost['best_epoch'] == run['selected_epoch'],
                'ECG checkpoint/cost identity differs')
        require(cost['environment']['matmul_precision'] == numeric['float32_matmul_precision'],
                'Recorded evaluator and queue matrix precision differ')
        path = local_path(run['checkpoint'], must_exist=True)
        require(pilot.sha256(path) == run['checkpoint_sha256'], 'Frozen ECG checkpoint checksum differs')
        checkpoint = torch.load(path, weights_only=False, map_location='cpu')
        cfg = portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True))
        require(pilot.effective_config(cfg) == pilot.effective_config(portable_config(cost['saved_config'])) and
                checkpoint['epoch'] == run['selected_epoch'] and
                cfg.get('temporal_consistency') == cost['saved_config'].get('temporal_consistency'),
                'ECG checkpoint config/epoch differs')
        validate_config(cfg, run, baseline)
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del model, checkpoint
        run.update(config=cfg, validation=selected_metrics(run, baseline['datasets']['ECG']['n_epochs']),
                   cost_record_sha256=pilot.sha256(local_path(run['cost_record'])))
        runs.append(run)
        print(f"ECG prerequisite OK: {run['model']} seed {run['seed']}, epoch {run['selected_epoch']}", flush=True)
    require(len(runs) == 18 and {(r['model'], r['seed']) for r in runs} ==
            {(m, s) for m in CELLS for s in spec['seeds']}, 'ECG registry is incomplete')
    for pair in spec['pairs']:
        for seed in spec['seeds']:
            configs = [next(r['config'] for r in runs if r['model'] == pair[k] and r['seed'] == seed)
                       for k in ('parent', 'variant')]
            compare_configs(*configs, pair['id'])
    data, metadata = load_validation(spec, baseline, runs[0]['config'])
    manifest = dict(version=spec['version'], specification=spec, output_root=portable_path(root),
        scope='ecg_four_pair_full_validation_clean_and_signed_deletion', runs=runs, data=metadata,
        environment=dict(pilot.environment(device), numerical_contract=numeric), source_sha256=source_identity(),
        source_hash_encoding='CRLF_normalized_to_LF', paper_baseline_version=baseline['version'],
        paper_baseline_dataset='ECG', training=False, official_test_opened=False, ordinary_inference=True,
        all_clean_anchors_required_before_deletion=True, expected_clean_anchors=18,
        expected_prediction_chunks=72, expected_paired_rows=1440,
        pending=['ECG input features', 'label-aligned time scaling', 'statistical inference with defensible dependence units',
                 'direct SSC window diagnostics', 'stage 7 gradients'],
        interpretation='four pair evaluations only; not four complete mechanism studies or independent-test evidence')
    guard(root, manifest)
    print('Preflight: all four ECG pairs; 30 validation records; high matrix precision; no model forward or tests.', flush=True)
    return dict(spec=spec, root=root, manifest=manifest, data=data)

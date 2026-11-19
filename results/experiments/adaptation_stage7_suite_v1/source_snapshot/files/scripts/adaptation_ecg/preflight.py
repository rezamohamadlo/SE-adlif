"""Read-only data/checkpoint/config/metrics checks, with no model forward."""
import contextlib
import csv
import hashlib
import io

import torch
import yaml
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_config, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ecg.data import load_validation, require

SPEC = 'config/studies/adaptation_ecg_alignment_pilot_v1.yaml'
CELLS = dict(MT_SE_adLIF='mt_se_adlif', AMT_SE_adLIF='amt_se_adlif', DTH_SE_adLIF='dth_se_adlif',
    LT_DTH_SE_adLIF='lt_dth_se_adlif', SA_MT_SE_adLIF='sa_mt_se_adlif', MT_TC_SE_adLIF='mt_se_adlif')


def selected_metrics(run, epochs):
    paths = list(local_path(run['cost_record']).parent.glob('logs/**/metrics.csv'))
    require(len(paths) == 1, 'Need exactly one original ECG metrics history')
    with paths[0].open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r.get('val_acc')]
    require(sorted(int(float(r['epoch'])) for r in rows) == list(range(epochs)),
            'Incomplete or overlapping ECG validation history')
    selected = next(r for r in rows if int(float(r['epoch'])) == run['selected_epoch'])
    require(float(selected['val_acc']) >= max(float(r['val_acc']) for r in rows)-1e-7,
            'ECG checkpoint is not validation selected')
    return dict(path=portable_path(paths[0]), sha256=pilot.sha256(paths[0]),
                accuracy=float(selected['val_acc']), loss=float(selected['val_loss']))


def validate_config(cfg, run, baseline):
    section, common, policy = baseline['datasets']['ECG'], baseline['common'], baseline['reproduction_policy']
    temporal = run['model'] == 'MT_TC_SE_adLIF'
    objective = 'mt_temporal_consistency_v1' if temporal else 'classification'
    expected = dict(random_seed=run['seed'], paper_baseline_version=baseline['version'],
        paper_baseline_dataset='ECG', two_layers=True, loss_agg='summed_membrane_potentials',
        early_stopping=policy['early_stopping'], evaluation_protocol=baseline['version'] +
        ('_mt_temporal_consistency_v1' if temporal else ''), model_variant=run['model'],
        **{k: section[k] for k in ('lr', 'batch_size', 'n_epochs')},
        **{k: common[k] for k in ('dropout', 'dt', 'alpha', 'gradient_clip_val', 'tracking_metric', 'tracking_mode')},
        **{k: policy[k] for k in ('scheduler', 'factor', 'patience')})
    require(all(cfg.get(k) == v for k, v in expected.items()), 'ECG saved baseline/protocol differs')
    require(cfg.get('paper_baseline') == baseline, 'Saved ECG paper baseline content differs')
    require(cfg.get('training_objective', 'classification') == objective and not cfg.get('auto_regression', False),
            'ECG objective differs')
    require(cfg.get('temporal_consistency') == (dict(weight=.1, window=5) if temporal else None),
            'ECG temporal training deviation differs')
    ds = cfg['dataset']
    require(ds['_target_'] == 'datasets.ecg.ECGLDM' and ds['num_classes'] == 6 and
            ds['batch_size'] == section['batch_size'] and ds['valid_fraction'] == section['valid_fraction'] and
            ds['random_seed'] == policy['split_seed'] and ds.get('burn_in_time', 0) == 0 and
            local_path(ds['data_path']) == local_path('data'), 'ECG saved data/split differs')
    arch = section['reference_architecture']
    for layer, inputs in [('l1', 4), ('l2', arch['neurons_per_layer'])]:
        require(cfg[layer]['cell'] == CELLS[run['model']] and cfg[layer]['input_size'] == inputs and
                cfg[layer]['n_neurons'] == arch['neurons_per_layer'] and
                all(cfg[layer][k] == arch[k] for k in ('q', 'c', 'tau_u_range', 'tau_w_range')) and
                all(cfg[layer][k] == common[k] for k in ('dt', 'alpha')),
                'ECG registered neuron architecture differs')
    require(cfg['l_out']['n_neurons'] == 6 and cfg['l_out']['input_size'] == arch['neurons_per_layer'] and
            cfg['l_out']['tau_u_range'] == [arch['output_tau']]*2 and
            cfg['l_out']['dt'] == common['dt'], 'ECG readout differs')


def compare_configs(parent, variant, pair):
    results = []
    allowed = {'cell'}
    if pair == 'amt_mt': allowed |= {'mix_rho_fast', 'mix_rho_slow'}
    if pair == 'lt_dth': allowed |= {'trace_rho_fast_range', 'trace_rho_slow_range'}
    for cfg in (parent, variant):
        item = pilot.effective_config(cfg)
        item.pop('model_variant', None)
        if pair == 'mt_tc':
            item.pop('training_objective', None)
            item.pop('evaluation_protocol', None)
        for layer in ('l1', 'l2'):
            item[layer] = {k: v for k, v in item[layer].items() if k not in allowed}
        for key in ('gradient_clip_val', 'factor', 'patience', 'tracking_metric', 'tracking_mode'):
            item[key] = cfg.get(key)
        results.append(item)
    require(results[0] == results[1], 'ECG pair settings differ beyond documented neuron/objective changes')


def source_identity():
    paths = {local_path(p) for p in pilot.source_identity()}
    paths.update((ROOT/'scripts/adaptation_ecg').glob('*.py'))
    paths.update(local_path(p) for p in [SPEC, 'config/paper_baseline.yaml', 'project_paths.py',
        'config/path_aliases.yaml', 'datasets/ecg.py', 'datasets/utils/pad_tensors.py',
        'scripts/ssc_resume_io.py', 'tests/test_adaptation_ecg.py'])
    return {portable_path(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(paths)}


def guard(root, manifest):
    path = root/'MANIFEST.json'
    if path.exists():
        pilot.validate_manifest(pilot.read_json(path), manifest)
    else:
        require(not root.exists() or not any(p.name != '.analysis.lock' for p in root.iterdir()),
                'Nonempty ECG output root has no manifest; preserve it and choose a new --output-root')


def prepare(args):
    from models.pl_module import MLPSNN
    spec = yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    baseline = yaml.safe_load(local_path(spec['inherits']).read_text(encoding='utf-8'))['paper_baseline']
    require(spec['dataset_section'] == 'ECG' and not spec['training'] and not spec['official_test_access'],
            'Only ECG validation diagnostics are supported')
    require(spec['conditions'] == [['clean', 0], ['delete_0.1', 0], ['delete_0.1', 1], ['delete_0.1', 2]] and
            spec['deletion_probability'] == .1, 'Frozen ECG pilot conditions differ')
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or
            (torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count())), 'Device unavailable')
    runs = []
    for frozen in spec['runs']:
        run = dict(frozen)
        cost = pilot.read_json(local_path(run['cost_record'], must_exist=True))
        require(cost['status'] == 'complete' and cost['dataset'] == 'ECG' and cost['seed'] == run['seed'] and
                pilot.cost_model_identity(cost) == run['model'] and
                portable_path(cost['checkpoint']) == run['checkpoint'] and
                cost['checkpoint_sha256'] == run['checkpoint_sha256'] and cost['best_epoch'] == run['selected_epoch'],
                'ECG cost/checkpoint identity differs')
        checkpoint_path = local_path(run['checkpoint'], must_exist=True)
        require(pilot.sha256(checkpoint_path) == run['checkpoint_sha256'], 'ECG checkpoint checksum differs')
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        cfg = portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True))
        require(pilot.effective_config(cfg) == pilot.effective_config(portable_config(cost['saved_config'])) and
                checkpoint['epoch'] == run['selected_epoch'] and
                cfg.get('temporal_consistency') == cost['saved_config'].get('temporal_consistency'),
                'ECG checkpoint saved config/epoch differs')
        validate_config(cfg, run, baseline)
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del model, checkpoint
        run.update(config=cfg, validation=selected_metrics(run, baseline['datasets']['ECG']['n_epochs']),
                   cost_record_sha256=pilot.sha256(local_path(run['cost_record'])))
        runs.append(run)
        print(f"ECG prerequisite OK: {run['model']} seed {run['seed']}, epoch {run['selected_epoch']}", flush=True)
    require({(r['model'], r['seed']) for r in runs} == {(m, s) for m in CELLS for s in spec['seeds']} and
            len(runs) == 18 and spec['seeds'] == [42, 123, 456], 'Incomplete ECG checkpoint registry')
    for pair in spec['pairs']:
        for seed in spec['seeds']:
            configs = [next(r['config'] for r in runs if r['model'] == pair[k] and r['seed'] == seed)
                       for k in ('parent', 'variant')]
            compare_configs(*configs, pair['id'])
    data, metadata = load_validation(spec, baseline, runs[0]['config'])
    root = experiment_directory(args.output_root or spec['output_root'])
    manifest = dict(version=spec['version'], specification=spec, output_root=portable_path(root),
        scope='ecg_validation_alignment_and_signed_deletion_pilot', readiness_runs=runs,
        runs=[r for r in runs if r['model'] in (spec['parent'], spec['variant'])], data=metadata,
        environment=pilot.environment(device), source_sha256=source_identity(), source_hash_encoding='CRLF_normalized_to_LF',
        paper_baseline_version=baseline['version'], paper_baseline_dataset='ECG',
        training=False, official_test_opened=False, ordinary_inference=True,
        comparison_loss='supervised CE on one temporal block per valid step; no auxiliary training penalties',
        limitations=['pilot only; not four completed ECG mechanism studies',
                    'no input features, time warping, regression, bootstrap or independent test claims',
                    'no patient IDs; no assumption of 30 independent patients',
                    'preserves original argmax of all-zero class vectors as class 0; no new exclusion mask'])
    guard(root, manifest)
    print('ECG readiness: 18 checkpoints; 30 validation records; six AMT/MT pilot anchors; no model forward.', flush=True)
    return spec, root, manifest, data

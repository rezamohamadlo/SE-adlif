"""Read-only stage-7 pilot artifacts, count inputs and frozen weights; no backward."""
import contextlib
import hashlib
import io

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_config, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_analysis.shd_data import SHDValidationData, preflight_shd, _select
from scripts.adaptation_gradient_pilot.methods import require, state_hash

SPEC = 'config/studies/adaptation_gradient_pilot_v1.yaml'
MODELS = ('MT_SE_adLIF', 'AMT_SE_adLIF')


def load_spec():
    spec = yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    baseline = yaml.safe_load(local_path(spec['inherits']).read_text(encoding='utf-8'))['paper_baseline']
    require(spec['dataset_section'] == 'SHD' and spec['report_label'] == 'SHD*' and spec['models'] == list(MODELS) and
            spec['seeds'] == [42] and spec['samples'] == 40 and spec['diagnostic_batch_size'] == 1 and
            spec['deletion_fraction'] == .01 and spec['random_replicates'] == 3 and spec['profile_bins'] == 20 and
            spec['burn_in_frames'] == 10 and spec['autograd_forward_loss_tolerance'] == 1e-6 and
            not spec['training'] and not spec['official_test_access'] and not spec['input_optimization'],
            'Frozen stage-7 pilot scope differs')
    return spec, baseline


def source_file(directory, suffix):
    require('..' not in suffix.replace('\\', '/').split('/'), 'Source traversal is forbidden')
    result = (directory/suffix).resolve()
    require(result.is_relative_to(directory.resolve()), 'Source file escapes its snapshot')
    return result


def source_directory(root, spec):
    snapshot = root/'shared/shd_clean_source'
    return snapshot if (snapshot/'PATH_INDEX.json').exists() else local_path(spec['source']['root'])


def verify_source(directory, spec):
    for suffix, checksum in spec['source']['files'].items():
        path = source_file(directory, suffix)
        require(path.is_file() and pilot.sha256(path) == checksum, 'Frozen clean source missing/changed: '+suffix)
    m, c = [pilot.read_json(directory/n) for n in ('MANIFEST.json', 'COMPLETE.json')]
    require(c['status'] == 'complete' and c['manifest_sha256'] == pilot.digest(m), 'Frozen source completion differs')
    for suffix, checksum in spec['source']['files'].items():
        if suffix != 'COMPLETE.json':
            require(c['outputs'].get(spec['source']['root']+'/'+suffix) == checksum, 'Source file is not completion-authenticated')
    return m


def read_cache(path):
    record = pilot.read_json(path)
    require(pilot.digest(record['value']) == record['value_sha256'], 'Cache content digest differs')
    return record


def guard(root, manifest):
    path = root/'MANIFEST.json'
    if path.exists(): pilot.validate_manifest(pilot.read_json(path), manifest)
    else:
        require(not root.exists() or not any(p.name != '.analysis.lock' for p in root.iterdir()),
                'Nonempty pilot root without matching manifest; preserve it and choose a new --output-root')


def source_identity(source):
    paths = {local_path(p) for p in source['source_sha256']}
    paths.update((ROOT/'scripts/adaptation_gradient_pilot').glob('*.py'))
    paths.update(local_path(p) for p in (SPEC, 'scripts/ssc_resume_io.py', 'tests/test_adaptation_gradient_pilot.py', 'run.py'))
    return {portable_path(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)}


def measurement_schema(cost, run, cfg):
    legacy = cost.get('version') == 'shd_costs_v1' and 'status' not in cost
    if legacy:
        identity = cost.get('measurement_identity', '')
        require(isinstance(identity,str) and len(identity)==64 and all(c in '0123456789abcdef' for c in identity) and
                cost.get('dataset') in (None,'SHD') and cost['saved_config'].get('paper_baseline_dataset')=='SHD' and
                cost['saved_config']['dataset'].get('_target_')=='datasets.shd.SHDLDM',
                'Legacy SHD measurement identity/schema differs')
    else:
        require(cost.get('status')=='complete' and cost.get('dataset')=='SHD','Modern SHD measurement is not complete')
    require(cost['seed']==run['seed'] and pilot.cost_model_identity(cost)==run['model'] and
            portable_path(cost['checkpoint'])==run['checkpoint'] and cost['checkpoint_sha256']==run['checkpoint_sha256'] and
            cost['best_epoch']==run['selected_epoch'] and cost['environment']['matmul_precision']=='high' and
            pilot.effective_config(portable_config(cost['saved_config']))==pilot.effective_config(cfg),
            'Saved measurement/checkpoint/config/precision identity differs')
    return 'legacy_shd_costs_v1_without_status_field' if legacy else 'complete_classification_cost_record'


def prepare(args):
    from models.pl_module import MLPSNN
    spec, baseline = load_spec()
    root = experiment_directory(args.output_root or spec['output_root'])
    source_dir = source_directory(root, spec)
    source = verify_source(source_dir, spec)
    audit_dir = root/'shared/stage6_closeout'
    audit_path = audit_dir/'STAGE6_CLOSEOUT_AUDIT.json' if (audit_dir/'STAGE6_CLOSEOUT_AUDIT.json').exists() else local_path(spec['stage6_audit']['record'])
    require(pilot.sha256(audit_path) == spec['stage6_audit']['sha256'], 'Scoped stage-6 closeout record changed')
    audit = pilot.read_json(audit_path)
    require(audit['stage6_scoped_complete'] and not audit['scientific_execution'], 'Need the recorded scoped closeout decision')
    document = audit_dir/'SCOPE_AUDIT.md' if (audit_dir/'SCOPE_AUDIT.md').exists() else local_path(audit['audit_path'], must_exist=True)
    require(pilot.sha256(document) == audit['audit_sha256'], 'Closeout audit document hash differs')
    for name, checksum in source['source_sha256'].items():
        require(hashlib.sha256(local_path(name, must_exist=True).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == checksum,
                'Completed SHD* dependency changed; preserve the source version: '+name)
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or
            (torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count())), 'Requested device unavailable')
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    require(torch.get_default_dtype() == torch.float32, 'Pilot requires the saved float32 model/input dtype')
    env = pilot.environment(device)
    require({k: v for k, v in env.items() if k != 'device'} == {k: v for k, v in source['environment'].items() if k != 'device'},
            'Pilot environment differs from reused source evidence')
    resolved = dict(source['resolved_study'], **spec)
    features = read_cache(source_file(source_dir, spec['clean_features_file']))['value']
    full_ids = source['sample_ids']
    require(len(full_ids) == 1631 and [f['sample_id'] for f in features] == full_ids, 'Full SHD* reference cohort differs')
    runs = [dict(r) for r in source['runs'] if r['model'] in MODELS and r['seed'] == 42]
    require(len(runs) == 2 and {r['model'] for r in runs} == set(MODELS), 'Need exactly the seed-42 MT/AMT checkpoints')
    for run in runs:
        path = local_path(run['checkpoint'], must_exist=True)
        require(pilot.sha256(path) == run['checkpoint_sha256'], 'Frozen checkpoint hash differs')
        checkpoint = torch.load(path, weights_only=False, map_location='cpu')
        cfg = portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True))
        require(cfg == run['config'] and checkpoint['epoch'] == run['selected_epoch'] and cfg['paper_baseline'] == baseline,
                'Saved checkpoint config/epoch/baseline differs')
        pilot.check_saved_config(cfg, run, source['resolved_study'])
        require(pilot.selected_metrics(run) == run['validation'], 'Validation selection/history differs')
        cost = pilot.read_json(local_path(run['cost_record'], must_exist=True))
        schema=measurement_schema(cost,run,cfg)
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.requires_grad_(False).eval()
        run.update(model_state_sha256=state_hash(model),cost_record_sha256=pilot.sha256(local_path(run['cost_record'])),
                   measurement_schema=schema)
        del model, checkpoint
        ref = next(r for r in spec['references'] if r['model'] == run['model'])
        cache = read_cache(source_file(source_dir, ref['file']))
        require(cache['identity']['checkpoint_sha256'] == run['checkpoint_sha256'] and
                cache['identity']['condition'] == 'clean' and cache['identity']['replicate'] == 0 and
                [r['sample_id'] for r in cache['value']] == full_ids, 'Clean source prediction identities differ')
        for row, f in zip(cache['value'], features):
            require(row['model'] == run['model'] and row['seed'] == 42 and
                    all(row[k] == f[k] for k in ('sample_id', 'target', 'speaker', 'input_sha256', 'original_support_frames', 'stored_frames')),
                    'Clean source inputs do not pair')
        run['reference_file'] = ref['file']
        print(f"Gradient pilot prerequisite OK: {run['model']} seed 42, epoch {run['selected_epoch']}", flush=True)
    comparable = []
    for run in runs:
        item = pilot.effective_config(run['config'])
        item.pop('model_variant', None)
        for layer in ('l1', 'l2'):
            item[layer] = {k: v for k, v in item[layer].items() if k not in ('cell', 'mix_rho_fast', 'mix_rho_slow')}
        for key in ('gradient_clip_val', 'factor', 'patience', 'tracking_metric', 'tracking_mode'):
            item[key] = run['config'].get(key)
        comparable.append(item)
    require(comparable[0] == comparable[1], 'Pilot parent/variant settings differ beyond registered neuron differences')
    metadata = preflight_shd(runs[0]['config'])
    require(metadata == source['data'], 'SHD* file, split or preprocessing differs from completed evidence')
    data = SHDValidationData(runs[0]['config'], sample_ids=full_ids, verify_hash=False)
    selected = _select(data.labels, full_ids, spec['samples'], spec['cohort_selection_seed'])
    classes = {str(k): sum(int(data.labels[i]) == k for i in selected) for k in range(20)}
    require(len(selected) == 40 and set(classes.values()) == {2}, 'Pilot must contain exactly two records per class')
    expected = {r['sample_id']: r for r in features}
    for _sid, _counts, row in pilot.input_rows(data, selected[:3], ('clean', 0, 'identity', 1.), source['resolved_study'], features=False):
        require(all(row[k] == expected[row['sample_id']][k] for k in ('target', 'speaker', 'input_sha256', 'original_support_frames', 'stored_frames')),
                'Input-only preprocessing contract differs from completed clean evidence')
    manifest = dict(version=spec['version'], specification=resolved, source_study=source['resolved_study'],
        output_root=portable_path(root), scope='stage7_shd_star_mt_amt_seed42_surrogate_input_gradient_pilot',
        runs=runs, data=metadata, pilot_ids=selected, class_counts=classes, full_validation_ids=full_ids,
        pilot_speaker_counts={str(k):sum(int(data.speakers[i])==int(k) for i in selected)
                              for k in sorted(set(int(data.speakers[i]) for i in selected))},
        paper_baseline_version=baseline['version'], paper_baseline_dataset='SHD', report_label='SHD*',
        source_sha256=source_identity(source), source_hash_encoding='CRLF_normalized_to_LF',
        environment=dict(env, float32_matmul_precision='high', cudnn_benchmark=False),
        batching=dict(saved_training_batch_size=baseline['datasets']['SHD']['batch_size'],
            original_batch_full_validation_anchors=True, diagnostic_batch_size=1,
            diagnostic_mode_source_drift='record per sample; not silently relabeled as original-batch validation',
            autograd_forward_equivalence_required=True),
        expected_gradients=80, maximum_paired_condition_records=200, training=False, official_test_opened=False,
        input_optimization=False, stage7_complete=False,
        limitations=['one seed, one pair and forty validation recordings are tooling evidence, not full stage 7',
            'SLAYER surrogate input sensitivity is not a true hard-spike finite-difference derivative or causal attribution',
            'single-record diagnostic batching is a declared evaluation deviation; preserve original-batch anchors separately',
            'physical support includes burn-in inputs; synthetic-padding gradients are recorded separately',
            'MT-derived absolute-gradient deletion inputs are shared by both models; selection uses model output sensitivity',
            'zero event-gradient guidance is explicitly ineligible, not fabricated; no direction-agreement completion gate',
            'no statistical significance, population confidence interval, TTA, weight training or optimized diagnostic inputs'])
    guard(root, manifest)
    print('Pilot preflight passed: SHD*, 40 validation records (2/class), seed 42, two frozen checkpoints.', flush=True)
    print('No forward, backward, gradient summary, deletion generation, tests or scientific execution.', flush=True)
    return dict(root=root, spec=resolved, manifest=manifest, data=data)

"""Reference/checkpoint/input preflight only, never forward/backward or tests."""
from argparse import Namespace
import hashlib

import yaml

from datasets.utils.pad_tensors import PadTensors
from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_batch_precision.contracts import batch_identity, precision_mode
from scripts.adaptation_batch_precision.preflight import prepare as reference_prepare
from scripts.adaptation_batch_precision.runtime import verify as verify_reference
from scripts.adaptation_gradient_pilot.methods import count_hash, require
from scripts.adaptation_gradient_pilot.preflight import guard, read_cache

SPEC = 'config/studies/adaptation_gradient_strict_v1.yaml'
TEST = 'tests/test_adaptation_gradient_strict.py'


def load_spec():
    spec = yaml.safe_load(local_path(SPEC, must_exist=True).read_text(encoding='utf-8'))
    expected = dict(inherits='config/paper_baseline.yaml', dataset_section='SHD', report_label='SHD*',
        precision_mode='highest_matmul', models=['MT_SE_adLIF','AMT_SE_adLIF'], seeds=[42], samples=40,
        repeats=2, focus_sample_id=3896, layouts=['singleton_natural','singleton_original_padding_focus'],
        diagnostic_batch_size=1, burn_in_frames=10, profile_bins=20, anchor_accuracy_tolerance=1e-6,
        anchor_loss_tolerance=.001, score_atol=2e-5, score_rtol=1e-5,
        gradient_relative_l2_limit=.01, gradient_cosine_minimum=.999,
        training=False, official_test_access=False, input_optimization=False, event_deletion=False, stage7_complete=False)
    require(all(spec[k] == v for k,v in expected.items()), 'Versioned strict gradient scope differs')
    return spec


def prepare(args):
    spec = load_spec();source = experiment_directory(spec['reference_root'])
    require(pilot.sha256(source/'MANIFEST.json') == spec['reference_manifest_sha256'] and
            pilot.sha256(source/'COMPLETE.json') == spec['reference_complete_sha256'], 'Strict reference study changed')
    verify_reference(source)
    ready = reference_prepare(Namespace(device=args.device, output_root=spec['reference_root']))
    old, data = ready['manifest'], ready['data']
    root = experiment_directory(args.output_root or spec['output_root'])
    legacy = ready['pilot_root']
    require(all(root != p and not root.is_relative_to(p) and not p.is_relative_to(root) for p in (source,legacy)),
            'Strict pilot must be outside both frozen studies')
    hashes = dict(old['source_sha256'])
    files = list((ROOT/'scripts/adaptation_gradient_strict').glob('*.py'))+[local_path(SPEC), local_path(TEST)]
    for path in files:hashes[portable_path(path)] = hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    for name, expected in hashes.items():
        require(hashlib.sha256(local_path(name, must_exist=True).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == expected,
                'Frozen dependency differs: '+name)
    old_manifest = pilot.read_json(legacy/'MANIFEST.json')
    anchors = {r['model']:read_cache(legacy/'anchors'/f"{r['model']}_seed_42.json")['value'] for r in old['runs']}
    original = {r['sample_id']:r for r in anchors[spec['models'][0]]}
    input_identities = {}
    for sid in old['pilot_ids']:
        item = data.load(sid)
        require(count_hash(item['counts']) == original[sid]['input_sha256'], 'Pilot physical input differs')
        input_identities[str(sid)] = batch_identity(PadTensors()([data.model_item(sid,item['counts'])]))
    strict_references = {r['model']:[] for r in old['runs']}
    for run in old['runs']:
        for b, ids in enumerate(old['original_batch_ids']):
            cache = read_cache(source/'jobs'/run['model']/spec['precision_mode']/f'original_batch_{b}.json')
            require(cache['identity']['sample_ids'] == ids and not cache['identity']['backend']['matmul_allow_tf32'],
                    'Same-mode original reference batch differs')
            strict_references[run['model']].extend(dict(sample_id=sid,**r) for sid,r in zip(ids,cache['value']))
        require([r['sample_id'] for r in strict_references[run['model']]] == old['full_validation_ids'],
                'Strict validation reference coverage differs')
    with precision_mode(spec['precision_mode'],old['source_backend']) as backend:
        require(backend['matmul_precision'] == 'highest' and not backend['matmul_allow_tf32'], 'Strict precision unavailable')
        backend = dict(backend)
    source_backend = next(r['backend'] for r in pilot.read_json(source/'FULL_VALIDATION_CHECKS.json')
                          if r['mode'] == spec['precision_mode'])
    require(backend == source_backend, 'Requested backend differs from strict reference')
    manifest = dict(version=spec['version'], specification=spec, output_root=portable_path(root),
        reference_manifest_sha256=pilot.digest(old), reference_complete_sha256=spec['reference_complete_sha256'],
        source_sha256=hashes, source_hash_encoding='CRLF_normalized_to_LF',
        environment=dict(old['environment'],float32_matmul_precision='highest'), backend=backend,
        source_backend=old['source_backend'], runs=old['runs'], pilot_ids=old['pilot_ids'],
        full_validation_ids=old['full_validation_ids'], original_batch_ids=old['original_batch_ids'],
        focus_original_batch_index=old['focus_original_batch_index'], focus_original_slot=old['focus_original_slot'],
        focus_padded_frames=old['focus_original_shape'][1], data=old['data'], input_identities=input_identities,
        paper_baseline_version=old_manifest['paper_baseline_version'], paper_baseline_dataset='SHD', report_label='SHD*',
        evaluation_deviations=['highest_matmul instead of saved source_high',
            'single-record diagnostic gradients', 'strict validation references belong to numerical diagnostic, not training selection'],
        expected_gradients=164, expected_natural_repeat_pairs=80, expected_padding_repeat_pairs=2,
        expected_padding_comparisons=4, expected_cross_precision_comparisons=80,
        training=False, official_test_opened=False, input_optimization=False, event_deletion=False, stage7_complete=False,
        threshold_role='proposed numerical tooling criteria; negative stability results are saved, not suppressed')
    guard(root,manifest)
    print('Strict gradient preflight passed: highest_matmul, 2 checkpoints, 40 fixed pilot records.',flush=True)
    print('Planned: 2 same-mode full-validation anchors; 164 input backward passes, including repeats/focus padding.',flush=True)
    print('No tests, forward, backward, gradient summaries or experiments executed.',flush=True)
    return dict(root=root,spec=spec,manifest=manifest,data=data,reference_root=source,legacy_root=legacy)

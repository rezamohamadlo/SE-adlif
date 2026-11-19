"""Authenticate the executed pilot and input layouts, without tests or forward."""
from argparse import Namespace
import hashlib

import torch
import yaml

from datasets.utils.pad_tensors import PadTensors
from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_batch_precision.contracts import (
    MODES, LAYOUTS, FOCUS, backend_settings, batch_identity, singleton_from_original,
)
from scripts.adaptation_gradient_pilot.methods import require
from scripts.adaptation_gradient_pilot.preflight import prepare as pilot_prepare, guard
from scripts.adaptation_gradient_pilot.runtime import verify as verify_pilot

SPEC = 'config/studies/adaptation_batch_precision_v1.yaml'
TEST = 'tests/test_adaptation_batch_precision.py'


def load_spec():
    spec = yaml.safe_load(local_path(SPEC, must_exist=True).read_text(encoding='utf-8'))
    require(spec['models'] == ['MT_SE_adLIF', 'AMT_SE_adLIF'] and spec['seeds'] == [42] and
            spec['samples'] == 40 and spec['focus_sample_id'] == 3896 and
            spec['precision_modes'] == list(MODES) and spec['layouts'] == list(LAYOUTS) and
            spec['focus_layouts'] == list(FOCUS) and spec['reproduction_loss_tolerance'] == 1e-6 and
            spec['inherits'] == 'config/paper_baseline.yaml' and spec['dataset_section'] == 'SHD' and
            spec['report_label'] == 'SHD*' and
            all(spec[k] is False for k in ('training', 'official_test_access', 'backward', 'input_optimization', 'stage7_complete')),
            'Versioned diagnostic scope differs')
    return spec


def prepare(args):
    spec = load_spec()
    source_root = experiment_directory(spec['pilot_root'])
    require(pilot.sha256(source_root/'MANIFEST.json') == spec['pilot_manifest_sha256'] and
            pilot.sha256(source_root/'COMPLETE.json') == spec['pilot_complete_sha256'], 'Executed pilot identity differs')
    verify_pilot(source_root)
    prepared = pilot_prepare(Namespace(device=args.device, output_root=spec['pilot_root']))
    old, data = prepared['manifest'], prepared['data']
    require(torch.device(args.device).type == 'cuda', 'This follow-up isolates batching on the original CUDA device type')
    require(old['runs'][0]['config']['paper_baseline'] ==
            yaml.safe_load(local_path(spec['inherits']).read_text())['paper_baseline'], 'Inherited baseline differs')
    source_backend = backend_settings()
    require(source_backend['matmul_precision'] == 'high' and source_backend['matmul_allow_tf32'] and
            not source_backend['cudnn_benchmark'], 'Original high precision backend is not available')
    root = experiment_directory(args.output_root or spec['output_root'])
    require(root != source_root and not root.is_relative_to(source_root) and not source_root.is_relative_to(root),
            'Follow-up must be separate from the frozen pilot')
    source_hashes = dict(old['source_sha256'])
    dependencies = list((ROOT/'scripts/adaptation_batch_precision').glob('*.py'))
    dependencies += [local_path(p) for p in (SPEC, TEST, 'project_paths.py', 'scripts/adaptation_analysis/records.py')]
    for path in dependencies:
        source_hashes[portable_path(path)] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    for name, sha in source_hashes.items():
        require(hashlib.sha256(local_path(name, must_exist=True).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == sha,
                'Frozen dependency changed: '+name)
    ids = old['full_validation_ids']; size = old['runs'][0]['config']['dataset']['batch_size']
    batches = [ids[i:i+size] for i in range(0, len(ids), size)]
    focus = spec['focus_sample_id']; group = next(b for b in batches if focus in b)
    # Input construction is preflight, not model execution. Authenticate focus
    # and complete companion batch; runtime authenticates every batch similarly.
    anchor = pilot.read_json(source_root/'anchors'/f"{spec['models'][0]}_seed_42.json")['value']
    expected = {r['sample_id']: r for r in anchor}
    items = []
    from scripts.adaptation_gradient_pilot.methods import count_hash
    for sid in group:
        counts = data.load(sid)['counts']
        require(count_hash(counts) == expected[sid]['input_sha256'], 'Original companion input differs')
        items.append(data.model_item(sid, counts))
    batch = PadTensors()(items)
    focus_slot = group.index(focus)
    focus_natural = singleton_from_original(batch, focus_slot, expected[focus]['stored_frames'], False)
    focus_padded = singleton_from_original(batch, focus_slot, expected[focus]['stored_frames'], True)
    natural_reference = PadTensors()([data.model_item(focus)])
    require(batch_identity(focus_natural) == batch_identity(natural_reference),
            'Slicing the original batch does not reproduce the pilot singleton input/target/mask')
    manifest = dict(version=spec['version'], specification=spec, output_root=portable_path(root),
        pilot_manifest_sha256=pilot.digest(old), pilot_complete_sha256=spec['pilot_complete_sha256'],
        source_sha256=source_hashes, source_hash_encoding='CRLF_normalized_to_LF',
        runs=old['runs'], pilot_ids=old['pilot_ids'], full_validation_ids=ids,
        data=old['data'], environment=old['environment'], source_backend=source_backend,
        original_batch_ids=batches, original_batch_sizes=[len(b) for b in batches],
        focus_original_batch_index=batches.index(group), focus_original_slot=group.index(focus),
        focus_original_shape=list(batch[0].shape),
        focus_input_layouts=dict(original=batch_identity(batch), singleton_natural=batch_identity(focus_natural),
                                singleton_original_padding=batch_identity(focus_padded)),
        expected_full_validation_anchors=6, expected_pilot_layout_rows=720,
        expected_focus_control_rows=24, expected_forward_calls=546,
        expected_full_validation_score_rows=9786,
        paper_baseline_version=old['paper_baseline_version'], paper_baseline_dataset='SHD', report_label='SHD*',
        training=False, backward=False, input_optimization=False, official_test_opened=False, stage7_complete=False,
        scope='diagnostic numerical follow-up; not replacement evidence, model selection or gradient expansion')
    guard(root, manifest)
    print('Batch/precision preflight passed: 2 checkpoints, 40 pilot IDs, 3 modes, 7 original batches.', flush=True)
    print('No tests, forward, backward or scientific results executed. Focus batch shape:', tuple(batch[0].shape), flush=True)
    return dict(root=root, spec=spec, manifest=manifest, data=data, pilot_root=source_root)

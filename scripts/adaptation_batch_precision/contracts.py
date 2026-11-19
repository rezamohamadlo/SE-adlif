"""Batch transformations and explicit backend settings; no model execution."""
from contextlib import contextmanager

import torch

from scripts.adaptation_gradient_pilot.methods import array_hash, require

MODES = ('source_high', 'highest_matmul', 'highest_matmul_cudnn_disabled')
LAYOUTS = ('original_batch', 'singleton_natural', 'singleton_original_padding')
FOCUS = ('original_repeat', 'singleton_repeat', 'replicated_original_padding', 'reversed_original')


def backend_settings():
    return dict(matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


@contextmanager
def precision_mode(mode, source):
    require(mode in MODES, 'Unregistered precision mode')
    previous = backend_settings()
    try:
        torch.set_float32_matmul_precision('high' if mode == 'source_high' else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = mode == 'source_high'
        torch.backends.cudnn.allow_tf32 = source['cudnn_allow_tf32'] if mode != MODES[2] else False
        torch.backends.cudnn.benchmark = source['cudnn_benchmark']
        torch.backends.cudnn.deterministic = source['cudnn_deterministic']
        yield backend_settings()
    finally:
        torch.set_float32_matmul_precision(previous['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = previous['matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = previous['cudnn_allow_tf32']
        torch.backends.cudnn.benchmark = previous['cudnn_benchmark']
        torch.backends.cudnn.deterministic = previous['cudnn_deterministic']


def singleton_from_original(batch, slot, stored_frames, retain_padding):
    x, target, blocks = batch
    require(0 <= slot < len(x) and 0 < stored_frames <= x.shape[1], 'Invalid original slot/length')
    end = x.shape[1] if retain_padding else stored_frames
    require(not torch.any(x[slot, stored_frames:]) and not torch.any(blocks[slot, stored_frames:]),
            'Cannot truncate non-padding input or blocks')
    return x[slot:slot+1, :end].clone(), target[slot:slot+1].clone(), blocks[slot:slot+1, :end].clone()


def repeated_target(batch, slot):
    x, target, blocks = batch
    require(0 <= slot < len(x), 'Invalid repeated-target slot')
    return tuple(v[slot:slot+1].repeat((len(x),)+(1,)*(v.ndim-1)) for v in (x, target, blocks))


def reversed_batch(batch):
    return tuple(v.flip(0) for v in batch)


def batch_identity(batch):
    x, target, blocks = batch
    require(x.ndim == 3 and x.shape[2] == 140 and x.dtype == torch.float32 and
            target.dtype == blocks.dtype == torch.int64 and blocks.shape == x.shape[:2] and
            target.shape == (len(x), 2), 'Batch dtype/shape contract differs')
    return dict(shape=list(x.shape), input_sha256=array_hash(x.numpy()),
                targets_sha256=array_hash(target.numpy()), blocks_sha256=array_hash(blocks.numpy()))


def compare_score(row, reference):
    return dict(prediction_changed=row['prediction'] != reference['prediction'],
                loss_difference=row['loss']-reference['loss'])


def require_original_anchor(anchor, original_spec):
    if anchor['mode'] == MODES[0]:
        require(anchor['samples'] == 1631 and anchor['source_prediction_mismatches'] == 0 and
                abs(anchor['accuracy_difference']) <= original_spec['accuracy_tolerance'] and
                abs(anchor['loss_difference']) <= original_spec['loss_tolerance'],
                'Source-high original anchor failed; no tolerances relaxed. Inspect saved jobs.')

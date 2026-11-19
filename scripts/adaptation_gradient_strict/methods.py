"""Explicit repeat criteria and frozen-weight surrogate input differentiation."""
import numpy as np
import torch

from scripts.adaptation_gradient_pilot.engine import ordinary, scores, supervised
from scripts.adaptation_gradient_pilot.methods import (
    array_hash, gradient_summary, require, require_frozen,
)


def score_check(observed, reference, spec):
    difference = observed['loss']-reference['loss']
    bound = spec['score_atol']+spec['score_rtol']*abs(reference['loss'])
    prediction_matches = observed['prediction'] == reference['prediction']
    finite = np.isfinite(observed['loss']) and np.isfinite(reference['loss'])
    return dict(loss_difference=difference, loss_bound=bound, prediction_matches=prediction_matches,
                loss_within_bound=bool(finite and abs(difference) <= bound),
                passed=bool(prediction_matches and finite and abs(difference) <= bound))


def gradient_comparison(first, second, physical_steps, spec):
    """Compare physical support only; never hide zero maps in a denominator."""
    a, b = [np.asarray(g) for g in (first, second)]
    require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] and
            0 < physical_steps <= min(len(a), len(b)) and np.isfinite(a).all() and np.isfinite(b).all(),
            'Gradient comparison needs finite aligned physical support')
    a, b = [g[:physical_steps].astype(np.float64).reshape(-1) for g in (a, b)]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    distance = float(np.linalg.norm(a-b))
    if na == nb == 0:
        cosine, relative, status, passed = None, 0., 'both_zero', True
    elif min(na, nb) == 0:
        cosine, relative, status, passed = None, 1., 'one_zero', False
    else:
        cosine = float(np.clip(np.dot(a, b)/(na*nb), -1., 1.))
        relative = distance/max(na, nb)
        status = 'both_nonzero'
        passed = relative <= spec['gradient_relative_l2_limit'] and cosine >= spec['gradient_cosine_minimum']
    return dict(first_physical_l2=na, second_physical_l2=nb, absolute_l2_difference=distance,
                symmetric_relative_l2=relative, cosine=cosine, zero_status=status,
                max_absolute_difference=float(np.max(np.abs(a-b))), passed=bool(passed),
                scope='physical input support including burn-in; excludes synthetic padding')


def input_gradient(model, batch, cfg, device, counts, spec, expected_state):
    require(not torch.is_inference_mode_enabled(), 'Input gradients cannot use inference_mode')
    require_frozen(model, expected_state)
    clean = ordinary(model, batch, cfg, device)[0]
    x, targets, blocks = [v.to(device) for v in batch]
    with torch.enable_grad():
        leaf = x.detach().clone().requires_grad_(True)
        losses, logits = supervised(model, leaf, targets, blocks, cfg)
        require(losses.numel() == 1, 'Strict pilot requires single-record CE')
        observed = scores(losses, logits, targets)[0]
        grad = torch.autograd.grad(losses.sum(), leaf, create_graph=False, retain_graph=False, allow_unused=False)[0]
        require(torch.isfinite(grad).all().item(), 'Nonfinite input surrogate gradient')
        gradient = grad[0].detach().cpu().numpy().copy()
    require_frozen(model, expected_state)
    return dict(clean=clean, autograd_forward=observed,
        autograd_forward_check=score_check(observed, clean, spec), model_state_sha256=expected_state,
        **gradient_summary(gradient, counts, spec['burn_in_frames'], spec['profile_bins'])), dict(gradient=gradient)


def validate_gradient(value, arrays, batch, counts, expected_state):
    gradient = arrays['gradient']
    require(gradient.dtype == np.float32 and gradient.shape == tuple(batch[0].shape[1:]) and
            np.isfinite(gradient).all() and array_hash(gradient) == value['gradient_sha256'] and
            value['model_state_sha256'] == expected_state and value['physical_steps'] == len(counts),
            'Gradient cache dtype/shape/hash/state/support differs')


def require_anchor(rows, references, spec):
    require(len(rows) == len(references) == 1631 and
            [r['sample_id'] for r in rows] == [r['sample_id'] for r in references], 'Strict anchor order/coverage differs')
    mismatch = sum(r['prediction'] != old['prediction'] for r, old in zip(rows, references))
    accuracy_delta = sum(r['correct']-old['correct'] for r, old in zip(rows, references))/len(rows)
    loss_delta = sum(r['loss']-old['loss'] for r, old in zip(rows, references))/len(rows)
    require(mismatch == 0 and abs(accuracy_delta) <= spec['anchor_accuracy_tolerance'] and
            abs(loss_delta) <= spec['anchor_loss_tolerance'],
            'Same-precision validation reference failed; gradients blocked, no tolerance bypass')
    return dict(status='passed', samples=len(rows), prediction_mismatches=mismatch,
        accuracy_difference=accuracy_delta, loss_difference=loss_delta,
        maximum_absolute_row_loss_drift=max(abs(r['loss']-old['loss']) for r, old in zip(rows, references)),
        reference_mode='highest_matmul', role='strict diagnostic reference; not source_high training-validation anchor')

"""Frozen-weight differentiation through ordinary MLPSNN.forward; no optimizer."""
import contextlib
import io

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from project_paths import local_path
from scripts.adaptation_gradient_pilot.methods import gradient_summary, require, require_frozen, state_hash


def load_model(run, device):
    from models.pl_module import MLPSNN
    torch.manual_seed(run['seed'])
    checkpoint = torch.load(local_path(run['checkpoint'], must_exist=True), weights_only=False, map_location='cpu')
    with contextlib.redirect_stdout(io.StringIO()):
        model = MLPSNN(OmegaConf.create(run['config']))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.requires_grad_(False).eval().to(device)
    require_frozen(model, run['model_state_sha256'])
    return model


def supervised(model, x, targets, blocks, cfg):
    from models.pl_module import aggregate_temporal_outputs
    output = model(x)
    logits = aggregate_temporal_outputs(output, blocks, targets.size(1), cfg['loss_agg'])[:, 1, :]
    labels = targets[:, 1].long()
    losses = F.cross_entropy(logits.float(), labels, reduction='none')
    require(torch.isfinite(output).all().item() and torch.isfinite(losses).all().item(), 'Nonfinite forward or supervised CE')
    return losses, logits


def scores(losses, logits, targets):
    labels = targets[:, 1].long()
    predictions = logits.argmax(-1)
    probability = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
    return [dict(loss=float(loss), prediction=int(pred), correct=int(pred == target), true_probability=float(p))
            for loss, pred, target, p in zip(losses.detach().cpu().tolist(), predictions.detach().cpu().tolist(),
                                            labels.cpu().tolist(), probability.detach().cpu().tolist())]


def ordinary(model, batch, cfg, device):
    x, targets, blocks = [v.to(device) for v in batch]
    with torch.no_grad():
        losses, logits = supervised(model, x, targets, blocks, cfg)
        return scores(losses, logits, targets)


def differentiate(model, batch, cfg, device, counts, spec, expected_state):
    require(not torch.is_inference_mode_enabled(), 'Input gradients cannot run inside inference_mode')
    require_frozen(model, expected_state)
    reference = ordinary(model, batch, cfg, device)[0]
    x, targets, blocks = [v.to(device) for v in batch]
    with torch.enable_grad():
        leaf = x.detach().clone().requires_grad_(True)
        losses, logits = supervised(model, leaf, targets, blocks, cfg)
        require(losses.numel() == 1, 'This bounded diagnostic uses exactly one record per backward pass')
        observed = scores(losses, logits, targets)[0]
        gradient = torch.autograd.grad(losses.sum(), leaf, create_graph=False, retain_graph=False, allow_unused=False)[0]
        require(torch.isfinite(gradient).all().item(), 'Nonfinite input surrogate gradient')
        g = gradient[0].detach().cpu().numpy().copy()
    require(observed['prediction'] == reference['prediction'] and
            abs(observed['loss']-reference['loss']) <= spec['autograd_forward_loss_tolerance'],
            'Enabling autograd changed the diagnostic-batch forward result')
    require_frozen(model, expected_state)
    summary = dict(clean=reference, autograd_forward=observed,
        autograd_forward_loss_difference=observed['loss']-reference['loss'],
        model_state_sha256=state_hash(model), **gradient_summary(g, counts, spec['burn_in_frames'], spec['profile_bins']))
    return summary, dict(gradient=g)

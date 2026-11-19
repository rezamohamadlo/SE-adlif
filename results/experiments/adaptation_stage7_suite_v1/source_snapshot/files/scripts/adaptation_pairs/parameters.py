"""Layer-level trace descriptors; no forward pass or optimizer."""
import math


def trace_record(rho, bin_ms):
    rho = float(rho)
    if not 0 < rho < 1 or not math.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError('Trace decay and physical bin width must be valid')
    frames = -1.0 / math.log(rho)
    return dict(rho=rho, e_folding_frames=frames, e_folding_ms=frames * bin_ms)


def extract(model, run, bin_ms):
    layers = []
    for name in ('l1','l2'):
        layer = getattr(model, name)
        learned = hasattr(layer, 'trace_decays')
        fast, slow = layer.trace_decays() if learned else (layer.rho_fast, layer.rho_slow)
        fast, slow = float(fast), float(slow)
        if not fast < slow:
            raise ValueError('Fast/slow trace ordering differs')
        layers.append(dict(layer=name, learned=learned, bin_ms=bin_ms,
                           fast=trace_record(fast,bin_ms), slow=trace_record(slow,bin_ms)))
    return dict(model=run['model'],seed=run['seed'],checkpoint=run['checkpoint'],
                checkpoint_sha256=run['checkpoint_sha256'],layers=layers,
                interpretation='Layer-level descriptors shared across samples; not independent per-sample observations. Trace decays operate per input frame; milliseconds use the saved input bin width, not neuron dt.')

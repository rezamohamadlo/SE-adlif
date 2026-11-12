"""Independent event deletion, preserving time support, zero padding and markers."""
import math
import torch


def thin_input_counts(inputs, probability, training):
    probability=float(probability)
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('Count-deletion probability must be finite within [0,1]')
    if not training or probability==0:
        return inputs
    positive=inputs.clamp_min(0)
    # These datasets supply integer-valued event counts. Fractional currents
    # are not treated as counts; filtering must happen AFTER this transform.
    if not torch.equal(positive,positive.round()) or not torch.isfinite(inputs).all():
        raise ValueError('Event deletion requires finite integer-valued original inputs')
    deleted=torch.binomial(positive,torch.full_like(positive,1-probability))
    return torch.where(inputs<0,inputs,deleted)

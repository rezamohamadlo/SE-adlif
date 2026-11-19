"""Training-only temporal regularization with target-boundary masking."""

import torch
import torch.nn.functional as F


def temporal_js_loss(outputs, targets, block_idx, window=5, ignore_index=-1):
    """JS divergence between overlapping, same-label temporal windows.

    ``block_idx`` maps frames to target positions, including ignored burn-in
    and batch padding. Windows crossing ignored frames or class transitions
    are excluded. Each eligible neighboring window pair has equal weight.
    This is an auxiliary objective, not the dataset's classification loss.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer')
    if (outputs.ndim != 3 or targets.ndim != 2 or block_idx.ndim != 2
            or outputs.shape[:2] != block_idx.shape
            or outputs.shape[0] != targets.shape[0]):
        raise ValueError('Expected outputs [B,T,C], targets [B,K], indices [B,T]')
    if outputs.shape[1] <= window:
        return outputs.sum() * 0
    labels = targets.long().gather(1, block_idx.long())
    label_windows = labels.unfold(1, window, 1)
    valid = (label_windows != ignore_index).all(-1)
    valid &= (label_windows == label_windows[..., :1]).all(-1)
    pairs = valid[:, 1:] & valid[:, :-1]
    pairs &= label_windows[:, 1:, 0] == label_windows[:, :-1, 0]
    probabilities = F.softmax(outputs.float(), dim=-1)
    windows = probabilities.unfold(1, window, 1).mean(-1)
    left, right = windows[:, :-1], windows[:, 1:]
    mean = (left + right) * 0.5
    # Keep logs finite under mixed precision or extreme logits.
    eps = torch.finfo(mean.dtype).tiny
    log_mean = mean.clamp_min(eps).log()
    js = 0.5 * ((left * (left.clamp_min(eps).log() - log_mean)).sum(-1)
                + (right * (right.clamp_min(eps).log() - log_mean)).sum(-1))
    weights = pairs.to(js.dtype)
    return (js * weights).sum() / weights.sum().clamp_min(1)

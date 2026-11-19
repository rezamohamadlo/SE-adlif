"""Pure native-frame permutation, with explicit per-region audit trails."""
import hashlib
import json
from numbers import Integral

import numpy as np


def identity(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def derive_seed(base, dataset, sample_id, width, replicate):
    # Deliberately excludes checkpoint, pair, model, training seed and PC.
    value = json.dumps([int(base), str(dataset), int(sample_id), int(width), int(replicate)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def permute(x, width, seed, *, burn=0, support=None, labels=None, signed=False):
    """Permute full windows after burn-in, optionally within constant-label runs.

    x is [time,channel]. support excludes all synthetic padding; suffix windows
    and burn-in stay fixed. Label runs are supplied explicitly, never inferred
    from input activity. The caller retains original targets and block masks.
    ECG execution is separately blocked until its label-region policy is reviewed.
    """
    a = np.asarray(x)
    if (a.ndim != 2 or a.dtype.kind not in 'biuf' or
            not np.isfinite(a).all()):
        raise ValueError('Expected finite numeric [time,channel] input')
    for name, value in [('width', width), ('seed', seed), ('burn', burn)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(name + ' must be an integer')
    support = len(a) if support is None else support
    if (isinstance(support, (bool, np.bool_)) or not isinstance(support, Integral) or
            not 0 <= support <= len(a) or width < 1 or seed < 0 or burn < 0):
        raise ValueError('Invalid support, width, seed or burn-in')
    if signed:
        if not np.isin(a, [-1, 0, 1]).all():
            raise ValueError('Signed inputs must be ternary')
    elif np.any(a < 0) or not np.equal(a, np.floor(a)).all():
        raise ValueError('Expected nonnegative integer counts')
    start = min(burn, support)
    if labels is None:
        regions = [(start, support)]
    else:
        y = np.asarray(labels)
        if y.shape != (len(a),) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError('Labels must be integer and aligned to the complete time axis')
        cuts = [start] + (np.flatnonzero(y[start + 1:support] != y[start:support - 1]) + start + 1).tolist() if start < support else [start]
        cuts += [support]
        regions = list(zip(cuts, cuts[1:]))
    rng = np.random.default_rng(int(seed))
    result, audit = a.copy(), []
    for lo, hi in regions:
        count = (hi - lo) // width
        order = np.arange(count)
        if count >= 2:
            # Rejection sampling from the uniform nonidentity distribution.
            while True:
                order = rng.permutation(count)
                if not np.array_equal(order, np.arange(count)):
                    break
            result[lo:lo + count * width] = a[lo:lo + count * width].reshape(count, width, a.shape[1])[order].reshape(-1, a.shape[1])
        audit.append(dict(start=lo, stop=hi, windows=count, order=order.tolist(),
                          tail_start=lo + count * width, eligible=count >= 2))
    if (not np.array_equal(a[:start], result[:start]) or
            not np.array_equal(a[support:], result[support:])):
        raise AssertionError('Burn-in/padding changed')
    for region in audit:
        tail = slice(region['tail_start'], region['stop'])
        if not np.array_equal(a[tail], result[tail]):
            raise AssertionError('Trailing window changed')
    counts = lambda v: (np.stack([(v == s).sum(0) for s in [-1, 1]]) if signed else v.sum(0, dtype=object))
    if not np.array_equal(counts(a), counts(result)):
        raise AssertionError('Per-channel event counts changed')
    eligible = any(r['eligible'] for r in audit)
    effective = not np.array_equal(a, result)
    return result, dict(seed=int(seed), width=int(width), support=int(support), burn=int(burn),
        regions=audit, eligible=eligible, effective=effective,
        status='effective' if effective else 'ineffective' if eligible else 'ineligible',
        original_sha256=identity(a), transformed_sha256=identity(result),
        event_count=int(np.count_nonzero(a)) if signed else int(a.sum(dtype=object)),
        channel_counts=counts(a).tolist())

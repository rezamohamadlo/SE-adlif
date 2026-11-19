"""Deterministic, count-preserving/controlled transforms for frozen-model tests."""
import math
from numbers import Integral

import numpy as np

from .features import validate_counts


def _restore_dtype(counts, original):
    # Compression of a binary input legitimately creates count values >1.
    if np.issubdtype(original.dtype, np.bool_):
        return counts
    result = counts.astype(original.dtype)
    if not np.array_equal(result.astype(np.int64), counts):
        raise ValueError("Input dtype cannot exactly represent transformed counts")
    return result


def thin_counts(x, probability, seed):
    """Delete individual events with binomial thinning, never whole count bins.

    The same (x, probability, seed) yields the same transformed input for both
    paired models. ``seed`` is a perturbation seed, separate from training seed.
    The input is never modified. Zero deletion is an exact dtype-preserving copy.
    """
    array, counts = validate_counts(x)
    if isinstance(probability, (bool, np.bool_)):
        raise ValueError("probability must be finite within [0, 1]")
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite within [0, 1]")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if probability == 0:
        return array.copy()
    if probability == 1:
        return np.zeros_like(array)
    rng = np.random.default_rng(int(seed))
    thinned = rng.binomial(counts, 1-probability)
    return _restore_dtype(thinned, array)


def warp_counts(x, scale):
    """Map frame t to floor(t*scale), rebinning event counts without interpolation.

    New length is ceil(T*scale). Every channel's total is preserved; duration,
    density, bin coincidences and other temporal features may all change. This
    is not an isolated manipulation of a single temporal statistic. Caller
    must pass physical support, then rebuild the original model's minimum pad.
    """
    array, counts = validate_counts(x)
    if isinstance(scale, (bool, np.bool_)):
        raise ValueError("scale must be finite and positive")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if scale == 1:
        return array.copy()
    scaled_length = len(array) * scale
    if not math.isfinite(scaled_length) or scaled_length > np.iinfo(np.intp).max:
        raise ValueError("Warped time axis exceeds supported size")
    new_length = int(math.ceil(scaled_length))
    totals = counts.sum(axis=0, dtype=object)
    if any(value > np.iinfo(np.int64).max for value in totals):
        raise ValueError("Channel count accumulation would overflow int64")
    positions = np.floor(np.arange(len(array), dtype=np.float64)*scale).astype(np.intp)
    if positions.max() >= new_length:
        raise ValueError("Floating-point mapping escaped warped time support")
    result = np.zeros((new_length, array.shape[1]), dtype=np.int64)
    np.add.at(result, positions, counts)
    return _restore_dtype(result, array)

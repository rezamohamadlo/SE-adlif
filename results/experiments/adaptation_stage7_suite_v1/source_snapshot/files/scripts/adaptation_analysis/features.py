"""Pure features of binned input counts; no model or dataset execution.

The caller must supply physical support explicitly when ``x`` contains minimum
or batch padding. A last nonzero frame is not a reliable recording duration.
"""
import math
from numbers import Integral

import numpy as np


def validate_counts(x):
    """Return a non-mutating array view and exact int64 count representation."""
    array = np.asarray(x)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("Expected nonempty count array [time, channels]")
    if not (np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError("Counts must be real numeric values")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.equal(array, np.floor(array)).all():
        raise ValueError("Counts must be integer valued; do not interpolate counts")
    if int(array.max()) > np.iinfo(np.int64).max:
        raise ValueError("Counts exceed int64 capacity")
    return array, array.astype(np.int64, copy=False)


def _window_frames(milliseconds, bin_ms):
    if isinstance(milliseconds, (bool, np.bool_)):
        raise ValueError("Window duration must be positive and finite")
    milliseconds = float(milliseconds)
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise ValueError("Window duration must be positive and finite")
    ratio = milliseconds / bin_ms
    rounded = round(ratio)
    if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Window duration must be an integer number of input bins")
    return int(rounded)


def measure_features(x, support_frames=None, short_ms=20, long_ms=200, bin_ms=4):
    """Describe physical count support with explicit undefined-value flags.

    ``fast_slow_contrast`` is the population SD of end-aligned short-minus-long
    total rates, divided by the mean total rate over physical support. Only
    complete windows are used, with at least two end positions required.

    ``autocorr_tau_ms`` is the first positive lag with centered, biased ACF at
    or below 1/e, searched through ``long_ms`` or the available support. It is
    a finite-window descriptor, not a measured neuronal memory constant. No
    crossing is right-censored and returns None; constant/empty activity is
    also undefined. Counts, fractions and durations remain available.
    """
    array, counts = validate_counts(x)
    if isinstance(bin_ms, (bool, np.bool_)):
        raise ValueError("bin_ms must be positive and finite")
    bin_ms = float(bin_ms)
    if not math.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError("bin_ms must be positive and finite")
    short_frames = _window_frames(short_ms, bin_ms)
    long_frames = _window_frames(long_ms, bin_ms)
    if short_frames >= long_frames:
        raise ValueError("Require short_ms < long_ms")
    supplied = support_frames is not None
    if support_frames is None:
        support_frames = array.shape[0]
    if (isinstance(support_frames, (bool, np.bool_))
            or not isinstance(support_frames, Integral)
            or not 0 <= support_frames <= array.shape[0]):
        raise ValueError("support_frames must be an integer within the input axis")
    support_frames = int(support_frames)
    if np.any(counts[support_frames:] != 0):
        raise ValueError("Nonzero frames outside physical support would be discarded")
    valid = counts[:support_frames]
    # Float64 reductions avoid integer overflow in descriptor arithmetic.
    channel_totals = valid.sum(axis=0, dtype=np.float64)
    time_counts = valid.sum(axis=1, dtype=np.float64)
    total = int(valid.sum(dtype=object))
    duration_ms = support_frames * bin_ms
    mean_rate = total / (duration_ms / 1000) if support_frames else None
    result = dict(
        feature_version="frame_temporal_features_v1",
        support_frames=support_frames,
        support_explicit=supplied,
        provided_frames=int(array.shape[0]),
        channels=int(array.shape[1]),
        bin_ms=bin_ms,
        duration_ms=duration_ms,
        total_spikes=total,
        mean_rate_hz=mean_rate,
        active_channel_fraction=float(np.mean(channel_totals > 0)),
        active_bin_fraction=float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0,
        fast_slow_contrast=None,
        fast_slow_valid=False,
        fast_slow_reason=None,
        autocorr_tau_ms=None,
        autocorr_valid=False,
        autocorr_reason=None,
        autocorr_censored=False,
        autocorr_horizon_ms=max(0, min(long_frames, support_frames - 1)) * bin_ms,
        short_ms=float(short_ms),
        long_ms=float(long_ms),
    )
    if support_frames == 0:
        result["fast_slow_reason"] = result["autocorr_reason"] = "empty_support"
        return result
    if total == 0:
        result["fast_slow_reason"] = result["autocorr_reason"] = "silent_input"
        return result
    if support_frames < long_frames + 1:
        result["fast_slow_reason"] = "insufficient_complete_windows"
    else:
        cumulative = np.r_[0.0, np.cumsum(time_counts)]
        ends = np.arange(long_frames, support_frames + 1)
        fast = (cumulative[ends] - cumulative[ends-short_frames]) / (short_frames*bin_ms/1000)
        slow = (cumulative[ends] - cumulative[ends-long_frames]) / (long_frames*bin_ms/1000)
        result["fast_slow_contrast"] = float(np.std(fast-slow, ddof=0) / mean_rate)
        result["fast_slow_valid"] = True
    centered = time_counts - time_counts.mean()
    variance_sum = float(np.dot(centered, centered))
    if variance_sum == 0:
        result["autocorr_reason"] = "constant_activity"
        return result
    max_lag = min(long_frames, support_frames - 1)
    acf = np.array([np.dot(centered[:-lag], centered[lag:]) / variance_sum
                    for lag in range(1, max_lag + 1)])
    crossings = np.flatnonzero(acf <= math.exp(-1))
    if crossings.size:
        result["autocorr_tau_ms"] = float((int(crossings[0])+1)*bin_ms)
        result["autocorr_valid"] = True
    else:
        result["autocorr_reason"] = "no_crossing_within_horizon"
        result["autocorr_censored"] = True
    return result

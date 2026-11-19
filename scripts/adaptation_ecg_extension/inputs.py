"""Signed ECG inputs with one explicit time map for events, labels and masks."""
import numpy as np
import torch

from datasets.utils.pad_tensors import PadTensors
from scripts.adaptation_ecg.data import array_hash, require


def time_map(length, scale):
    require(isinstance(length, int) and length > 0 and np.isfinite(scale) and scale > 0,
            'Invalid time-resampling length/scale')
    target = max(1, int(np.floor(length*scale+.5)))
    # Nearest source-bin center; deterministic half-bin ties choose the later bin.
    mapping = np.minimum(length-1, ((np.arange(target, dtype=np.int64)*2+1)*length)//(2*target))
    return mapping


def aligned_resample(x, labels, mask, scale):
    require(x.ndim == 2 and len(x) == len(labels) == len(mask) and mask.dtype == np.bool_,
            'Input/label/mask alignment differs')
    mapping = time_map(len(x), scale)
    return x[mapping].copy(), labels[mapping].copy(), mask[mapping].copy(), mapping


class ExtensionData:
    def __init__(self, base, specification):
        self.base, self.spec, self.ids = base, specification, base.ids
        self.items = {}

    def item(self, sample_id, condition, replicate):
        key = sample_id, condition, replicate
        if key not in self.items:
            require(sample_id in self.ids, 'ECG sample is outside the frozen validation split')
            labels = self.base.y[sample_id].copy()
            mask = np.ones(len(labels), dtype=bool)
            if condition in ('clean', 'delete_0.1'):
                x = self.base.input(sample_id, condition, replicate, self.spec)
                mapping = np.arange(len(labels), dtype=np.int64)
            else:
                require(condition in ('time_scale_0.9', 'time_scale_1.1') and replicate == 0,
                        'Unknown ECG resampling condition')
                scale = self.spec['time_scaling'][condition]
                x, labels, mask, mapping = aligned_resample(
                    self.base.input(sample_id, 'clean', 0, self.spec), labels, mask, scale)
            require(np.isin(x, [-1, 0, 1]).all(), 'Time resampling changed the signed ECG alphabet')
            blocks = np.arange(1, len(labels)+1, dtype=np.int64)
            blocks[~mask] = 0
            labels[~mask] = -1
            identity = dict(sample_id=sample_id, condition=condition, replicate=replicate,
                input_sha256=array_hash(x), targets_sha256=array_hash(labels), blocks_sha256=array_hash(blocks),
                mask_sha256=array_hash(mask), time_map_sha256=array_hash(mapping),
                source_targets_sha256=array_hash(self.base.y[sample_id]))
            self.items[key] = (x, labels, mask, blocks, identity)
        return self.items[key]

    def batch(self, ids, condition, replicate):
        items = [self.item(i, condition, replicate) for i in ids]
        return PadTensors(batch_first=True)([(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(blocks))
            for x, y, _mask, blocks, _identity in items])

    def validate_rows(self, rows, run, condition, replicate):
        require([r['sample_id'] for r in rows] == self.ids, 'ECG prediction coverage/order differs')
        for row in rows:
            _x, labels, mask, _blocks, identity = self.item(row['sample_id'], condition, replicate)
            require(all(row[k] == v for k, v in identity.items()) and row['model'] == run['model'] and
                    row['seed'] == run['seed'], 'ECG prediction identity differs')
            n = int(mask.sum())
            require(row['valid_steps'] == n and len(row['predictions']) == n and
                    all(0 <= p < 6 for p in row['predictions']) and
                    row['correct_steps'] == sum(p == int(y) for p, y in zip(row['predictions'], labels[mask])) and
                    np.isfinite(row['loss_sum']) and row['loss_sum'] >= 0 and
                    row['loss'] == row['loss_sum']/n and row['accuracy'] == row['correct_steps']/n,
                    'ECG cached temporal metrics differ')


def signed_features(x, labels, mask, definition):
    """Native-step descriptors, independent of every model outcome."""
    require(np.isin(x, [-1, 0, 1]).all() and mask.all(),
            'This fixed full-record feature protocol requires all stored steps to be scored')
    activity = np.abs(x).sum(axis=1, dtype=np.float64)
    short, long = definition['short_steps'], definition['long_steps']
    require(isinstance(short, int) and isinstance(long, int) and 0 < short < long, 'Invalid feature windows')
    total, n = int(activity.sum()), len(x)
    result = dict(feature_version='ecg_signed_activity_steps_v1', time_unit='model_step', stored_steps=n,
        scored_steps=int(mask.sum()), total_events=total, positive_events=int((x > 0).sum()),
        negative_events=int((x < 0).sum()), events_per_step=total/n,
        polarity_balance=None if total == 0 else float(x.sum(dtype=np.float64)/total),
        class_fractions=[float((labels[mask] == k).mean()) for k in range(6)],
        short_steps=short, long_steps=long, fast_slow_contrast=None, fast_slow_reason=None,
        autocorr_tau_steps=None, autocorr_censored=False, autocorr_reason=None,
        autocorr_horizon_steps=min(long, n-1))
    if not total:
        result.update(fast_slow_reason='silent_activity', autocorr_reason='silent_activity')
        return result
    if n < long+1:
        result['fast_slow_reason'] = 'insufficient_complete_windows'
    else:
        cumulative = np.r_[0., np.cumsum(activity)]
        ends = np.arange(long, n+1)
        fast = (cumulative[ends]-cumulative[ends-short])/short
        slow = (cumulative[ends]-cumulative[ends-long])/long
        result['fast_slow_contrast'] = float(np.std(fast-slow, ddof=0)/(total/n))
    centered = activity-activity.mean()
    variance = float(centered @ centered)
    if variance == 0:
        result['autocorr_reason'] = 'constant_activity'
    else:
        for lag in range(1, min(long, n-1)+1):
            if float(centered[:-lag] @ centered[lag:])/variance <= np.exp(-1):
                result['autocorr_tau_steps'] = lag
                break
        if result['autocorr_tau_steps'] is None:
            result.update(autocorr_censored=True, autocorr_reason='no_crossing_within_horizon')
    return result

"""Read-only SHD* validation inputs without datamodule downloads or disk caches.

The conversion and random split match datasets.shd.SHDLDM: microsecond event
conversion, Tonic downsampling/ToFrame, then minimum padding and block masks.
The unpadded count frames are retained separately for input feature analysis.
"""
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from tonic.datasets.hsd import SHD
from tonic.io import make_structured_array
from tonic.transforms import Downsample, ToFrame

from project_paths import local_path, portable_path

OFFICIAL_TRAIN_SHA256 = '2bddb4bd46732f09982b7d1631b7c29c19853c73d3d240e3eb32bba909bdd6c1'
SELECTION_SEED = 20261110


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_shd_config(cfg):
    """Reject another split/preprocessing rather than silently change a run."""
    data = dict(cfg['dataset'] if 'dataset' in cfg else cfg)
    target = data.get('_target_', 'datasets.shd.SHDLDM')
    expected = dict(validate_on=0.2, random_seed=42, spatial_factor=0.2,
                    time_factor=0.001, window_size=4, pad_to_min_size=250,
                    ignore_first_timesteps=10, num_classes=20)
    if target != 'datasets.shd.SHDLDM':
        raise ValueError('Pilot requires the saved SHD classification dataset')
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f'SHD pilot requires {key}={value}; got {data.get(key)!r}')
    if data.get('additional_test_set_validation', False):
        raise ValueError('Official-test validation is not permitted in the pilot')
    if data.get('train_augmentation') is not None:
        raise ValueError('Pilot requires ordinary SHD*, not the speaker-augmentation study')
    if data.get('speaker_split_manifest') or data.get('speaker_split_fingerprint'):
        raise ValueError('Speaker-holdout split is a different study')
    if not data.get('data_path'):
        raise ValueError('Saved dataset.data_path is required')
    return data


def validation_indices(size, fraction=0.2, split_seed=42):
    """Exact random_split order: training slice first, validation slice last."""
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError('Dataset size must be a positive integer')
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must lie strictly between zero and one')
    count = math.floor(int(size) * fraction)
    if count < 1:
        raise ValueError('Validation split would be empty')
    indices = torch.randperm(int(size), generator=torch.Generator().manual_seed(split_seed)).tolist()
    return indices[int(size)-count:]


def _read_metadata(path):
    with h5py.File(path, 'r') as dataset:
        for key in ('labels', 'spikes/times', 'spikes/units', 'extra/speaker'):
            if key not in dataset:
                raise ValueError(f'Missing SHD HDF5 field: {key}')
        labels = np.asarray(dataset['labels'][:], dtype=np.int64)
        speakers = np.asarray(dataset['extra/speaker'][:], dtype=np.int64)
        if (labels.ndim != 1 or speakers.shape != labels.shape
                or len(dataset['spikes/times']) != len(labels)
                or len(dataset['spikes/units']) != len(labels)):
            raise ValueError('SHD sample axes disagree')
    if np.any(labels < 0) or np.any(labels >= 20):
        raise ValueError('SHD labels must lie in [0,19]')
    return labels, speakers


def _select(labels, validation_ids, samples, selection_seed):
    if samples is None:
        return list(validation_ids)
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError('Sample count must be a positive integer or None')
    if samples > len(validation_ids):
        raise ValueError(f'Requested {samples} samples but validation contains {len(validation_ids)}')
    if samples == len(validation_ids):
        return list(validation_ids)
    rng = np.random.default_rng(selection_seed)
    classes = sorted(set(int(labels[i]) for i in validation_ids))
    pools = {label: list(rng.permutation([i for i in validation_ids if labels[i] == label]))
             for label in classes}
    chosen = set()
    # Round robin avoids a pilot dominated by one class; IDs remain in the
    # original validation order so every PC gets the same sample sequence.
    while len(chosen) < samples:
        for label in classes:
            if pools[label] and len(chosen) < samples:
                chosen.add(int(pools[label].pop()))
    return [int(i) for i in validation_ids if i in chosen]


def select_validation_ids(path, samples=256, selection_seed=SELECTION_SEED):
    labels, _ = _read_metadata(Path(path))
    return _select(labels, validation_indices(len(labels)), samples, selection_seed)


def preflight_shd(cfg, expected_sha256=OFFICIAL_TRAIN_SHA256, verify_hash=True):
    data = validate_shd_config(cfg)
    path = local_path(data['data_path']) / 'SHDWrapper' / 'shd_train.h5'
    if not path.is_file():
        raise FileNotFoundError(f'Missing SHD training HDF5: {path}. Transfer the exact official file; no download is performed.')
    actual = file_sha256(path) if verify_hash else None
    if verify_hash and actual != expected_sha256:
        raise ValueError(f'SHD training data SHA-256 differs: expected {expected_sha256}, got {actual}')
    labels, speakers = _read_metadata(path)
    indices = validation_indices(len(labels), data['validate_on'], data['random_seed'])
    identity = dict(dataset_sha256=expected_sha256, validation_indices=indices,
                    validate_on=data['validate_on'], split_seed=data['random_seed'])
    return dict(path=portable_path(path), sha256=actual,
                expected_sha256=expected_sha256, hash_verified=verify_hash,
                total_samples=len(labels), train_samples=len(labels)-len(indices),
                validation_samples=len(indices), split_seed=data['random_seed'],
                validation_indices=indices,
                validation_class_counts={str(label): int(np.sum(labels[indices] == label))
                                         for label in sorted(set(labels[indices].tolist()))},
                validation_speaker_counts={str(speaker): int(np.sum(speakers[indices] == speaker))
                                           for speaker in sorted(set(speakers[indices].tolist()))},
                split_fingerprint=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                input_channels=140, bin_ms=4.0, min_frames=250, ignore_first_timesteps=10,
                official_test_opened=False)


class SHDValidationData:
    """Selected original-row SHD validation inputs; reads only train HDF5."""

    def __init__(self, cfg, sample_ids=None, samples=256, selection_seed=SELECTION_SEED,
                 expected_sha256=OFFICIAL_TRAIN_SHA256, verify_hash=True):
        self.config = validate_shd_config(cfg)
        self.metadata = preflight_shd(cfg, expected_sha256, verify_hash)
        self.path = local_path(self.metadata['path'], must_exist=True)
        self.labels, self.speakers = _read_metadata(self.path)
        self.validation_ids = self.metadata['validation_indices']
        if sample_ids is None:
            self.sample_ids = _select(self.labels, self.validation_ids, samples, selection_seed)
        else:
            self.sample_ids = [int(i) for i in sample_ids]
            if (not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids)
                    or not set(self.sample_ids).issubset(self.validation_ids)):
                raise ValueError('Requested IDs must be unique original rows from saved validation split')
        self.metadata.update(sample_ids=self.sample_ids, selection_seed=selection_seed,
                             selection='class_round_robin_then_original_validation_order')
        self._downsample = Downsample(time_factor=self.config['time_factor'],
                                      spatial_factor=self.config['spatial_factor'])
        self._to_frame = ToFrame(sensor_size=(140, 1, 1), time_window=self.config['window_size'])

    def __len__(self):
        return len(self.sample_ids)

    def load(self, sample_id):
        sample_id = int(sample_id)
        if sample_id not in self.sample_ids:
            raise ValueError('Sample ID was not frozen in this pilot subset')
        with h5py.File(self.path, 'r') as dataset:
            seconds = np.asarray(dataset['spikes/times'][sample_id], dtype=np.float64)
            units = np.asarray(dataset['spikes/units'][sample_id])
        if (seconds.shape != units.shape or not np.isfinite(seconds).all()
                or np.any(seconds < 0) or np.any(np.diff(seconds) < 0)
                or np.any(units < 0) or np.any(units >= 700)):
            raise ValueError(f'Invalid original SHD events at row {sample_id}')
        if len(seconds):
            events = make_structured_array(seconds*1e6, units, 1, dtype=SHD.dtype)
            frames = self._to_frame(self._downsample(events)).reshape(-1, 140)
            counts = np.asarray(frames, dtype=np.int64)
            duration_ms = float((seconds[-1]-seconds[0])*1000)
        else:
            counts = np.zeros((0, 140), dtype=np.int64)
            duration_ms = 0.0
        if np.any(counts < 0):
            raise ValueError('Negative frame count indicates preprocessing overflow')
        support = counts.shape[0]
        return dict(sample_id=sample_id, target=int(self.labels[sample_id]),
                    speaker=int(self.speakers[sample_id]), counts=counts,
                    support_frames=support, stored_frames=max(support, 250),
                    bin_ms=4.0, duration_ms=duration_ms,
                    raw_event_count=len(seconds), framed_event_count=int(counts.sum()),
                    frame_support_ms=support*4.0,
                    framing='Saved Tonic ToFrame defaults; incomplete terminal slice may be excluded')

    def model_item(self, sample_id, counts=None):
        """Return original dataset tuple; caller may collate with PadTensors."""
        sample_id = int(sample_id)
        if sample_id not in self.sample_ids:
            raise ValueError('Sample ID was not frozen in this pilot subset')
        if counts is None:
            counts = self.load(sample_id)['counts']
        values = np.asarray(counts)
        if (values.ndim != 2 or values.shape[1] != 140 or not np.isfinite(values).all()
                or np.any(values < 0) or not np.equal(values, np.floor(values)).all()):
            raise ValueError('Model inputs must be finite nonnegative [T,140] event counts')
        tensor = torch.from_numpy(values.astype(np.float32, copy=True))
        if tensor.shape[0] < 250:
            tensor = torch.cat((tensor, torch.zeros((250-tensor.shape[0], 140))), dim=0)
        blocks = torch.ones(tensor.shape[0], dtype=torch.int64)
        blocks[:10] = 0
        return tensor, torch.tensor(int(self.labels[sample_id]), dtype=torch.int64), blocks

"""Version 1.0.0: frozen, recording-aligned SHD speaker holdout metadata.

No event preprocessing, downloads, training or performance-based selection.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from project_paths import local_path, portable_path

PROTOCOL = 'shd_speaker_holdout_v1'
DEVELOPMENT_PROTOCOL = 'shd_speaker_development_v2'
ALGORITHM = 'sha256_seed_speaker_rank_v1'
SPLIT_SEED = 42


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def split_metadata(speakers, labels, split_seed=SPLIT_SEED):
    speakers, labels = np.asarray(speakers), np.asarray(labels)
    if (speakers.ndim != 1 or labels.ndim != 1 or len(speakers) != len(labels)
            or not len(labels) or speakers.dtype.kind not in 'iu'
            or labels.dtype.kind not in 'iu'):
        raise ValueError('Expected aligned, nonempty 1D integer speaker IDs and labels')
    available = sorted(int(s) for s in np.unique(speakers))
    if len(available) < 3:
        raise ValueError('Speaker holdout requires at least three training speakers')
    if set(int(y) for y in np.unique(labels)) != set(range(20)):
        raise ValueError('Official training labels must cover all 20 SHD classes (0..19)')
    # Numeric ordering, then rank each ID by a fully specified UTF-8 hash.
    ranked = sorted(available, key=lambda s: (
        hashlib.sha256(f'{ALGORITHM}:{int(split_seed)}:{s}'.encode('utf-8')).hexdigest(), s))
    validation = sorted(ranked[:2])
    held_out = np.isin(speakers, validation)
    subsets = {}
    for name, mask in [('train', ~held_out), ('validation', held_out)]:
        indices = np.flatnonzero(mask).tolist()  # Original HDF5 row order.
        counts = np.bincount(labels[mask].astype(np.int64), minlength=20).tolist()
        missing = [c for c, count in enumerate(counts) if count == 0]
        if missing:
            raise ValueError(f'{name} lacks classes {missing}; fixed speakers {validation}; no fallback')
        subsets[name] = dict(indices=indices, sample_count=len(indices), class_counts=counts,
                             speaker_ids=sorted(int(s) for s in np.unique(speakers[mask])))
    return dict(protocol=PROTOCOL, algorithm_version=ALGORITHM, split_seed=int(split_seed),
                speaker_order='ascending integer ID; rank by SHA-256 then integer ID',
                available_speaker_ids=available, validation_speaker_ids=validation,
                total_training_recordings=len(labels), subsets=subsets,
                validation_fraction=subsets['validation']['sample_count'] / len(labels))


def read_metadata(path):
    with h5py.File(path, 'r') as stream:
        required = ('labels', 'extra/speaker', 'spikes/times', 'spikes/units', 'extra/keys')
        if any(key not in stream for key in required):
            raise ValueError(f'Missing SHD recording metadata in {path}; required {required}')
        n = len(stream['labels'])
        if any(stream[key].shape != (n,) for key in required[:4]):
            raise ValueError(f'SHD metadata/event rows are not aligned in {path}')
        if stream['extra/keys'].shape != (20,):
            raise ValueError(f'Expected 20 class keys in {path}')
        return stream['extra/speaker'][:], stream['labels'][:]


def build_manifest(data_path):
    location = local_path(data_path) / 'SHDWrapper'
    train, test = location / 'shd_train.h5', location / 'shd_test.h5'
    for path in (train, test):
        if not path.is_file():
            raise FileNotFoundError(f'Raw SHD file unavailable: {path}; no random-split fallback')
    speakers, labels = read_metadata(train)
    manifest = split_metadata(speakers, labels)
    # Content identity is portable: paths are routing metadata, not fingerprint inputs.
    manifest['dataset_identity'] = dict(
        dataset='SHD', source='official train/test HDF5',
        train_sha256=file_sha256(train), test_sha256=file_sha256(test),
        training_metadata_sha256=digest(dict(speakers=speakers.tolist(), labels=labels.tolist())),
        speaker_field='extra/speaker', recording_axis='HDF5 row index shared with labels/spikes',
    )
    manifest['fingerprint'] = digest(manifest)
    return manifest


def build_development_manifest(data_path, validation_speakers, role):
    """Predeclared two-speaker split using ONLY official training recordings."""
    if role not in ('development', 'pair_sensitivity'):
        raise ValueError('Unknown speaker development role')
    selected = sorted(validation_speakers)
    if len(selected) != 2 or len(set(selected)) != 2 or any(type(s) is not int for s in selected):
        raise ValueError('Exactly two distinct integer speaker IDs must be declared in advance')
    train = local_path(data_path)/'SHDWrapper/shd_train.h5'
    if not train.is_file():
        raise FileNotFoundError('Official SHD training file is required; no fallback or test-file access')
    speakers, labels = read_metadata(train)
    available = sorted(int(s) for s in np.unique(speakers))
    if len(available) != 10 or not set(selected).issubset(available):
        raise ValueError('Expected ten official-training speakers including the two frozen IDs')
    heldout = np.isin(speakers, selected)
    subsets = {}
    for name, mask in [('train', ~heldout), ('validation', heldout)]:
        counts = np.bincount(labels[mask].astype(np.int64), minlength=20).tolist()
        if len(counts)!=20 or any(c==0 for c in counts):
            raise ValueError(f'{name} lacks full class coverage; do not search for another pair')
        subsets[name] = dict(indices=np.flatnonzero(mask).tolist(), sample_count=int(mask.sum()),
                             class_counts=counts, speaker_ids=sorted(int(s) for s in np.unique(speakers[mask])))
    if (set(subsets['train']['indices']) & set(subsets['validation']['indices'])
            or len(subsets['train']['speaker_ids'])!=8
            or set(subsets['train']['speaker_ids']) & set(selected)):
        raise ValueError('Speaker or recording overlap in frozen split')
    manifest = dict(protocol=DEVELOPMENT_PROTOCOL, algorithm_version='predeclared_two_speaker_ids_v2',
                    role=role, split_seed=42, available_speaker_ids=available,
                    validation_speaker_ids=selected, total_training_recordings=len(labels), subsets=subsets,
                    validation_fraction=subsets['validation']['sample_count']/len(labels),
                    official_test_opened=False,
                    dataset_identity=dict(dataset='SHD', source='official train HDF5 only',
                        train_sha256=file_sha256(train),
                        training_metadata_sha256=digest(dict(speakers=speakers.tolist(), labels=labels.tolist())),
                        speaker_field='extra/speaker', recording_axis='Original HDF5 row index'))
    manifest['fingerprint']=digest(manifest)
    return manifest


def verify_manifest(manifest, data_path, expected_fingerprint):
    rebuilt = (build_development_manifest(data_path,manifest['validation_speaker_ids'],manifest['role'])
               if manifest.get('protocol')==DEVELOPMENT_PROTOCOL else build_manifest(data_path))
    if manifest.get('fingerprint') != expected_fingerprint or manifest != rebuilt:
        raise ValueError('Speaker split manifest/dataset/fingerprint changed; refusing reuse')
    return manifest


def load_manifest(path, data_path, expected_fingerprint):
    with local_path(path, must_exist=True).open(encoding='utf-8') as stream:
        return verify_manifest(json.load(stream), data_path, expected_fingerprint)


def check_loader(dataset, manifest):
    """Check the installed wrapper's index space and public speaker metadata."""
    expected = local_path(dataset.location_on_system) / dataset.data_filename
    speakers, labels = read_metadata(expected)
    if (dataset.data_filename != 'shd_train.h5' or len(dataset) != len(labels)
            or not np.array_equal(np.asarray(dataset.speaker), speakers)
            or digest(dict(speakers=speakers.tolist(), labels=labels.tolist())) !=
            manifest['dataset_identity']['training_metadata_sha256']):
        raise ValueError('Installed SHD loader is incompatible with recording-aligned speaker metadata')


def save_manifest(path, manifest):
    path = local_path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError(f'Frozen split changed: {portable_path(path)}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)

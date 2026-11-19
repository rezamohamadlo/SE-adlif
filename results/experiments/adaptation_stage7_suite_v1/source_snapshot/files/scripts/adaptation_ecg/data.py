"""Train-MAT-only ECG loader; never instantiate ECGLDM or open the test MAT."""
import hashlib

import numpy as np
import torch
from scipy.io import loadmat

from project_paths import local_path, portable_path
from datasets.ecg import QTDB_ECGWrapper
from datasets.utils.pad_tensors import PadTensors


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = f'{value.dtype.str}:{value.shape}:'.encode('ascii')
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def load_validation(spec, baseline, cfg):
    from scripts.adaptation_analysis.pilot import sha256
    path = local_path(spec['dataset_file'], must_exist=True)
    require(path.name == 'QTDB_train.mat', 'Only the frozen training MAT may be opened')
    require(sha256(path) == spec['dataset_sha256'], 'ECG training MAT checksum differs')
    mat = loadmat(path)
    x, one_hot, ticks = (mat[key] for key in ('x', 'y', 't'))
    require(list(x.shape) == spec['expected_input_shape'], 'ECG input shape differs')
    require(one_hot.shape == (*x.shape[:2], 6) and ticks.shape == x.shape[:2],
            'ECG temporal label/time alignment differs')
    require(np.isfinite(x).all() and np.isin(x, [-1, 0, 1]).all(), 'Expected signed ECG events')
    require(np.isin(one_hot, [0, 1]).all() and np.isin(one_hot.sum(-1), [0, 1]).all(),
            'ECG targets must be binary with at most one active class per model step')
    require(np.array_equal(ticks, np.broadcast_to(np.arange(x.shape[1]), ticks.shape)),
            'Unexpected ECG time grid; physical time must not be guessed')
    section = baseline['datasets']['ECG']
    n_val = int(len(x) * section['valid_fraction'])
    # Exact random_split algorithm and order: train first, validation last.
    ids = torch.randperm(len(x), generator=torch.Generator().manual_seed(
        baseline['reproduction_policy']['split_seed'])).tolist()[-n_val:]
    require(n_val == spec['expected_validation_samples'], 'ECG split size differs')
    require(cfg['dataset'].get('burn_in_time', 0) == 0,
            'Nonzero ECG burn-in requires an explicit target-mask protocol review')
    y = one_hot.argmax(-1).astype(np.int64)
    data = ValidationData(x, y, ids)
    # Input-only contract comparison with the existing wrapper/collator; no DataModule setup.
    legacy = QTDB_ECGWrapper(torch.utils.data.TensorDataset(torch.from_numpy(x * 1.0),
                            torch.from_numpy(y)), burn_in_time=0)
    first = ids[:3]
    a = data.batch(first, 'clean', 0, spec)
    b = PadTensors(batch_first=True)([legacy[i] for i in first])
    require(all(torch.equal(left, right) for left, right in zip(a, b)),
            'ECG validation loader differs from the saved preprocessing contract')
    meta = dict(path=portable_path(path), sha256=spec['dataset_sha256'],
        input_shape=list(x.shape), input_values=[-1, 0, 1], validation_indices=ids,
        train_records=len(x)-n_val, validation_records=n_val, split_seed=42,
        split='last floor(N * 0.05) indices of torch.randperm with a fresh generator',
        steps_per_record=x.shape[1], channels=x.shape[2], classes=6,
        burn_in_steps=0, padding='no collator padding: all stored records have the same saved length',
        zero_label_policy='preserve original argmax: all-zero class vectors map to class 0 and remain scored',
        zero_label_positions_validation=int((one_hot[ids].sum(-1) == 0).sum()),
        zero_label_positions_by_validation_record=[int((one_hot[i].sum(-1) == 0).sum()) for i in ids],
        max_i='not used by the original loader; no new truncation or mask inferred',
        time_unit='model step; physical milliseconds not independently established',
        patient_identity='not supplied by this MAT; rows are not independent patient clusters',
        loader_contract_records=first, loader_contract='passed, input-only', official_test_opened=False)
    return data, meta


def delete_events(x, probability, sample_id, replicate, selection_seed):
    """Drop events without changing polarity, record length, targets or block IDs."""
    require(0 <= probability <= 1 and np.isin(x, [-1, 0, 1]).all(), 'Invalid signed-event deletion')
    rng = np.random.default_rng(np.random.SeedSequence([selection_seed, sample_id, replicate]))
    result = x.copy()
    result[(x != 0) & (rng.random(x.shape) < probability)] = 0
    return result


class ValidationData:
    def __init__(self, x, y, ids):
        self.x, self.y, self.ids = x, y, list(ids)

    def input(self, sample_id, condition, replicate, spec):
        require(sample_id in self.ids, 'Requested sample is outside the frozen validation split')
        x = self.x[sample_id].astype(np.float32)
        if condition == 'delete_0.1':
            return delete_events(x, spec['deletion_probability'], sample_id, replicate, spec['selection_seed'])
        require(condition == 'clean' and replicate == 0, 'Unsupported ECG condition')
        return x

    def batch(self, ids, condition, replicate, spec):
        items = [(torch.from_numpy(self.input(i, condition, replicate, spec)),
                  torch.from_numpy(self.y[i]), torch.arange(1, len(self.y[i])+1)) for i in ids]
        return PadTensors(batch_first=True)(items)

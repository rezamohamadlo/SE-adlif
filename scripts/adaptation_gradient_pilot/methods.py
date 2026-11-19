"""Explicit surrogate-gradient summaries and exact-budget count deletions."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = f'{value.dtype.str}:{value.shape}:'.encode('ascii')
    return hashlib.sha256(header+value.tobytes()).hexdigest()


def count_hash(value):
    value = np.asarray(value)
    require(value.ndim == 2 and np.isfinite(value).all() and np.all(value >= 0) and np.equal(value, np.floor(value)).all(),
            'Counts must be finite nonnegative integers')
    return hashlib.sha256(str(value.shape).encode()+value.astype('<i8', copy=False).tobytes()).hexdigest()


def state_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode()+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
    return result.hexdigest()


def require_frozen(model, expected):
    require(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            'Pilot requires eval mode, frozen weights and no parameter gradients')
    require(state_hash(model) == expected, 'Frozen model state changed')


def gradient_summary(gradient, counts, burn_in=10, bins=20):
    gradient = np.asarray(gradient)
    require(gradient.ndim == 2 and gradient.shape[1] == counts.shape[1] and len(gradient) >= len(counts) and
            np.isfinite(gradient).all(), 'Nonfinite or misaligned input surrogate gradient')
    physical = gradient[:len(counts)].astype(np.float64)
    absolute = np.abs(physical)
    mass = float(absolute.sum())
    event_mass = float(absolute[counts > 0].sum())
    if mass == 0:
        temporal, channels = None, None
    else:
        positions = np.minimum(bins-1, np.arange(len(counts))*bins//max(1, len(counts)))
        temporal = (np.bincount(positions, weights=absolute.sum(1), minlength=bins)/mass).tolist()
        channels = (absolute.sum(0)/mass).tolist()
    return dict(gradient_sha256=array_hash(gradient), physical_steps=len(counts), stored_steps=len(gradient),
        physical_abs_gradient_sum=mass, physical_gradient_l2=float(np.linalg.norm(physical)),
        event_cell_abs_gradient_sum=event_mass, empty_cell_abs_gradient_sum=float(absolute[counts == 0].sum()),
        burn_in_input_abs_gradient_sum=float(absolute[:burn_in].sum()),
        synthetic_padding_abs_gradient_sum=float(np.abs(gradient[len(counts):].astype(np.float64)).sum()),
        relative_time_abs_mass=temporal, channel_abs_mass=channels,
        normalization_status='defined' if mass > 0 else 'zero_physical_gradient',
        deletion_ranking_status='defined' if event_mass > 0 else 'no_nonzero_gradient_on_existing_events')


def event_budget(counts, fraction):
    count_hash(counts)
    require(0 < fraction <= 1, 'Invalid deletion fraction')
    total = int(np.asarray(counts).sum(dtype=np.int64))
    return 0 if total == 0 else max(1, int(np.floor(total*fraction)))


def exact_random_delta(counts, budget, seed):
    count_hash(counts)
    total = int(counts.sum(dtype=np.int64))
    require(0 <= budget <= total, 'Deletion budget exceeds available events')
    if budget == 0: return np.zeros_like(counts, dtype=np.int64)
    events = np.random.default_rng(seed).choice(total, size=budget, replace=False)
    bins = np.searchsorted(np.cumsum(counts.reshape(-1), dtype=np.int64), events, side='right')
    removed = np.bincount(bins, minlength=counts.size).reshape(counts.shape)
    return -removed.astype(np.int64)


def guided_delta(counts, gradient, budget):
    count_hash(counts)
    require(counts.shape == gradient.shape and np.isfinite(gradient).all(), 'Invalid deletion ranking gradient')
    available = np.flatnonzero(counts.reshape(-1) > 0)
    scores = np.abs(gradient.reshape(-1)[available]).astype(np.float64)
    require(0 <= budget <= int(counts.sum(dtype=np.int64)), 'Invalid guided deletion budget')
    if budget == 0 or not np.any(scores > 0): return None
    ranking = available[np.lexsort((available, -scores))]
    delta, remaining = np.zeros(counts.size, dtype=np.int64), budget
    for i in ranking:
        amount = min(remaining, int(counts.reshape(-1)[i]))
        delta[i] = -amount
        remaining -= amount
        if remaining == 0: break
    require(remaining == 0, 'Guided deletion did not consume its exact budget')
    return delta.reshape(counts.shape)


def validate_delta(counts, delta, budget):
    require(delta.shape == counts.shape and np.issubdtype(delta.dtype, np.integer) and np.all(delta <= 0) and
            np.all(counts+delta >= 0) and -int(delta.sum(dtype=np.int64)) == budget,
            'Deletion must preserve length, nonnegative integer counts and exact matched budget')


def publish(path, temporary, replace=os.replace, sleep=time.sleep):
    for attempt in range(12):
        try:
            replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 11:
                error.add_note('Uncommitted payload preserved at '+str(temporary))
                raise
            sleep(min(.05*2**attempt, 1.))


def atomic_npz(path, arrays, *, replace=os.replace, sleep=time.sleep):
    """Closed unique same-directory payload; bounded Windows sharing retries."""
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='.'+path.name+'-', suffix='.tmp', dir=path.parent)
    with os.fdopen(descriptor, 'wb') as stream:
        np.savez_compressed(stream, **arrays)
    publish(path, name, replace, sleep)


def atomic_copy(source, target):
    target = Path(target)
    descriptor, name = tempfile.mkstemp(prefix='.'+target.name+'-', suffix='.tmp', dir=target.parent)
    with os.fdopen(descriptor, 'wb') as destination, open(source, 'rb') as original:
        shutil.copyfileobj(original, destination)
    publish(target, name)


def cached_array(path, identity, produce, write):
    """Embedded identities allow recovery of a published NPZ lacking its JSON sidecar."""
    from scripts.adaptation_analysis import pilot
    from project_paths import portable_path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = path.with_suffix('.json')
    require(path.exists() or not metadata.exists(), 'Array file missing from an existing cache; transfer the exact file')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            pilot.validate_manifest(json.loads(saved['identity_json'].item()), identity)
            value = json.loads(saved['summary_json'].item())
            arrays = {k: saved[k].copy() for k in saved.files if k not in ('identity_json', 'summary_json')}
    else:
        value, arrays = produce()
        require(not any(k in arrays for k in ('identity_json', 'summary_json')), 'Reserved array metadata key')
        atomic_npz(path, dict(arrays, identity_json=np.array(json.dumps(identity, sort_keys=True, allow_nan=False)),
                             summary_json=np.array(json.dumps(value, sort_keys=True, allow_nan=False))))
    header = dict(identity=identity, value=value, value_sha256=pilot.digest(value), artifact=portable_path(path),
                  artifact_sha256=pilot.sha256(path))
    if metadata.exists(): pilot.validate_manifest(pilot.read_json(metadata), header)
    else: write(metadata, header)
    return value, arrays, [path, metadata]

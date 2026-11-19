"""Opt-in fresh-run evidence; reads state without drawing RNG or running a model."""
import hashlib
import json
import os
from datetime import datetime, timezone

import torch

from extended_seed_study import atomic_json
from project_paths import local_path


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_fingerprint(model):
    """Hash tensor values separately from trainability, including scalar buffers."""
    parameters = dict(model.named_parameters())
    tensors = {}
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value_hash = hashlib.sha256(_bytes(tensor)).hexdigest()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        tensors[name] = dict(sha256=value_hash, shape=shape, dtype=dtype,
                             parameter=name in parameters,
                             requires_grad=parameters[name].requires_grad if name in parameters else False)
        digest.update(json.dumps([name, dtype, shape, value_hash], separators=(',', ':')).encode('utf-8'))
    return dict(model_state_sha256=digest.hexdigest(), tensor_count=len(tensors), tensors=tensors)


def record_initial_state(directory, model, cfg, precision):
    """Create/reuse initial evidence; never describe resumed weights as initialization."""
    if cfg.get('ckpt_path') or cfg.get('init_ckpt_path'):
        raise ValueError('Initial state evidence is for fresh training only')
    state = state_fingerprint(model)
    device = next(model.parameters()).device
    record = dict(version='fresh_initial_state_v1',
        recorded_at=datetime.now(timezone.utc).isoformat(),
        model_variant=cfg['model_variant'], training_seed=int(cfg['random_seed']),
        dataset=cfg['paper_baseline_dataset'],
        split_fingerprint=cfg.get('dataset', {}).get('speaker_split_fingerprint'),
        protocol=cfg.get('evaluation_protocol'), initialization_mode='fresh',
        optimizer_updates=0, precision=str(precision), model_device=str(device),
        cpu_rng_sha256=hashlib.sha256(_bytes(torch.get_rng_state())).hexdigest(),
        cuda_rng_sha256=(hashlib.sha256(_bytes(torch.cuda.get_rng_state(device))).hexdigest()
                         if device.type == 'cuda' else None),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cuda_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), **state)
    path = local_path(directory)/'INITIAL_MODEL_STATE.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        comparable = lambda value: {k: v for k, v in value.items() if k != 'recorded_at'}
        if comparable(previous) != comparable(record):
            raise ValueError('Existing initial-state evidence conflicts; refusing to overwrite')
        return previous
    atomic_json(path, record)
    return record

"""Measure trusted, validation-selected SHD* checkpoints without training.

Defaults cover the canonical seven-model, three-seed suite. Results are separate
from training artifacts. --dry-run checks checkpoint identities without inference.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time

import hydra
from omegaconf import OmegaConf
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from datasets.utils.pad_tensors import PadTensors
from models.pl_module import MLPSNN
from project_paths import local_path, portable_config, portable_path
from cost_records import load_measurement, record_path, seed_directory

ROOT = Path(__file__).resolve().parent
MODELS = {'SE_adLIF': 'SE-adLIF reference', 'DTH_SE_adLIF': 'DTH',
          'LG_DTH_SE_adLIF': 'LG-DTH', 'DA_SE_adLIF': 'DA',
          'FP_SE_adLIF': 'FP', 'MR_SE_adLIF': 'MR (K=2)', 'MT_SE_adLIF': 'MT'}
VERSION = 'shd_costs_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_directory(model, seed):
    return seed_directory(ROOT, model, 'SHD', seed)


def selected_checkpoint(directory):
    record = directory / 'REFERENCE_RESULT.json'
    if record.exists():
        path = directory / json.loads(record.read_text(encoding='utf-8'))['checkpoint']
    else:
        text = (directory / 'TRAINING_SUMMARY.md').read_text(encoding='utf-8')
        match = re.search(r'## Best checkpoint\s+```text\s+([^\n]+)', text)
        if not match:
            raise ValueError(f'No selected checkpoint record: {directory}')
        path = directory / match[1].strip()
    if not path.is_file():
        raise FileNotFoundError(f'Copy the exact selected checkpoint: {path}')
    return path


def saved_test_accuracy(directory):
    record = directory / 'REFERENCE_RESULT.json'
    if record.exists():
        return json.loads(record.read_text(encoding='utf-8'))['test_acc']
    paths = list(directory.glob('logs/**/metrics.csv'))
    if len(paths) != 1:
        raise ValueError(f'Ambiguous metrics histories: {directory}')
    with paths[0].open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r.get('test_acc')]
    if len(rows) != 1:
        raise ValueError(f'Expected one final test record: {directory}')
    return float(rows[0]['test_acc'])


def dataset_signature(cfg):
    settings = OmegaConf.to_container(cfg.dataset, resolve=True)
    # SHDLDM's constructor defaults this legacy-optional flag to false.
    settings.setdefault('additional_test_set_validation', False)
    for key in ['data_path', 'num_workers', 'cache_namespace']:
        settings.pop(key, None)
    return settings


def load_run(model_name, seed):
    directory = run_directory(model_name, seed)
    path = selected_checkpoint(directory)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(portable_config(OmegaConf.to_container(
        OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True)))
    expected_cell = {'SE_adLIF': 'se_adlif'}.get(model_name, model_name.lower())
    if (cfg.l1.cell != expected_cell or cfg.l2.cell != expected_cell
            or cfg.random_seed != seed or cfg.dataset.validate_on != 0.2
            or cfg.dataset.random_seed != 42 or cfg.dataset.batch_size != 256
            or cfg.get('early_stopping', False) or cfg.n_epochs != 300):
        raise ValueError(f'Unexpected model/seed/SHD* baseline settings: {path}')
    if cfg.dataset._target_ != 'datasets.shd.SHDLDM':
        raise ValueError('This evaluator supports SHD* classification only')
    if cfg.get('auto_regression', False) or not cfg.two_layers:
        raise ValueError('Expected two hidden layers and classification')
    return directory, path, checkpoint, cfg


class ScalarOperations(TorchDispatchMode):
    """Count logical scalar ATen operations, excluding matrix multiplies.

    A special function/comparison counts as ONE logical operation, not one
    hardware FLOP. Reductions count scalar additions plus mean division.
    Bias in fused addmm counts separately. Shape/memory operations are excluded.
    Unknown operations fail rather than silently undercounting.
    """
    ELEMENTWISE = {'add', 'sub', 'rsub', 'mul', 'div', 'neg', 'abs', 'exp',
                   'sigmoid', 'tanh', 'sin', 'sqrt', 'pow', 'square', 'reciprocal',
                   'heaviside', 'gt', 'ge', 'lt', 'le', 'eq', 'ne', 'relu'}
    MEMORY = {'detach', 'alias', 'view', '_unsafe_view', 'reshape', 't', 'transpose',
              'permute', 'unsqueeze', 'squeeze', 'expand', 'slice', 'select', 'unbind',
              'index_select', 'cat', 'stack', 'clone', 'copy_', 'lift_fresh',
              'lift_fresh_copy', '_to_copy', 'to', 'empty', 'empty_like',
              'empty_strided', 'zeros', 'zeros_like', 'ones_like', 'full_like',
              'fill_', 'new_zeros', 'new_empty', 'constant_pad_nd'}

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        name = func._schema.name.split('::')[-1]
        n = result.numel() if isinstance(result, torch.Tensor) else 0
        if name in self.ELEMENTWISE:
            self.counts[name] += n
            if name in {'add', 'sub'} and kwargs.get('alpha', 1) != 1:
                self.counts['mul'] += n
        elif name in {'sum', 'mean'}:
            self.counts[name] += args[0].numel() - n + (n if name == 'mean' else 0)
        elif name in {'clamp', 'clamp_min', 'clamp_max'}:
            bounds = sum(x is not None for x in args[1:]) + sum(
                v is not None for k, v in kwargs.items() if k in {'min', 'max'})
            self.counts[name] += n * max(1, bounds)
        elif name in {'_softmax', 'softmax'}:
            width = args[0].shape[args[1]]
            self.counts[name] += (n // width) * (5 * width - 2)
        elif name == 'dropout':
            if args[2]:
                raise ValueError('Operation profiling requires evaluation mode')
        elif name == 'linear':
            if (len(args) > 2 and args[2] is not None) or kwargs.get('bias') is not None:
                self.counts['bias_add'] += n
        elif name == 'addmm':
            self.counts['bias_add'] += n
        elif name not in self.MEMORY | {'mm', 'matmul', 'bmm'}:
            raise NotImplementedError(f'Unaccounted scalar operation: {func}')
        return result


def profile_cell(layer, batch, steps):
    state = layer.initial_state(batch, device=layer.weight.device)
    x = layer.weight.new_zeros((batch, layer.in_features))
    counters = []
    for _ in range(steps):
        with ScalarOperations() as counter:
            _, state = layer(x, state)
        counters.append(counter.counts)
    return counters


def operation_profiles(model):
    profiles = {}
    for name in ['l1', 'l2', 'out_layer']:
        layer = getattr(model, name)
        period = getattr(layer, 'adaptation_update_interval', 1)
        # The first step can broadcast singleton initial membrane state.
        one = profile_cell(layer, 1, 2 * period + 2)
        two = profile_cell(layer, 2, 2 * period + 2)
        three = profile_cell(layer, 3, 2 * period + 2)
        keys = set().union(*(set(c) for c in one + two + three))
        affine = []
        for t in range(period + 1):
            slope = {k: two[t][k] - one[t][k] for k in keys}
            intercept = {k: 2 * one[t][k] - two[t][k] for k in keys}
            affine.append({'per_sample': slope, 'per_batch': intercept})
        for t, actual in enumerate(three):
            entry = affine[0 if t == 0 else 1 + (t - 1) % period]
            assert all(actual[k] == 3 * entry['per_sample'][k] + entry['per_batch'][k]
                       for k in keys), (name, t, 'non-affine operation count')
        profiles[name] = dict(period=period, steps=affine)
    return profiles


def batch_scalar_operations(profiles, batch, timesteps):
    result = Counter()
    for profile in profiles.values():
        for t in range(timesteps):
            entry = profile['steps'][0 if t == 0 else 1 + (t - 1) % profile['period']]
            for k in entry['per_sample']:
                result[k] += batch * entry['per_sample'][k] + entry['per_batch'][k]
    return result


class SpikeCounter:
    def __init__(self, model, lengths, timesteps):
        self.values = {}
        self.handles = []
        for name in ['l1', 'l2', 'out_layer']:
            layer = getattr(model, name)
            slots = torch.zeros(6, device=layer.weight.device, dtype=torch.float64)
            self.values[name] = slots
            step = [0]

            def hook(cell, args, output, slots=slots, step=step, name=name):
                x, state = args
                t = step[0]
                valid = (lengths > t).unsqueeze(1)
                # Event multiplicity in framed input is retained, not binarized.
                ff = x.sum(1) * cell.out_features
                rec = (state[1].sum(1) * cell.out_features
                       if getattr(cell, 'use_recurrent', False) else torch.zeros_like(ff))
                slots[0] += (ff + rec).sum()
                slots[1] += ((ff + rec) * valid.squeeze(1)).sum()
                if name != 'out_layer':
                    spikes = output[0]
                    slots[2] += spikes.sum()
                    slots[3] += (spikes * valid).sum()
                    slots[4] += ((spikes != 0) & (spikes != 1)).sum()
                    slots[5] += (~torch.isfinite(spikes)).sum()
                step[0] += 1

            self.handles.append(layer.register_forward_hook(hook))
        self.timesteps = timesteps

    def close(self):
        for handle in self.handles:
            handle.remove()
        values = {k: v.cpu().tolist() for k, v in self.values.items()}
        if any(v[4] or v[5] for v in values.values()):
            raise ValueError('Hidden outputs are not finite binary spikes')
        return values


def collate_lengths(items):
    return (*PadTensors()(items), torch.tensor([len(x[0]) for x in items]))


def prepare_batches(cfg):
    data_path = local_path(cfg.dataset.data_path, must_exist=True)
    # Refuse automatic downloads; the evaluator must use existing local data.
    if not (data_path / 'SHDWrapper/shd_test.h5').is_file():
        raise FileNotFoundError('Local data/SHDWrapper/shd_test.h5 is required')
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    data_cfg.data_path = str(data_path)
    data_cfg.num_workers = 0
    dm = hydra.utils.instantiate(data_cfg)
    loader = dm.test_dataloader()
    loader.collate_fn = collate_lengths
    batches = []
    fingerprint = hashlib.sha256()
    for batch in loader:
        for tensor in batch:
            fingerprint.update(str(tuple(tensor.shape)).encode())
            fingerprint.update(tensor.contiguous().numpy().tobytes())
        batches.append(batch)
    return batches, fingerprint.hexdigest()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(model, batches, device, repeats, warmup):
    profiles = operation_profiles(model)
    counts = Counter()
    scalar_ops = Counter()
    for cpu_batch in batches:
        inputs, targets, block_idx, lengths = [x.to(device) for x in cpu_batch]
        b, t = inputs.shape[:2]
        counter = SpikeCounter(model, lengths, t)
        try:
            outputs = model(inputs)
        finally:
            activity = counter.close()
        reduced, _, _ = model.process_predictions_and_compute_losses(outputs, targets, block_idx)
        mask = targets.flatten() != -1
        counts['correct'] += int((reduced.argmax(-1)[mask] == targets.flatten()[mask]).sum())
        counts['targets'] += int(mask.sum())
        counts['samples'] += b
        counts['padded_sample_timesteps'] += b * t
        counts['sequence_sample_timesteps'] += int(lengths.sum())
        scalar_ops.update(batch_scalar_operations(profiles, b, t))
        for name, values in activity.items():
            counts['synops_padded'] += values[0]
            counts['synops_sequence'] += values[1]
            if name != 'out_layer':
                counts[f'{name}_spikes_padded'] += values[2]
                counts[f'{name}_spikes_sequence'] += values[3]
    # Hooks removed. GPU inputs prepared before each timed region. The timed
    # scope is MLPSNN.forward only, without loss, aggregation or data transfer.
    x = batches[0][0].to(device)
    for _ in range(warmup):
        model(x)
    synchronize(device)
    timings = []
    for _ in range(repeats):
        elapsed = 0.0
        for cpu_batch in batches:
            x = cpu_batch[0].to(device)
            synchronize(device)
            start = time.perf_counter()
            model(x)
            synchronize(device)
            elapsed += time.perf_counter() - start
        timings.append(1000 * elapsed / counts['samples'])
    n = counts['samples']
    macs = sum(layer.weight.numel() + (
        layer.recurrent.numel() if getattr(layer, 'use_recurrent', False) else 0)
        for layer in [model.l1, model.l2, model.out_layer])
    return dict(
        totals=dict(counts), nonmatrix_scalar_operations_by_kind=dict(scalar_ops),
        operation_profiles=profiles,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        spikes_l1_per_sample=counts['l1_spikes_sequence'] / n,
        spikes_l2_per_sample=counts['l2_spikes_sequence'] / n,
        spikes_total_per_sample=(counts['l1_spikes_sequence'] + counts['l2_spikes_sequence']) / n,
        firing_l1_percent=100 * counts['l1_spikes_sequence'] / (counts['sequence_sample_timesteps'] * model.l1.out_features),
        firing_l2_percent=100 * counts['l2_spikes_sequence'] / (counts['sequence_sample_timesteps'] * model.l2.out_features),
        synops_per_sample=counts['synops_sequence'] / n,
        synops_padded_per_sample=counts['synops_padded'] / n,
        nonmatrix_ops_per_sample=sum(scalar_ops.values()) / n,
        dense_gflops_test_padded=2 * macs * counts['padded_sample_timesteps'] / n / 1e9,
        forward_ms_per_sample=statistics.mean(timings), timing_repeat_ms_per_sample=timings,
        reproduced_test_acc=counts['correct'] / counts['targets'])


def mean_sd(values, digits=2, scale=1):
    values = [v / scale for v in values]
    return f'{statistics.mean(values):,.{digits}f} ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â± {statistics.stdev(values):,.{digits}f}'


def update_reports(output=None):
    records = [load_measurement(ROOT, model, 'SHD', seed, output)
               for model in MODELS for seed in [42, 123, 456]]
    identities = {r['measurement_identity'] for r in records}
    if len(identities) != 1 or any(r['version'] != VERSION for r in records):
        raise ValueError('Cannot aggregate different data or measurement protocols')
    rows = []
    for model, label in MODELS.items():
        group = [r for r in records if r['model'] == model]
        def metric(key, **kwargs):
            return mean_sd([r['measurements'][key] for r in group], **kwargs)
        rows.append('| ' + label + ' | ' + ' | '.join([
            metric('spikes_total_per_sample', digits=0),
            metric('spikes_l1_per_sample', digits=0) + ' / ' + metric('spikes_l2_per_sample', digits=0),
            metric('firing_l1_percent') + ' / ' + metric('firing_l2_percent'),
            metric('synops_per_sample', scale=1e6),
            metric('nonmatrix_ops_per_sample', scale=1e6),
            metric('forward_ms_per_sample', digits=3)]) + ' |')
    table = ('| Model | Total hidden spikes/sample | Spikes/sample, L1 / L2 | Firing fraction (%), L1 / L2 | Estimated synaptic operations/sample (M) | Non-matrix logical operations/sample (M), padded | Inference time (ms/sample) |\n'
             '|---|---|---|---|---|---|---|\n' + '\n'.join(rows))
    report = ROOT / 'reports/reference-vs-models.md'
    text = report.read_text(encoding='utf-8')
    text, n = re.subn(r'\| Model \| Total hidden spikes/sample[^\n]*\n\|[-| :]+\n(?:\|[^\n]*\n?){7}',
                      lambda _: table + '\n', text)
    if n != 1:
        raise ValueError('Cannot identify activity table; report was not changed')
    start = text.index('Fill the pending columns') if 'Fill the pending columns' in text else text.index('Measured on the full official SHD* test set')
    end = text.index('| Shared setting |', start)
    evidence = portable_path(output) if output is not None else 'results/valid'
    text = text[:start] + (
        f'Measured on the full official SHD* test set, n=3 training seeds per model; entries are mean ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â± sample SD. Per-seed JSONs, exact checkpoint hashes, operation breakdowns and timing repeats: `{evidence}/`.\n\n'
        'Spikes and firing fractions use each stored sequence length before batch padding, including the configured minimum 250 frames and initial ignored frames. This is not the raw event-duration length. Additional batch-padding spikes and costs are retained separately in the JSON records. All models use identical test batches, batch size 256, float32 and validation-selected checkpoints.\n\n'
        'Synaptic operations are an ideal dense-fan-out event-driven addition estimate: input frame event multiplicity and hidden spikes times the number of receiving neurons, including the actual previous-step recurrent spikes. Recurrent spikes without a subsequent destination timestep are not counted; zero-valued weights are not pruned. Delay filtering and neuron updates are counted separately. These estimates do not describe the dense GPU execution.\n\n'
        'Non-matrix logical operations count the executed cell code on all batch-padded timesteps, including parameter transforms, bias additions, surrogate forward arithmetic, reductions and state/threshold updates. A transcendental, comparison or scalar arithmetic operation each counts as one logical operation; softmax uses an explicit scalar reduction convention. Tensor allocation, indexing, memory movement and matrix multiplication are excluded. These are not hardware FLOPs and must not be added to event-driven SynOps as an energy estimate.\n\n'
        'Runtime is synchronized wall time for the uninstrumented `MLPSNN.forward`, after warm-up; data loading/transfer, loss and temporal prediction aggregation are excluded. Hardware, repeat counts and software versions are recorded in JSON. Runtime SD across seeds also reflects system variability; timing was sequential, not an energy benchmark.\n\n'
    ) + text[end:]
    report.write_text(text, encoding='utf-8', newline='\n')
    # Update reference-only SHD* activity cells while keeping other datasets pending.
    report = ROOT / 'reports/paper-vs-reference.md'
    text = report.read_text(encoding='utf-8')
    ref = rows[0].split(' | ')[1:]
    ref[-1] = ref[-1].rstrip(' |')
    text = re.sub(r'^\| SHD\* \| 450,760 \| 450,020 \|.*$',
                  lambda _: '| SHD* | 450,760 | 450,020 | ' + ' | '.join(ref) + ' |', text, flags=re.M)
    text = text.replace('Estimated synaptic operations/sample | Neuron-state operations/sample',
                        'Estimated synaptic operations/sample (M) | Non-matrix logical operations/sample (M), padded')
    text = text.replace('Pending entries will be filled from saved configurations/checkpoints and a separate inference evaluation. LG-DTH training has completed.',
                        'SHD* activity and timing are measured from all three saved reference checkpoints; SSC/ECG measurements remain pending.')
    report.write_text(text, encoding='utf-8', newline='\n')
    report = ROOT / 'results/not_improved/LG_DTH_SE_adLIF/RESULTS.md'
    if not report.exists():
        report = ROOT / 'results/valid/LG_DTH_SE_adLIF/RESULTS.md'
    text = report.read_text(encoding='utf-8')
    group = [r['measurements'] for r in records if r['model'] == 'LG_DTH_SE_adLIF']
    replacements = {
        'Total hidden spikes/sample': mean_sd([r['spikes_total_per_sample'] for r in group], digits=0),
        'Spikes/sample, L1 / L2': mean_sd([r['spikes_l1_per_sample'] for r in group], digits=0) + ' / ' + mean_sd([r['spikes_l2_per_sample'] for r in group], digits=0),
        'Firing fraction (%), L1 / L2': mean_sd([r['firing_l1_percent'] for r in group]) + ' / ' + mean_sd([r['firing_l2_percent'] for r in group]),
        'Estimated synaptic operations/sample': mean_sd([r['synops_per_sample'] for r in group], digits=0),
        'Neuron-state operations/sample': mean_sd([r['nonmatrix_ops_per_sample'] for r in group], digits=0),
        'Inference time (ms/sample)': mean_sd([r['forward_ms_per_sample'] for r in group], digits=3)}
    for name, value in replacements.items():
        lookup = ('Non-matrix logical operations/sample, padded'
                  if name == 'Neuron-state operations/sample' and '| Non-matrix logical operations/sample, padded |' in text
                  else name)
        text = re.sub(r'^\| ' + re.escape(lookup) + r' \|[^\n]*',
                      lambda _: f'| {name} | {value} | Mean ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â± sample SD across three seeds; definitions in linked comparison |', text, flags=re.M)
    text = text.replace('Neuron-state operations/sample', 'Non-matrix logical operations/sample, padded')
    text = re.sub(r'(?:These pending measurements|All three seeds measured)[^\n]+',
                  'All three seeds measured on the official test set. Activity uses pre-batch sequence lengths, including configured minimum padding; non-matrix operations and runtime include batch padding. See the [measurement protocol](../../../../reports/reference-vs-models.md#spike-activity-and-inference-cost).', text)
    report.write_text(text, encoding='utf-8', newline='\n')
    if output is not None:
        save_json(output / 'AGGREGATE.json', dict(version=VERSION, records=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', help='Optional separate benchmark output; default saves beside each training summary')
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--update-reports', action='store_true')
    parser.add_argument('--reports-only', action='store_true', help='Read saved measurements without inference')
    args = parser.parse_args()
    if args.timing_repeats < 2 or args.warmup < 1:
        parser.error('Use at least two timing repeats and one warm-up')
    output = local_path(args.output) if args.output else None
    if args.reports_only:
        update_reports(output)
        return
    if args.dry_run:
        for model in args.models:
            for seed in args.seeds:
                _, path, _, _ = load_run(model, seed)
                print(model, seed, portable_path(path), flush=True)
        return
    session = output if output is not None else ROOT / 'results/valid'
    session.mkdir(parents=True, exist_ok=True)
    # OS lock is released on exit/crash. Do not bypass a held lock.
    with (session / '.evaluation.lock').open('a+b') as lock:
        if platform.system() == 'Windows':
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device(args.device)
        torch.set_float32_matmul_precision('high')
        environment = dict(python=platform.python_version(), torch=torch.__version__,
                           cuda=torch.version.cuda, device=str(device),
                           gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                           timing_repeats=args.timing_repeats, warmup=args.warmup,
                           matmul_precision=torch.get_float32_matmul_precision(),
                           implementation_sha256=sha256(Path(__file__)),
                           model_source_sha256={portable_path(p): sha256(p) for p in
                               sorted((ROOT / 'models').rglob('*.py'))},
                           timing_scope='MLPSNN.forward; excludes transfer, loading, loss and prediction aggregation',
                           scalar_convention='Logical scalar ATen operations; nonlinearities/comparisons each count as one, not FLOPs',
                           sequence_scope='Stored transformed length before batch padding; includes minimum 250 frames and initial ignored frames')
        batches = None
        dataset_settings = None
        for model_name in args.models:
            for seed in args.seeds:
                directory, path, checkpoint, cfg = load_run(model_name, seed)
                settings = dataset_signature(cfg)
                if batches is None:
                    print('Preparing identical official test batches...', flush=True)
                    batches, data_hash = prepare_batches(cfg)
                    dataset_settings = settings
                if settings != dataset_settings:
                    raise ValueError('Dataset/preprocessing differs across checkpoints')
                identity = hashlib.sha256(json.dumps(dict(version=VERSION, environment=environment,
                    data_hash=data_hash, settings=settings), sort_keys=True).encode()).hexdigest()
                destination = record_path(ROOT, model_name, 'SHD', seed, output)
                ckpt_hash = sha256(path)
                if destination.exists():
                    old = json.loads(destination.read_text(encoding='utf-8'))
                    if old['measurement_identity'] != identity or old['checkpoint_sha256'] != ckpt_hash:
                        raise ValueError(f'Existing measurement differs; use a new --output: {destination}')
                    print(f'{model_name} seed {seed}: verified existing measurement', flush=True)
                    continue
                torch.manual_seed(seed)
                with redirect_stdout(io.StringIO()):
                    model = MLPSNN(cfg)
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                model.to(device).eval()
                print(f'{model_name} seed {seed}: measuring...', flush=True)
                with torch.inference_mode():
                    measurements = measure(model, batches, device, args.timing_repeats, args.warmup)
                expected = saved_test_accuracy(directory)
                if abs(measurements['reproduced_test_acc'] - expected) > 1e-6:
                    raise ValueError(f'Test accuracy differs: {measurements["reproduced_test_acc"]} vs {expected}')
                record = dict(version=VERSION, measurement_identity=identity, model=model_name, seed=seed,
                    created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=portable_path(path),
                    checkpoint_sha256=ckpt_hash, best_epoch=checkpoint['epoch'],
                    saved_config=portable_config(OmegaConf.to_container(cfg, resolve=True)),
                    environment=environment, data_sha256=data_hash, measurements=measurements)
                save_json(destination, record)
                print(f'{model_name} seed {seed}: spikes/sample={measurements["spikes_total_per_sample"]:.1f}; '
                      f'forward ms/sample={measurements["forward_ms_per_sample"]:.3f}', flush=True)
                del model, checkpoint
        if args.update_reports:
            update_reports(output)
            print('Updated comparison reports and LG-DTH results.', flush=True)


if __name__ == '__main__':
    main()

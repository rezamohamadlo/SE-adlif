"""Measure frozen-checkpoint TTA costs; never trains or changes ordinary costs."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import statistics
import time

from omegaconf import OmegaConf
import torch

import evaluate_tta as tta
import measure_reference_costs as reference
import measure_run_costs as ordinary
import measure_shd_costs as shared
from project_paths import ROOT, local_path, portable_path
from ssc_loss_diagnostic import atomic_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evidence_for(job, gains):
    """Require existing official-test evidence for this exact checkpoint/policy."""
    model = job['config']['l1']['cell'].removesuffix('_se_adlif')
    paths = sorted((ROOT / 'results/experiments').glob(
        f'{model}_tta_*/{job["dataset"]}/seed_{job["seed"]}/test/RESULT.json'))
    matches = []
    for path in paths:
        record = json.loads(path.read_text(encoding='utf-8'))
        manifest = json.loads(path.with_name('MANIFEST.json').read_text(encoding='utf-8'))
        if (record['checkpoint_sha256'] == job['checkpoint_sha256']
                and record['gains'] == gains and record['split'] == 'test'
                and manifest['checkpoint_sha256'] == job['checkpoint_sha256']):
            matches.append((path, record))
    if not matches:
        raise ValueError(f'No matching saved TTA test result: {job["dataset"]} seed {job["seed"]}')
    first = matches[0][1]
    for _, record in matches[1:]:
        for key in ('tta_accuracy', 'baseline_accuracy', 'data_sha256', 'evaluated_targets'):
            if record['metrics'][key] != first['metrics'][key]:
                raise ValueError('Conflicting saved TTA test evidence; resolve before measuring')
    path, record = matches[-1]
    return dict(path=portable_path(path), sha256=shared.sha256(path), metrics=record['metrics'])


def test_fingerprint(batches):
    """Use evaluate_tta's signature; the fourth tensor is measurement-only lengths."""
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in batch[:3]:
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_result(metrics, evidence):
    expected = evidence['metrics']
    if (metrics['data_sha256'] != expected['data_sha256']
            or metrics['evaluated_targets'] != expected['evaluated_targets']):
        raise ValueError('Test data differs from saved TTA evaluation')
    for key in ('baseline_accuracy', 'tta_accuracy'):
        if abs(metrics[key] - expected[key]) > 1e-7:
            raise ValueError(f'Reproduced {key} differs from saved TTA evaluation')


@torch.inference_mode()
def measure(model, batches, device, event_input, gains, repeats=3, warmup=2):
    if not batches:
        raise ValueError('Empty test data')
    profiles = shared.operation_profiles(model)
    operations, counts = Counter(), Counter()
    views = [Counter() for _ in gains]
    for cpu_batch in batches:
        x, targets, blocks, lengths = [tensor.to(device) for tensor in cpu_batch]
        b, steps = x.shape[:2]
        labels = targets.flatten().long()
        mask = labels != -1
        probabilities = []
        for index, gain in enumerate(gains):
            counter = reference.ReferenceCounter(model, lengths, event_input)
            try:
                outputs = model(x if gain == 1 else x * gain)
            finally:
                activity = counter.close()
            # Gains change event amplitudes, not event multiplicity. ReferenceCounter
            # sees scaled amplitudes; restore original input counts in its L1 total.
            if event_input:
                valid = torch.arange(steps, device=device)[None, :] < lengths[:, None]
                correction = x.sum(-1) * model.l1.out_features * (1 - gain)
                activity['l1'][0] += correction.sum().item()
                activity['l1'][1] += (correction * valid).sum().item()
            for name, values in activity.items():
                views[index]['synops_sequence'] += values[1]
                views[index]['synops_padded'] += values[0]
                if name != 'out_layer':
                    views[index][f'{name}_spikes_sequence'] += values[3]
            logits = tta.aggregate_temporal_outputs(
                outputs, blocks, targets.size(1), model.output_func)
            logits = logits.reshape(-1, outputs.size(-1))[mask].float()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite outputs')
            probabilities.append(logits.softmax(-1))
            if gain == 1:
                counts['baseline_correct'] += (logits.argmax(-1) == labels[mask]).sum().item()
        counts['tta_correct'] += (torch.stack(probabilities).mean(0).argmax(-1) == labels[mask]).sum().item()
        counts['targets'] += mask.sum().item()
        counts['samples'] += b
        counts['timesteps'] += b * steps
        counts['stored_timesteps'] += lengths.sum().item()
        counts['gain_multiplications'] += x.numel() * sum(g != 1 for g in gains)
        operations.update(shared.batch_scalar_operations(profiles, b, steps))
    if not counts['targets']:
        raise ValueError('No labeled test targets')
    n = counts['samples']
    macs = sum(layer.weight.numel() + (layer.recurrent.numel() if getattr(layer, 'use_recurrent', False) else 0)
               for layer in (model.l1, model.l2, model.out_layer))
    dense = 2 * macs * counts['timesteps'] / n / 1e9
    arithmetic = sum(v for k, v in operations.items() if k in reference.ARITHMETIC) / n / 1e9

    # Paired measurements without activity hooks. Transfers happen before timing.
    times = {key: [] for key in ('single_forward', 'tta_forwards', 'single_pipeline', 'tta_pipeline')}
    def timed(call):
        shared.synchronize(device)
        start = time.perf_counter()
        call()
        shared.synchronize(device)
        return time.perf_counter() - start

    def forwards(xs):
        for value in xs:
            model(value)

    for index, cpu_batch in enumerate(batches):
        x, target, block = [tensor.to(device) for tensor in cpu_batch[:3]]
        xs = [x if gain == 1 else x * gain for gain in gains]
        calls = dict(single_forward=lambda: model(x), tta_forwards=lambda: forwards(xs),
                     single_pipeline=lambda: tta.batch_predictions(model, (x, target, block), model.output_func, [1.0]),
                     tta_pipeline=lambda: tta.batch_predictions(model, (x, target, block), model.output_func, gains))
        if index == 0:
            for _ in range(warmup):
                for call in calls.values():
                    call()
        for repeat in range(repeats):
            if index == 0:
                for values in times.values():
                    values.append(0.)
            keys = list(calls)
            if (repeat + index) % 2:
                keys.reverse()
            for key in keys:
                times[key][repeat] += timed(calls[key]) * 1000 / n
        print(f'Measured timing batch {index + 1}/{len(batches)}', flush=True)
    view_metrics = [dict(gain=g, synops_per_sample=v['synops_sequence']/n,
                         spikes_per_sample=(v['l1_spikes_sequence']+v['l2_spikes_sequence'])/n)
                    for g, v in zip(gains, views)]
    return dict(samples=n, evaluated_targets=counts['targets'], data_sha256=test_fingerprint(batches),
                baseline_accuracy=counts['baseline_correct']/counts['targets'],
                tta_accuracy=counts['tta_correct']/counts['targets'],
                total_parameters=sum(p.numel() for p in model.parameters()),
                single_dense_gflops=dense, tta_dense_gflops=len(gains)*dense,
                single_arithmetic_gflops=dense+arithmetic,
                tta_arithmetic_gflops=len(gains)*(dense+arithmetic),
                single_synops_per_sample=view_metrics[gains.index(1.0)]['synops_per_sample'],
                gain_scaling_gflops=counts['gain_multiplications']/n/1e9,
                tta_synops_per_sample=sum(v['synops_per_sample'] for v in view_metrics),
                tta_spikes_per_sample=sum(v['spikes_per_sample'] for v in view_metrics),
                tta_analog_input_dense_macs_per_sample=(0 if event_input else
                    len(gains)*model.l1.weight.numel()*counts['timesteps']/n),
                per_view=view_metrics,
                forward_scalar_operations_by_kind=dict(operations),
                input_counts=dict(counts),
                timing_ms_per_sample={k: statistics.mean(v) for k, v in times.items()},
                timing_repeats_ms_per_sample=times)


def save_record(directory, stem, identity, metrics):
    record = dict(status='complete', fingerprint=fingerprint(identity), identity=identity,
                  created_utc=datetime.now(timezone.utc).isoformat(), measurements=metrics,
                  scopes=dict(denominator='One original input sequence, not one gain view',
                      forward_timing='Uninstrumented forwards; pre-scaled inputs; excludes transfer/loading/aggregation',
                      pipeline_timing='Gain scaling, forwards, temporal reduction, class softmax and averaging; excludes transfer/loading/loss/final decision',
                      compute='Executed batch-padded forwards; matrix MAC=2 FLOPs; aggregation excluded; gain scaling separate',
                      activity='Stored sequence extent including minimum padding and burn-in; batch padding excluded',
                      synops='Ideal dense fan-out additions; original event multiplicity preserved under gain scaling; ECG analog projection excluded'))
    lines = ['# TTA test costs', '', 'Frozen validation-selected checkpoint; no training.', '',
             '| Metric | Per original input sequence |', '|---|---:|']
    for key in ('single_dense_gflops', 'tta_dense_gflops', 'single_arithmetic_gflops',
                'tta_arithmetic_gflops', 'gain_scaling_gflops', 'single_synops_per_sample', 'tta_synops_per_sample',
                'tta_spikes_per_sample', 'tta_analog_input_dense_macs_per_sample'):
        lines.append(f'| {key} | {metrics[key]:.9f} |')
    for key, value in metrics['timing_ms_per_sample'].items():
        lines.append(f'| {key} ms/sample | {value:.6f} |')
    lines += ['', 'GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging',
              'are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.',
              'SynOps count event multiplicity with original input counts, not scaled amplitudes;',
              'ECG analog input projection is excluded. No energy measurement is implied.']
    tmp = directory / f'{stem}.md.tmp'
    tmp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    tmp.replace(directory / f'{stem}.md')
    atomic_json(directory / f'{stem}.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('DTH_SE_adLIF', 'MT_SE_adLIF'), required=True)
    parser.add_argument('--datasets', nargs='+', choices=('SHD', 'SSC', 'ECG'), required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=(42, 123, 456), default=[42, 123, 456])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--record-name', default='TTA_COST_RESULTS', help='New basename for changed measurement settings')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 1:
        parser.error('repeats and warmup must be positive')
    if not args.record_name.startswith('TTA_COST_RESULTS') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.record_name):
        parser.error('record-name must start with TTA_COST_RESULTS and contain only letters, digits, _ or -')
    policy = OmegaConf.to_container(OmegaConf.load(ROOT/'config/tta.yaml'), resolve=True)
    jobs = []
    for model in dict.fromkeys(args.models):
        for dataset in dict.fromkeys(args.datasets):
            for seed in dict.fromkeys(args.seeds):
                job = tta.inspect_run(dataset, seed, model)
                if job is None:
                    raise ValueError(f'Missing selected checkpoint: {model}/{dataset}/{seed}')
                gains = tta.gains_for(policy, dataset)
                evidence = evidence_for(job, gains)
                directory = local_path(job['checkpoint']).parent.parent
                print(f'Ready: {model} {dataset} seed {seed}; {portable_path(directory / (args.record_name+".json"))}', flush=True)
                jobs.append((job, gains, evidence, directory))
    if args.dry_run:
        print('Preflight complete; no inference or writes.')
        return
    sources = {portable_path(p): shared.sha256(p) for p in
               [Path(__file__), Path(tta.__file__), Path(reference.__file__), Path(shared.__file__),
                Path(ordinary.__file__), ROOT/'project_paths.py', ROOT/'config/tta.yaml',
                *sorted((ROOT/'models').rglob('*.py')), *sorted((ROOT/'datasets').rglob('*.py'))]}
    groups = {}
    for job, gains, evidence, directory in jobs:
        cfg = OmegaConf.create(job['config'])
        device = torch.device(str(cfg.device))
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Checkpoint requests CUDA but CUDA is unavailable')
        torch.set_float32_matmul_precision('high')
        identity = dict(version='tta_costs_v1', job=job, gains=gains, evidence=evidence,
                        repeats=args.repeats, warmup=args.warmup, sources=sources,
                        environment=dict(python=platform.python_version(), torch=torch.__version__,
                            packages={name: importlib.metadata.version(name) for name in
                                      ('numpy', 'pytorch-lightning', 'tonic', 'hydra-core')},
                            cuda=torch.version.cuda, device=str(device), precision=torch.get_float32_matmul_precision(),
                            gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor()))
        with ordinary.measurement_lock(directory):
            path = directory / f'{args.record_name}.json'
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                if record.get('status') != 'complete' or record.get('fingerprint') != fingerprint(identity):
                    raise ValueError('TTA cost identity changed; preserve existing record and use a new --record-name')
                print(f'Reusing {portable_path(path)}', flush=True)
            else:
                torch.manual_seed(job['seed'])
                state = torch.load(local_path(job['checkpoint']), map_location='cpu', weights_only=False)
                with redirect_stdout(io.StringIO()):
                    model = tta.MLPSNN(cfg)
                model.load_state_dict(state['state_dict'], strict=True)
                model.to(device).eval()
                batches, _ = ordinary.prepare_batches(job['dataset'], cfg)
                if test_fingerprint(batches) != evidence['metrics']['data_sha256']:
                    raise ValueError('Test batches differ from saved TTA evidence')
                metrics = measure(model, batches, device, job['dataset'] != 'ECG', gains,
                                  args.repeats, args.warmup)
                verify_result(metrics, evidence)
                record = save_record(directory, args.record_name, identity, metrics)
                del model, batches, state
            groups.setdefault((job['dataset'], job['config']['l1']['cell']), []).append(record['measurements'])
    print('\nDataset / model | n | forward ms | pipeline ms | dense GFLOPs | million SynOps')
    for key, values in groups.items():
        def fmt(items):
            return f'{statistics.mean(items):.6f} +/- {statistics.stdev(items):.6f}' if len(items)>1 else f'{items[0]:.6f} (n=1)'
        print(f'{key} | {len(values)} | '+ ' | '.join([
            fmt([v['timing_ms_per_sample']['tta_forwards'] for v in values]),
            fmt([v['timing_ms_per_sample']['tta_pipeline'] for v in values]),
            fmt([v['tta_dense_gflops'] for v in values]),
            fmt([v['tta_synops_per_sample']/1e6 for v in values])]))


if __name__ == '__main__':
    main()

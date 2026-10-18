"""Evaluate SSC/ECG reference checkpoint activity and test-set forward costs.

Preserves the SHD evaluator and all training artifacts. ECG has analog inputs;
its first projection is dense, not a spike-derived synaptic-operation estimate.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
import re
import statistics
import time

import hydra
from omegaconf import OmegaConf
import torch

import measure_shd_costs as shared
from project_paths import local_path, portable_config, portable_path

ROOT = shared.ROOT
VERSION = 'reference_test_costs_v1'
ARITHMETIC = {'add', 'sub', 'rsub', 'mul', 'div', 'neg', 'reciprocal',
              'sum', 'mean', 'bias_add', 'square'}


class ReferenceCounter:
    def __init__(self, model, lengths, event_input):
        self.values = {}
        self.handles = []
        for name in ['l1', 'l2', 'out_layer']:
            layer = getattr(model, name)
            slots = torch.zeros(6, device=layer.weight.device, dtype=torch.float64)
            self.values[name] = slots
            step = [0]

            def hook(cell, args, output, slots=slots, step=step, name=name):
                x, state = args
                valid = (lengths > step[0]).unsqueeze(1)
                ff = (x.sum(1) * cell.out_features if event_input or name != 'l1'
                      else torch.zeros(x.shape[0], device=x.device))
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

    def close(self):
        for handle in self.handles:
            handle.remove()
        result = {k: v.cpu().tolist() for k, v in self.values.items()}
        if any(v[4] or v[5] for v in result.values()):
            raise ValueError('Expected finite binary hidden spikes')
        return result


def load_run(dataset, seed):
    directory = ROOT / f'results/valid/reference/{dataset}/seed_{seed}'
    path = shared.selected_checkpoint(directory)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(portable_config(OmegaConf.to_container(
        OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True)))
    expected = {'SSC': 'datasets.ssc.SSCLDM', 'ECG': 'datasets.ecg.ECGLDM'}[dataset]
    if (cfg.dataset._target_ != expected or cfg.random_seed != seed or not cfg.two_layers
            or cfg.l1.cell != 'se_adlif' or cfg.l2.cell != 'se_adlif'
            or cfg.get('auto_regression', False)):
        raise ValueError(f'Unexpected reference configuration: {path}')
    return directory, path, checkpoint, cfg


def data_signature(cfg):
    values = OmegaConf.to_container(cfg.dataset, resolve=True)
    for key in ['data_path', 'cache_root', 'num_workers']:
        values.pop(key, None)
    values.setdefault('burn_in_time', 0) if 'ECG' in values['_target_'] else None
    return values


def prepare_batches(dataset, cfg):
    data_path = local_path(cfg.dataset.data_path, must_exist=True)
    needed = (['SSCWrapper/ssc_train.h5', 'SSCWrapper/ssc_valid.h5', 'SSCWrapper/ssc_test.h5']
              if dataset == 'SSC' else ['QTDB/QTDB_train.mat', 'QTDB/QTDB_test.mat'])
    if any(not (data_path / path).is_file() for path in needed):
        raise FileNotFoundError(f'Copy existing local {dataset} data before evaluation; no download permitted')
    values = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    values.data_path = str(data_path)
    values.num_workers = 0
    if values.get('cache_root'):
        values.cache_root = str(local_path(values.cache_root))
    dm = hydra.utils.instantiate(values)
    dm.setup('test')
    loader = dm.test_dataloader()
    loader.collate_fn = shared.collate_lengths
    batches = []
    digest = hashlib.sha256()
    for i, batch in enumerate(loader):
        for tensor in batch:
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        batches.append(batch)
        if (i + 1) % 10 == 0:
            print(f'{dataset}: prepared {i + 1}/{len(loader)} test batches', flush=True)
    return batches, digest.hexdigest()


def measure(model, batches, device, event_input, repeats, warmup):
    profiles = shared.operation_profiles(model)
    counts, operations = Counter(), Counter()
    for cpu_batch in batches:
        inputs, targets, block_idx, lengths = [x.to(device) for x in cpu_batch]
        b, t = inputs.shape[:2]
        counter = ReferenceCounter(model, lengths, event_input)
        try:
            outputs = model(inputs)
        finally:
            activity = counter.close()
        reduced, _, _ = model.process_predictions_and_compute_losses(outputs, targets, block_idx)
        target = targets.flatten()
        mask = target != -1
        counts['correct'] += int((reduced.argmax(-1)[mask] == target[mask]).sum())
        counts['targets'] += int(mask.sum())
        counts['samples'] += b
        counts['padded_sample_timesteps'] += b * t
        counts['sequence_sample_timesteps'] += int(lengths.sum())
        operations.update(shared.batch_scalar_operations(profiles, b, t))
        for name, values in activity.items():
            counts['synops_padded'] += values[0]
            counts['synops_sequence'] += values[1]
            if name != 'out_layer':
                counts[f'{name}_spikes_padded'] += values[2]
                counts[f'{name}_spikes_sequence'] += values[3]
    x = batches[0][0].to(device)
    for _ in range(warmup):
        model(x)
    shared.synchronize(device)
    times = []
    for _ in range(repeats):
        elapsed = 0
        for cpu_batch in batches:
            x = cpu_batch[0].to(device)
            shared.synchronize(device)
            start = time.perf_counter()
            model(x)
            shared.synchronize(device)
            elapsed += time.perf_counter() - start
        times.append(1000 * elapsed / counts['samples'])
    n = counts['samples']
    macs = sum(cell.weight.numel() + (cell.recurrent.numel() if getattr(cell, 'use_recurrent', False) else 0)
               for cell in [model.l1, model.l2, model.out_layer])
    dense = 2 * macs * counts['padded_sample_timesteps'] / n / 1e9
    arith = sum(v for k, v in operations.items() if k in ARITHMETIC) / n / 1e9
    return dict(totals=dict(counts), operation_profiles=profiles,
        nonmatrix_scalar_operations_by_kind=dict(operations),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        spikes_total_per_sample=(counts['l1_spikes_sequence'] + counts['l2_spikes_sequence']) / n,
        spikes_l1_per_sample=counts['l1_spikes_sequence'] / n,
        spikes_l2_per_sample=counts['l2_spikes_sequence'] / n,
        firing_l1_percent=100 * counts['l1_spikes_sequence'] / (counts['sequence_sample_timesteps'] * model.l1.out_features),
        firing_l2_percent=100 * counts['l2_spikes_sequence'] / (counts['sequence_sample_timesteps'] * model.l2.out_features),
        synops_per_sample=counts['synops_sequence'] / n,
        nonmatrix_ops_per_sample=sum(operations.values()) / n,
        dense_gflops_test_padded=dense, arithmetic_gflops_test_padded=dense + arith,
        logical_gops_test_padded=dense + sum(operations.values()) / n / 1e9,
        analog_input_dense_macs_per_sample=(0 if event_input else
            model.l1.weight.numel() * counts['padded_sample_timesteps'] / n),
        forward_ms_per_sample=statistics.mean(times), timing_repeat_ms_per_sample=times,
        reproduced_test_acc=counts['correct'] / counts['targets'])


def update_report(output):
    records = {ds: [json.loads((output / ds / f'seed_{s}.json').read_text(encoding='utf-8'))
                    for s in [42, 123, 456]] for ds in ['SSC', 'ECG']}
    shd = [json.loads((ROOT / f'results/measurements/shd_costs_v1/SE_adLIF/seed_{s}.json').read_text(encoding='utf-8'))
           for s in [42, 123, 456]]
    records['SHD*'] = shd
    p = ROOT / 'reports/paper-vs-reference.md'
    text = p.read_text(encoding='utf-8')
    for ds in ['SSC', 'ECG']:
        group = [r['measurements'] for r in records[ds]]
        def metric(key, **kwargs):
            return shared.mean_sd([r[key] for r in group], **kwargs)
        row = '| ' + ds + ' | ' + ' | '.join([str(group[0]['total_parameters']), str(group[0]['trainable_parameters']),
            metric('spikes_total_per_sample', digits=0),
            metric('spikes_l1_per_sample', digits=0) + ' / ' + metric('spikes_l2_per_sample', digits=0),
            metric('firing_l1_percent') + ' / ' + metric('firing_l2_percent'),
            metric('synops_per_sample', scale=1e6), metric('nonmatrix_ops_per_sample', scale=1e6),
            metric('forward_ms_per_sample', digits=3)]) + ' |'
        # Restrict replacement to the activity table, not published accuracies.
        text = re.sub(r'^\| ' + ds + r' \| (?:Pending|[\d,]+) \| (?:Pending|[\d,]+) \|[^\n]+$',
                      lambda _: row, text, flags=re.M)
    marker = '## Test-set forward computation'
    text = text.split(marker)[0].rstrip() if marker in text else text.rstrip()
    rows = []
    for ds in ['SHD*', 'SSC', 'ECG']:
        group = [r['measurements'] for r in records[ds]]
        n = group[0]['totals']['samples']
        dense = [m['dense_gflops_test_padded'] for m in group]
        arithmetic = [m['dense_gflops_test_padded'] + sum(v for k, v in m['nonmatrix_scalar_operations_by_kind'].items()
                      if k in ARITHMETIC) / m['totals']['samples'] / 1e9 for m in group]
        logical = [m['dense_gflops_test_padded'] + m['nonmatrix_ops_per_sample'] / 1e9 for m in group]
        rows.append(f'| {ds} | {n:,} | {shared.mean_sd(dense, digits=6)} | {shared.mean_sd(arithmetic, digits=6)} | {shared.mean_sd(logical, digits=6)} |')
    text = text.replace('SHD* activity and timing are measured from all three saved reference checkpoints; SSC/ECG measurements remain pending.',
                        'Activity and timing are measured on the official test sets from all three saved reference checkpoints per dataset.')
    text = text.replace('Activity, operation and timing values will be reported', 'Activity, operation and timing values are reported')
    text += '\n\n' + marker + '\n\n' + (
        '| Dataset | Test sequences | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Matrix + all logical scalar GOp/sample |\n'
        '|---|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n'
        'Entries are mean ± sample SD across three seeds, using actual batch-padded test lengths. The arithmetic estimate adds counted add/subtract/multiply/divide, negation, reciprocal, sum/mean reduction and bias operations in the cell forward code to the matrix estimate. Nonlinear functions (e.g. exp/sigmoid) and comparisons are excluded from that arithmetic GFLOP figure; they are included as one logical operation each in the last column. Memory traffic is excluded from all columns. That logical total is a counting convention, not hardware FLOPs or energy. Loss and temporal prediction aggregation are excluded from both estimates. Thus neither column should be called an exact full end-to-end inference FLOP count.\n\n'
        'SSC inputs are framed event counts. ECG inputs are analog: its first projection is counted as dense MACs, not spike events. The ECG SynOps activity-table cell therefore covers hidden-driven connections only; its analog first-projection MACs are stored separately. All hidden spikes count stored pre-batch sequence lengths (including configured minimum padding and burn-in); padded totals are retained separately. ECG accuracy is per labeled timestep, while the compute denominator is one input sequence. Dataset-specific batch sizes and lengths mean cross-dataset runtimes are not matched-workload comparisons.\n\n'
        f'SSC/ECG checkpoint hashes, resolved configs, per-seed counts, timing repeats and reproducible command: [measurement records](../{portable_path(output)}/README.md). Existing validation estimates remain separately labeled above. The ECG training-budget/stopping mismatch is unchanged. No checkpoint was retrained or selected using these test results.\n')
    p.write_text(text, encoding='utf-8', newline='\n')
    shared.save_json(output / 'AGGREGATE.json', dict(version=VERSION, datasets=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['SSC', 'ECG'], default=['SSC', 'ECG'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='results/measurements/reference_test_costs_v1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--update-reports', action='store_true')
    parser.add_argument('--reports-only', action='store_true', help='Refresh reports from existing records, without inference')
    args = parser.parse_args()
    output = local_path(args.output)
    if args.reports_only:
        for ds in ['SSC', 'ECG']:
            records = [json.loads((output / ds / f'seed_{s}.json').read_text(encoding='utf-8'))
                       for s in [42, 123, 456]]
            if len({r['identity'] for r in records}) != 1 or any(
                    r['version'] != VERSION or r['dataset'] != ds or r['seed'] != s
                    for r, s in zip(records, [42, 123, 456])):
                raise ValueError(f'Inconsistent saved measurement records: {ds}')
        update_report(output)
        return
    if args.dry_run:
        for ds in args.datasets:
            for seed in args.seeds:
                _, path, _, _ = load_run(ds, seed)
                print(ds, seed, portable_path(path), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.evaluation.lock').open('a+b') as lock:
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
            evaluator_sha256=shared.sha256(__file__), shared_evaluator_sha256=shared.sha256(shared.__file__),
            model_source_sha256={portable_path(p): shared.sha256(p) for p in sorted((ROOT / 'models').rglob('*.py'))},
            matmul_precision=torch.get_float32_matmul_precision(), timing_repeats=3, warmup=2)
        for ds in args.datasets:
            batches = None
            for seed in args.seeds:
                directory, path, checkpoint, cfg = load_run(ds, seed)
                if batches is None:
                    print(f'{ds}: preparing official test set...', flush=True)
                    batches, data_hash = prepare_batches(ds, cfg)
                    signature = data_signature(cfg)
                if data_signature(cfg) != signature:
                    raise ValueError(f'{ds}: differing preprocessing across seeds')
                identity = hashlib.sha256(json.dumps(dict(version=VERSION, environment=environment,
                    data_hash=data_hash, signature=signature), sort_keys=True).encode()).hexdigest()
                destination = output / ds / f'seed_{seed}.json'
                ckpt_hash = shared.sha256(path)
                if destination.exists():
                    old = json.loads(destination.read_text(encoding='utf-8'))
                    if old['identity'] != identity or old['checkpoint_sha256'] != ckpt_hash:
                        raise ValueError('Changed measurement inputs: choose a new --output')
                    print(f'{ds} seed {seed}: verified existing measurement', flush=True)
                    continue
                torch.manual_seed(seed)
                with redirect_stdout(io.StringIO()):
                    model = shared.MLPSNN(cfg)
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                model.to(device).eval()
                print(f'{ds} seed {seed}: measuring...', flush=True)
                with torch.inference_mode():
                    metrics = measure(model, batches, device, ds == 'SSC', 3, 2)
                expected = shared.saved_test_accuracy(directory)
                if abs(metrics['reproduced_test_acc'] - expected) > 1e-6:
                    raise ValueError(f'{ds} seed {seed}: accuracy differs: {metrics["reproduced_test_acc"]} vs {expected}')
                shared.save_json(destination, dict(version=VERSION, identity=identity, dataset=ds, seed=seed,
                    created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=portable_path(path),
                    checkpoint_sha256=ckpt_hash, best_epoch=checkpoint['epoch'],
                    saved_config=portable_config(OmegaConf.to_container(cfg, resolve=True)),
                    baseline_matched=ds != 'ECG', environment=environment, data_sha256=data_hash,
                    measurements=metrics))
                print(f'{ds} seed {seed}: dense GFLOPs/sample={metrics["dense_gflops_test_padded"]:.6f}; '
                      f'test accuracy={metrics["reproduced_test_acc"]:.6f}', flush=True)
                del model, checkpoint
            del batches
        if args.update_reports:
            update_report(output)
            print('Updated paper-vs-reference test-set activity and forward-cost tables.', flush=True)


if __name__ == '__main__':
    main()

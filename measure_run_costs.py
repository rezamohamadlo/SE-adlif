"""Save test-set activity and compute measurements for one completed runner job."""
import argparse
from contextlib import redirect_stdout, contextmanager
from datetime import datetime, timezone
import io
import json
import platform
from pathlib import Path

from omegaconf import OmegaConf
import torch

import measure_reference_costs as reference
import measure_shd_costs as shared
from models.pl_module import layer_map
from project_paths import local_path, portable_config, portable_path
from cost_records import normalized_record

VERSION = 'runner_test_costs_v1'
TARGETS = {'SHD': 'datasets.shd.SHDLDM', 'SSC': 'datasets.ssc.SSCLDM',
           'ECG': 'datasets.ecg.ECGLDM'}


def update_training_summary(directory, record):
    """Mirror completed test measurements without changing training history."""
    path = directory / 'TRAINING_SUMMARY.md'
    if not path.exists():
        return
    start, end = '<!-- test-costs:start -->', '<!-- test-costs:end -->'
    text = path.read_text(encoding='utf-8')
    if start in text:
        before, rest = text.split(start, 1)
        if end not in rest:
            raise ValueError('Incomplete test-cost section in training summary')
        _, after = rest.split(end, 1)
    else:
        before, after = text, ''
    metrics = record['measurements']
    labels = {
        'total_parameters': 'Total parameters',
        'trainable_parameters': 'Trainable parameters',
        'spikes_total_per_sample': 'Hidden spikes/sample',
        'spikes_l1_per_sample': 'L1 spikes/sample',
        'spikes_l2_per_sample': 'L2 spikes/sample',
        'firing_l1_percent': 'L1 firing fraction (%)',
        'firing_l2_percent': 'L2 firing fraction (%)',
        'synops_per_sample': 'Synaptic operations/sample',
        'analog_input_dense_macs_per_sample': 'Analog input dense MACs/sample',
        'dense_gflops_test_padded': 'Dense matrix GFLOPs/sample',
        'arithmetic_gflops_test_padded': 'Matrix + scalar arithmetic GFLOPs/sample',
        'logical_gops_test_padded': 'Logical GOp/sample',
        'forward_ms_per_sample': 'Forward runtime (ms/sample)',
    }
    lines = [start, '## Test-set spike activity and inference cost', '',
             f"Validation-selected checkpoint, epoch {record['best_epoch']} (zero-based). No training performed.", '',
             '| Measurement | Value |', '|---|---:|']
    lines += [f'| {label} | {metrics[key]:,.6f} |' for key, label in labels.items()]
    lines += ['', 'Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.',
              'ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.',
              'Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.',
              'These test measurements are separate from the validation GFLOPs estimate above.',
              'Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).', end]
    temporary = path.with_suffix('.md.costs.tmp')
    temporary.write_text(before.rstrip() + '\n\n' + '\n'.join(lines) + after.rstrip() + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare_batches(dataset, cfg):
    return shared.prepare_batches(cfg) if dataset == 'SHD' else reference.prepare_batches(dataset, cfg)


@contextmanager
def measurement_lock(directory):
    with (directory / '.evaluation.lock').open('a+b') as lock:
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
        yield


def collect(directory, dataset, dry_run=False):
    directory = local_path(directory, must_exist=True)
    if dry_run:
        return _collect(directory, dataset, True)
    with measurement_lock(directory):
        return _collect(directory, dataset, False)


def _collect(directory, dataset, dry_run=False):
    directory = local_path(directory, must_exist=True)
    if not (directory / 'completed.txt').is_file():
        raise ValueError('Only completed training/test runs may be measured')
    path = shared.selected_checkpoint(directory)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(portable_config(OmegaConf.to_container(
        OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True)))
    if cfg.dataset._target_ != TARGETS[dataset] or not cfg.two_layers or cfg.get('auto_regression', False):
        raise ValueError('Expected a supported two-hidden-layer classification checkpoint')
    if cfg.l1.cell not in layer_map or cfg.l2.cell not in layer_map:
        raise ValueError('Unregistered hidden cell')
    if dry_run:
        print(f'Would measure {dataset}: {portable_path(path)}; no inference or writes.')
        return None
    digest = shared.sha256(path)
    result_path = directory / 'COST_RESULTS.json'
    if result_path.exists():
        old = normalized_record(json.loads(result_path.read_text(encoding='utf-8')))
        if (old.get('status') != 'complete' or old.get('checkpoint_sha256') != digest
                or old.get('dataset') != dataset or old.get('seed') != cfg.random_seed):
            raise ValueError('Existing cost record has a different checkpoint/protocol; preserve it before remeasurement')
        print(f'Reusing verified cost record: {portable_path(result_path)}')
        update_training_summary(directory, old)
        return old
    torch.manual_seed(cfg.random_seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device(str(cfg.device))
    with redirect_stdout(io.StringIO()):
        model = shared.MLPSNN(cfg)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    print(f'Measuring {dataset} test activity/cost: {portable_path(path)}', flush=True)
    batches, data_hash = prepare_batches(dataset, cfg)
    with torch.inference_mode():
        metrics = reference.measure(model, batches, device, dataset != 'ECG', 3, 2)
    expected = shared.saved_test_accuracy(directory)
    if abs(metrics['reproduced_test_acc'] - expected) > 1e-6:
        raise ValueError(f'Reproduced test accuracy differs: {metrics["reproduced_test_acc"]} vs {expected}')
    record = dict(version=VERSION, status='complete', dataset=dataset, seed=int(cfg.random_seed),
        created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=portable_path(path),
        checkpoint_sha256=digest, best_epoch=int(checkpoint['epoch']),
        saved_config=portable_config(OmegaConf.to_container(cfg, resolve=True)), data_sha256=data_hash,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            matmul_precision=torch.get_float32_matmul_precision(), timing_repeats=3, warmup=2,
            source_sha256={portable_path(p): shared.sha256(p) for p in
                [Path(__file__), Path(__file__).with_name('cost_records.py'), Path(shared.__file__), Path(reference.__file__),
                 *sorted((shared.ROOT / 'models').rglob('*.py'))]}),
        scopes=dict(activity='Stored sequence extent before batch padding, including configured minimum padding and burn-in',
            compute='Executed batch-padded timesteps; MLPSNN.forward only',
            timing='Synchronized forward wall time; excludes transfer/loading/loss/temporal aggregation',
            synops='Dense fan-out event estimate; ECG analog input projection excluded and stored as dense MACs',
            arithmetic='Matrix MAC=2 FLOPs; selected scalar arithmetic; nonlinearities/comparisons excluded',
            logical='Matrix operations plus scalar operations, nonlinearities/comparisons each counted as one; not hardware FLOPs'),
        measurements=metrics)
    lines = ['# Test-set activity and forward costs', '',
             'Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.', '',
             '| Measurement | Value |', '|---|---:|']
    for key in ['total_parameters', 'trainable_parameters', 'spikes_total_per_sample',
                'spikes_l1_per_sample', 'spikes_l2_per_sample', 'firing_l1_percent', 'firing_l2_percent',
                'synops_per_sample', 'analog_input_dense_macs_per_sample', 'dense_gflops_test_padded',
                'arithmetic_gflops_test_padded', 'logical_gops_test_padded', 'forward_ms_per_sample']:
        lines.append(f'| {key} | {metrics[key]:.6f} |')
    lines += ['', 'Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.',
              'ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.']
    temporary = directory / 'COST_RESULTS.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(directory / 'COST_RESULTS.md')
    # JSON is the completion record; publish it after the human-readable summary.
    shared.save_json(result_path, record)
    update_training_summary(directory, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dataset', required=True, choices=TARGETS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    collect(args.run_dir, args.dataset, args.dry_run)


if __name__ == '__main__':
    main()

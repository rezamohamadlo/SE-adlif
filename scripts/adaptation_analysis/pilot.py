"""Bounded, frozen-checkpoint SHD* validation analysis. Never trains or opens test data."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from project_paths import local_path, portable_path, portable_config
from scripts.adaptation_analysis.features import measure_features
from scripts.adaptation_analysis.perturbations import thin_counts, warp_counts

STUDY_FILE = ROOT / 'config/studies/adaptation_mechanism_pilot_v1.yaml'
PCS = ('pc1', 'pc2', 'pc3')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pc', choices=PCS, required=True)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', default='results/experiments/adaptation_mechanism_analysis_v1')
    parser.add_argument('--check-only', action='store_true', help='Verify actual prerequisites; no inference or output writes')
    parser.add_argument('--self-check', action='store_true', help='Run pure synthetic helper checks and exit; combine with --check-only for preflight')
    return parser


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def output_directory(output_root, pc):
    if pc not in PCS:
        raise ValueError('Unknown PC assignment')
    if '..' in str(output_root).replace('\\', '/').split('/'):
        raise ValueError('Output traversal is not permitted')
    base = local_path(output_root)
    allowed = (ROOT / 'results/experiments').resolve()
    if base == allowed or not base.is_relative_to(allowed):
        raise ValueError('Analysis output root must be a study directory inside results/experiments')
    # Existing training records must never become analysis destinations.
    if any(base.rglob('*.ckpt')) or any(base.rglob('metrics.csv')):
        raise ValueError('Analysis output root contains training evidence; choose a new study directory')
    return base / pc


def validate_manifest(existing, requested):
    if existing != requested:
        raise ValueError('Analysis manifest differs. Preserve existing evidence and choose a new --output-root.')


def paired_rows(parent, variant):
    """Strict one-to-one pairing, never an inner join that hides missing cases."""
    def index(rows):
        result = {}
        for row in rows:
            key = row['sample_id']
            if key in result:
                raise ValueError(f'Duplicate sample ID {key}')
            result[key] = row
        return result
    left, right = index(parent), index(variant)
    if not left or set(left) != set(right):
        raise ValueError('Paired sample IDs differ or are empty')
    result = []
    for sample_id, a in left.items():
        b = right[sample_id]
        if ('seed' in a or 'seed' in b) and a.get('seed') != b.get('seed'):
            raise ValueError(f'Paired training seed differs for sample {sample_id}')
        for key in ('target', 'input_sha256', 'condition', 'replicate'):
            if a[key] != b[key]:
                raise ValueError(f'Paired {key} differs for sample {sample_id}')
        for row in (a, b):
            if not np.isfinite(row['loss']) or row['correct'] not in (0, 1):
                raise ValueError('Invalid per-sample evaluation result')
        row = {k: v for k, v in a.items() if k not in ('model', 'loss', 'correct', 'prediction', 'true_probability')}
        row.update(parent_loss=a['loss'], variant_loss=b['loss'], benefit=a['loss']-b['loss'],
                   parent_correct=a['correct'], variant_correct=b['correct'],
                   correctness_difference=b['correct']-a['correct'])
        result.append(row)
    return result


@contextlib.contextmanager
def study_lock(path):
    """OS-owned lock; stale lock-file existence does not imply a held lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another analysis owns {portable_path(path)}') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def self_check():
    x = np.zeros((100, 3), dtype=np.int64)
    x[::4, 0], x[::7, 1] = 2, 3
    assert np.array_equal(thin_counts(x, 0, 1), x)
    assert np.array_equal(warp_counts(x, 1), x)
    assert np.array_equal(warp_counts(x, .9).sum(0), x.sum(0))
    assert np.array_equal(thin_counts(x, .1, 42), thin_counts(x, .1, 42))
    assert measure_features(x)['total_spikes'] == int(x.sum())
    a = dict(sample_id=1, target=0, input_sha256='same', condition='clean', replicate=0, loss=2., correct=0)
    b = dict(a, loss=1., correct=1)
    assert paired_rows([a], [b])[0]['benefit'] == 1
    validate_manifest({'a': 1}, {'a': 1})
    print('Self-check passed (synthetic arrays only; no data/model loading or output writes).', flush=True)


def selected_metrics(run):
    csv_paths = list(local_path(run['cost_record']).parent.glob('logs/**/metrics.csv'))
    if len(csv_paths) != 1:
        raise ValueError(f'Expected one unmixed metrics history for {run["model"]} seed {run["seed"]}')
    with csv_paths[0].open(newline='', encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream) if r.get('val_acc')]
    epochs = [int(float(row['epoch'])) for row in rows]
    if sorted(epochs) != list(range(300)):
        raise ValueError(f'Incomplete or overlapping validation history: {portable_path(csv_paths[0])}')
    selected = next(row for row in rows if int(float(row['epoch'])) == run['selected_epoch'])
    acc = float(selected['val_acc'])
    if acc < max(float(row['val_acc']) for row in rows)-1e-7:
        raise ValueError('Recorded checkpoint is not validation-selected')
    return dict(path=portable_path(csv_paths[0]), sha256=sha256(csv_paths[0]),
                epoch=run['selected_epoch'], accuracy=acc, loss=float(selected['val_loss']))


def effective_config(cfg):
    keys = ('random_seed', 'dataset', 'two_layers', 'auto_regression', 'dropout', 'l1', 'l2', 'l_out',
            'n_epochs', 'batch_size', 'loss_agg', 'lr', 'scheduler', 'early_stopping', 'training_objective',
            'paper_baseline_version', 'paper_baseline_dataset', 'evaluation_protocol', 'model_variant')
    return {key: cfg.get(key) for key in keys}


def cost_model_identity(cost):
    # Historical AMT evaluator records put the model identity in saved_config.
    # Validate the actual recorded identity instead of inventing a default.
    model = cost.get('model', cost.get('saved_config', {}).get('model_variant'))
    if not model:
        raise ValueError('Measurement record has no saved model identity')
    return model


def check_saved_config(cfg, run, study):
    from scripts.adaptation_analysis.shd_data import validate_shd_config
    validate_shd_config(cfg)
    required = dict(random_seed=run['seed'], two_layers=True, dropout=.15, n_epochs=300,
                    batch_size=256, lr=.01, early_stopping=False,
                    loss_agg='sum_softmax_over_time', paper_baseline_version=study['baseline_version'],
                    paper_baseline_dataset='SHD', evaluation_protocol='paper_classification_v1',
                    model_variant=run['model'])
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f'{run["model"]} seed {run["seed"]}: saved {key} differs from frozen study')
    cell = {'MT_SE_adLIF': 'mt_se_adlif', 'AMT_SE_adLIF': 'amt_se_adlif'}[run['model']]
    if any(cfg[layer]['cell'] != cell for layer in ('l1', 'l2')):
        raise ValueError('Checkpoint neuron identity differs')
    if cfg.get('auto_regression', False) or cfg.get('training_objective', 'classification') != 'classification':
        raise ValueError('Unexpected task or training objective')
    if cfg['dataset']['batch_size'] != 256:
        raise ValueError('Saved validation batch size differs')


def preflight(args, study):
    import torch
    from omegaconf import OmegaConf
    from scripts.adaptation_analysis.shd_data import preflight_shd, select_validation_ids
    if not 1 <= args.samples <= study['max_samples']:
        raise ValueError(f'--samples must be between 1 and {study["max_samples"]}')
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(study['seeds']):
        raise ValueError('--seeds must be unique members of 42, 123, 456')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f'Requested CUDA device is unavailable: {args.device}')
    elif device.type != 'cpu':
        raise ValueError('Use cpu or a CUDA device')
    runs = []
    for frozen in study['runs']:
        if frozen['seed'] not in args.seeds:
            continue
        run = dict(frozen)
        path = local_path(run['checkpoint'], must_exist=True)
        cost = read_json(local_path(run['cost_record'], must_exist=True))
        if (cost_model_identity(cost) != run['model'] or cost['seed'] != run['seed']
                or portable_path(cost['checkpoint']) != run['checkpoint']
                or cost['checkpoint_sha256'] != run['checkpoint_sha256']
                or cost['best_epoch'] != run['selected_epoch']):
            raise ValueError('Frozen checkpoint and measurement identity disagree')
        if sha256(path) != run['checkpoint_sha256']:
            raise ValueError(f'Checkpoint SHA-256 mismatch: {run["checkpoint"]}')
        # Only authenticated, trusted local project checkpoints are deserialized.
        checkpoint = torch.load(path, weights_only=False, map_location='cpu')
        cfg = portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True))
        saved = portable_config(cost['saved_config'])
        if effective_config(cfg) != effective_config(saved):
            raise ValueError('Checkpoint config and saved measurement config disagree')
        check_saved_config(cfg, run, study)
        if checkpoint['epoch'] != run['selected_epoch']:
            raise ValueError('Checkpoint epoch differs')
        run.update(config=cfg, validation=selected_metrics(run))
        runs.append(run)
        del checkpoint
        print(f'Prerequisite OK: {run["model"]} seed {run["seed"]}, validation epoch {run["selected_epoch"]}', flush=True)
    # Pair invariants: only the explicit AMT neuron/mixing additions may differ.
    for seed in args.seeds:
        configs = [dict(next(r['config'] for r in runs if r['seed'] == seed and r['model'] == model))
                   for model in (study['parent'], study['variant'])]
        for cfg in configs:
            for layer in ('l1', 'l2'):
                cfg[layer] = {k: v for k, v in cfg[layer].items() if k not in ('cell', 'mix_rho_fast', 'mix_rho_slow')}
        keys = ('dataset', 'two_layers', 'dropout', 'n_epochs', 'batch_size', 'loss_agg', 'lr', 'scheduler',
                'early_stopping', 'l1', 'l2', 'l_out', 'factor', 'patience', 'gradient_clip_val')
        if {k: configs[0].get(k) for k in keys} != {k: configs[1].get(k) for k in keys}:
            raise ValueError('Parent and variant do not share matched settings')
    data = preflight_shd(runs[0]['config'], study['dataset_sha256'])
    if data['path'] != study['dataset_file']:
        raise ValueError('Dataset path differs from frozen study')
    sample_ids = select_validation_ids(local_path(data['path']), args.samples, study['selection_seed'])
    print(f'SHD*: {data["validation_samples"]} clean validation anchors; {len(sample_ids)} pilot recordings; no official test access.', flush=True)
    return runs, data, sample_ids


def source_identity():
    paths = [STUDY_FILE, ROOT / 'project_paths.py', ROOT / 'config/paper_baseline.yaml',
             ROOT / 'datasets/utils/pad_tensors.py']
    paths += sorted((ROOT / 'scripts/adaptation_analysis').glob('*.py'))
    # Include registered forward implementations and their local utilities.
    paths += sorted((ROOT / 'models').rglob('*.py'))
    paths += sorted((ROOT / 'module').rglob('*.py'))
    paths += sorted((ROOT / 'functional').rglob('*.py'))
    return {portable_path(path): sha256(path) for path in paths}


def environment(device):
    import importlib.metadata
    import torch
    packages = {p: importlib.metadata.version(p) for p in
                ('torch', 'numpy', 'tonic', 'h5py', 'omegaconf', 'pytorch-lightning', 'torchmetrics', 'matplotlib')}
    return dict(python=platform.python_version(), packages=packages, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                cuda=torch.version.cuda, platform=platform.system())


def conditions(pc, study):
    values = [('clean', 0, 'identity', 1.)]
    if pc == 'pc2':
        values += [(f'time_scale_{scale:g}', 0, 'warp', scale) for scale in study['arms'][pc]['scales'] if scale != 1]
    if pc == 'pc3':
        values += [(f'delete_{p:g}', rep, 'thin', p) for p in study['arms'][pc]['probabilities'] if p != 0
                   for rep in range(study['arms'][pc]['replicates'])]
    return values


def input_rows(data, ids, condition, study, features=True):
    name, replicate, transform, value = condition
    result = []
    for sample_id in ids:
        source = data.load(sample_id)
        x = source.pop('counts')
        perturb_seed = int(digest([study['selection_seed'], sample_id, name, replicate])[:16], 16)
        if transform == 'warp':
            x = warp_counts(x, value)
        elif transform == 'thin':
            x = thin_counts(x, value, perturb_seed)
        row = {k: v for k, v in source.items() if k not in ('framing', 'duration_ms', 'support_frames', 'stored_frames', 'frame_support_ms')}
        row.update(condition=name, replicate=replicate, perturbation_seed=perturb_seed,
                   input_sha256=hashlib.sha256(str(x.shape).encode() + x.astype('<i8', copy=False).tobytes()).hexdigest(),
                   original_duration_ms=source['duration_ms'], original_support_frames=source['support_frames'],
                   transformed_support_frames=len(x), stored_frames=max(len(x), 250))
        if features:
            row.update(measure_features(x, **study['features']))
        result.append((sample_id, x, row))
    return result


def infer(model, cfg, data, inputs, device, label):
    import torch
    import torch.nn.functional as F
    from datasets.utils.pad_tensors import PadTensors
    from models.pl_module import aggregate_temporal_outputs
    from tqdm import tqdm
    rows = []
    batch_size = cfg['dataset']['batch_size']
    with torch.inference_mode():
        for start in tqdm(range(0, len(inputs), batch_size), desc=label, unit='batch', dynamic_ncols=True):
            batch = inputs[start:start+batch_size]
            x, targets, blocks = PadTensors()([data.model_item(sample_id, values) for sample_id, values, _ in batch])
            x, targets, blocks = x.to(device), targets.to(device), blocks.to(device)
            outputs = model(x)
            logits = aggregate_temporal_outputs(outputs, blocks, targets.size(1), cfg['loss_agg'])[:, 1, :]
            target = targets[:, 1].long()
            losses = F.cross_entropy(logits, target, reduction='none')
            predictions = logits.argmax(-1)
            probabilities = logits.softmax(-1).gather(1, target[:, None]).squeeze(1)
            if not torch.isfinite(losses).all():
                raise ValueError('Nonfinite validation losses')
            for (_, _, row), loss, prediction, probability in zip(batch, losses.cpu().tolist(), predictions.cpu().tolist(), probabilities.cpu().tolist()):
                rows.append(dict(row, model=cfg['model_variant'], seed=cfg['random_seed'], loss=loss,
                                 correct=int(prediction == row['target']), prediction=prediction, true_probability=probability))
    return rows


def cached_json(path, identity, produce):
    if path.exists():
        record = read_json(path)
        validate_manifest(record['identity'], identity)
        if digest(record['value']) != record['value_sha256']:
            raise ValueError(f'Cached output damaged: {portable_path(path)}')
        return record['value']
    value = produce()
    write_json(path, dict(identity=identity, value=value, value_sha256=digest(value)))
    return value


def write_csv(path, rows):
    temporary = path.with_suffix('.csv.tmp')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_analysis(args, study, runs, data_meta, sample_ids, destination):
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    from scripts.adaptation_analysis.shd_data import SHDValidationData
    from scripts.adaptation_analysis.reporting import write_report
    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    env = environment(device)
    manifest = dict(version=study['version'], pc=args.pc, arm=study['arms'][args.pc], samples=args.samples,
                    seeds=args.seeds, sample_ids=sample_ids, data=data_meta, source_sha256=source_identity(),
                    runs=runs, environment=env, ordinary_inference=True, training=False, official_test_opened=False)
    with study_lock(destination / '.analysis.lock'):
        manifest_path = destination / 'MANIFEST.json'
        if manifest_path.exists():
            validate_manifest(read_json(manifest_path), manifest)
        else:
            write_json(manifest_path, manifest)
        complete_path = destination / 'PILOT_COMPLETE.json'
        if complete_path.exists():
            complete = read_json(complete_path)
            if complete['manifest_sha256'] != digest(manifest):
                raise ValueError('Completion identity differs')
            for path, checksum in complete['outputs'].items():
                if sha256(local_path(path, must_exist=True)) != checksum:
                    raise ValueError(f'Completed output changed: {path}')
            print(f'Already complete and verified: {portable_path(destination)}', flush=True)
            return
        raw_dir = destination / 'diagnostic_rows'
        raw_dir.mkdir(exist_ok=True)
        start = time.monotonic()
        all_data = SHDValidationData(runs[0]['config'], sample_ids=data_meta['validation_indices'], verify_hash=False)
        subset = SHDValidationData(runs[0]['config'], sample_ids=sample_ids, verify_hash=False)
        clean = ('clean', 0, 'identity', 1.)
        anchor_inputs = input_rows(all_data, all_data.sample_ids, clean, study, features=False)
        pilot_inputs = {(condition[0], condition[1]): input_rows(subset, sample_ids, condition, study)
                        for condition in conditions(args.pc, study)}
        evaluated = {}
        anchors = []
        outputs = [manifest_path]
        for index, run in enumerate(runs):
            print(f'[{index+1}/{len(runs)}] Frozen {run["model"]}, seed {run["seed"]}', flush=True)
            write_json(destination / 'PROGRESS.json', dict(status='running', model=run['model'], seed=run['seed'],
                       completed_models=index, total_models=len(runs), elapsed_seconds=time.monotonic()-start))
            torch.manual_seed(run['seed'])
            checkpoint = torch.load(local_path(run['checkpoint']), weights_only=False, map_location='cpu')
            cfg = OmegaConf.create(run['config'])
            with contextlib.redirect_stdout(io.StringIO()):
                model = MLPSNN(cfg)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            del checkpoint
            model.requires_grad_(False).eval().to(device)
            identity = dict(manifest_sha256=digest(manifest), checkpoint_sha256=run['checkpoint_sha256'])
            stem = f'{run["model"]}_seed_{run["seed"]}'
            anchor_path = raw_dir / f'{stem}_full_validation.json'
            anchor = cached_json(anchor_path, identity, lambda: infer(model, run['config'], all_data, anchor_inputs, device, 'Full clean validation'))
            accuracy = sum(row['correct'] for row in anchor)/len(anchor)
            loss = float(np.mean([row['loss'] for row in anchor]))
            accuracy_delta = accuracy-run['validation']['accuracy']
            loss_delta = loss-run['validation']['loss']
            if abs(accuracy_delta) > study['accuracy_tolerance'] or abs(loss_delta) > study['loss_tolerance']:
                raise ValueError(f'Clean validation reproduction failed for {stem}: accuracy {accuracy:.9f} vs {run["validation"]["accuracy"]:.9f}; loss {loss:.9f} vs {run["validation"]["loss"]:.9f}. No scientific interpretation is allowed.')
            anchors.append(dict(model=run['model'], seed=run['seed'], n=len(anchor), accuracy=accuracy, loss=loss,
                                expected=run['validation'], accuracy_difference=accuracy_delta, loss_difference=loss_delta))
            outputs.append(anchor_path)
            print(f'Clean anchor passed: accuracy {accuracy*100:.2f}%, loss {loss:.6f}', flush=True)
            for (name, replicate), inputs in pilot_inputs.items():
                rows_path = raw_dir / f'{stem}_{name}_rep_{replicate}.json'
                rows = cached_json(rows_path, dict(identity, condition=name, replicate=replicate),
                                   lambda: infer(model, run['config'], subset, inputs, device, f'{name} rep {replicate}'))
                evaluated[run['model'], run['seed'], name, replicate] = rows
                outputs.append(rows_path)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        paired = []
        for seed in args.seeds:
            for name, replicate in pilot_inputs:
                paired += paired_rows(evaluated[study['parent'], seed, name, replicate],
                                      evaluated[study['variant'], seed, name, replicate])
        anchor_path = destination / 'VALIDATION_ANCHORS.json'
        write_json(anchor_path, anchors)
        paired_path = raw_dir / 'paired_samples.csv'
        write_csv(paired_path, paired)
        outputs += [anchor_path, paired_path]
        outputs += write_report(destination, paired, args.pc, dict(parent=study['parent'], variant=study['variant'],
                               dataset='SHD* validation only', n_recordings=args.samples, seeds=args.seeds,
                               environment=env, elapsed_seconds=time.monotonic()-start,
                               caution='Exploratory frozen-model pilot; no test accuracy, TTA, training, or causal architectural proof.'))
        write_json(complete_path, dict(status='complete', manifest_sha256=digest(manifest),
                   completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                   outputs={portable_path(path): sha256(path) for path in outputs}))
        write_json(destination / 'PROGRESS.json', dict(status='complete', elapsed_seconds=time.monotonic()-start))
        print(f'Pilot finished and stopped: {portable_path(destination)}', flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_check:
        self_check()
        if not args.check_only:
            return 0
    study = yaml.safe_load(STUDY_FILE.read_text(encoding='utf-8'))
    destination = output_directory(args.output_root, args.pc)
    runs, data, sample_ids = preflight(args, study)
    if args.check_only:
        print(f'Preflight passed for {args.pc}. No inference or output writes. Destination: {portable_path(destination)}', flush=True)
        return 0
    run_analysis(args, study, runs, data, sample_ids, destination)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'Error: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)

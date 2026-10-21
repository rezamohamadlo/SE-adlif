"""Evaluation-only MT/DTH/MT-DTH input-gain TTA; invoked through runner.py --tta."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import statistics

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from models.pl_module import MLPSNN, aggregate_temporal_outputs
from project_paths import ROOT, local_path, portable_config, portable_path
from cost_records import seed_directory
from search_shd import search_lock
from ssc_loss_diagnostic import atomic_json


def gains_for(policy, dataset):
    gains = list(policy['datasets'][dataset]['gains'])
    if len(gains) != len(set(gains)) or gains.count(1.0) != 1:
        raise ValueError('TTA views must be distinct and contain identity gain exactly once')
    if not gains or any(not isinstance(g, (float, int)) or not 0 < g < 2 for g in gains):
        raise ValueError('Expected finite, positive input gains below 2')
    return gains


@torch.inference_mode()
def batch_predictions(model, batch, loss_agg, gains):
    inputs, targets, block_idx = batch
    labels = targets.flatten().long()
    mask = labels != -1
    probabilities, identity = [], None
    for gain in gains:
        # Multiplication preserves padding, timing, masks and target alignment.
        outputs = model(inputs if gain == 1.0 else inputs * gain)
        logits = aggregate_temporal_outputs(outputs, block_idx, targets.size(1), loss_agg)
        logits = logits.reshape(-1, outputs.size(-1))[mask].float()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite TTA model outputs')
        probabilities.append(logits.softmax(-1))
        if gain == 1.0:
            identity = logits
    return identity, torch.stack(probabilities).mean(0), labels[mask]


def inspect_run(dataset, seed, model='MT_SE_adLIF'):
    directory = seed_directory(ROOT, model, dataset, seed)
    choices = [p for p in (directory / 'ckpt').glob('*.ckpt') if p.name != 'last.ckpt']
    if not choices:
        return None
    if len(choices) != 1:
        raise ValueError(f'Ambiguous best checkpoint in {portable_path(directory)}')
    checkpoint = choices[0]
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = portable_config(OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']), resolve=True))
    cell = {'MT_SE_adLIF': 'mt_se_adlif', 'DTH_SE_adLIF': 'dth_se_adlif', 'MT_DTH_SE_adLIF': 'mt_dth_se_adlif'}[model]
    if (cfg['random_seed'] != seed or cfg['exp_name'].split('_')[0] != dataset
            or cfg['l1']['cell'] != cell or cfg['l2']['cell'] != cell):
        raise ValueError('Checkpoint does not match requested model/dataset/seed')
    metrics = directory / 'logs/mlp_snn/version_0/metrics.csv'
    if not metrics.exists():
        raise ValueError('Missing original validation metrics; cannot verify checkpoint selection')
    with metrics.open() as stream:
        values = [r for r in csv.DictReader(stream) if r.get('val_acc')]
    if not values or len(values) != len(set(r['epoch'] for r in values)):
        raise ValueError('Missing or conflicting validation history')
    if (not (directory / 'completed.txt').exists()
            or {int(float(r['epoch'])) for r in values} != set(range(int(cfg['n_epochs'])))):
        raise ValueError('Selected run is not a verified full-budget training completion')
    best = max(float(r['val_acc']) for r in values)
    callbacks = [v for k, v in state.get('callbacks', {}).items() if 'ModelCheckpoint' in str(k)]
    if not any(abs(float(v['best_model_score']) - best) < 1e-7 for v in callbacks):
        raise ValueError('Selected checkpoint score does not match validation selection')
    if dataset == 'SSC' and cfg['loss_agg'] != 'mean_membrane_potentials':
        raise ValueError('Current SSC TTA requires a mean-loss checkpoint')
    for file in (['SHDWrapper/shd_train.h5', 'SHDWrapper/shd_test.h5'] if dataset == 'SHD' else
                 ['SSCWrapper/ssc_train.h5', 'SSCWrapper/ssc_valid.h5', 'SSCWrapper/ssc_test.h5']
                 if dataset == 'SSC' else ['QTDB/QTDB_train.mat', 'QTDB/QTDB_test.mat']):
        local_path(str(Path(cfg['dataset']['data_path']) / file), must_exist=True)
    return dict(dataset=dataset, seed=seed, checkpoint=portable_path(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                selected_epoch=state['epoch'], config=cfg)


def evaluate(job, gains, split):
    state = torch.load(local_path(job['checkpoint']), map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(job['config'])
    device = torch.device(str(cfg.device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Saved checkpoint requests CUDA but CUDA is unavailable')
    torch.manual_seed(job['seed'])
    torch.set_float32_matmul_precision('high')
    with contextlib.redirect_stdout(io.StringIO()):
        model = MLPSNN(cfg)
    model.load_state_dict(state['state_dict'], strict=True)
    model.to(device).eval()
    dc = dict(job['config']['dataset'])
    for key in ('data_path', 'cache_root'):
        if dc.get(key):
            dc[key] = str(local_path(dc[key]))
    dc['num_workers'] = 0
    dm = hydra.utils.instantiate(dc)
    dm.prepare_data()
    dm.setup('validate' if split == 'validation' else 'test')
    loader = dm.val_dataloader() if split == 'validation' else dm.test_dataloader()
    if isinstance(loader, list):
        raise ValueError('Expected one evaluation split, not additional test validation')
    count, correct_base, correct_tta, loss_base, loss_tta = 0, 0, 0, 0., 0.
    data_digest = hashlib.sha256()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            for tensor in batch:
                data_digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
                data_digest.update(tensor.contiguous().numpy().tobytes())
            base, probabilities, labels = batch_predictions(
                model, tuple(t.to(device) for t in batch), cfg.loss_agg, gains)
            count += labels.numel()
            correct_base += (base.argmax(-1) == labels).sum().item()
            correct_tta += (probabilities.argmax(-1) == labels).sum().item()
            loss_base += F.cross_entropy(base, labels, reduction='sum').item()
            loss_tta += F.nll_loss(probabilities.clamp_min(1e-12).log(), labels, reduction='sum').item()
            if index % 10 == 0:
                print(f"{job['dataset']} seed {job['seed']} {split}: batch {index+1}/{len(loader)}", flush=True)
    if not count:
        raise ValueError('Empty evaluation split')
    return dict(evaluated_targets=count, baseline_accuracy=correct_base/count,
                tta_accuracy=correct_tta/count, difference_pp=100*(correct_tta-correct_base)/count,
                baseline_loss=loss_base/count, tta_nll=loss_tta/count,
                data_sha256=data_digest.hexdigest(), device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                matmul_precision=torch.get_float32_matmul_precision())


def write_report(output, model='MT_SE_adLIF'):
    records = [json.loads(p.read_text()) for p in sorted(output.glob('*/*/*/RESULT.json'))]
    lines = [f'# {model} test-time augmentation', '',
             'Separate exploratory evaluation. Identity plus input-gain views; frozen weights. '
             'Validation and test results are separate. Previously observed test scores do not become unseen.', '',
             '| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |',
             '|---|---:|---|---:|---:|---:|']
    groups = {}
    for r in records:
        m = r['metrics']
        groups.setdefault((r['dataset'], r['split']), []).append(m['tta_accuracy']*100)
        lines.append(f"| {'SHD*' if r['dataset']=='SHD' else r['dataset']} | {r['seed']} | {r['split']} | "
                     f"{100*m['baseline_accuracy']:.2f} | {100*m['tta_accuracy']:.2f} | {m['difference_pp']:+.3f} |")
    lines += ['', '| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |',
              '|---|---|---:|---:|---:|']
    for (dataset, split), values in sorted(groups.items()):
        sd = f'{statistics.stdev(values):.2f}' if len(values)>1 else 'unavailable'
        lines.append(f"| {'SHD*' if dataset=='SHD' else dataset} | {split} | {len(values)} | {statistics.mean(values):.2f} | {sd} |")
    lines += ['', 'Three model forwards per batch; not a cost-matched comparison to one-view inference. '
              'No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.']
    (output / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('MT_SE_adLIF', 'DTH_SE_adLIF', 'MT_DTH_SE_adLIF'), default='MT_SE_adLIF')
    parser.add_argument('--datasets', nargs='+', choices=('SHD','SSC','ECG'), required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=(42,123,456), required=True)
    parser.add_argument('--split', choices=('validation','test'), default='validation')
    parser.add_argument('--output')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    policy = OmegaConf.to_container(OmegaConf.load(ROOT / 'config/tta.yaml'), resolve=True)
    jobs, missing = [], []
    for dataset in dict.fromkeys(args.datasets):
        gains_for(policy, dataset)
        for seed in dict.fromkeys(args.seeds):
            job = inspect_run(dataset, seed, args.model)
            if job is None:
                missing.append(f'{dataset} seed {seed}')
                print(f'Missing: {dataset} seed {seed}', flush=True)
            else:
                jobs.append(job)
                print(f"Ready: {dataset} seed {seed}, {args.split}, views={gains_for(policy,dataset)}", flush=True)
    if args.dry_run:
        print(f'{len(jobs)} ready, {len(missing)} missing; no inference, training or result writes.')
        return
    if missing:
        raise ValueError('No evaluation started. Transfer missing selected checkpoints first: '+', '.join(missing))
    if not jobs:
        raise ValueError('No selected checkpoints')
    output = local_path(args.output or f'results/experiments/{args.model.removesuffix("_SE_adLIF").lower()}_tta_v1')
    environment = {p: importlib.metadata.version(p) for p in ('torch','numpy','pytorch-lightning','tonic')}
    sources = {name: hashlib.sha256(local_path(name).read_bytes()).hexdigest() for name in
               ('evaluate_tta.py','models/pl_module.py','models/ours/mt_adlif.py','models/ours/dth_adlif.py','models/ours/mt_dth_adlif.py','models/alif.py',
                'models/li.py','models/helpers.py','module/tau_trainers.py','datasets/shd.py',
                'datasets/ssc.py','datasets/ecg.py','datasets/utils/pad_tensors.py')}
    with search_lock(output):
        for job in jobs:
            gains = gains_for(policy, job['dataset'])
            identity = dict(**job, split=args.split, policy=policy, environment=environment, sources=sources)
            fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            directory = output/job['dataset']/f"seed_{job['seed']}"/args.split
            manifest = directory/'MANIFEST.json'
            if manifest.exists():
                if json.loads(manifest.read_text())['fingerprint'] != fingerprint:
                    raise ValueError('TTA configuration changed; use a new output directory')
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError('Refusing an unguarded existing evaluation directory')
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(manifest, dict(fingerprint=fingerprint, **identity))
            if not (directory/'RESULT.json').exists():
                metrics = evaluate(job, gains, args.split)
                atomic_json(directory/'RESULT.json', dict(dataset=job['dataset'],seed=job['seed'],
                            split=args.split,checkpoint=job['checkpoint'],
                            checkpoint_sha256=job['checkpoint_sha256'],gains=gains,metrics=metrics))
            result = json.loads((directory/'RESULT.json').read_text())
            print(f"{job['dataset']} seed {job['seed']} {args.split}: "
                  f"baseline={100*result['metrics']['baseline_accuracy']:.2f}%, "
                  f"TTA={100*result['metrics']['tta_accuracy']:.2f}%", flush=True)
            write_report(output, args.model)


if __name__ == '__main__':
    main()

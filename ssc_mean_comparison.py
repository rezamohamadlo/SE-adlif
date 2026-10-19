"""Run two fresh, validation-only SSC mean-loss jobs: DTH and MT, seed 42."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from project_paths import local_path, portable_config, portable_path
from search_shd import search_lock
from ssc_loss_diagnostic import atomic_json, stream_training, validation_result

ROOT = Path(__file__).resolve().parent
MODELS = ('DTH_SE_adLIF', 'MT_SE_adLIF')
REFERENCE = 'results/valid/reference/SSC/seed_42'


def job(model):
    overrides = [f'experiment=SSC_{model}_mean', 'random_seed=42']
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = portable_config(OmegaConf.to_container(
            compose(config_name='main', overrides=overrides), resolve=True))
    return local_path(cfg['logdir']), overrides, cfg


def verify_reference(cfg):
    reference = local_path(REFERENCE, must_exist=True)
    saved = json.loads((reference / 'FULL_RUN_CONFIG.json').read_text(encoding='utf-8'))
    shared = ('random_seed', 'lr', 'batch_size', 'n_epochs', 'early_stopping',
              'loss_agg', 'dropout', 'gradient_clip_val', 'scheduler', 'patience',
              'factor', 'two_layers', 'alpha', 'c', 'dt', 'l_out', 'dataset')
    differences = [key for key in shared if cfg.get(key) != saved.get(key)]
    for layer in ('l1', 'l2'):
        for key, value in saved[layer].items():
            if key != 'cell' and cfg[layer].get(key) != value:
                differences.append(f'{layer}.{key}')
    if differences:
        raise ValueError(f'Mismatch against the promoted reference: {differences}')
    original = json.loads((reference / 'DIAGNOSTIC_MANIFEST.json').read_text())
    for package, version in original['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Environment differs from reference: {package}; resolve before training')
    for name, digest in original['source_sha256'].items():
        if hashlib.sha256(local_path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Shared training source differs from reference: {name}')
    if cfg['run_test'] or cfg['ckpt_path'] or cfg['init_ckpt_path']:
        raise ValueError('Expected fresh weights, fresh optimizer and separate final test evaluation')
    record = json.loads((reference / 'TEST_EVALUATION_METADATA.json').read_text())
    checkpoint = local_path(record['checkpoint'], must_exist=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record['checkpoint_sha256']:
        raise ValueError('Promoted reference checkpoint identity mismatch')


def report():
    lines = ['# SSC mean-loss comparison - seed 42', '',
             'Protocol: ssc_mean_loss_v1; base: paper_classification_v1, SSC. '
             'Reference reused from a completed diagnostic; DTH/MT train fresh. '
             'This comparison was planned after observing the reference test score.', '',
             '| Model | Best validation (%) | Final test (%) | Status |',
             '|---|---:|---:|---|']
    ref = local_path(REFERENCE)
    v = json.loads((ref / 'VALIDATION_RESULTS.json').read_text())
    t = json.loads((ref / 'TEST_RESULTS.json').read_text())
    lines.append(f"| Reference | {100*v['best_validation_accuracy']:.2f} | "
                 f"{100*t['metrics']['test_acc']:.2f} | complete; diagnostic provenance retained |")
    for model in MODELS:
        d, _, cfg = job(model)
        result = validation_result(d, cfg['n_epochs'])
        lines.append(f'| {model} | ' +
                     (f"{100*result['best_validation_accuracy']:.2f} | pending | validation training complete |"
                      if result else 'pending | pending | not completed |'))
    local_path('results/valid/SSC_MEAN_COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    jobs = [job(model) for model in MODELS]
    for _, _, cfg in jobs:
        verify_reference(cfg)
    for name in ('ssc_train.h5', 'ssc_valid.h5', 'ssc_test.h5'):
        local_path(f'data/SSCWrapper/{name}', must_exist=True)
    if args.dry_run:
        for directory, _, cfg in jobs:
            print(json.dumps(dict(directory=portable_path(directory), experiment=cfg['exp_name'],
                                  seed=42, lr=cfg['lr'], batch=cfg['batch_size'], epochs=cfg['n_epochs'],
                                  loss=cfg['loss_agg'], early_stopping=cfg['early_stopping'],
                                  run_test=cfg['run_test'], protocol=cfg['evaluation_protocol'])))
        return
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    packages = {p: importlib.metadata.version(p) for p in
                ('numpy', 'torch', 'tonic', 'pytorch-lightning', 'h5py', 'torchmetrics')}
    sources = {str(p): hashlib.sha256(local_path(str(p)).read_bytes()).hexdigest() for p in
               ('run.py', 'models/pl_module.py', 'models/alif.py', 'models/ours/dth_adlif.py',
                'models/ours/mt_adlif.py', 'datasets/ssc.py', 'module/tau_trainers.py')}
    with search_lock(local_path('results/valid/.ssc_mean_queue')):
        for directory, overrides, cfg in jobs:
            identity = dict(config=cfg, packages=packages, sources=sources)
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            manifest = directory / 'MEAN_COMPARISON_MANIFEST.json'
            if manifest.exists():
                if json.loads(manifest.read_text())['fingerprint'] != fingerprint:
                    raise ValueError('Changed config/environment/source; use a separate experiment directory')
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError(f'Refusing nonempty unguarded output: {portable_path(directory)}')
            directory.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                atomic_json(manifest, dict(fingerprint=fingerprint, **identity))
                atomic_json(directory / 'FULL_RUN_CONFIG.json', cfg)
            if validation_result(directory, cfg['n_epochs']):
                print(f'Complete: {portable_path(directory)}', flush=True)
                report()
                continue
            last = directory / 'ckpt/last.ckpt'
            run_overrides = list(overrides)
            if last.exists():
                if not args.resume:
                    raise ValueError('Use --resume to restore this interrupted job from its own last.ckpt')
                run_overrides.append(f'ckpt_path={json.dumps(portable_path(last))}')
            elif (directory / 'logs/mlp_snn/version_0/metrics.csv').exists():
                raise ValueError('Partial metrics without last.ckpt; preserve evidence and use a new directory')
            print(f"{'Resume' if last.exists() else 'Fresh'}: {portable_path(directory)}", flush=True)
            code = stream_training([sys.executable, '-u', str(ROOT / 'run.py'), *run_overrides],
                                   directory / 'console.log')
            result = validation_result(directory, cfg['n_epochs'])
            report()
            if code or not result:
                raise RuntimeError(f'Incomplete job: {portable_path(directory)}; see console.log')
            atomic_json(directory / 'VALIDATION_RESULTS.json', result)


if __name__ == '__main__':
    main()

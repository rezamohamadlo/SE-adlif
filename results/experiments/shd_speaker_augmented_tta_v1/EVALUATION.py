"""User-requested frozen-checkpoint TTA for completed augmented speaker-holdout runs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
import evaluate_tta as tta
from extended_seed_study import source_identity
from project_paths import local_path, portable_path
import runner
from search_shd import search_lock
import shd_speaker_holdout as speaker
from ssc_loss_diagnostic import atomic_json

OUTPUT = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)


def preflight():
    # These changes live only inside this evaluation process.
    tta.MODEL_CELLS['LG_DTH_SE_adLIF'] = 'lg_dth_se_adlif'
    for model in speaker.MODELS:
        tta.EXPERIMENT_ROOTS[model] = 'results/experiments/shd_speaker_augmented_v1'
    jobs = []
    with speaker.selected_profile('augmented_v1'):
        manifest = speaker.preflight_data()
        with speaker.configured_runner(runner, manifest):
            for model in speaker.MODELS:
                for item in runner.build_plan(model, seeds=SEEDS, datasets=('SHD',)):
                    if item['status'] != 'complete':
                        raise ValueError('TTA requires a completed ordinary run')
                    job = tta.inspect_run('SHD', item['seed'], model)
                    costs = json.loads((item['directory'] / 'COST_RESULTS.json').read_text(encoding='utf-8'))
                    if (local_path(costs['checkpoint']) != local_path(job['checkpoint'])
                            or costs['checkpoint_sha256'] != job['checkpoint_sha256']):
                        raise ValueError('TTA checkpoint differs from the original measured selection')
                    job.update(model=model, ordinary_accuracy=speaker.test_accuracy(item['directory']) / 100,
                               split_fingerprint=manifest['fingerprint'])
                    jobs.append(job)
                    print(f'Ready: {model} seed {job["seed"]}, selected epoch {job["selected_epoch"]}', flush=True)
    return jobs


def report():
    summaries = []
    lines = ['# SHD speaker holdout + training augmentation: exploratory test-time augmentation', '',
             'Same validation-selected checkpoints. Fixed gains 1.00/0.95/1.05; mean class probabilities.',
             'No training or parameter adaptation. Official test results have previously been observed.',
             'TTA uses three model forwards per batch. Ordinary and TTA results remain separate.', '',
             '| Model | Seed | Selected epoch (zero-based) | Ordinary test (%) | TTA test (%) | Difference (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for model in speaker.MODELS:
        base_values, tta_values = [], []
        for seed in SEEDS:
            path = OUTPUT / model / 'SHD' / f'seed_{seed}' / 'test' / 'RESULT.json'
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            metrics = record['metrics']
            base_values.append(100 * metrics['baseline_accuracy'])
            tta_values.append(100 * metrics['tta_accuracy'])
            lines.append(f'| {model} | {seed} | {record["selected_epoch"]} | '
                         f'{base_values[-1]:.4f} | {tta_values[-1]:.4f} | {metrics["difference_pp"]:+.4f} |')
        if len(tta_values) == len(SEEDS):
            summaries += [f'{model}, n=3: ordinary mean +/- sample SD '
                      f'{statistics.mean(base_values):.4f} +/- {statistics.stdev(base_values):.4f}%; '
                      f'TTA {statistics.mean(tta_values):.4f} +/- {statistics.stdev(tta_values):.4f}%.', '']
    lines += ['', *summaries]
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = preflight()
    policy = OmegaConf.to_container(OmegaConf.load(ROOT / 'config/tta.yaml'), resolve=True)
    gains = tta.gains_for(policy, 'SHD')
    if args.dry_run:
        print(f'Validated {len(jobs)} checkpoints and fixed gains {gains}; no inference or training.')
        return
    provenance = dict(source=source_identity(ROOT),
                      recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      environment={p: importlib.metadata.version(p) for p in
                                   ('torch', 'numpy', 'pytorch-lightning', 'tonic')})
    with search_lock(OUTPUT):
        for job in jobs:
            directory = OUTPUT / job['model'] / 'SHD' / f'seed_{job["seed"]}' / 'test'
            identity = dict(**job, policy=policy, split='test', provenance=provenance)
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            manifest_path = directory / 'MANIFEST.json'
            if manifest_path.exists():
                if json.loads(manifest_path.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
                    raise ValueError('Evaluation identity changed; use a new versioned output directory')
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError('Refusing an unguarded evaluation directory')
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(manifest_path, dict(**identity, fingerprint=fingerprint))
            result_path = directory / 'RESULT.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if (result.get('fingerprint') != fingerprint
                        or result.get('checkpoint_sha256') != job['checkpoint_sha256']):
                    raise ValueError('Cached result disagrees with its evaluation manifest')
            else:
                metrics = tta.evaluate(job, gains, 'test')
                if abs(metrics['baseline_accuracy'] - job['ordinary_accuracy']) > 1e-6:
                    atomic_json(directory / 'IDENTITY_MISMATCH.json', dict(job=job, metrics=metrics))
                    raise ValueError('Identity inference does not reproduce ordinary test accuracy')
                result = dict(model=job['model'], dataset='SHD', seed=job['seed'], split='test',
                              evaluation_protocol='shd_speaker_augmented_tta_v1',
                              selected_epoch=job['selected_epoch'], checkpoint=job['checkpoint'],
                              checkpoint_sha256=job['checkpoint_sha256'], gains=gains,
                              fingerprint=fingerprint, metrics=metrics)
                atomic_json(result_path, result)
            print(f'{job["model"]} seed {job["seed"]}: ordinary '
                  f'{100*result["metrics"]["baseline_accuracy"]:.4f}%, TTA '
                  f'{100*result["metrics"]["tta_accuracy"]:.4f}%', flush=True)
            report()
    print(f'Saved: {portable_path(OUTPUT / "RESULTS.md")}', flush=True)


if __name__ == '__main__':
    main()

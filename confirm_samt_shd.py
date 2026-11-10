"""Fresh three-seed SA-MT confirmation of the explicitly chosen first trial."""
import argparse
from contextlib import nullcontext
import copy
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from project_paths import local_path, portable_path, portable_config
from search_shd import search_lock, atomic_text
from tune_shd import check_directory, launch

SOURCE = 'results/experiments/samt_shd_seed123_epoch188_continuation_plateau80_v6/lr0.0004_drop0.1/seed_123/SWEEP_CONFIG.json'
OUTPUT = 'results/experiments/samt_shd_first_trial_fresh_v1'
SEEDS = (42, 123, 456)


def build_plan(output=OUTPUT):
    output = local_path(output)
    source = local_path(SOURCE, must_exist=True)
    saved = json.loads(source.read_text())
    original = portable_config(saved['config'])
    assert saved['candidate']['id'] == 'lr0.0004_drop0.1'
    assert (original['lr'], original['dropout'], original['batch_size']) == (.0004, .1, 256)
    assert (original['scheduler'], original['tracking_metric'], original['tracking_mode'], original['factor'], original['patience']) == ('plateau', 'val_acc', 'max', .9, 10)
    assert original['dataset']['validate_on'] == .2 and original['dataset']['random_seed'] == 42
    assert all(original[layer]['cell'] == 'sa_mt_se_adlif' for layer in ('l1', 'l2'))
    plan=[]
    for seed in SEEDS:
        d=output/'SA_MT_SE_adLIF'/'SHD'/f'seed_{seed}'
        cfg=copy.deepcopy(original)
        cfg.update(random_seed=seed, n_epochs=300, init_ckpt_path=None, ckpt_path=None,
                   early_stopping=False, run_test=True, logdir=portable_path(d),
                   launcher_logdir=portable_path(output), launcher_script='confirm_samt_shd.py',
                   evaluation_protocol='samt_first_trial_fresh_exploratory_v1')
        cfg.pop('hydra',None)
        cfg['paper_baseline_deviations']=['model_variant=SA_MT_SE_adLIF',
            'fresh confirmation of explicitly chosen continuation trial lr0.0004_drop0.1',
            'lr=0.0004; dropout=0.1; plateau factor=0.9 patience=10; 300 epochs',
            'configuration chosen after test inspection; exploratory evidence']
        manifest=dict(search='samt_first_trial_fresh_exploratory_v1',
                      source_manifest=SOURCE, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      config=portable_config(cfg))
        plan.append(dict(directory=d,manifest=manifest,status=check_directory(d,manifest)))
    return plan


def report(plan,output):
    rows=[]
    for item in plan:
        d=item['directory']; values=[]
        path=d/'logs/mlp_snn/version_0/metrics.csv'
        if path.exists():
            with path.open() as stream:
                values=[float(r['test_acc'])*100 for r in csv.DictReader(stream) if r.get('test_acc')]
        complete=(d/'completed.txt').exists()
        if complete and len(values)!=1:
            raise ValueError(f'Expected one test result: {d}')
        rows.append(dict(seed=item['manifest']['config']['random_seed'],
                         status='complete' if complete else 'pending',
                         test_accuracy_percent=values[0] if complete else None))
    scores=[r['test_accuracy_percent'] for r in rows if r['status']=='complete']
    summary=dict(note='Fresh training; chosen after inspecting continuation test scores; exploratory',
                 runs=rows,n=len(scores),mean_percent=statistics.mean(scores) if scores else None,
                 sample_sd_pp=statistics.stdev(scores) if len(scores)>1 else None)
    atomic_text(output/'SUMMARY.json',json.dumps(summary,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=OUTPUT)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    output=local_path(args.output)
    try:
        with nullcontext() if args.dry_run else search_lock(output):
            plan=build_plan(output)
            data=local_path(plan[0]['manifest']['config']['dataset']['data_path'])
            for name in ('shd_train.h5','shd_test.h5'):
                if not (data/'SHDWrapper'/name).is_file():
                    raise FileNotFoundError(name)
            for item in plan:
                print(f"Seed {item['manifest']['config']['random_seed']}: {item['status']}; fresh, 300 epochs",flush=True)
            if args.dry_run:
                print('Validated 3 runs; no writes or training.')
                return
            for item in plan:
                try:
                    launch(item['directory'],item['manifest'],item['status'])
                finally:
                    report(plan,output)
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        parser.exit(1,f'Error: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130,'Interrupted; repeat the same command to resume.\n')


if __name__=='__main__':
    main()

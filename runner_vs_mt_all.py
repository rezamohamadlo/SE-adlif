"""Sequential VS-MT experiments: SSC, then SHD*, then ECG; official runner lifecycle."""
import argparse
from contextlib import redirect_stdout
import io
import json

import runner
import runner_vs_mt_ssc as ssc
from project_paths import local_path, portable_path

OUTPUT = 'results/experiments/vs_mt_v1'
ORDER = ('SSC', 'SHD', 'ECG')
FILES = {'SSC': ('SSCWrapper/ssc_train.h5', 'SSCWrapper/ssc_valid.h5', 'SSCWrapper/ssc_test.h5'),
         'SHD': ('SHDWrapper/shd_train.h5', 'SHDWrapper/shd_test.h5'),
         'ECG': ('QTDB/QTDB_train.mat', 'QTDB/QTDB_test.mat')}


def preflight(datasets, seeds):
    """Resolve every selected job and check local data before launching anything."""
    plans = {}
    for dataset in ORDER:
        if dataset not in datasets:
            continue
        if dataset == 'SSC':
            items, _ = ssc.plans(seeds)
        else:
            with ssc.output_root(OUTPUT):
                items = runner.build_plan(ssc.MODEL, seeds=seeds, datasets=[dataset])
        data = local_path(items[0]['config']['dataset']['data_path'])
        for name in FILES[dataset]:
            if not (data / name).is_file():
                raise FileNotFoundError(f'Missing {dataset} data: {data / name}')
        plans[dataset] = items
    return plans


def commands(datasets, seeds, dry_run=False, cost_option=None):
    result = []
    for dataset in ORDER:
        if dataset not in datasets:
            continue
        args = ['--datasets', dataset, '--seeds', *map(str, seeds)]
        if dataset != 'SSC':
            args = ['--models', ssc.MODEL, *args, '--logdir', OUTPUT]
        if dry_run:
            args.append('--dry-run')
        if cost_option:
            args.append(cost_option)
        result.append((dataset, args))
    return result


def save_parameters(items):
    """Read completed validation-selected checkpoints; perform no inference."""
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    from measure_shd_costs import selected_checkpoint, sha256
    for item in items:
        directory = item['directory']
        if not (directory / 'completed.txt').is_file():
            continue
        path = selected_checkpoint(directory)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        cfg = OmegaConf.create(checkpoint['hyper_parameters']['cfg'])
        if (cfg.random_seed != item['seed'] or cfg.l1.cell != 'vs_mt_se_adlif'
                or cfg.l2.cell != 'vs_mt_se_adlif'
                or cfg.dataset._target_ != item['config']['dataset']['_target_']):
            raise ValueError(f'Unexpected VS-MT checkpoint identity: {path}')
        with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
            model = MLPSNN(cfg).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        record = dict(checkpoint=portable_path(path), checkpoint_sha256=sha256(path),
                      seed=item['seed'], dataset=item['dataset'], layers=ssc.parameter_record(model))
        (directory / 'NEURON_PARAMETERS.json').write_text(json.dumps(record, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=ORDER, default=list(ORDER))
    parser.add_argument('--seeds', nargs='+', type=int, choices=runner.SEEDS, default=list(runner.SEEDS))
    parser.add_argument('--dry-run', action='store_true')
    costs = parser.add_mutually_exclusive_group()
    costs.add_argument('--no-measure-costs', action='store_true')
    costs.add_argument('--measure-existing-costs', action='store_true')
    args = parser.parse_args(argv)
    seeds = list(dict.fromkeys(args.seeds))
    planned = preflight(args.datasets, seeds)
    option = ('--no-measure-costs' if args.no_measure_costs else
              '--measure-existing-costs' if args.measure_existing_costs else None)
    jobs = commands(args.datasets, seeds, args.dry_run, option)
    if args.dry_run:
        for dataset, command in jobs:
            (ssc.main if dataset == 'SSC' else runner.main)(command)
        return
    from extended_seed_study import study_lock
    # A separate lock protects this sequence. SSC also uses its original lock;
    # SHD/ECG use this output root, preserving an existing SSC run's identity.
    with study_lock(local_path(OUTPUT)):
        planned = preflight(args.datasets, seeds)
        for dataset, command in jobs:
            print(f'Starting VS-MT {dataset}; later datasets wait for completion.', flush=True)
            (ssc.main if dataset == 'SSC' else runner.main)(command)
            if dataset != 'SSC':
                save_parameters(planned[dataset])


if __name__ == '__main__':
    main()

"""Opt-in SSC study operations, invoked exclusively through runner.py."""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from project_paths import local_path, portable_config, portable_path

ROOT = Path(__file__).resolve().parent
STAGES = ('lr', 'select-lr', 'dropout', 'select-dropout', 'check', 'select-winner',
          'confirm', 'freeze-test', 'test', 'report')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def definition():
    spec = OmegaConf.to_container(OmegaConf.load(ROOT / 'config/studies/ssc_validation_search_v1.yaml'))
    if (spec['name'] != 'ssc_validation_search_v1'
            or spec['output_root'] != 'results/experiments/ssc_validation_search_v1'
            or spec['paper_baseline_version'] != 'paper_classification_v1'
            or spec['models'] != ['MT_SE_adLIF', 'SA_MT_SE_adLIF', 'AMT_SE_adLIF']
            or spec['learning_rates'] != [.003, .006, .009]
            or spec['dropouts'] != [.1, .15, .2]
            or spec['screening_seeds'] != [42, 123] or spec['check_seed'] != 456
            or spec['confirmation_seeds'] != [1009, 2027, 3037, 4049, 5051]
            or spec['confirmation_assignment'] != 'paired_seeds_alternating_pc3_pc4'
            or spec['reference'] != 'SE_adLIF' or spec['historical_baseline_reuse'] is not False):
        raise ValueError('Changed SSC study definition requires a new version and output root')
    return spec


def study_root():
    return local_path(definition()['output_root'])


def candidate(model, lr, dropout=.15):
    return dict(model=model, lr=lr, dropout=dropout)


def candidate_id(c):
    return f"{c['model']}_lr{c['lr']:g}_drop{c['dropout']:g}".replace('.', 'p')


def manifest_path(stage):
    return study_root() / 'selection' / f'{stage}.json'


def read_frozen(stage):
    path = manifest_path(stage)
    if not path.is_file():
        raise ValueError(f'Missing frozen {stage} selection: {portable_path(path)}; merge both PCs\' records first')
    record = json.loads(path.read_text(encoding='utf-8'))
    check = dict(record)
    checksum = check.pop('sha256', None)
    if (checksum != digest(check) or check.get('study_sha256') != digest(definition())
            or check.get('stage') != stage or check.get('schema') != 'ssc_selection_v1'):
        raise ValueError(f'Invalid frozen study manifest: {path}')
    return record


def freeze(stage, payload, dry_run):
    from extended_seed_study import atomic_json
    record = dict(schema='ssc_selection_v1', study_sha256=digest(definition()), stage=stage, **payload)
    record['sha256'] = digest(record)
    path = manifest_path(stage)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != record:
        raise ValueError(f'Frozen selection changed; preserve it and use a new study: {path}')
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, record)
    print(json.dumps(record, indent=2))
    return record


def job(stage, pc, c, seed, selection=None):
    relative = f"{definition()['output_root']}/{pc}/{stage}/{candidate_id(c)}"
    metadata = dict(name=definition()['name'], definition_sha256=digest(definition()),
                    stage=stage, pc=pc, candidate=c, selection_sha256=selection)
    return dict(stage=stage, pc=pc, candidate=c, seed=seed, seeds=[seed],
                logdir=relative, metadata=metadata)


def jobs(stage, pc):
    if pc not in ('pc3', 'pc4'):
        raise ValueError('Select --study-pc pc3 or pc4')
    spec = definition()
    seed = spec['screening_seeds'][0 if pc == 'pc3' else 1]
    if stage == 'lr':
        return [job(stage, pc, candidate(model, lr), seed)
                for model in spec['models'] for lr in spec['learning_rates']]
    if stage == 'dropout':
        selected = read_frozen('select-lr')
        return [job(stage, pc, dict(c, dropout=drop), seed, selected['sha256'])
                for c in selected['candidates'] for drop in (.1, .2)]
    if stage == 'check':
        selected = read_frozen('select-dropout')
        index = 0 if pc == 'pc3' else 1
        return [job(stage, pc, selected['candidates'][index], spec['check_seed'], selected['sha256'])]
    if stage == 'confirm':
        selected = read_frozen('select-winner')
        cands = [selected['winner'], candidate(spec['reference'], .006)]
        return [job(stage, pc, c, s, selected['sha256'])
                for i, s in enumerate(spec['confirmation_seeds'])
                if i % 2 == (0 if pc == 'pc3' else 1) for c in cands]
    raise ValueError(f'Not a training stage: {stage}')


def all_jobs(stage):
    return [j for pc in ('pc3', 'pc4') for j in jobs(stage, pc)]


def job_directory(j):
    return local_path(j['logdir']) / j['candidate']['model'] / 'SSC' / f"seed_{j['seed']}"


def data_record(pc):
    path = study_root() / pc / 'DATA_PROVENANCE.json'
    if not path.exists():
        raise ValueError(f'Missing SSC data provenance: {portable_path(path)}')
    record = json.loads(path.read_text(encoding='utf-8'))
    if (record.get('schema') != 'ssc_training_data_v1' or set(record.get('files', {})) != {'train', 'valid'}
            or any(not isinstance(h, str) or len(h) != 64 for h in record['files'].values())):
        raise ValueError('Invalid SSC data provenance')
    return record


def prepare_data_record(pc):
    from extended_seed_study import atomic_json
    files = {split: ROOT / 'data/SSCWrapper' / f'ssc_{split}.h5' for split in ('train', 'valid')}
    if any(not p.is_file() for p in files.values()):
        raise ValueError('Missing data/SSCWrapper/ssc_train.h5 or ssc_valid.h5; transfer the exact dataset')
    record = dict(schema='ssc_training_data_v1', files={k: sha256(p) for k, p in files.items()})
    path = study_root() / pc / 'DATA_PROVENANCE.json'
    if path.exists():
        if data_record(pc) != record:
            raise ValueError('SSC training/validation data changed; refusing to reuse this study')
    else:
        atomic_json(path, record)


def deviations(c):
    values = [] if c['model'] == 'SE_adLIF' else [f"model_variant={c['model']}"]
    return values + ['loss_agg=mean_membrane_potentials (released SSC implementation)',
                     'study=ssc_validation_search_v1', f"tuning.lr={c['lr']}",
                     f"tuning.dropout={c['dropout']}"]


def job_overrides(j, model, dataset, seed):
    if (model != j['candidate']['model'] or dataset != 'SSC' or seed != j['seed']):
        raise ValueError('SSC job does not match the requested model/dataset/seed')
    c = j['candidate']
    def hydra_value(value):
        if isinstance(value, dict):
            return '{' + ','.join(f'{k}:{hydra_value(v)}' for k, v in value.items()) + '}'
        return json.dumps(value)
    return [f"lr={c['lr']}", f"dropout={c['dropout']}",
            f"++ssc_study={hydra_value(j['metadata'])}",
            f"++evaluation_protocol=ssc_mean_loss_v1_ssc_validation_search_v1_{j['stage']}",
            f"++paper_baseline_deviations={json.dumps(deviations(c))}"]


@contextmanager
def composed(c):
    overrides = [f"experiment=SSC_{c['model']}", '+comparison=SSC',
                 f"lr={c['lr']}", f"dropout={c['dropout']}"]
    if GlobalHydra.instance().is_initialized():
        yield portable_config(OmegaConf.to_container(compose(config_name='main', overrides=overrides), resolve=True))
    else:
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            yield portable_config(OmegaConf.to_container(compose(config_name='main', overrides=overrides), resolve=True))


def validate_training_config(cfg):
    spec = definition()
    meta = cfg.get('ssc_study', {})
    c = meta.get('candidate', {})
    if (meta.get('name') != spec['name'] or meta.get('definition_sha256') != digest(spec)
            or meta.get('stage') not in ('lr', 'dropout', 'check', 'confirm')
            or meta.get('pc') not in ('pc3', 'pc4')
            or set(c) != {'model', 'lr', 'dropout'}
            or c['model'] not in [*spec['models'], spec['reference']]
            or c['lr'] not in spec['learning_rates'] or c['dropout'] not in spec['dropouts']
            or cfg.get('run_test') is not False or cfg.get('validation_only') is not True
            or cfg.get('paper_baseline_version') != spec['paper_baseline_version']
            or cfg.get('seed_cohort') or cfg.get('evaluation_file') or cfg.get('evaluation')
            or cfg.get('model_variant') != c['model']
            or cfg.get('paper_baseline_deviations') != deviations(c)
            or cfg.get('evaluation_protocol') != f"ssc_mean_loss_v1_ssc_validation_search_v1_{meta['stage']}"):
        raise ValueError('Unsafe SSC study training configuration')
    if meta['stage'] != 'confirm' and c['model'] == spec['reference']:
        raise ValueError('Reference is only scheduled for frozen confirmation')
    if c['model'] == spec['reference'] and c != candidate(spec['reference'], .006):
        raise ValueError('Confirmation reference must retain the baseline')
    if meta['stage'] == 'lr' and c['dropout'] != .15:
        raise ValueError('LR screening must retain baseline dropout')
    if meta['stage'] in ('lr', 'dropout'):
        allowed = [spec['screening_seeds'][0 if meta['pc'] == 'pc3' else 1]]
    else:
        allowed = [spec['check_seed']] if meta['stage'] == 'check' else spec['confirmation_seeds']
    if type(cfg.get('random_seed')) is not int or cfg['random_seed'] not in allowed:
        raise ValueError('Invalid SSC study training seed')
    matching = [j for j in jobs(meta['stage'], meta['pc'])
                if j['candidate'] == c and j['seed'] == cfg['random_seed'] and j['metadata'] == meta]
    if len(matching) != 1:
        raise ValueError('SSC training job differs from the frozen stage assignment')
    expected_dir = job_directory(job(meta['stage'], meta['pc'], c, cfg['random_seed'], meta.get('selection_sha256')))
    if local_path(cfg['logdir']) != expected_dir:
        raise ValueError('SSC study output path does not match candidate/PC/seed')
    if (cfg.get('launcher_script') != 'runner.py' or cfg.get('launcher_logdir') != matching[0]['logdir']
            or cfg.get('run_mode') != ('reference' if c['model'] == spec['reference'] else 'ours')):
        raise ValueError('SSC launcher metadata differs from the frozen assignment')
    # Compare every baseline/config field, not just LR and batch. Routing and
    # provenance are the only extra fields permitted in the resolved config.
    with composed(c) as baseline:
        route = {'logdir', 'ckpt_path', 'random_seed', 'run_test', 'paper_baseline_deviations'}
        if any(cfg.get(k) != value for k, value in baseline.items() if k not in route):
            raise ValueError('SSC study changed an undeclared baseline setting')
        extras = set(cfg) - set(baseline)
        if extras - {'launcher_script', 'model_variant', 'run_mode', 'launcher_logdir',
                     'evaluation_protocol', 'validation_only', 'ssc_study'}:
            raise ValueError('Undeclared SSC study configuration fields')


@contextmanager
def runner_job(runner, j):
    keys = ('RUN_LOGDIR', 'SEED_COHORT', 'VALIDATION_ONLY', 'SSC_STUDY_JOB')
    old = [getattr(runner, k) for k in keys]
    runner.RUN_LOGDIR, runner.SEED_COHORT = j['logdir'], None
    runner.VALIDATION_ONLY, runner.SSC_STUDY_JOB = True, j
    try:
        yield
    finally:
        for key, value in zip(keys, old):
            setattr(runner, key, value)


def evidence(j, checkpoints=False):
    from extended_seed_study import read_completed, read_provenance, validation_rows
    directory = job_directory(j)
    if not (directory / 'validation_completed.txt').exists():
        raise ValueError(f'Pending validation run: {portable_path(directory)}')
    record, cfg = read_completed(directory, require_checkpoints=checkpoints)
    read_provenance(directory, ROOT)
    if (cfg['ssc_study'] != j['metadata'] or record.get('ssc_study') != j['metadata']
            or cfg['random_seed'] != j['seed']
            or len(validation_rows(directory, cfg)) != cfg['n_epochs']
            or any(not isinstance(record.get(k), str) or len(record[k]) != 64
                   for k in ('checkpoint_sha256', 'last_checkpoint_sha256'))):
        raise ValueError(f'Invalid SSC study completion evidence: {directory}')
    if checkpoints:
        for key, hash_key in [('best_checkpoint', 'checkpoint_sha256'), ('last_checkpoint', 'last_checkpoint_sha256')]:
            if sha256(local_path(record[key], must_exist=True)) != record[hash_key]:
                raise ValueError(f'Checkpoint contents changed: {record[key]}')
    return dict(candidate=j['candidate'], seed=j['seed'], directory=portable_path(directory),
                accuracy=record['best_validation_accuracy'],
                training_data_sha256=digest(data_record(j['pc'])),
                checkpoint=record['best_checkpoint'], checkpoint_sha256=record['checkpoint_sha256'],
                validation_sha256=sha256(directory / 'VALIDATION_RESULTS.json'),
                config_sha256=sha256(directory / 'FULL_RUN_CONFIG.json'),
                metrics_sha256=sha256(directory / 'logs/mlp_snn/version_0/metrics.csv'))


def ranking(records):
    if len({r.get('training_data_sha256') for r in records}) > 1:
        raise ValueError('SSC PCs used different training/validation data')
    groups = {}
    for r in records:
        groups.setdefault(candidate_id(r['candidate']), []).append(r)
    ranked = []
    for key, group in groups.items():
        if len({r['seed'] for r in group}) != len(group):
            raise ValueError('Repeated seed in selection evidence')
        vals = [r['accuracy'] for r in group]
        ranked.append(dict(candidate=group[0]['candidate'], mean=statistics.mean(vals),
                           sd=statistics.stdev(vals) if len(vals) > 1 else 0, n=len(vals)))
    return sorted(ranked, key=lambda r: (-r['mean'], r['sd'], candidate_id(r['candidate'])))


def select(stage, dry_run):
    if stage == 'select-lr':
        records = [evidence(j) for j in all_jobs('lr')]
        ranked = ranking(records)
        best = []
        for row in ranked:
            if row['candidate']['model'] not in [c['model'] for c in best]:
                best.append(row['candidate'])
        return freeze(stage, dict(candidates=best[:2], ranking=ranked, evidence=records), dry_run)
    if stage == 'select-dropout':
        prior = read_frozen('select-lr')
        records = [evidence(j) for j in all_jobs('dropout')]
        records += [r for r in prior['evidence'] if r['candidate'] in prior['candidates']]
        verify_snapshots(records)
        ranked = ranking(records)
        best = []
        for row in ranked:
            if row['candidate']['model'] not in [c['model'] for c in best]:
                best.append(row['candidate'])
        return freeze(stage, dict(candidates=best, ranking=ranked, evidence=records,
                                 parent_sha256=prior['sha256']), dry_run)
    prior = read_frozen('select-dropout')
    records = [evidence(j) for j in all_jobs('check')]
    records += [r for r in prior['evidence'] if r['candidate'] in prior['candidates']]
    verify_snapshots(records)
    ranked = ranking(records)
    baselines = read_frozen('select-lr')['evidence']
    improvements = []
    for c in prior['candidates']:
        selected = [r['accuracy'] for r in records if r['candidate'] == c and r['seed'] in (42, 123)]
        baseline = [r['accuracy'] for r in baselines
                    if r['candidate'] == candidate(c['model'], .006)]
        improvements.append(statistics.mean(selected) - statistics.mean(baseline))
    if max(improvements) <= 0:
        raise ValueError('Neither finalist improved its own baseline validation mean; stop and review before confirmation')
    return freeze(stage, dict(winner=ranked[0]['candidate'], ranking=ranked, evidence=records,
                             parent_sha256=prior['sha256'],
                             screening_improvements=improvements,
                             caveat='Development-selected winner; test performance is not yet known.'), dry_run)


def verify_snapshots(records):
    for r in records:
        directory = local_path(r['directory'], must_exist=True)
        for filename, key in [('VALIDATION_RESULTS.json', 'validation_sha256'),
                              ('FULL_RUN_CONFIG.json', 'config_sha256'),
                              ('logs/mlp_snn/version_0/metrics.csv', 'metrics_sha256')]:
            if sha256(directory / filename) != r[key]:
                raise ValueError(f'Frozen validation evidence changed: {directory / filename}')
        cfg = json.loads((directory / 'FULL_RUN_CONFIG.json').read_text(encoding='utf-8'))
        if r['training_data_sha256'] != digest(data_record(cfg['ssc_study']['pc'])):
            raise ValueError('Frozen SSC data provenance changed')


def freeze_test(dry_run):
    winner = read_frozen('select-winner')
    # Completion records contain hashes written after trusted checkpoint inspection.
    # Both PCs merge small evidence; each verifies its exact local weights at test.
    records = [evidence(j) for j in all_jobs('confirm')]
    if len({r['training_data_sha256'] for r in records}) != 1:
        raise ValueError('Confirmation PCs used different SSC training/validation data')
    return freeze('freeze-test', dict(winner_sha256=winner['sha256'], evidence=records,
                                     inference='ordinary', test_selection_forbidden=True), dry_run)


def validate_final_evidence(directory):
    """Permit test/cost evidence only after frozen confirmation; never for screening."""
    cfg = json.loads((directory / 'FULL_RUN_CONFIG.json').read_text(encoding='utf-8'))
    validate_training_config(cfg)
    if cfg['ssc_study']['stage'] != 'confirm' or (directory / 'completed.txt').exists():
        raise ValueError('Test evidence is forbidden in SSC screening')
    frozen = read_frozen('freeze-test')
    matches = [r for r in frozen['evidence'] if local_path(r['directory']) == directory.resolve()]
    if len(matches) != 1:
        raise ValueError('Run is not in the frozen confirmation checkpoint list')
    r = matches[0]
    verify_snapshots([r])
    test = json.loads((directory / 'TEST_RESULTS.json').read_text(encoding='utf-8'))
    if (test.get('schema') != 'ssc_frozen_test_v1' or test.get('freeze_sha256') != frozen['sha256']
            or test.get('checkpoint') != r['checkpoint'] or test.get('checkpoint_sha256') != r['checkpoint_sha256']
            or test.get('inference') != 'ordinary' or test.get('seed') != cfg['random_seed']
            or test.get('model') != cfg['model_variant']
            or not isinstance(test.get('test_data_sha256'), str) or len(test['test_data_sha256']) != 64
            or any(type(test.get('metrics', {}).get(k)) not in (int, float)
                   or not math.isfinite(test['metrics'][k]) for k in ('test_acc', 'test_loss'))
            or not 0 <= test['metrics']['test_acc'] <= 1):
        raise ValueError('Invalid frozen SSC final-test evidence')
    if test['metrics']['test_loss'] < 0:
        raise ValueError('Invalid negative classification test loss')
    from extended_seed_study import read_provenance
    provenance = read_provenance(directory)[0]
    if any(test.get('environment', {}).get(k) != provenance.get(k)
           for k in ('git_commit', 'tracked_diff_sha256', 'source_files_sha256')):
        raise ValueError('Final test used a different source revision')
    return test


def evaluate(j, frozen, dry_run, runner, measure_existing=False):
    from extended_seed_study import atomic_json, load_run_checkpoint
    directory = job_directory(j)
    r = evidence(j, checkpoints=True)
    expected = [e for e in frozen['evidence'] if e['directory'] == r['directory']]
    if expected != [r]:
        raise ValueError('Confirmation evidence differs from the frozen checkpoint list')
    if (directory / 'TEST_RESULTS.json').exists():
        validate_final_evidence(directory)
        if measure_existing and not dry_run:
            runner.collect_costs(directory, 'SSC')
        print(f"Verified existing test: {portable_path(directory)}")
        return
    if dry_run:
        print(f"Would test frozen checkpoint without training: {r['checkpoint']}")
        return
    import hydra
    import pytorch_lightning as pl
    import torch
    from models.pl_module import MLPSNN
    cfg = json.loads((directory / 'FULL_RUN_CONFIG.json').read_text(encoding='utf-8'))
    test_data = local_path(cfg['dataset']['data_path']) / 'SSCWrapper/ssc_test.h5'
    if not test_data.is_file():
        raise ValueError(f'Missing exact SSC test file: {portable_path(test_data)}')
    test_data_hash = sha256(test_data)
    state = load_run_checkpoint(r['checkpoint'], cfg)
    config = OmegaConf.create(cfg)
    pl.seed_everything(config.random_seed, workers=True)
    torch.set_float32_matmul_precision('high')
    model = MLPSNN(config)
    model.load_state_dict(state['state_dict'], strict=True)
    datamodule = hydra.utils.instantiate(config.dataset, data_path=str(local_path(config.dataset.data_path)))
    device = torch.device(str(config.device))
    trainer = pl.Trainer(accelerator='gpu' if device.type == 'cuda' else 'cpu',
                         devices=[device.index or 0] if device.type == 'cuda' else 1,
                         logger=False, enable_checkpointing=False)
    metrics = trainer.test(model, datamodule=datamodule)[0]
    from extended_seed_study import source_identity
    record = dict(schema='ssc_frozen_test_v1', freeze_sha256=frozen['sha256'],
                  checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
                  seed=j['seed'], model=j['candidate']['model'], inference='ordinary', metrics=metrics,
                  test_data_sha256=test_data_hash, environment=dict(**source_identity(ROOT),
                      torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                      matmul_precision=torch.get_float32_matmul_precision()))
    atomic_json(directory / 'TEST_RESULTS.json', record)
    validate_final_evidence(directory)
    runner.collect_costs(directory, 'SSC')


def report(dry_run):
    from scipy.stats import t
    frozen = read_frozen('freeze-test')
    rows = []
    test_hashes = set()
    for r in frozen['evidence']:
        test = validate_final_evidence(local_path(r['directory']))
        test_hashes.add(test.get('test_data_sha256'))
        rows.append(dict(model=test['model'], seed=test['seed'], accuracy=100 * test['metrics']['test_acc']))
    winner = read_frozen('select-winner')['winner']['model']
    if len(test_hashes) != 1 or None in test_hashes:
        raise ValueError('Final tests used different or unidentified SSC test files')
    ref = definition()['reference']
    groups = {model: {r['seed']: r['accuracy'] for r in rows if r['model'] == model} for model in (winner, ref)}
    seeds = definition()['confirmation_seeds']
    if any(set(g) != set(seeds) for g in groups.values()):
        raise ValueError('Final report requires all five paired seeds')
    diffs = [groups[winner][seed] - groups[ref][seed] for seed in seeds]
    mean = statistics.mean(diffs)
    half = float(t.ppf(.975, len(seeds) - 1)) * statistics.stdev(diffs) / math.sqrt(len(seeds))
    lines = ['# SSC frozen confirmation', '', 'Ordinary inference; five fresh matching training seeds.',
             'Development followed previously observed test results; the official test set is not untouched.',
             'The winner is tuned and the reference is fixed; this does not isolate a neuron mechanism.', '',
             '| Model | ' + ' | '.join(map(str, seeds)) + ' | Mean +/- sample SD (%, Δ paper) |',
             '|---|' + '---:|' * (len(seeds) + 1)]
    for model in (winner, ref):
        vals = [groups[model][seed] for seed in seeds]
        lines.append(f'| {model} | ' + ' | '.join(f'{v:.2f}' for v in vals)
                     + f' | {statistics.mean(vals):.2f} +/- {statistics.stdev(vals):.2f} ({statistics.mean(vals)-80.44:+.2f} pp) |')
    lines += ['', 'Δ paper = local mean minus published SSC mean 80.44%, in percentage points.', '',
              f'Paired winner minus reference: {mean:+.3f} pp; 95% paired t interval '
              f'[{mean-half:+.3f}, {mean+half:+.3f}] pp. Small-sample interval, n=5.', '',
              'Per-seed paired differences (pp): ' + ', '.join(f'{seed}: {d:+.3f}' for seed, d in zip(seeds, diffs)), '',
              'Cost evidence is retained beside each seed as COST_RESULTS.json/.md; missing costs remain pending.']
    text = '\n'.join(lines) + '\n'
    if not dry_run:
        (study_root() / 'CONFIRMATION_RESULTS.md').write_text(text, encoding='utf-8')
    print(text)


def execute_training(stage, pc, dry_run, runner):
    from extended_seed_study import study_lock
    planned = jobs(stage, pc)
    def preflight():
        count = 0
        for j in planned:
            with runner_job(runner, j):
                items = runner.build_plan(j['candidate']['model'], seeds=[j['seed']], datasets=['SSC'])
                for item in items:
                    print(f"{pc} {stage} {candidate_id(j['candidate'])} seed {j['seed']}: {item['status']}")
                count += len(items)
        return count
    count = preflight()
    if dry_run:
        print(f'Validated {count} runs; no training, inference or writes.')
        return
    # Dataset presence is machine-local; git does not transfer ignored HDF5 files.
    with study_lock(study_root() / pc):
        prepare_data_record(pc)
        preflight()
        for j in planned:
            with runner_job(runner, j):
                runner.run_validation(j['candidate']['model'], 2, [j['seed']], ['SSC'], 1)
    print(f'Finished {pc} {stage}. Merge both PCs\' small validation records before selection.')


def main(argv, runner):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssc-study-stage', choices=STAGES, required=True)
    parser.add_argument('--study-pc', choices=('pc3', 'pc4'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--measure-existing-costs', action='store_true')
    args = parser.parse_args(argv)
    stage = args.ssc_study_stage
    try:
        definition()
        if args.measure_existing_costs and stage != 'test':
            raise ValueError('--measure-existing-costs is only valid for the frozen test stage')
        if stage in ('lr', 'dropout', 'check', 'confirm'):
            return execute_training(stage, args.study_pc, args.dry_run, runner)
        if stage == 'test':
            from extended_seed_study import study_lock
            if args.study_pc not in ('pc3', 'pc4'):
                raise ValueError('Frozen testing requires --study-pc')
            frozen = read_frozen('freeze-test')
            assigned = jobs('confirm', args.study_pc)
            # Preflight every assigned checkpoint before the first test forward.
            for j in assigned:
                r = evidence(j, checkpoints=True)
                if [e for e in frozen['evidence'] if e['directory'] == r['directory']] != [r]:
                    raise ValueError('Confirmation evidence differs from frozen checkpoint list')
                if (job_directory(j) / 'TEST_RESULTS.json').exists():
                    validate_final_evidence(job_directory(j))
            if args.dry_run:
                for j in assigned:
                    evaluate(j, frozen, True, runner)
            else:
                with study_lock(study_root() / args.study_pc):
                    for j in assigned:
                        evaluate(j, frozen, False, runner, args.measure_existing_costs)
            return
        if args.study_pc:
            raise ValueError('Selection, freeze and report use merged results; omit --study-pc')
        from extended_seed_study import study_lock
        def operation():
            if stage.startswith('select-'):
                return select(stage, args.dry_run)
            if stage == 'freeze-test':
                return freeze_test(args.dry_run)
            return report(args.dry_run)
        if args.dry_run:
            return operation()
        with study_lock(study_root() / 'selection'):
            return operation()
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')

"""Read-only SSC queue preparation; no model forward, fitting or resampling."""
import contextlib
import io

import torch
from omegaconf import OmegaConf

from project_paths import local_path, portable_path, portable_config
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ssc.data import SSCValidationData, metadata, check_legacy_contract, validate_config
from scripts.adaptation_ssc.evaluate import selected_metrics
from scripts.adaptation_ssc_suite.common import load_spec, require, source_identity, validate_source, guard

CELLS = dict(MT_SE_adLIF='mt_se_adlif', AMT_SE_adLIF='amt_se_adlif', DTH_SE_adLIF='dth_se_adlif',
             LT_DTH_SE_adLIF='lt_dth_se_adlif', SA_MT_SE_adLIF='sa_mt_se_adlif', MT_TC_SE_adLIF='mt_se_adlif')


def validate_config_and_objective(cfg, run, spec, baseline):
    validate_config(cfg, baseline)
    section = baseline['datasets']['SSC']
    temporal = run['model'] == 'MT_TC_SE_adLIF'
    objective = spec['variant_training_objective'] if temporal else 'classification'
    protocol = spec['evaluation_protocol'] + ('_' + objective if temporal else '')
    expected = dict(random_seed=run['seed'], model_variant=run['model'], two_layers=True,
        paper_baseline_version=baseline['version'], paper_baseline_dataset='SSC', evaluation_protocol=protocol,
        dropout=baseline['common']['dropout'], loss_agg=spec['loss_agg'],
        early_stopping=baseline['reproduction_policy']['early_stopping'],
        **{key: section[key] for key in ('lr', 'batch_size', 'n_epochs')})
    require(all(cfg.get(k) == v for k, v in expected.items()), 'SSC saved baseline/protocol differs')
    require(cfg.get('training_objective', 'classification') == objective and not cfg.get('auto_regression', False),
            'Unexpected saved training objective')
    require(cfg.get('temporal_consistency') == (spec['variant_temporal_consistency'] if temporal else None),
            'Saved temporal-objective parameters differ')
    for layer, input_size in [('l1', 140), ('l2', 720)]:
        require(cfg[layer]['cell'] == CELLS[run['model']] and cfg[layer]['n_neurons'] ==
                section['reference_architecture']['neurons_per_layer'] and cfg[layer]['input_size'] == input_size,
                'Registered SSC architecture differs')
    require(cfg['l_out']['n_neurons'] == 35, 'SSC classification output differs')


def compare_pair_configs(parent, variant, pair):
    stripped = []
    allowed = {'cell'}
    if pair['id'] == 'amt_mt':
        allowed |= {'mix_rho_fast', 'mix_rho_slow'}
    if pair['id'] == 'lt_dth':
        allowed |= {'trace_rho_fast_range', 'trace_rho_slow_range'}
    for cfg in (parent, variant):
        item = pilot.effective_config(cfg)
        item.pop('model_variant', None)
        if pair['id'] == 'mt_tc':
            item.pop('training_objective', None)
            item.pop('evaluation_protocol', None)
        for layer in ('l1', 'l2'):
            item[layer] = {k: v for k, v in item[layer].items() if k not in allowed}
        for key in ('gradient_clip_val', 'factor', 'patience', 'tracking_metric', 'tracking_mode'):
            item[key] = cfg.get(key)
        stripped.append(item)
    require(stripped[0] == stripped[1], 'Pair settings differ beyond the explicit neuron/objective differences')


def prepare(args):
    from models.pl_module import MLPSNN
    spec, baseline = load_spec()
    destination = experiment_directory(args.output_root or spec['output_root'])
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or
            (torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count())), 'Requested device unavailable')
    snapshot = destination / 'shared/amt_mt_full_validation'
    source_directory = snapshot if (snapshot / 'PATH_INDEX.json').exists() else local_path(spec['source_evaluation']['root'])
    source, _, _ = validate_source(spec, snapshot if source_directory == snapshot else None)
    environment = pilot.environment(device)
    require({k: v for k, v in environment.items() if k != 'device'} ==
            {k: v for k, v in source['environment'].items() if k != 'device'},
            'Reused source environment differs; this suite requires its recorded software/GPU contract')
    for path, checksum in source['source_sha256'].items():
        import hashlib
        require(hashlib.sha256(local_path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == checksum,
                'Completed source code changed; preserve the source study')
    runs = []
    for frozen in spec['runs']:
        run = dict(frozen)
        record = pilot.read_json(local_path(run['cost_record'], must_exist=True))
        require(record['status'] == 'complete' and record['dataset'] == 'SSC' and record['seed'] == run['seed'] and
                pilot.cost_model_identity(record) == run['model'] and
                portable_path(record['checkpoint']) == run['checkpoint'] and
                record['checkpoint_sha256'] == run['checkpoint_sha256'] and record['best_epoch'] == run['selected_epoch'],
                'Checkpoint/cost identities differ')
        path = local_path(run['checkpoint'], must_exist=True)
        require(pilot.sha256(path) == run['checkpoint_sha256'], 'Frozen checkpoint checksum differs')
        checkpoint = torch.load(path, weights_only=False, map_location='cpu')
        cfg = portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True))
        require(pilot.effective_config(cfg) == pilot.effective_config(portable_config(record['saved_config'])) and
                cfg.get('temporal_consistency') == record['saved_config'].get('temporal_consistency') and
                checkpoint['epoch'] == run['selected_epoch'], 'Checkpoint saved config/epoch differs')
        validate_config_and_objective(cfg, run, spec, baseline)
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.requires_grad_(False).eval()
        require((hasattr(model.l1, 'b_slow') and hasattr(model.l2, 'b_slow')) ==
                (run['model'] == 'SA_MT_SE_adLIF'), 'SA-MT parameter identity differs')
        require((hasattr(model.l1, 'trace_decays') and hasattr(model.l2, 'trace_decays')) ==
                (run['model'] == 'LT_DTH_SE_adLIF'), 'LT-DTH learned parameter identity differs')
        del model, checkpoint
        run.update(config=cfg, validation=selected_metrics(run, baseline['datasets']['SSC']['n_epochs']))
        runs.append(run)
        print(f"SSC prerequisite OK: {run['model']} seed {run['seed']}, epoch {run['selected_epoch']}", flush=True)
    require(len(runs) == 18 and len({(r['model'], r['seed']) for r in runs}) == 18, 'Incomplete checkpoint registry')
    for pair in spec['pairs']:
        for seed in spec['seeds']:
            both = [next(r['config'] for r in runs if r['seed'] == seed and r['model'] == name)
                    for name in (pair['parent'], pair['variant'])]
            compare_pair_configs(*both, pair)
    for old in source['runs']:
        current = next(r for r in runs if r['model'] == old['model'] and r['seed'] == old['seed'])
        require(all(current[k] == old[k] for k in ('checkpoint', 'checkpoint_sha256', 'selected_epoch', 'validation')) and
                pilot.effective_config(current['config']) == pilot.effective_config(old['config']),
                'Reused AMT/MT run differs from current frozen checkpoint registry')
    meta, labels, speakers = metadata(runs[0]['config'], spec, baseline)
    require(meta['speaker_groups'] == spec['expected_speaker_groups'], 'Official SSC speaker cohort differs')
    data = SSCValidationData(runs[0]['config'], meta, labels, speakers, meta['validation_indices'])
    contract = check_legacy_contract(data)
    jobs = ['contract_tests', 'snapshot_source', 'amt_mt_statistics', 'lt_dth_evaluation', 'lt_dth_statistics',
            'sa_mt_evaluation', 'sa_mt_statistics', 'mt_tc_evaluation', 'mt_tc_statistics', 'summary']
    manifest = dict(version=spec['version'], specification=spec, scope='ssc_validation_mechanism_suite',
        output_root=portable_path(destination),
        sample_ids=meta['validation_indices'], data=meta, runs=runs, source_sha256=source_identity(),
        source_hash_encoding='CRLF_normalized_to_LF', environment=environment, loader_preflight=contract,
        jobs=jobs, training=False, official_test_opened=False, ordinary_inference=True,
        comparison_loss=spec['loss_agg'], intentional_training_difference='MT-TC objective only; JS excluded from comparison loss')
    guard(destination, manifest)
    return dict(spec=spec, root=destination, manifest=manifest, data=data, source_manifest=source,
                source_directory=source_directory)

"""Full-validation MT-TC/MT diagnostics; no training or official test access."""
import argparse
import contextlib
import hashlib
import io
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import yaml
from project_paths import local_path, portable_path, portable_config
from scripts.adaptation_analysis import pilot, evaluate as original
from scripts.adaptation_analysis.records import experiment_directory,verify_package

STUDY='config/studies/adaptation_mt_tc_full_validation_v1.yaml'


def load_study():
    definition=yaml.safe_load(local_path(STUDY).read_text(encoding='utf-8'))
    if definition['inherits'] != original.STUDY:
        raise ValueError('Unexpected inherited study')
    study=dict(original.load_study(),**definition)
    if study['training'] or study['test_set_access'] or (study['parent'],study['variant']) != ('MT_SE_adLIF','MT_TC_SE_adLIF'):
        raise ValueError('Only registered frozen MT-TC/MT diagnostics are supported')
    return study


def validate_objective(cfg,run,study):
    variant=run['model']==study['variant']
    expected=study['variant_training_objective'] if variant else 'classification'
    if cfg.get('training_objective','classification')!=expected:
        raise ValueError('Unexpected training objective')
    if variant and cfg.get('temporal_consistency')!=study['variant_temporal_consistency']:
        raise ValueError('Temporal-consistency weight/window differs from frozen study')
    if not variant and cfg.get('temporal_consistency') is not None:
        raise ValueError('Parent unexpectedly has temporal-consistency settings')


def check_config(cfg, run, study):
    from scripts.adaptation_analysis.shd_data import validate_shd_config
    validate_shd_config(cfg)
    baseline=yaml.safe_load(local_path(study['paper_baseline']).read_text(encoding='utf-8'))['paper_baseline']
    data=baseline['datasets']['SHD']; common=baseline['common']; policy=baseline['reproduction_policy']
    expected=dict(random_seed=run['seed'],model_variant=run['model'],two_layers=True,auto_regression=False,
                  paper_baseline_version=baseline['version'],paper_baseline_dataset='SHD',
                  evaluation_protocol=baseline['version']+('_'+study['variant_training_objective'] if run['model']==study['variant'] else ''),loss_agg='sum_softmax_over_time',
                  early_stopping=policy['early_stopping'],dropout=common['dropout'],
                  **{key:data[key] for key in ('lr','batch_size','n_epochs')})
    for key,value in expected.items():
        if cfg.get(key,False if key=='auto_regression' else None) != value:
            raise ValueError('Saved config differs from paper baseline: '+key)
    if cfg['dataset']['batch_size'] != data['batch_size']:
        raise ValueError('Validation batch size differs')
    cell={'MT_SE_adLIF':'mt_se_adlif','MT_TC_SE_adLIF':'mt_se_adlif'}[run['model']]
    if any(cfg[layer]['cell'] != cell for layer in ('l1','l2')):
        raise ValueError('Registered neuron identity differs')
    validate_objective(cfg,run,study)


def compare_configs(parent,variant):
    configs=[]
    for cfg in (parent,variant):
        current=pilot.effective_config(cfg)
        current.pop('model_variant',None)
        current.pop('training_objective',None)
        current.pop('evaluation_protocol',None)
        for layer in ('l1','l2'):
            current[layer]={k:v for k,v in current[layer].items() if k not in ('cell',)}
        for key in ('factor','patience','gradient_clip_val','tracking_metric','tracking_mode'):
            current[key]=cfg.get(key)
        configs.append(current)
    if configs[0] != configs[1]:
        raise ValueError('Parent and variant settings differ beyond the explicitly registered training-objective difference')


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    from scripts.adaptation_analysis.shd_data import preflight_shd
    study=load_study(); destination=experiment_directory(args.output_root or study['output_root'])
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and
        (not torch.cuda.is_available() or (device.index or 0)>=torch.cuda.device_count())):
        raise ValueError('Requested device is unavailable')
    runs=[]
    for frozen in study['runs']:
        run=dict(frozen);cost=pilot.read_json(local_path(run['cost_record'],must_exist=True))
        if (pilot.cost_model_identity(cost)!=run['model'] or cost['seed']!=run['seed'] or
            portable_path(cost['checkpoint'])!=run['checkpoint'] or cost['checkpoint_sha256']!=run['checkpoint_sha256'] or
            cost['best_epoch']!=run['selected_epoch']):
            raise ValueError('Frozen checkpoint and cost identity disagree')
        path=local_path(run['checkpoint'],must_exist=True)
        if pilot.sha256(path)!=run['checkpoint_sha256']: raise ValueError('Checkpoint checksum differs')
        checkpoint=torch.load(path,weights_only=False,map_location='cpu')
        cfg=portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']),resolve=True))
        if pilot.effective_config(cfg)!=pilot.effective_config(portable_config(cost['saved_config'])):
            raise ValueError('Checkpoint and saved measurement config disagree')
        if cfg.get('temporal_consistency')!=cost['saved_config'].get('temporal_consistency'):
            raise ValueError('Checkpoint/measurement temporal objective parameters differ')
        if checkpoint['epoch']!=run['selected_epoch']:raise ValueError('Selected epoch differs')
        check_config(cfg,run,study)
        # Strict state loading and parameter inspection only; never forward/fit.
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            model=MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model.requires_grad_(False).eval()
        if any(hasattr(getattr(model,layer),'b_slow') for layer in ('l1','l2')):
            raise ValueError('Expected ordinary MT neuron parameters for both models')
        del model
        run.update(config=cfg,validation=pilot.selected_metrics(run));runs.append(run)
        del checkpoint
        print(f"Prerequisite OK: {run['model']} seed {run['seed']}, epoch {run['selected_epoch']}",flush=True)
    for seed in study['seeds']:
        cohort=[r for r in runs if r['seed']==seed]
        if len(cohort)!=2 or {r['model'] for r in cohort}!={study['parent'],study['variant']}:
            raise ValueError('Incomplete model/seed cohort')
        compare_configs(cohort[0]['config'],cohort[1]['config'])
    data=preflight_shd(runs[0]['config'],study['dataset_sha256'])
    if data['path']!=study['dataset_file']:raise ValueError('Frozen dataset path differs')
    ids=original.full_sample_ids(data,study)
    sources=original.source_identity(original.load_study())
    for path in list((ROOT/'scripts/adaptation_mt_tc').glob('*.py'))+[local_path(STUDY)]:
        sources[portable_path(path)]=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    manifest=dict(version=study['version'],scope='full_validation',analyses=study['analyses'],study_file=STUDY,
        resolved_study=study,sample_ids=ids,samples=len(ids),seeds=study['seeds'],data=data,runs=runs,
        paper_baseline_version=study['baseline_version'],paper_baseline_dataset='SHD',source_sha256=sources,
        source_hash_encoding='CRLF_normalized_to_LF',environment=pilot.environment(device),training=False,
        official_test_opened=False,ordinary_inference=True,condition_definitions=study['conditions'])
    manifest['comparison_loss']=study['comparison_loss']
    manifest['intentional_training_difference']=study['intentional_training_difference']
    from scripts.adaptation_mt_tc.reuse import validate_reuse
    manifest['reuse']=validate_reuse(study,manifest)
    print('Reuse preflight passed: 18 MT prediction chunks and 6 shared feature chunks; new inference is MT-TC only.',flush=True)
    path=destination/'MANIFEST.json'
    if path.exists():pilot.validate_manifest(pilot.read_json(path),manifest)
    elif destination.exists() and any(p.name!='.analysis.lock' for p in destination.iterdir()):
        raise ValueError('Nonempty destination without matching manifest')
    print(f'SHD*: ALL {len(ids)} validation samples; 3 training seeds; 6 conditions/replicates. No training or official test access.')
    return study,destination,manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--output-root')
    flags=parser.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true');flags.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    if args.verify_only:
        print({k:len(v) for k,v in verify_package(args.output_root or load_study()['output_root']).items()});return 0
    study,destination,manifest=prepare(args)
    if args.check_only:
        print('Preflight passed; no inference or output writes.');return 0
    from scripts.adaptation_mt_tc.runtime import evaluate
    evaluate(study,destination,manifest)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr,flush=True);raise SystemExit(1)

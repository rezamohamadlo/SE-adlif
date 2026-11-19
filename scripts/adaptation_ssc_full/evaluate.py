"""Full SSC AMT/MT validation diagnostics. Agent may run --check-only; execution is user-launched."""
import argparse
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import yaml
from project_paths import local_path,portable_path,portable_config
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory,verify_package
from scripts.adaptation_ssc.data import metadata,SSCValidationData,check_legacy_contract

STUDY='config/studies/adaptation_ssc_amt_full_validation_v1.yaml'


def load_study():
    study=yaml.safe_load(local_path(STUDY).read_text(encoding='utf-8'))
    baseline=yaml.safe_load(local_path(study['inherits']).read_text(encoding='utf-8'))['paper_baseline']
    if study['dataset_section']!='SSC' or study['training'] or study['test_set_access']:
        raise ValueError('Only frozen official-validation SSC diagnostics are supported')
    if (study.get('sample_selection')!='all_official_validation_in_original_file_order' or
        study.get('diagnostic_scope')!='full_official_validation' or
        any(key in study for key in ('samples','max_samples'))):
        raise ValueError('Full SSC study must retain all official validation records')
    definitions=yaml.safe_load(local_path(study['feature_definition_source']).read_text(encoding='utf-8'))
    study['features']=definitions['features']
    study['conditions']=[list(c) for c in pilot.conditions('pc2',definitions)+pilot.conditions('pc3',definitions)[1:]]
    study.update(minimum_frames=baseline['datasets']['SSC']['pad_to_min_size'],
                 paper_baseline_version=baseline['version'],paper_baseline_dataset='SSC')
    return study,baseline


def selected_metrics(run,epochs):
    paths=list(local_path(run['cost_record']).parent.glob('logs/**/metrics.csv'))
    if len(paths)!=1:raise ValueError('Need one original SSC metric history')
    with paths[0].open(newline='',encoding='utf-8') as f:rows=[r for r in csv.DictReader(f) if r.get('val_acc')]
    if sorted(int(float(r['epoch'])) for r in rows)!=list(range(epochs)):
        raise ValueError('Incomplete/mixed SSC validation history')
    row=next(r for r in rows if int(float(r['epoch']))==run['selected_epoch'])
    if float(row['val_acc'])<max(float(r['val_acc']) for r in rows)-1e-7:
        raise ValueError('SSC checkpoint is not validation selected')
    return dict(path=portable_path(paths[0]),sha256=pilot.sha256(paths[0]),epoch=run['selected_epoch'],
                accuracy=float(row['val_acc']),loss=float(row['val_loss']))


def check_config(cfg,run,study,baseline):
    from scripts.adaptation_ssc.data import validate_config
    validate_config(cfg,baseline)
    section=baseline['datasets']['SSC']
    expected=dict(random_seed=run['seed'],model_variant=run['model'],two_layers=True,
        paper_baseline_version=baseline['version'],paper_baseline_dataset='SSC',evaluation_protocol=study['evaluation_protocol'],
        loss_agg=study['loss_agg'],dropout=baseline['common']['dropout'],early_stopping=baseline['reproduction_policy']['early_stopping'],
        **{k:section[k] for k in ('lr','batch_size','n_epochs')})
    for key,value in expected.items():
        if cfg.get(key)!=value:raise ValueError('Saved SSC baseline differs: '+key)
    if cfg.get('auto_regression',False) or cfg.get('training_objective','classification')!='classification':
        raise ValueError('Unexpected regression/training objective')
    cell={study['parent']:'mt_se_adlif',study['variant']:'amt_se_adlif'}[run['model']]
    for layer in ('l1','l2'):
        if cfg[layer]['cell']!=cell or cfg[layer]['n_neurons']!=section['reference_architecture']['neurons_per_layer']:
            raise ValueError('SSC registered cell/architecture differs')


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    study,baseline=load_study();seeds=args.seeds or study['seeds']
    if len(set(seeds))!=len(seeds) or not set(seeds).issubset(study['seeds']) or not seeds:
        raise ValueError('Seeds must be unique members of 42/123/456')
    seeds=sorted(seeds)
    root=args.output_root or study['output_root']
    if not args.output_root and seeds!=study['seeds']:root+='_seed_cohorts/'+'_'.join(map(str,seeds))
    destination=experiment_directory(root);device=torch.device(args.device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and (not torch.cuda.is_available() or (device.index or 0)>=torch.cuda.device_count())):
        raise ValueError('Requested device is unavailable')
    runs=[]
    for frozen in study['runs']:
        if frozen['seed'] not in seeds:continue
        run=dict(frozen);cost=pilot.read_json(local_path(run['cost_record'],must_exist=True))
        if (cost.get('status')!='complete' or cost['dataset']!='SSC' or cost['seed']!=run['seed'] or
            pilot.cost_model_identity(cost)!=run['model'] or portable_path(cost['checkpoint'])!=run['checkpoint'] or
            cost['checkpoint_sha256']!=run['checkpoint_sha256'] or cost['best_epoch']!=run['selected_epoch']):
            raise ValueError('SSC checkpoint/measurement identities disagree')
        path=local_path(run['checkpoint'],must_exist=True)
        if pilot.sha256(path)!=run['checkpoint_sha256']:raise ValueError('SSC checkpoint checksum differs')
        checkpoint=torch.load(path,weights_only=False,map_location='cpu')
        cfg=portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']),resolve=True))
        if pilot.effective_config(cfg)!=pilot.effective_config(portable_config(cost['saved_config'])) or checkpoint['epoch']!=run['selected_epoch']:
            raise ValueError('Saved checkpoint config/epoch differs')
        check_config(cfg,run,study,baseline)
        with torch.random.fork_rng(devices=[]),contextlib.redirect_stdout(io.StringIO()):model=MLPSNN(OmegaConf.create(cfg))
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.requires_grad_(False).eval()
        del model,checkpoint
        run.update(config=cfg,validation=selected_metrics(run,baseline['datasets']['SSC']['n_epochs']));runs.append(run)
        print(f"SSC prerequisite OK: {run['model']} seed {run['seed']}, epoch {run['selected_epoch']}",flush=True)
    if len(runs)!=2*len(seeds):raise ValueError('Incomplete SSC model/seed cohort')
    for seed in seeds:
        configs=[]
        for run in [r for r in runs if r['seed']==seed]:
            cfg=pilot.effective_config(run['config']);cfg.pop('model_variant',None)
            for layer in ('l1','l2'):cfg[layer]={k:v for k,v in cfg[layer].items() if k not in ('cell','mix_rho_fast','mix_rho_slow')}
            for key in ('gradient_clip_val','factor','patience'):cfg[key]=run['config'].get(key)
            configs.append(cfg)
        if configs[0]!=configs[1]:raise ValueError('SSC parent/variant common settings differ')
    meta,labels,speakers=metadata(runs[0]['config'],study,baseline)
    ids=list(meta['validation_indices']);samples=len(ids)
    data=SSCValidationData(runs[0]['config'],meta,labels,speakers,meta['validation_indices'])
    contract=check_legacy_contract(data)
    paths=[local_path(STUDY),local_path(study['inherits']),local_path(study['feature_definition_source']),
           local_path('project_paths.py'),local_path('datasets/ssc.py'),local_path('datasets/utils/transforms.py')]
    paths+=list((ROOT/'scripts/adaptation_ssc').glob('*.py'))
    paths+=list((ROOT/'scripts/adaptation_ssc_full').glob('*.py'))
    paths.append(local_path('config/path_aliases.yaml'))
    sources=pilot.source_identity()
    paths += [local_path(name) for name in sources]
    identity={portable_path(p):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(set(paths))}
    manifest=dict(version=study['version'],study_file=STUDY,resolved_study=study,scope='ssc_full_validation',
        samples=samples,sample_ids=ids,seeds=seeds,expected_training_seeds=study['seeds'],data=meta,runs=runs,
        paper_baseline_version=baseline['version'],paper_baseline_dataset='SSC',source_sha256=identity,
        source_hash_encoding='CRLF_normalized_to_LF',environment=pilot.environment(device),loader_preflight=contract,
        training=False,official_test_opened=False,ordinary_inference=True,condition_definitions=study['conditions'])
    p=destination/'MANIFEST.json'
    if p.exists():pilot.validate_manifest(pilot.read_json(p),manifest)
    elif destination.exists() and any(p.name!='.analysis.lock' for p in destination.iterdir()):
        raise ValueError('Nonempty SSC output directory without matching manifest')
    print(f'SSC: official validation {len(labels)} records under every condition ({samples} samples); seeds {seeds}. Loader matches legacy preprocessing on 3 records; no model forward.',flush=True)
    return study,destination,manifest,data


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--output-root')
    parser.add_argument('--seeds',type=int,nargs='+')
    flags=parser.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true');flags.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    if args.verify_only:
        study,_=load_study();root=args.output_root or study['output_root']
        seeds=sorted(args.seeds or study['seeds'])
        if len(set(seeds))!=len(seeds) or not set(seeds).issubset(study['seeds']):
            raise ValueError('Invalid verification seed cohort')
        if not args.output_root and seeds!=study['seeds']:root+='_seed_cohorts/'+'_'.join(map(str,seeds))
        print('Verified SSC full-validation outputs:',sum(len(v) for v in verify_package(root).values()));return 0
    study,destination,manifest,data=prepare(args)
    if args.check_only:print('SSC preflight passed. No inference, features, statistics, tests or result writes.');return 0
    from scripts.adaptation_ssc_full.runtime import execute
    execute(study,destination,manifest,data);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr,flush=True);raise SystemExit(1)

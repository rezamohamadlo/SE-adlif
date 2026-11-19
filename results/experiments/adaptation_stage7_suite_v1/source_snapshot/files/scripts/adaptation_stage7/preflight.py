"""Authenticate all 54 frozen checkpoints and existing validation evidence."""
import contextlib
import hashlib
import io

import torch
import yaml
from omegaconf import OmegaConf

from project_paths import ROOT, local_path, portable_config, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_gradient_pilot.methods import require, state_hash
from scripts.adaptation_stage7.data import Adapter

SPEC='config/studies/adaptation_stage7_suite_v1.yaml'


def source_manifest(entry):
    root=local_path(entry['root'],must_exist=True)
    require(pilot.sha256(root/'MANIFEST.json')==entry['manifest_sha256'] and
            pilot.sha256(root/'COMPLETE.json')==entry['complete_sha256'],'Source package identity differs')
    m,c=[pilot.read_json(root/n) for n in ('MANIFEST.json','COMPLETE.json')]
    require(c['status']=='complete' and c['manifest_sha256']==pilot.digest(m),'Source completion differs')
    for name,h in c['outputs'].items():require(pilot.sha256(local_path(name,must_exist=True))==h,'Source evidence changed: '+name)
    for name,h in m['source_sha256'].items():
        require(hashlib.sha256(local_path(name,must_exist=True).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,
                'Frozen scientific source changed: '+name)
    return m


def prepare(args):
    from models.pl_module import MLPSNN
    spec=yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    require(spec['datasets']==['SHD','SSC','ECG'] and spec['seeds']==[42,123,456] and len(spec['models'])==6 and
            spec['samples']==dict(SHD=40,SSC=70,ECG=30) and spec['repeat_records']==8 and
            spec['diagnostic_records']==4 and spec['optimization_steps']==6 and spec['event_fraction']==.01 and
            spec['random_replicates']==3 and not spec['training'] and not spec['official_test_access'],'Stage-7 scope differs')
    baseline=yaml.safe_load(local_path(spec['inherits']).read_text())['paper_baseline']
    root=experiment_directory(args.output_root or spec['output_root']);device=torch.device(args.device)
    require(device.type=='cuda' and torch.cuda.is_available(),'This registered suite uses CUDA float32')
    require(torch.get_default_dtype()==torch.float32,'Registered suite requires native float32 inputs/model weights')
    runs=[];adapters={};hashes={};source_records=[]
    for ds in spec['datasets']:
        sources=[source_manifest(e) for e in spec['sources'][ds]]
        canonical={}
        for entry,m in zip(spec['sources'][ds],sources):
            source_records.append(dict(dataset=ds,**entry))
            require(not root.is_relative_to(local_path(entry['root'])),'Output nested inside frozen source')
            for name,h in m['source_sha256'].items():
                require(name not in hashes or hashes[name]==h,'Incompatible source versions: '+name);hashes[name]=h
            for run in m['runs']:
                key=run['model'],run['seed']
                if key in canonical:
                    require(all(canonical[key][k]==run[k] for k in ('checkpoint_sha256','config','selected_epoch','validation')),'Shared parent differs')
                else:canonical[key]=run
        require(set(canonical)=={(m,s) for m in spec['models'] for s in spec['seeds']},'Need all 18 dataset checkpoints')
        for name in spec['models']:
            for seed in spec['seeds']:
                old=canonical[name,seed];path=local_path(old['checkpoint'],must_exist=True)
                require(pilot.sha256(path)==old['checkpoint_sha256'],'Checkpoint identity changed')
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                cfg=portable_config(OmegaConf.to_container(OmegaConf.create(checkpoint['hyper_parameters']['cfg']),resolve=True))
                require(cfg==old['config'] and checkpoint['epoch']==old['selected_epoch'] and
                        cfg['paper_baseline_version']==baseline['version'] and cfg['paper_baseline_dataset']==ds and
                        cfg['paper_baseline']==baseline,'Saved configuration/baseline identity differs')
                with torch.random.fork_rng(devices=[]),contextlib.redirect_stdout(io.StringIO()):model=MLPSNN(OmegaConf.create(cfg))
                model.load_state_dict(checkpoint['state_dict'],strict=True);model.requires_grad_(False).eval()
                runs.append(dict(dataset=ds,model=name,seed=seed,checkpoint=portable_path(path),checkpoint_sha256=old['checkpoint_sha256'],
                    selected_epoch=old['selected_epoch'],config=cfg,validation=old['validation'],model_state_sha256=state_hash(model)))
                del model,checkpoint
        adapters[ds]=Adapter(ds,sources[0],canonical[spec['models'][0],42]['config'],baseline,spec)
        print(f'Stage-7 {ds}: 18 checkpoints; {len(adapters[ds].selected)} gradient records; {len(adapters[ds].ids)} reference records.',flush=True)
    for path in list((ROOT/'scripts/adaptation_stage7').glob('*.py'))+[local_path(SPEC),local_path('tests/test_adaptation_stage7.py')]:
        hashes[portable_path(path)]=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    manifest=dict(version=spec['version'],specification=spec,output_root=portable_path(root),runs=runs,
        sources=source_records,source_sha256=hashes,source_hash_encoding='CRLF_normalized_to_LF',
        cohorts={ds:dict(selected=a.selected,repeat_ids=a.repeat_ids,diagnostic_ids=a.diagnostic_ids,data=a.meta) for ds,a in adapters.items()},
        paper_baseline_version=baseline['version'],precision='highest',training=False,official_test_opened=False,
        protocol='exploratory fixed-weight gradients and budgeted temporal diagnostic inputs; source_high baselines preserved',
        execution_authorization='User explicitly authorized agent commands/experiments to finish stage 7 on 2026-11-16')
    manifest['software']={k:v for k,v in pilot.environment(device).items() if k not in ('device','gpu')}
    if (root/'MANIFEST.json').exists():pilot.validate_manifest(pilot.read_json(root/'MANIFEST.json'),manifest)
    else:require(not root.exists() or all(p.name in ('.suite.lock','.analysis.lock') for p in root.iterdir()),
                 'Nonempty suite root without manifest')
    return dict(root=root,spec=spec,manifest=manifest,adapters=adapters,device=device)

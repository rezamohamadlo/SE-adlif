"""User-launched MT/MT-TC window diagnostics. --check-only never runs forward."""
import argparse
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import yaml
from project_paths import local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory,verify_package,verify_completion
from scripts.adaptation_mt_tc import evaluate as source_evaluator
from scripts.adaptation_tc_windows.metrics import describe

SPEC='config/studies/adaptation_mt_tc_windows_v1.yaml'
METRICS=('adjacent_js_nats','true_class_concentration','true_class_mean_probability','window_accuracy','top_class_switch_fraction')


def prepare(args,spec):
    # Existing prerequisite checks load trusted weights strictly, but do not forward.
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        study,_,source=source_evaluator.prepare(SimpleNamespace(device=args.device,output_root=spec['source']))
    for line in captured.getvalue().splitlines():
        if line.startswith('Prerequisite OK:'):print(line,flush=True)
    verify_package(spec['source'])
    if pilot.digest(source)!=spec['source_manifest_sha256'] or pilot.sha256(local_path(spec['source'])/'COMPLETE.json')!=spec['source_completion_sha256']:
        raise ValueError('Frozen source identity differs')
    if (spec['window_frames'],spec['stride_frames'],spec['minimum_windows'],spec['burn_in_frames'],spec['physical_bin_ms'])!=(5,5,2,10,4.):
        raise ValueError('Unsupported versioned window definition')
    if spec['training'] or spec['official_test_access'] or not spec['new_inference']:
        raise ValueError('Unexpected window-diagnostic execution definition')
    if source['data']['bin_ms']!=spec['physical_bin_ms']:raise ValueError('Physical frame width differs')
    destination=experiment_directory(args.output_root or spec['output_root'])
    if destination==experiment_directory(spec['source']):raise ValueError('Cannot overwrite source evaluation')
    paths=list((ROOT/'scripts/adaptation_tc_windows').glob('*.py'))+[local_path(SPEC)]
    identity=dict(source['source_sha256'])
    for p in paths:identity[portable_path(p)]=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    manifest=dict(version=spec['version'],specification=spec,source_manifest_sha256=pilot.digest(source),
        source_completion_sha256=spec['source_completion_sha256'],source_sha256=identity,
        sample_ids=source['sample_ids'],runs=source['runs'],data=source['data'],environment=source['environment'],
        paper_baseline_version=source['paper_baseline_version'],paper_baseline_dataset='SHD',
        training=False,new_inference=True,official_test_opened=False,scope='clean_validation_window_diagnostics')
    p=destination/'MANIFEST.json'
    if p.exists():pilot.validate_manifest(pilot.read_json(p),manifest)
    elif destination.exists() and any(p.name!='.analysis.lock' for p in destination.iterdir()):
        raise ValueError('Nonempty destination without matching manifest')
    return destination,manifest,study


def infer_windows(model,run,data,inputs,device,spec):
    import torch
    import torch.nn.functional as F
    from datasets.utils.pad_tensors import PadTensors
    from models.pl_module import aggregate_temporal_outputs
    rows=[];cfg=run['config'];size=cfg['dataset']['batch_size']
    with torch.inference_mode():
        for start in range(0,len(inputs),size):
            batch=inputs[start:start+size]
            x,targets,blocks=PadTensors()([data.model_item(sid,values) for sid,values,_ in batch])
            x,targets,blocks=x.to(device),targets.to(device),blocks.to(device)
            output=model(x)
            logits=aggregate_temporal_outputs(output,blocks,targets.size(1),cfg['loss_agg'])[:,1,:]
            losses=F.cross_entropy(logits,targets[:,1].long(),reduction='none')
            if not torch.isfinite(output).all() or not torch.isfinite(losses).all():raise ValueError('Nonfinite diagnostic output')
            probabilities=output.float().softmax(-1).cpu().numpy();pred=logits.argmax(-1).cpu().tolist()
            for i,(_,_,row) in enumerate(batch):
                descriptors=describe(probabilities[i],row['target'],row['original_support_frames'],spec['window_frames'],spec['burn_in_frames'])
                rows.append(dict(row,model=run['model'],seed=run['seed'],loss=float(losses[i].cpu()),
                    prediction=pred[i],correct=int(pred[i]==row['target']),**descriptors))
            print(f"{run['model']} seed {run['seed']}: {min(start+size,len(inputs))}/{len(inputs)}",flush=True)
    return rows


def verify_anchor(rows,run,spec,study):
    old=pilot.read_json(local_path(spec['source'])/'diagnostic_rows'/f"{run['model']}_seed_{run['seed']}_clean_rep_0.json")['value']
    old={r['sample_id']:r for r in old}
    if len(rows)!=1631 or {r['sample_id'] for r in rows}!=set(old):raise ValueError('Diagnostic sample coverage differs')
    for row in rows:
        expected=old[row['sample_id']]
        if any(row[k]!=expected[k] for k in ('target','speaker','input_sha256','prediction')):
            raise ValueError('Diagnostic clean input/prediction differs from completed evaluation')
    delta=float(np.mean([r['loss']-old[r['sample_id']]['loss'] for r in rows]))
    if abs(delta)>study['loss_tolerance']:raise ValueError('Diagnostic supervised loss differs')
    return dict(model=run['model'],seed=run['seed'],samples=len(rows),mean_loss_difference=delta,
                max_absolute_sample_loss_difference=max(abs(r['loss']-old[r['sample_id']]['loss']) for r in rows))


def summarize(rows,seeds):
    groups=[];differences=[]
    models=('MT_SE_adLIF','MT_TC_SE_adLIF')
    for seed in seeds:
        by_model={model:{r['sample_id']:r for r in rows if r['seed']==seed and r['model']==model} for model in models}
        if set(by_model[models[0]])!=set(by_model[models[1]]):raise ValueError('Window records do not pair')
        for model,index in by_model.items():
            groups.append(dict(seed=seed,model=model,**{key:dict(n_finite=sum(r[key] is not None for r in index.values()),
                mean=float(np.mean([r[key] for r in index.values() if r[key] is not None])) if any(r[key] is not None for r in index.values()) else None) for key in METRICS}))
        for key in METRICS:
            paired=[(by_model[models[0]][i][key],by_model[models[1]][i][key]) for i in by_model[models[0]]]
            paired=[b-a for a,b in paired if a is not None and b is not None]
            differences.append(dict(seed=seed,metric=key,n_finite_pairs=len(paired),
                mean_variant_minus_parent=float(np.mean(paired)) if paired else None))
    aggregate=[]
    for key in METRICS:
        valid=[d['mean_variant_minus_parent'] for d in differences if d['metric']==key and d['mean_variant_minus_parent'] is not None]
        aggregate.append(dict(metric=key,n_training_seeds=len(valid),mean=float(np.mean(valid)) if valid else None,
            sample_sd=float(np.std(valid,ddof=1)) if len(valid)>1 else None))
    return dict(groups=groups,paired_per_seed_differences=differences,across_seed_differences=aggregate,
                limitations=['Descriptive only: windows and samples are not independent training seeds.',
                'Stable low-confidence or consistently incorrect evidence is not stable useful class evidence; inspect true-class mean and window accuracy jointly.',
                'JS is an output diagnostic in nats, never a prediction-loss penalty here.',
                'No diagnostic-benefit regression: parent-derived confidence and prediction loss can be mathematically coupled.',
                'Exploratory after descriptive results; not causal or independent final-test confirmation.'])


def execute(destination,manifest,study,spec):
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    from scripts.adaptation_analysis.shd_data import SHDValidationData
    device=torch.device(manifest['environment']['device'])
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
    with pilot.study_lock(destination/'.analysis.lock'):
        p=destination/'MANIFEST.json'
        if p.exists():pilot.validate_manifest(pilot.read_json(p),manifest)
        else:pilot.write_json(p,manifest)
        if (destination/'COMPLETE.json').exists():
            verify_completion(destination,filename='COMPLETE.json');print('Already complete and verified.');return
        started=time.monotonic();raw=destination/'diagnostic_rows';raw.mkdir(exist_ok=True)
        data=SHDValidationData(manifest['runs'][0]['config'],sample_ids=manifest['sample_ids'],verify_hash=False)
        inputs=pilot.input_rows(data,data.sample_ids,('clean',0,'identity',1.),study,features=False)
        outputs=[p];all_rows=[];anchors=[]
        for run in manifest['runs']:
            pilot.write_json(destination/'PROGRESS.json',dict(status='running',model=run['model'],seed=run['seed']))
            path=raw/f"{run['model']}_seed_{run['seed']}.json"
            def produce():
                torch.manual_seed(run['seed'])
                checkpoint=torch.load(local_path(run['checkpoint']),weights_only=False,map_location='cpu')
                with contextlib.redirect_stdout(io.StringIO()):model=MLPSNN(OmegaConf.create(run['config']))
                model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
                model.requires_grad_(False).eval().to(device)
                value=infer_windows(model,run,data,inputs,device,spec)
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
                return value
            rows=pilot.cached_json(path,dict(manifest_sha256=pilot.digest(manifest),checkpoint_sha256=run['checkpoint_sha256']),produce)
            anchors.append(verify_anchor(rows,run,spec,study));all_rows+=rows;outputs.append(path)
        summary=summarize(all_rows,study['seeds'])
        summary.update(version=spec['version'],dataset='SHD* clean validation',samples=len(manifest['sample_ids']),
                       training_seeds=study['seeds'],window_frames=5,stride_frames=5,burn_in_frames=10)
        pilot.write_json(destination/'SUMMARY.json',summary);pilot.write_json(destination/'ANCHORS.json',anchors)
        text=['# MT/MT-TC clean window evidence diagnostics','','Nonoverlapping 5-frame windows (20 ms), after burn-in and excluding synthetic padding. Each sample has equal weight within a model/seed; windows are not pooled as independent observations. Negative MT-TC minus MT JS means smoother adjacent probability evidence, not necessarily correct evidence.','','| Seed | Metric | Finite pairs | MT-TC minus MT |','|---|---|---:|---:|']
        for d in summary['paired_per_seed_differences']:
            value='undefined' if d['mean_variant_minus_parent'] is None else f"{d['mean_variant_minus_parent']:+.8f}"
            text.append(f"| {d['seed']} | {d['metric']} | {d['n_finite_pairs']} | {value} |")
        text+=['','## Across training seeds','','Mean and sample SD of per-seed paired differences; no window/sample pooling. JS is in nats; other metrics are dimensionless fractions (accuracy difference .01 = 1 pp). SD is not a confidence interval.','','| Metric | Training seeds | Mean MT-TC minus MT | Sample SD |','|---|---:|---:|---:|']
        for d in summary['across_seed_differences']:
            mean='undefined' if d['mean'] is None else f"{d['mean']:+.8f}"
            sd='undefined' if d['sample_sd'] is None else f"{d['sample_sd']:.8f}"
            text.append(f"| {d['metric']} | {d['n_training_seeds']} | {mean} | {sd} |")
        text+=['','Limitations:']+['- '+s for s in summary['limitations']]
        (destination/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
        outputs += [destination/'SUMMARY.json',destination/'SUMMARY.md',destination/'ANCHORS.json']
        pilot.write_json(destination/'COMPLETE.json',dict(status='complete',manifest_sha256=pilot.digest(manifest),
            completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            outputs={portable_path(p):pilot.sha256(p) for p in outputs}))
        pilot.write_json(destination/'PROGRESS.json',dict(status='complete'))
        print('Window diagnostics completed:',portable_path(destination))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda:0');parser.add_argument('--output-root')
    flags=parser.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true');flags.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv);spec=yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    if args.verify_only:
        print('Verified:',len(verify_completion(experiment_directory(args.output_root or spec['output_root']),filename='COMPLETE.json')));return 0
    destination,manifest,study=prepare(args,spec)
    if args.check_only:
        print('Window preflight passed: 1631 clean samples, 6 checkpoints; no forward, diagnostic calculations, tests or output writes.');return 0
    execute(destination,manifest,study,spec);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr,flush=True);raise SystemExit(1)

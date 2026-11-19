"""Resumable fixed-weight gradients, projected diagnostic inputs and transfer controls."""
from datetime import datetime,timezone
import json
import time

import numpy as np
import torch

from project_paths import local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_pilot.engine import load_model
from scripts.adaptation_gradient_pilot.methods import array_hash,cached_array,require,require_frozen
from scripts.adaptation_gradient_pilot.preflight import read_cache
from scripts.adaptation_gradient_strict.methods import gradient_comparison
from scripts.adaptation_stage7.methods import forward,backward,budget_for,propose,random_relocation,validate_native
from scripts.ssc_resume_io import AtomicJSONWriter


def seal(root,manifest,outputs,**extra):
    writer=AtomicJSONWriter(root,pilot.write_json)
    writer(root/'COMPLETE.json',dict(status='complete',manifest_sha256=pilot.digest(manifest),
        completed_utc=datetime.now(timezone.utc).isoformat(),outputs={portable_path(p):pilot.sha256(p) for p in dict.fromkeys(outputs)},**extra))


def verify(root):
    root=local_path(root,must_exist=True);m,c=[pilot.read_json(root/n) for n in ('MANIFEST.json','COMPLETE.json')]
    require(c['status']=='complete' and c['manifest_sha256']==pilot.digest(m) and
            c['outputs'].get(portable_path(root/'MANIFEST.json'))==pilot.sha256(root/'MANIFEST.json'),'Completion identity differs')
    for name,h in c['outputs'].items():
        path=local_path(name,must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path)==h,'Output changed: '+name)
    return c


class Cache:
    def __init__(self,root,manifest):
        self.root=root;self.manifest=manifest;self.outputs=[];self.write=AtomicJSONWriter(root,pilot.write_json)
        if (root/'MANIFEST.json').exists():pilot.validate_manifest(pilot.read_json(root/'MANIFEST.json'),manifest)
        else:
            require(not root.exists() or all(p.name in ('.analysis.lock','.suite.lock') for p in root.iterdir()),
                    'Nonempty cache without manifest; preserve evidence and choose a new study root')
            root.mkdir(parents=True,exist_ok=True);self.write(root/'MANIFEST.json',manifest)
        self.outputs.append(root/'MANIFEST.json')

    def json(self,suffix,identity,produce):
        path=self.root/suffix;path.parent.mkdir(parents=True,exist_ok=True)
        contract=dict(manifest_sha256=pilot.digest(self.manifest),**identity)
        if path.exists():record=read_cache(path);pilot.validate_manifest(record['identity'],contract);value=record['value']
        else:
            value=produce();self.write(path,dict(identity=contract,value=value,value_sha256=pilot.digest(value)))
        self.outputs.append(path);return value

    def array(self,suffix,identity,produce):
        value,arrays,paths=cached_array(self.root/suffix,dict(manifest_sha256=pilot.digest(self.manifest),**identity),produce,self.write)
        self.outputs+=paths;return value,arrays

    def report(self,name,value):
        path=self.root/name;self.write(path,value);self.outputs.append(path)

    def progress(self,phase,**extra):
        self.write(self.root/'PROGRESS.json',dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra))


def job_root(root,run):return root/'jobs'/run['dataset']/run['model']/f"seed_{run['seed']}"


def run_job(prepared,run):
    root,spec,global_m,device=[prepared[k] for k in ('root','spec','manifest','device')]
    a=prepared['adapters'][run['dataset']];destination=job_root(root,run)
    if (destination/'COMPLETE.json').exists():verify(destination);return
    manifest=dict(version=spec['version'],suite_manifest_sha256=pilot.digest(global_m),run=run,
        cohort=global_m['cohorts'][run['dataset']],specification=spec,precision='highest',training=False)
    with pilot.study_lock(destination/'.analysis.lock'):
        cache=Cache(destination,manifest);start=time.monotonic();model=load_model(run,device)
        try:
            cache.report('RUNTIME_PROVENANCE.json',dict(environment=pilot.environment(device),matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                authorization='Explicit user permission for agent execution to finish stage 7, 2026-11-16'))
            full=[];size=run['config']['dataset']['batch_size']
            for b,start_id in enumerate(range(0,len(a.ids),size)):
                ids=a.ids[start_id:start_id+size];cache.progress('strict_full_validation_reference',batch=b+1)
                rows=cache.json(f'reference/batch_{b}.json',dict(sample_ids=ids),
                    lambda:[dict(sample_id=sid,**s) for sid,s in zip(ids,forward(model,a.batch(ids),run['config'],device))])
                require([r['sample_id'] for r in rows]==ids,'Reference sample order differs');full+=rows
            cache.report('STRICT_REFERENCE.json',dict(records=len(full),valid_steps=sum(r['valid_steps'] for r in full),
                record_mean_loss=sum(r['loss'] for r in full)/len(full),
                step_weighted_accuracy=sum(r['correct_steps'] for r in full)/sum(r['valid_steps'] for r in full),
                saved_training_validation=run['validation'],
                interpretation='New highest-precision reference; does not replace frozen source_high evidence'))
            physical={sid:a.input(sid) for sid in a.selected};gradients={};records=[];repeats=[]
            for number,sid in enumerate(a.selected,1):
                x=physical[sid];batch=a.batch([sid],{sid:x})
                for rep in range(2 if sid in a.repeat_ids else 1):
                    cache.progress('input_gradients',record=number,total=len(a.selected),repeat=rep)
                    def produce_gradient():
                        v,g=backward(model,batch,run['config'],device,x,a.burn,spec,run['model_state_sha256'])
                        v.update(sample_id=sid,repeat=rep,input_sha256=array_hash(x))
                        return v,dict(gradient=g)
                    value,arrays=cache.array(f'gradients/sample_{sid}_rep_{rep}.npz',dict(sample_id=sid,repeat=rep,input_sha256=array_hash(x)),produce_gradient)
                    g=arrays['gradient'];require(array_hash(g)==value['gradient_sha256'] and np.isfinite(g).all(),'Gradient cache differs')
                    gradients[sid,rep]=(value,g);records.append(value)
                if sid in a.repeat_ids:
                    first,second=[gradients[sid,rep] for rep in [0,1]]
                    metric=gradient_comparison(first[1],second[1],len(x),spec)
                    metric.update(sample_id=sid,forward_passed=all(r[0]['autograd_forward_check']['passed'] for r in [first,second]))
                    metric['passed']=metric['passed'] and metric['forward_passed'];repeats.append(metric)
            cache.report('GRADIENT_SUMMARIES.json',records);cache.report('REPEAT_CHECKS.json',repeats)
            diagnostics=[]
            for sid in a.diagnostic_ids:
                original=physical[sid];budget=budget_for(original,a.domain,spec['event_fraction'])
                def optimize():
                    current=original.copy();current_score=gradients[sid,0][0]['clean'];trace=[];allocation=[]
                    stable=next(r['passed'] for r in repeats if r['sample_id']==sid)
                    if not stable:return dict(sample_id=sid,status='skipped_unstable_gradient',budget=budget,moves=0,allocation=[],trace=[],clean=current_score,optimized=current_score),dict(input=current)
                    for iteration in range(spec['optimization_steps']):
                        path=cache.root/'optimization_candidates'/f'sample_{sid}_step_{iteration}.npz'
                        def make_step():
                            if iteration==0:g=gradients[sid,0][1];check=gradients[sid,0][0]['autograd_forward_check']
                            else:
                                value,g=backward(model,a.batch([sid],{sid:current}),run['config'],device,current,a.burn,spec,run['model_state_sha256'])
                                check=value['autograd_forward_check']
                            candidate,proposed_alloc=propose(original,g,a.domain,budget)
                            moves=validate_native(original,candidate,a.domain,budget)
                            feasible=moves>0 and not np.array_equal(candidate,current) and check['passed']
                            score=forward(model,a.batch([sid],{sid:candidate}),run['config'],device)[0] if feasible else current_score
                            threshold=spec['score_atol']+spec['score_rtol']*abs(current_score['loss'])
                            accepted=feasible and score['loss']>current_score['loss']+threshold
                            status=('autograd_forward_failed' if not check['passed'] else 'accepted' if accepted else
                                'rejected_no_measured_ce_increase' if feasible else 'no_new_feasible_positive_surrogate_move')
                            return dict(score=score,accepted=bool(accepted),moves=moves,allocation=proposed_alloc,status=status,
                                previous_loss=current_score['loss'],acceptance_bound=threshold,autograd_forward_check=check),dict(input=candidate,gradient=g)
                        step,payload=cache.array(path.relative_to(cache.root).as_posix(),dict(sample_id=sid,iteration=iteration,
                            current_sha256=array_hash(current),current_score_sha256=pilot.digest(current_score)),make_step)
                        trace.append(dict(iteration=iteration,status=step['status'],loss=step['score']['loss'],
                            previous_loss=step['previous_loss'],acceptance_bound=step['acceptance_bound'],moves=step['moves'],
                            candidate_sha256=array_hash(payload['input']),candidate_path=portable_path(path)))
                        if not step['accepted']:break
                        current,current_score,allocation=payload['input'],step['score'],step['allocation']
                    moves=validate_native(original,current,a.domain,budget)
                    return dict(sample_id=sid,status='improved' if moves else 'no_accepted_improvement',budget=budget,moves=moves,
                        allocation=allocation,trace=trace,clean=gradients[sid,0][0]['clean'],optimized=current_score),dict(input=current)
                cache.progress('constrained_input_optimization',sample_id=sid)
                value,payload=cache.array(f'optimized/sample_{sid}.npz',dict(sample_id=sid,original_sha256=array_hash(original)),optimize)
                require(validate_native(original,payload['input'],a.domain,budget)==value['moves'],'Optimized input budget differs')
                for row in value['trace']:
                    path=local_path(row['candidate_path'],must_exist=True)
                    require(path.is_relative_to(destination),'Candidate trace escapes its job')
                    record=read_cache(path.with_suffix('.json'))
                    require(pilot.sha256(path)==record['artifact_sha256'],'Candidate trace file changed')
                    cache.outputs += [path,path.with_suffix('.json')]
                diagnostics.append(value)
                for rep in range(spec['random_replicates']):
                    seed=int(pilot.digest([spec['version'],run['dataset'],run['model'],run['seed'],sid,rep])[:16],16)
                    def make_control():
                        x=random_relocation(original,value['allocation'],a.domain,seed)
                        return dict(sample_id=sid,replicate=rep,seed=seed,moves=value['moves']),dict(input=x)
                    control,arrays=cache.array(f'controls/sample_{sid}_rep_{rep}.npz',dict(sample_id=sid,replicate=rep,seed=seed),make_control)
                    require(validate_native(original,arrays['input'],a.domain,budget)==value['moves'],'Random relocation not budget-matched')
            require_frozen(model,run['model_state_sha256'])
            cache.report('DIAGNOSTICS.json',diagnostics)
            cache.report('FROZEN_STATE.json',dict(status='unchanged',model_state_sha256=run['model_state_sha256'],all_parameter_gradients_none=True))
            cache.report('SUMMARY.json',dict(dataset=run['dataset'],model=run['model'],seed=run['seed'],
                reference_records=len(full),gradient_records=len(records),physical_records=len(a.selected),repeat_checks=len(repeats),
                repeat_failures=sum(not r['passed'] for r in repeats),autograd_failures=sum(not r['autograd_forward_check']['passed'] for r in records),
                optimized_attempts=len(diagnostics),accepted_diagnostic_inputs=sum(r['moves']>0 for r in diagnostics),
                optimization_autograd_failures=sum(any(step['status']=='autograd_forward_failed' for step in r['trace']) for r in diagnostics),
                skipped_unstable_inputs=sum(r['status']=='skipped_unstable_gradient' for r in diagnostics),training=False,stage7_complete=False))
            seal(destination,manifest,cache.outputs,elapsed_seconds=time.monotonic()-start)
            cache.progress('complete');print('Completed',run['dataset'],run['model'],run['seed'],flush=True)
        finally:
            del model;torch.cuda.empty_cache()


def load_input(path):
    with np.load(path,allow_pickle=False) as z:return z['input'].copy()


def run_transfer(prepared,dataset,seed):
    root,spec,m,device=[prepared[k] for k in ('root','spec','manifest','device')];a=prepared['adapters'][dataset]
    runs=[r for r in m['runs'] if r['dataset']==dataset and r['seed']==seed]
    if not all((job_root(root,r)/'COMPLETE.json').exists() for r in runs):return
    destination=root/'transfers'/dataset/f'seed_{seed}'
    if (destination/'COMPLETE.json').exists():verify(destination);return
    dependencies={r['model']:pilot.sha256(job_root(root,r)/'COMPLETE.json') for r in runs}
    manifest=dict(version=spec['version'],suite_manifest_sha256=pilot.digest(m),dataset=dataset,seed=seed,
        dependencies=dependencies,diagnostic_ids=a.diagnostic_ids,training=False)
    with pilot.study_lock(destination/'.analysis.lock'):
        cache=Cache(destination,manifest);all_rows=[]
        for target in runs:
            model=load_model(target,device)
            try:
                for sid in a.diagnostic_ids:
                    original=a.input(sid);batch=a.batch([sid],{sid:original})
                    clean=cache.json(f"clean/{target['model']}_{sid}.json",dict(target=target['model'],sample_id=sid),
                        lambda:forward(model,batch,target['config'],device)[0])
                    expected=read_cache(job_root(root,target)/'gradients'/f'sample_{sid}_rep_0.json')['value']['clean']
                    require(clean['predictions']==expected['predictions'] and
                            abs(clean['loss']-expected['loss'])<=spec['score_atol']+spec['score_rtol']*abs(expected['loss']),
                            'Transfer fresh clean reference failed; no target comparison can bypass it')
                    for origin in runs:
                        origin_root=job_root(root,origin);diagnostic=next(r for r in pilot.read_json(origin_root/'DIAGNOSTICS.json') if r['sample_id']==sid)
                        for condition,rep in [('optimized',0)]+[('random',i) for i in range(spec['random_replicates'])]:
                            source=origin_root/('optimized' if condition=='optimized' else 'controls')/(f'sample_{sid}.npz' if condition=='optimized' else f'sample_{sid}_rep_{rep}.npz')
                            x=load_input(source);require(validate_native(original,x,a.domain,diagnostic['budget'])==diagnostic['moves'],'Transfer input domain/budget failed')
                            identity=dict(target=target['model'],origin=origin['model'],sample_id=sid,condition=condition,replicate=rep,
                                input_sha256=array_hash(x),source_artifact_sha256=pilot.sha256(source))
                            cache.progress('shared_diagnostic_transfer',**identity)
                            score=cache.json(f"scores/{target['model']}/{origin['model']}_{sid}_{condition}_{rep}.json",identity,
                                lambda:forward(model,a.batch([sid],{sid:x}),target['config'],device)[0])
                            all_rows.append(dict(dataset=dataset,seed=seed,target=target['model'],origin=origin['model'],sample_id=sid,
                                condition=condition,replicate=rep,input_sha256=array_hash(x),moves=diagnostic['moves'],
                                loss=score['loss'],clean_loss=clean['loss'],loss_change=score['loss']-clean['loss'],accuracy=score['accuracy']))
                require_frozen(model,target['model_state_sha256'])
            finally:del model;torch.cuda.empty_cache()
        require(len(all_rows)==6*6*4*4,'Transfer comparison coverage differs')
        cache.report('TRANSFER_ROWS.json',all_rows)
        seal(destination,manifest,cache.outputs);cache.progress('complete')

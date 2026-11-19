"""User-launched strict anchors, repeated input gradients and stability reports."""
from datetime import datetime, timezone
from collections import Counter
import json
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from datasets.utils.pad_tensors import PadTensors
from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_batch_precision.contracts import batch_identity, precision_mode
from scripts.adaptation_gradient_pilot.engine import load_model, ordinary
from scripts.adaptation_gradient_pilot.methods import (
    atomic_copy, array_hash, cached_array, count_hash, require, require_frozen,
)
from scripts.adaptation_gradient_pilot.preflight import guard, read_cache
from scripts.adaptation_gradient_strict.methods import (
    gradient_comparison, input_gradient, require_anchor, score_check, validate_gradient,
)
from scripts.ssc_resume_io import AtomicJSONWriter


def verify(directory):
    root=experiment_directory(directory)
    m,c=[pilot.read_json(root/n) for n in ('MANIFEST.json','COMPLETE.json')]
    require(c['status']=='complete' and c['manifest_sha256']==pilot.digest(m) and
            c['outputs'].get(portable_path(root/'MANIFEST.json'))==pilot.sha256(root/'MANIFEST.json'),
            'Strict pilot completion identity differs')
    for name,sha in c['outputs'].items():
        path=local_path(name,must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path)==sha,'Strict pilot output changed: '+name)
    s=pilot.read_json(root/'SUMMARY.json')
    require(s['gradient_records']==164 and s['samples']==40 and s['natural_repeat_pairs']==80 and
            s['padding_repeat_pairs']==2 and s['padding_comparisons']==4 and s['cross_precision_comparisons']==80 and
            not s['stage7_complete'] and not s['event_deletion'],'Strict pilot scope/coverage differs')
    checks=pilot.read_json(root/'STABILITY_CHECKS.json')
    require(Counter(r['kind'] for r in checks)==dict(natural_repeat=80,padding_repeat=2,padding_layout=4,cross_precision_legacy=80),
            'Strict pilot comparison coverage differs')
    gradients=pilot.read_json(root/'GRADIENT_SUMMARIES.json')
    require(len(gradients)==164 and len(pilot.read_json(root/'GRADIENT_INDEX.json'))==164,
            'Strict gradient array coverage differs')
    stable=(all(r['passed'] for r in checks if r['required_for_stability']) and
            all(r['autograd_forward_check']['passed'] and r['strict_singleton_reference_check']['passed'] for r in gradients))
    require(s['stability_criteria_passed']==c['stability_criteria_passed']==stable,'Stored stability outcome differs')
    anchors=pilot.read_json(root/'FULL_VALIDATION_ANCHORS.json')
    require(len(anchors)==2 and all(r['status']=='passed' and r['samples']==1631 for r in anchors),
            'Same-precision full-validation anchors did not pass')
    require(pilot.read_json(root/'CONTRACT_TESTS.json')['status']=='passed','No passing user contracts')
    # Completion authenticates saved negative findings too; it never implies stability passed.
    return c


def execute(prepared):
    root,spec,m,data,source,legacy=[prepared[k] for k in
        ('root','spec','manifest','data','reference_root','legacy_root')]
    started=time.monotonic();device=torch.device(m['environment']['device']);models={}
    with pilot.study_lock(root/'.analysis.lock'):
        guard(root,m)
        if (root/'COMPLETE.json').exists():
            print('Already complete; verified outputs:',len(verify(root)['outputs']));return
        write=AtomicJSONWriter(root,pilot.write_json)
        if not (root/'MANIFEST.json').exists():write(root/'MANIFEST.json',m)
        identity=pilot.digest(m);outputs=[root/'MANIFEST.json']
        def progress(phase,**kw):
            write(root/'PROGRESS.json',dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**kw))
        def cached(path,contract,produce):
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                record=read_cache(path);pilot.validate_manifest(record['identity'],contract);value=record['value']
            else:
                value=produce();write(path,dict(identity=contract,value=value,value_sha256=pilot.digest(value)))
            outputs.append(path);return value
        try:
            progress('user_contract_tests')
            tests=subprocess.run([sys.executable,'-m','unittest','tests.test_adaptation_gradient_strict'],cwd=ROOT,
                capture_output=True,text=True,encoding='utf-8',errors='replace')
            log=root/'CONTRACT_TESTS.txt';log.write_text(tests.stdout+tests.stderr,encoding='utf-8')
            write(root/'CONTRACT_TESTS.json',dict(status='passed' if tests.returncode==0 else 'failed',
                returncode=tests.returncode,log_sha256=pilot.sha256(log)))
            outputs += [log,root/'CONTRACT_TESTS.json']
            require(tests.returncode==0,'Strict gradient contracts failed; inspect CONTRACT_TESTS.txt')
            require(pilot.sha256(source/'MANIFEST.json')==spec['reference_manifest_sha256'] and
                    pilot.sha256(source/'COMPLETE.json')==spec['reference_complete_sha256'],
                    'Frozen strict reference identity changed after preflight')
            source_spec=pilot.read_json(source/'MANIFEST.json')['specification']
            require(pilot.sha256(legacy/'MANIFEST.json')==source_spec['pilot_manifest_sha256'] and
                    pilot.sha256(legacy/'COMPLETE.json')==source_spec['pilot_complete_sha256'],
                    'Frozen source-high pilot identity changed after preflight')
            path_index={}
            def copy_selected(study,suffix,prefix):
                path=study/suffix;destination=root/'shared'/prefix/suffix
                ledger=pilot.read_json(study/'COMPLETE.json')['outputs']
                expected=pilot.sha256(study/'COMPLETE.json') if suffix=='COMPLETE.json' else ledger[portable_path(path)]
                require(pilot.sha256(path)==expected,'Source subset changed')
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():atomic_copy(path,destination)
                require(pilot.sha256(destination)==expected,'Source snapshot differs')
                outputs.append(destination);path_index[portable_path(path)]=portable_path(destination)
                return destination
            for suffix in ['MANIFEST.json','COMPLETE.json','SCORES.json','FULL_VALIDATION_CHECKS.json']:
                copy_selected(source,suffix,'strict_reference')
            for run in m['runs']:
                for b in range(len(m['original_batch_ids'])):
                    copy_selected(source,f"jobs/{run['model']}/{spec['precision_mode']}/original_batch_{b}.json",'strict_reference')
            for suffix in ['MANIFEST.json','COMPLETE.json','GRADIENT_INDEX.json','GRADIENT_SUMMARIES.json']:
                copy_selected(legacy,suffix,'source_high_pilot')
            old_index=pilot.read_json(legacy/'GRADIENT_INDEX.json')
            old_arrays={}
            for row in old_index:
                original=local_path(row['path'],must_exist=True)
                require(original.is_relative_to(legacy) and pilot.sha256(original)==row['sha256'],'Legacy gradient identity differs')
                suffix=original.relative_to(legacy).as_posix()
                copied=copy_selected(legacy,suffix,'source_high_pilot')
                with np.load(copied,allow_pickle=False) as z:
                    g=z['gradient'].copy();summary=json.loads(z['summary_json'].item())
                require(array_hash(g)==summary['gradient_sha256'],'Legacy embedded gradient hash differs')
                old_arrays[row['model'],row['sample_id']]=g
            write(root/'shared/PATH_INDEX.json',dict(path_index=path_index,scope='authenticated selected metadata/references and all 80 legacy gradient arrays; manifests unchanged'))
            outputs.append(root/'shared/PATH_INDEX.json')
            legacy_summaries={(r['model'],r['sample_id']):r for r in pilot.read_json(root/'shared/source_high_pilot/GRADIENT_SUMMARIES.json')}
            strict_single={(r['model'],r['sample_id']):r for r in pilot.read_json(root/'shared/strict_reference/SCORES.json')
                           if r['mode']==spec['precision_mode'] and r['layout']=='singleton_natural'}
            references={}
            for run in m['runs']:
                rows=[]
                for b,ids in enumerate(m['original_batch_ids']):
                    cache=read_cache(root/'shared/strict_reference/jobs'/run['model']/spec['precision_mode']/f'original_batch_{b}.json')
                    rows += [dict(sample_id=sid,**r) for sid,r in zip(ids,cache['value'])]
                references[run['model']]=rows
            counts={sid:data.load(sid)['counts'] for sid in m['pilot_ids']}
            for run in m['runs']:
                for sid in m['pilot_ids']:
                    require(count_hash(counts[sid])==legacy_summaries[run['model'],sid]['input_sha256'],
                            'Pilot physical input differs from legacy identity')
            for run in m['runs']:models[run['model']]=load_model(run,device)
            values={};arrays={};gradient_index=[]
            with precision_mode(spec['precision_mode'],m['source_backend']) as actual_backend:
                pilot.validate_manifest(actual_backend,m['backend'])
                anchors=[]
                # Both full-validation anchors must pass before any backward.
                for run in m['runs']:
                    model=models[run['model']];anchor_rows=[]
                    for b,ids in enumerate(m['original_batch_ids']):
                        progress('same_mode_full_validation',model=run['model'],batch=b+1,total=7)
                        print(f"Strict anchor {run['model']}: batch {b+1}/7",flush=True)
                        def produce_anchor():
                            items=[]
                            for sid in ids:
                                item=data.load(sid);items.append(data.model_item(sid,item['counts']))
                            batch=PadTensors()(items)
                            reference=read_cache(root/'shared/strict_reference/jobs'/run['model']/spec['precision_mode']/f'original_batch_{b}.json')
                            require(batch_identity(batch)==reference['identity']['batch'],
                                    'Same-mode full-validation input/target/block hashes differ')
                            score=ordinary(model,batch,run['config'],device)
                            return [dict(sample_id=sid,**r) for sid,r in zip(ids,score)]
                        path=root/'anchors'/run['model']/'seed_42'/f'batch_{b}.json'
                        anchor_rows += cached(path,dict(manifest_sha256=identity,model=run['model'],
                            checkpoint_sha256=run['checkpoint_sha256'],backend=actual_backend,sample_ids=ids),produce_anchor)
                    result=require_anchor(anchor_rows,references[run['model']],spec)
                    anchors.append(dict(model=run['model'],**result));require_frozen(model,run['model_state_sha256'])
                write(root/'FULL_VALIDATION_ANCHORS.json',anchors);outputs.append(root/'FULL_VALIDATION_ANCHORS.json')
                for run in m['runs']:
                    model=models[run['model']]
                    jobs=[('singleton_natural',sid,rep) for sid in m['pilot_ids'] for rep in range(spec['repeats'])]
                    jobs += [('singleton_original_padding_focus',spec['focus_sample_id'],rep) for rep in range(spec['repeats'])]
                    for layout,sid,rep in jobs:
                        x=counts[sid];batch=PadTensors()([data.model_item(sid,x)])
                        require(batch_identity(batch)==m['input_identities'][str(sid)],'Natural singleton input/target/mask changed')
                        if layout=='singleton_original_padding_focus':
                            extra=m['focus_padded_frames']-batch[0].shape[1]
                            require(extra>=0,'Original padding shorter than singleton')
                            batch=(F.pad(batch[0],(0,0,0,extra)),batch[1],F.pad(batch[2],(0,extra)))
                            old_manifest=pilot.read_json(root/'shared/strict_reference/MANIFEST.json')
                            require(batch_identity(batch)==old_manifest['focus_input_layouts']['singleton_original_padding'],
                                    'Focus padded input/target/mask differs from numerical diagnostic')
                        progress('input_backward_repeats',model=run['model'],layout=layout,sample_id=sid,repeat=rep,
                                 completed=len(gradient_index),total=m['expected_gradients'])
                        print(f"Strict input gradient {len(gradient_index)+1}/164: {run['model']} {layout} id={sid} repeat={rep}",flush=True)
                        path=root/'gradients'/run['model']/'seed_42'/layout/f'sample_{sid}_rep_{rep}.npz'
                        contract=dict(manifest_sha256=identity,checkpoint_sha256=run['checkpoint_sha256'],backend=actual_backend,
                            model=run['model'],seed=42,layout=layout,sample_id=sid,repeat=rep,
                            physical_input_sha256=count_hash(x),batch=batch_identity(batch))
                        def produce_gradient():
                            value,payload=input_gradient(model,batch,run['config'],device,x,spec,run['model_state_sha256'])
                            ref=strict_single[run['model'],sid]
                            value.update(model=run['model'],seed=42,sample_id=sid,layout=layout,repeat=rep,
                                input_sha256=count_hash(x),strict_singleton_reference_check=score_check(value['clean'],ref,spec),
                                reference_mode=spec['precision_mode'])
                            return value,payload
                        value,payload,paths=cached_array(path,contract,produce_gradient,write);outputs += paths
                        validate_gradient(value,payload,batch,x,run['model_state_sha256'])
                        require_frozen(model,run['model_state_sha256'])
                        key=(run['model'],layout,sid,rep);values[key]=value;arrays[key]=payload['gradient']
                        gradient_index.append(dict(model=run['model'],seed=42,layout=layout,sample_id=sid,repeat=rep,
                            path=portable_path(path),sha256=pilot.sha256(path)))
            require(len(values)==164,'Gradient job coverage differs')
            checks=[]
            def add_comparison(kind,model,sid,a,b,first_score,second_score,rep=None):
                metric=gradient_comparison(a,b,len(counts[sid]),spec)
                forward=score_check(second_score,first_score,spec)
                checks.append(dict(kind=kind,model=model,seed=42,sample_id=sid,repeat=rep,
                    gradient_passed=metric.pop('passed'),**metric,
                    **{'score_'+k:v for k,v in forward.items()},
                    required_for_stability=kind!='cross_precision_legacy'))
                checks[-1]['passed']=checks[-1]['gradient_passed'] and forward['passed']
            for run in m['runs']:
                model=run['model']
                for sid in m['pilot_ids']:
                    a,b=[(model,'singleton_natural',sid,rep) for rep in range(2)]
                    add_comparison('natural_repeat',model,sid,arrays[a],arrays[b],values[a]['clean'],values[b]['clean'])
                    add_comparison('cross_precision_legacy',model,sid,old_arrays[model,sid],arrays[a],
                        legacy_summaries[model,sid]['clean'],values[a]['clean'])
                sid=spec['focus_sample_id']
                a,b=[(model,'singleton_original_padding_focus',sid,rep) for rep in range(2)]
                add_comparison('padding_repeat',model,sid,arrays[a],arrays[b],values[a]['clean'],values[b]['clean'])
                for rep in range(2):
                    a,b=(model,'singleton_natural',sid,rep),(model,'singleton_original_padding_focus',sid,rep)
                    add_comparison('padding_layout',model,sid,arrays[a],arrays[b],values[a]['clean'],values[b]['clean'],rep)
            required=[r for r in checks if r['required_for_stability']]
            require(Counter(r['kind'] for r in checks)==dict(natural_repeat=80,padding_repeat=2,padding_layout=4,cross_precision_legacy=80),
                    'Stability comparison coverage differs')
            autograd_failures=[dict(model=v['model'],layout=v['layout'],sample_id=v['sample_id'],repeat=v['repeat'])
                for v in values.values() if not v['autograd_forward_check']['passed']]
            reference_failures=[dict(model=v['model'],layout=v['layout'],sample_id=v['sample_id'],repeat=v['repeat'])
                for v in values.values() if not v['strict_singleton_reference_check']['passed']]
            passed=all(r['passed'] for r in required) and not autograd_failures and not reference_failures
            summary=dict(status='executed_stability_passed_pending_review' if passed else 'executed_stability_failed_requires_review',
                samples=40,seeds=[42],gradient_records=164,natural_repeat_pairs=80,padding_repeat_pairs=2,
                padding_comparisons=4,cross_precision_comparisons=80,required_stability_checks=len(required),
                failed_stability_checks=[r for r in required if not r['passed']],autograd_forward_failures=autograd_failures,
                strict_reference_failures=reference_failures,stability_criteria_passed=passed,
                nonzero_physical_gradients=sum(v['physical_abs_gradient_sum']>0 for v in values.values()),
                zero_physical_gradients=sum(v['physical_abs_gradient_sum']==0 for v in values.values()),
                criteria={k:spec[k] for k in ('score_atol','score_rtol','gradient_relative_l2_limit','gradient_cosine_minimum')},
                criteria_role='proposed engineering criteria; not significance, proof of correct hard-spike gradients or causality',
                backend=m['backend'],training=False,event_deletion=False,input_optimization=False,stage7_complete=False,
                next_step='Read-only review of repeat, forward and padding stability before preparing any perturbations or expanding coverage.')
            for name,value in [('GRADIENT_INDEX.json',gradient_index),('GRADIENT_SUMMARIES.json',list(values.values())),
                               ('STABILITY_CHECKS.json',checks),('SUMMARY.json',summary),
                               ('FROZEN_STATE_CHECKS.json',[dict(model=r['model'],status='unchanged',
                                   model_state_sha256=r['model_state_sha256'],all_parameter_gradients_none=True) for r in m['runs']])]:
                write(root/name,value);outputs.append(root/name)
            pilot.write_csv(root/'STABILITY_CHECKS.csv',checks);outputs.append(root/'STABILITY_CHECKS.csv')
            text=['# Fixed-precision SHD* input-gradient repeat pilot','',
                'MT/AMT, seed 42, same 40 validation recordings. No training, official test, event deletion or input optimization.',
                'Two full-validation references passed under highest_matmul; these do not replace source_high training-validation anchors.',
                f"Saved gradients: 164; repeat/padding checks: {len(required)}; cross-precision legacy comparisons: 80.",
                f"Declared stability criteria passed: {passed}. Autograd-forward failures: {len(autograd_failures)}; strict reference failures: {len(reference_failures)}.",
                'Zero physical gradients are explicit; no normalized maps or cosine values are fabricated for them.',
                'All model state hashes unchanged. Criteria are engineering thresholds, not scientific significance or causal attribution.',
                'Completion authenticates negative results too. Review SUMMARY.json, STABILITY_CHECKS.csv and raw GRADIENT_INDEX.json before expansion.',
                'Source-high historical arrays, same-mode reference subset and portable path index are preserved under shared/.',
                'Stage 7 remains incomplete. No event-deletion or optimized-input queue launches automatically.','']
            (root/'SUMMARY.md').write_text('\n'.join(text),encoding='utf-8');outputs.append(root/'SUMMARY.md')
            write(root/'COMPLETE.json',dict(status='complete',manifest_sha256=identity,
                completed_utc=datetime.now(timezone.utc).isoformat(),attempt_elapsed_seconds=time.monotonic()-started,
                stability_criteria_passed=passed,stage7_complete=False,
                outputs={portable_path(p):pilot.sha256(p) for p in dict.fromkeys(outputs)}))
            progress('complete',registered_outputs=len(verify(root)['outputs']),stability_criteria_passed=passed)
            print('Strict gradient pilot complete:',portable_path(root/'SUMMARY.md'),'stability passed:',passed,flush=True)
        except Exception as error:
            progress('failed',error_type=type(error).__name__,error=str(error));raise
        finally:
            models.clear()
            if torch.cuda.is_available():torch.cuda.empty_cache()

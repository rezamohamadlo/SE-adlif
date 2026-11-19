"""User-launched full anchors, input backward passes and shared deletion controls."""
from datetime import datetime, timezone
import subprocess
import sys
import time

import numpy as np
import torch

from datasets.utils.pad_tensors import PadTensors
from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_gradient_pilot.engine import load_model, ordinary, differentiate
from scripts.adaptation_gradient_pilot.methods import (
    array_hash, atomic_copy, cached_array, count_hash, event_budget, exact_random_delta, guided_delta, require, require_frozen, validate_delta,
)
from scripts.adaptation_gradient_pilot.preflight import guard, read_cache, source_directory, source_file, verify_source
from scripts.adaptation_gradient_pilot.reporting import summarize, write_figures
from scripts.ssc_resume_io import AtomicJSONWriter

IDENTITY_KEYS = ('sample_id', 'target', 'speaker', 'input_sha256', 'original_support_frames', 'stored_frames')


def snapshot(root, spec, write):
    source = source_directory(root, spec)
    verify_source(source, spec)
    target = root/'shared/shd_clean_source'
    target.mkdir(parents=True, exist_ok=True)
    paths, index = [], {}
    for suffix, checksum in spec['source']['files'].items():
        path = source_file(target, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): atomic_copy(source_file(source, suffix), path)
        require(pilot.sha256(path) == checksum, 'Selected source snapshot differs')
        paths.append(path);index[spec['source']['root']+'/'+suffix]=portable_path(path)
    p = target/'PATH_INDEX.json'
    record = dict(path_index=index, scope='authenticated clean seed-42 subset; original manifests unchanged')
    if p.exists(): pilot.validate_manifest(pilot.read_json(p), record)
    else: write(p, record)
    paths.append(p)
    destination = root/'shared/stage6_closeout'
    destination.mkdir(exist_ok=True)
    p = destination/'STAGE6_CLOSEOUT_AUDIT.json'
    if not p.exists(): atomic_copy(local_path(spec['stage6_audit']['record']), p)
    require(pilot.sha256(p) == spec['stage6_audit']['sha256'], 'Closeout snapshot differs')
    audit=pilot.read_json(p);document=destination/'SCOPE_AUDIT.md'
    if not document.exists(): atomic_copy(local_path(audit['audit_path'],must_exist=True),document)
    require(pilot.sha256(document)==audit['audit_sha256'],'Closeout document snapshot differs')
    return target, paths+[p,document]


def verify(directory):
    root = experiment_directory(directory)
    m, c = [pilot.read_json(root/n) for n in ('MANIFEST.json', 'COMPLETE.json')]
    require(c['status'] == 'complete' and c['manifest_sha256'] == pilot.digest(m) and
            c['outputs'].get(portable_path(root/'MANIFEST.json')) == pilot.sha256(root/'MANIFEST.json'), 'Pilot completion identity differs')
    for name, checksum in c['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path) == checksum, 'Pilot output changed: '+name)
    verify_source(root/'shared/shd_clean_source', m['specification'])
    anchors = pilot.read_json(root/'FULL_VALIDATION_ANCHORS.json')
    require(len(anchors) == 2 and all(r['status']=='passed' and r['samples']==1631 for r in anchors), 'Original-batch anchors failed')
    summary = pilot.read_json(root/'SUMMARY.json')
    require(summary['gradient_records']==80 and summary['samples']==40 and summary['n_training_seeds']==1 and
            not summary['stage7_complete'] and not summary['input_optimization'], 'Pilot scope/coverage differs')
    require(pilot.read_json(root/'CONTRACT_TESTS.json')['status']=='passed', 'No passing contract evidence')
    return c


def execute(prepared):
    root, spec, manifest, data = (prepared[k] for k in ('root','spec','manifest','data'))
    started = time.monotonic()
    with pilot.study_lock(root/'.analysis.lock'):
        guard(root, manifest)
        if (root/'COMPLETE.json').exists():
            print('Pilot already complete; verified outputs:', len(verify(root)['outputs']))
            return
        write = AtomicJSONWriter(root, pilot.write_json)
        if not (root/'MANIFEST.json').exists(): write(root/'MANIFEST.json', manifest)
        identity, outputs, models = pilot.digest(manifest), [root/'MANIFEST.json'], {}

        def progress(phase, **details):
            write(root/'PROGRESS.json', dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**details))

        def cached(path, contract, produce):
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                record=read_cache(path);pilot.validate_manifest(record['identity'],contract);value=record['value']
            else:
                value=produce();write(path,dict(identity=contract,value=value,value_sha256=pilot.digest(value)))
            outputs.append(path)
            return value

        try:
            progress('contract_tests')
            tests=subprocess.run([sys.executable,'-m','unittest','tests.test_adaptation_gradient_pilot'],cwd=ROOT,
                capture_output=True,text=True,encoding='utf-8',errors='replace')
            (root/'CONTRACT_TESTS.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
            write(root/'CONTRACT_TESTS.json',dict(status='passed' if tests.returncode==0 else 'failed',returncode=tests.returncode,
                log_sha256=pilot.sha256(root/'CONTRACT_TESTS.txt')))
            outputs += [root/'CONTRACT_TESTS.txt',root/'CONTRACT_TESTS.json']
            require(tests.returncode==0,'Gradient pilot contracts failed; inspect CONTRACT_TESTS.txt')
            evidence,copied=snapshot(root,spec,write);outputs+=copied
            source=pilot.read_json(evidence/'MANIFEST.json')
            features=read_cache(source_file(evidence,spec['clean_features_file']))['value']
            feature={r['sample_id']:r for r in features}
            references={r['model']:{v['sample_id']:v for v in read_cache(source_file(evidence,r['reference_file']))['value']}
                        for r in manifest['runs']}
            device=torch.device(manifest['environment']['device'])
            for run in manifest['runs']:models[run['model']]=load_model(run,device)
            anchors=[]
            for run in manifest['runs']:
                model=models[run['model']];progress('original_batch_full_validation',model=run['model'])
                print('Full original-batch anchor:',run['model'],flush=True)
                def full_anchor():
                    rows=[];size=run['config']['dataset']['batch_size'];ids=manifest['full_validation_ids']
                    for start in range(0,len(ids),size):
                        inputs=pilot.input_rows(data,ids[start:start+size],('clean',0,'identity',1.),manifest['source_study'],features=False)
                        batch=PadTensors()([data.model_item(sid,x) for sid,x,_row in inputs])
                        result=ordinary(model,batch,run['config'],device)
                        rows.extend(dict(**{k:row[k] for k in IDENTITY_KEYS},model=run['model'],seed=42,**score)
                                    for (_sid,_x,row),score in zip(inputs,result))
                    return rows
                path=root/'anchors'/f"{run['model']}_seed_42.json"
                rows=cached(path,dict(manifest_sha256=identity,checkpoint_sha256=run['checkpoint_sha256'],kind='original_batch_full_validation'),full_anchor)
                require([r['sample_id'] for r in rows]==manifest['full_validation_ids'],'Full anchor coverage/order differs')
                for row in rows:
                    old=references[run['model']][row['sample_id']]
                    require(all(row[k]==old[k] for k in (*IDENTITY_KEYS,'prediction')),'Original-batch clean input/prediction differs')
                accuracy=sum(r['correct'] for r in rows)/len(rows);loss=sum(r['loss'] for r in rows)/len(rows)
                require(abs(accuracy-run['validation']['accuracy'])<=spec['accuracy_tolerance'] and
                        abs(loss-run['validation']['loss'])<=spec['loss_tolerance'],'Original-batch validation anchor failed')
                require_frozen(model,run['model_state_sha256'])
                anchors.append(dict(status='passed',model=run['model'],seed=42,samples=len(rows),accuracy=accuracy,loss=loss,
                    accuracy_difference=accuracy-run['validation']['accuracy'],loss_difference=loss-run['validation']['loss']))
            write(root/'FULL_VALIDATION_ANCHORS.json',anchors);outputs.append(root/'FULL_VALIDATION_ANCHORS.json')
            counts={};input_identity={}
            for i in manifest['pilot_ids']:
                _sid,x,row=pilot.input_rows(data,[i],('clean',0,'identity',1.),manifest['source_study'],features=False)[0]
                require(all(row[k]==feature[i][k] for k in IDENTITY_KEYS),'Pilot reconstructed clean input differs')
                counts[i]=x;input_identity[i]={k:row[k] for k in IDENTITY_KEYS}
            gradients={};gradient_arrays={};gradient_index=[]
            for run in manifest['runs']:
                model=models[run['model']]
                for number,i in enumerate(manifest['pilot_ids'],1):
                    progress('input_surrogate_backward',model=run['model'],record=number,total=40,sample_id=i)
                    print(f"Input gradient {run['model']}: {number}/40, id={i}",flush=True)
                    path=root/'gradients'/run['model']/'seed_42'/f'sample_{i}.npz'
                    contract=dict(manifest_sha256=identity,checkpoint_sha256=run['checkpoint_sha256'],
                                  sample_id=i,input_sha256=input_identity[i]['input_sha256'],diagnostic_batch_size=1)
                    def produce_gradient():
                        batch=PadTensors()([data.model_item(i,counts[i])])
                        value,arrays=differentiate(model,batch,run['config'],device,counts[i],spec,run['model_state_sha256'])
                        old=references[run['model']][i]
                        value.update(input_identity[i],model=run['model'],seed=42,original_batch_reference={k:old[k] for k in ('loss','prediction','correct','true_probability')},
                                     source_batch_loss_difference=value['clean']['loss']-old['loss'])
                        return value,arrays
                    value,arrays,paths=cached_array(path,contract,produce_gradient,write);outputs+=paths
                    g=arrays['gradient']
                    require(g.shape==(max(len(counts[i]),manifest['data']['min_frames']),140) and np.isfinite(g).all() and
                            array_hash(g)==value['gradient_sha256'] and value['model_state_sha256']==run['model_state_sha256'] and
                            value['model']==run['model'] and value['seed']==42 and
                            all(value[k]==v for k,v in input_identity[i].items()),'Gradient cache identity/shape/state differs')
                    gradients[run['model'],i]=value;gradient_arrays[run['model'],i]=g
                    gradient_index.append(dict(model=run['model'],seed=42,sample_id=i,path=portable_path(path),sha256=pilot.sha256(path)))
                require_frozen(model,run['model_state_sha256'])
            write(root/'GRADIENT_INDEX.json',gradient_index);outputs.append(root/'GRADIENT_INDEX.json')
            write(root/'GRADIENT_SUMMARIES.json',list(gradients.values()));outputs.append(root/'GRADIENT_SUMMARIES.json')
            cases=[]
            for number,i in enumerate(manifest['pilot_ids'],1):
                progress('shared_deletion_controls',record=number,total=40,sample_id=i)
                x=counts[i];budget=event_budget(x,spec['deletion_fraction'])
                controls=[]
                parent_gradient=gradient_arrays[spec['models'][0],i][:len(x)]
                for condition,rep in [('mt_abs_gradient_deletion',0)]+[('exact_random_deletion',j) for j in range(3)]:
                    seed=int(pilot.digest([spec['version'],spec['cohort_selection_seed'],i,condition,rep])[:16],16)
                    path=root/'shared_inputs'/f'sample_{i}'/condition/f'rep_{rep}.npz'
                    contract=dict(manifest_sha256=identity,sample_id=i,source_input_sha256=input_identity[i]['input_sha256'],
                        condition=condition,replicate=rep,budget=budget,seed=seed,
                        parent_gradient_sha256=gradients[spec['models'][0],i]['gradient_sha256'])
                    if condition=='mt_abs_gradient_deletion' and (budget==0 or gradients[spec['models'][0],i]['event_cell_abs_gradient_sum']==0):
                        continue
                    def produce_delta():
                        delta=guided_delta(x,parent_gradient,budget) if condition=='mt_abs_gradient_deletion' else exact_random_delta(x,budget,seed)
                        require(delta is not None,'Eligible guidance unexpectedly missing')
                        validate_delta(x,delta,budget)
                        return dict(condition=condition,replicate=rep,budget=budget,seed=seed,
                            input_sha256=count_hash(x+delta),delta_sha256=array_hash(delta),
                            removed_burn_in_events=-int(delta[:spec['burn_in_frames']].sum()),
                            domain='nonnegative integer counts, original length'),dict(delta=delta)
                    value,arrays,paths=cached_array(path,contract,produce_delta,write);outputs+=paths
                    delta=arrays['delta'];validate_delta(x,delta,budget)
                    require(array_hash(delta)==value['delta_sha256'] and count_hash(x+delta)==value['input_sha256'],'Shared deletion input cache differs')
                    controls.append(dict(value,delta=delta))
                path=root/'evaluations'/f'sample_{i}.json'
                contract=dict(manifest_sha256=identity,sample_id=i,inputs=[{k:v for k,v in r.items() if k!='delta'} for r in controls])
                def evaluate_case():
                    conditions=[dict(condition='clean',replicate=0,budget=0,input_sha256=input_identity[i]['input_sha256'],
                        evaluations=[dict(model=m,**gradients[m,i]['clean'],actual_loss_change=0.,surrogate_linear_prediction=0.) for m in spec['models']])]
                    for control in controls:
                        delta=control['delta'];batch=PadTensors()([data.model_item(i,x+delta)])
                        evaluations=[]
                        for run in manifest['runs']:
                            model=models[run['model']];score=ordinary(model,batch,run['config'],device)[0]
                            predicted=float(np.sum(gradient_arrays[run['model'],i][:len(x)].astype(np.float64)*delta))
                            evaluations.append(dict(model=run['model'],**score,actual_loss_change=score['loss']-gradients[run['model'],i]['clean']['loss'],
                                                    surrogate_linear_prediction=predicted))
                        conditions.append(dict(condition=control['condition'],replicate=control['replicate'],budget=budget,
                                               input_sha256=control['input_sha256'],evaluations=evaluations))
                    return dict(sample_id=i,target=input_identity[i]['target'],speaker=input_identity[i]['speaker'],
                                guided_status=gradients[spec['models'][0],i]['deletion_ranking_status'],conditions=conditions)
                case=cached(path,contract,evaluate_case)
                require(case['sample_id']==i and len(case['conditions'])==1+len(controls),'Pilot evaluation coverage differs')
                expected=[dict(condition='clean',replicate=0,budget=0,input_sha256=input_identity[i]['input_sha256'])]+controls
                for actual,wanted in zip(case['conditions'],expected):
                    require(all(actual[k]==wanted[k] for k in ('condition','replicate','budget','input_sha256')) and
                            [r['model'] for r in actual['evaluations']]==spec['models'],'Shared evaluation input/model identity differs')
                    for r in actual['evaluations']:
                        require(np.isfinite(r['loss']) and r['loss']>=0 and 0<=r['prediction']<20 and
                                r['correct']==int(r['prediction']==case['target']) and 0<=r['true_probability']<=1,
                                'Invalid cached diagnostic prediction')
                cases.append(case)
            for run in manifest['runs']:require_frozen(models[run['model']],run['model_state_sha256'])
            write(root/'FROZEN_STATE_CHECKS.json',[dict(model=r['model'],seed=42,status='unchanged',model_state_sha256=r['model_state_sha256'],
                all_parameter_gradients_none=True) for r in manifest['runs']]);outputs.append(root/'FROZEN_STATE_CHECKS.json')
            summary,paired=summarize(gradients,cases,spec['models'],manifest['pilot_ids'])
            pilot.write_csv(root/'PAIRED_RECORDS.csv',paired);outputs.append(root/'PAIRED_RECORDS.csv')
            outputs+=write_figures(root,gradients,summary,spec['models'],manifest['pilot_ids'])
            write(root/'SUMMARY.json',summary)
            lines=['# Stage-7 SHD* input-gradient tooling pilot','',
                'MT/AMT, seed 42, 40 validation recordings (two per class). No weight training or official test.',
                'Two original-batch full-validation anchors passed. Input gradients and controlled deletions use declared single-record batching.',
                'These are SLAYER surrogate CE sensitivities, not true hard-spike finite differences or causal attention.',
                'Synthetic-padding derivatives and burn-in-input derivatives are recorded separately; padding is not perturbed.','',
                '| Model | Nonzero physical gradients | Source-batch prediction differences | Max source-batch CE drift | Max autograd forward CE drift |',
                '|---|---:|---:|---:|---:|']
            for r in summary['groups']:
                lines.append(f"| {r['model']} | {r['nonzero_physical_gradients']} / 40 | {r['source_batch_prediction_mismatches']} | "
                             f"{r['max_absolute_source_batch_loss_drift']:.8g} | {r['max_autograd_forward_loss_drift']:.8g} |")
            lines+=['','Shared 1% exact event-deletion budget; MT guidance and three uniform random realizations. '
                'Guided/random contrasts use the same eligible records. Predicted linear and actual finite CE changes may disagree; agreement is not a completion gate.',
                '',f"Gradients: {summary['gradient_records']}; evaluation records: {summary['evaluation_records']}; paired conditions: {summary['paired_condition_records']}.",
                'Raw signed arrays: GRADIENT_INDEX.json. Scalar traces and all limits: SUMMARY.json and PAIRED_RECORDS.csv.',
                'Figures: relative normalized physical-input gradient mass, diagnostic/source batch drift, and matched guided/random effects.',
                'This pilot does not complete stage 7. Review numerical behavior and gradient quality before expanding pairs/seeds/data or optimizing inputs.','']
            (root/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8');outputs += [root/'SUMMARY.json',root/'SUMMARY.md']
            write(root/'COMPLETE.json',dict(status='complete',manifest_sha256=identity,completed_utc=datetime.now(timezone.utc).isoformat(),
                attempt_elapsed_seconds=time.monotonic()-started,stage7_complete=False,input_optimization=False,
                outputs={portable_path(p):pilot.sha256(p) for p in dict.fromkeys(outputs)}))
            progress('complete',registered_outputs=len(verify(root)['outputs']))
            print('Stage-7 pilot completed:',portable_path(root/'SUMMARY.md'),flush=True)
        except Exception as error:
            progress('failed',error_type=type(error).__name__,error=str(error));raise
        finally:
            models.clear()
            if torch.cuda.is_available():torch.cuda.empty_cache()

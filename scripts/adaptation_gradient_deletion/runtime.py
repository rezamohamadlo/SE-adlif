"""User-launched fresh-clean gates and shared matched event-deletion evaluation."""
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
from scripts.adaptation_batch_precision.contracts import batch_identity, precision_mode
from scripts.adaptation_gradient_deletion.contracts import (
    atomic_csv, build_delta, control_seed, linear_prediction, load_primary, plan_controls, require_clean_checks,
)
from scripts.adaptation_gradient_deletion.reporting import summarize, write_figure
from scripts.adaptation_gradient_pilot.engine import load_model, ordinary
from scripts.adaptation_gradient_pilot.methods import (
    array_hash, atomic_copy, cached_array, count_hash, require, require_frozen, validate_delta,
)
from scripts.adaptation_gradient_pilot.preflight import guard, read_cache
from scripts.adaptation_gradient_strict.methods import score_check
from scripts.ssc_resume_io import AtomicJSONWriter


def validate_score(row,target):
    require(np.isfinite(row['loss']) and row['loss']>=0 and 0<=row['prediction']<20 and
            row['correct']==int(row['prediction']==target) and 0<=row['true_probability']<=1,
            'Invalid cached prediction/loss')


def verify(directory):
    root=experiment_directory(directory)
    m,c=[pilot.read_json(root/n) for n in ('MANIFEST.json','COMPLETE.json')]
    require(c['status']=='complete' and c['manifest_sha256']==pilot.digest(m) and
            c['outputs'].get(portable_path(root/'MANIFEST.json'))==pilot.sha256(root/'MANIFEST.json'),
            'Deletion completion identity differs')
    for name,sha in c['outputs'].items():
        path=local_path(name,must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path)==sha,'Deletion output changed: '+name)
    s=pilot.read_json(root/'SUMMARY.json');eligible=s['guided_eligible_records']
    require(s['samples']==40 and s['n_training_seeds']==1 and s['primary_repeat']==0 and 0<=eligible<=40 and
            s['shared_deletion_inputs']==120+eligible and s['paired_condition_records']==160+eligible and
            s['model_evaluation_records']==2*(160+eligible) and s['contrast_records']==2*eligible and
            not s['backward'] and not s['stage7_complete'],'Deletion coverage/scope differs')
    checks=pilot.read_json(root/'CLEAN_REFERENCE_CHECKS.json')
    require_clean_checks(checks)
    require(pilot.read_json(root/'CONTRACT_TESTS.json')['status']=='passed','No passing user contracts')
    return c


def execute(prepared):
    root,spec,m,data,source=[prepared[k] for k in ('root','spec','manifest','data','source_root')]
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
            tests=subprocess.run([sys.executable,'-m','unittest','tests.test_adaptation_gradient_deletion'],cwd=ROOT,
                capture_output=True,text=True,encoding='utf-8',errors='replace')
            log=root/'CONTRACT_TESTS.txt';log.write_text(tests.stdout+tests.stderr,encoding='utf-8')
            write(root/'CONTRACT_TESTS.json',dict(status='passed' if tests.returncode==0 else 'failed',
                returncode=tests.returncode,log_sha256=pilot.sha256(log)))
            outputs += [log,root/'CONTRACT_TESTS.json']
            require(tests.returncode==0,'Deletion contracts failed; inspect CONTRACT_TESTS.txt')
            require(pilot.sha256(source/'MANIFEST.json')==spec['source_manifest_sha256'] and
                    pilot.sha256(source/'COMPLETE.json')==spec['source_complete_sha256'],'Frozen gradient source changed after preflight')
            ledger=pilot.read_json(source/'COMPLETE.json')['outputs'];path_index={};copied={}
            def copy_selected(suffix):
                original=(source/suffix).resolve();require(original.is_relative_to(source),'Snapshot source escapes study')
                target=root/'shared/strict_source'/suffix;target.parent.mkdir(parents=True,exist_ok=True)
                expected=spec['source_complete_sha256'] if suffix=='COMPLETE.json' else ledger[portable_path(original)]
                require(pilot.sha256(original)==expected,'Snapshot source changed')
                if not target.exists():atomic_copy(original,target)
                require(pilot.sha256(target)==expected,'Snapshot output differs')
                outputs.append(target);path_index[portable_path(original)]=portable_path(target);return target
            metadata=['MANIFEST.json','COMPLETE.json','SUMMARY.json','FULL_VALIDATION_ANCHORS.json','FROZEN_STATE_CHECKS.json',
                      'GRADIENT_INDEX.json','GRADIENT_SUMMARIES.json','STABILITY_CHECKS.json','CONTRACT_TESTS.json','CONTRACT_TESTS.txt']
            for name in metadata:copy_selected(name)
            source_manifest=pilot.read_json(source/'MANIFEST.json')
            for run in m['runs']:
                for b in range(len(source_manifest['original_batch_ids'])):
                    copy_selected(f"anchors/{run['model']}/seed_42/batch_{b}.json")
            for item in m['primary_gradients']:
                original=local_path(item['path'],must_exist=True);require(original.is_relative_to(source),'Primary escapes source')
                copied[item['model'],item['sample_id']]=copy_selected(original.relative_to(source).as_posix())
            write(root/'shared/PATH_INDEX.json',dict(path_index=path_index,
                scope='authenticated selected metadata, full anchor chunks and 80 primary NPZ arrays; historical manifests unchanged'))
            outputs.append(root/'shared/PATH_INDEX.json')
            gradients={};reference={};counts={sid:data.load(sid)['counts'] for sid in m['pilot_ids']}
            for item in m['primary_gradients']:
                key=item['model'],item['sample_id']
                value,g=load_primary(copied[key],item,m['source_manifest_sha256'],m['backend'])
                require(value['input_sha256']==count_hash(counts[key[1]]),'Primary physical input hash differs')
                gradients[key]=g;reference[key]=value
            for run in m['runs']:models[run['model']]=load_model(run,device)
            clean={};checks=[];cases=[];control_index=[]
            with precision_mode(spec['precision_mode'],m['source_backend']) as backend:
                pilot.validate_manifest(backend,m['backend'])
                # All eighty fresh clean gates precede any perturbation generation/evaluation.
                for run in m['runs']:
                    for sid in m['pilot_ids']:
                        progress('fresh_clean_references',model=run['model'],sample_id=sid,completed=len(checks),total=80)
                        batch=PadTensors()([data.model_item(sid,counts[sid])])
                        require(batch_identity(batch)==m['input_identities'][str(sid)],'Clean input/target/mask differs')
                        path=root/'clean'/run['model']/'seed_42'/f'sample_{sid}.json'
                        contract=dict(manifest_sha256=identity,checkpoint_sha256=run['checkpoint_sha256'],backend=backend,
                            sample_id=sid,batch=batch_identity(batch))
                        score=cached(path,contract,lambda:ordinary(models[run['model']],batch,run['config'],device)[0])
                        validate_score(score,int(data.labels[sid]));require_frozen(models[run['model']],run['model_state_sha256'])
                        check=score_check(score,reference[run['model'],sid]['clean'],spec)
                        checks.append(dict(model=run['model'],seed=42,sample_id=sid,**check));clean[run['model'],sid]=score
                write(root/'CLEAN_REFERENCE_CHECKS.json',checks);outputs.append(root/'CLEAN_REFERENCE_CHECKS.json')
                require_clean_checks(checks)
                for number,sid in enumerate(m['pilot_ids'],1):
                    progress('matched_deletion',record=number,total=40,sample_id=sid)
                    print(f'Strict matched deletion {number}/40, id={sid}',flush=True)
                    x=counts[sid];parent=gradients[spec['models'][0],sid][:len(x)]
                    budget,eligible,plan=plan_controls(x,parent,spec);controls=[]
                    for condition,rep in plan:
                        seed=control_seed(spec,sid,condition,rep)
                        path=root/'shared_inputs'/'SHD'/'seed_42'/f'sample_{sid}'/condition/f'rep_{rep}.npz'
                        contract=dict(manifest_sha256=identity,sample_id=sid,condition=condition,replicate=rep,seed=seed,
                            budget=budget,source_input_sha256=count_hash(x),
                            parent_gradient_sha256=reference[spec['models'][0],sid]['gradient_sha256'])
                        def produce_delta():
                            delta=build_delta(x,parent,budget,condition,seed)
                            return dict(condition=condition,replicate=rep,seed=seed,budget=budget,
                                input_sha256=count_hash(x+delta),delta_sha256=array_hash(delta),
                                removed_burn_in_events=-int(delta[:10].sum()),physical_steps=len(x)),dict(delta=delta)
                        value,payload,paths=cached_array(path,contract,produce_delta,write);outputs += paths
                        delta=payload['delta'];validate_delta(x,delta,budget)
                        require(array_hash(delta)==value['delta_sha256'] and count_hash(x+delta)==value['input_sha256'],
                                'Shared deletion delta/input differs')
                        controls.append(dict(value,delta=delta));control_index.append(dict(sample_id=sid,condition=condition,
                            replicate=rep,path=portable_path(path),sha256=pilot.sha256(path)))
                    path=root/'evaluations'/'SHD'/'seed_42'/f'sample_{sid}.json'
                    contract=dict(manifest_sha256=identity,sample_id=sid,backend=backend,
                        clean_sha256={model:pilot.digest(clean[model,sid]) for model in spec['models']},
                        inputs=[{k:v for k,v in control.items() if k!='delta'} for control in controls])
                    def evaluate_case():
                        conditions=[dict(condition='clean',replicate=0,budget=0,input_sha256=count_hash(x),
                            evaluations=[dict(model=model,**clean[model,sid],actual_loss_change=0.,surrogate_linear_prediction=0.) for model in spec['models']])]
                        original=PadTensors()([data.model_item(sid,x)])
                        for control in controls:
                            delta=control['delta'];batch=PadTensors()([data.model_item(sid,x+delta)])
                            require(torch.equal(batch[1],original[1]) and torch.equal(batch[2],original[2]) and
                                    batch[0].shape==original[0].shape,'Deletion changed target, blocks or input length')
                            scores=[]
                            for run in m['runs']:
                                model=run['model'];score=ordinary(models[model],batch,run['config'],device)[0]
                                require_frozen(models[model],run['model_state_sha256'])
                                predicted=linear_prediction(gradients[model,sid],delta)
                                scores.append(dict(model=model,**score,actual_loss_change=score['loss']-clean[model,sid]['loss'],
                                    surrogate_linear_prediction=predicted))
                            conditions.append(dict(condition=control['condition'],replicate=control['replicate'],budget=budget,
                                input_sha256=control['input_sha256'],evaluations=scores))
                        return dict(sample_id=sid,target=int(data.labels[sid]),speaker=int(data.speakers[sid]),
                            guidance_eligible=eligible,conditions=conditions)
                    case=cached(path,contract,evaluate_case)
                    expected=[dict(condition='clean',replicate=0,budget=0,input_sha256=count_hash(x))]+controls
                    require(case['sample_id']==sid and case['guidance_eligible']==eligible and len(case['conditions'])==len(expected),
                            'Case coverage/guidance differs')
                    for actual,wanted in zip(case['conditions'],expected):
                        require(all(actual[k]==wanted[k] for k in ('condition','replicate','budget','input_sha256')) and
                                [r['model'] for r in actual['evaluations']]==spec['models'],'Shared case identity differs')
                        for score in actual['evaluations']:validate_score(score,case['target'])
                    cases.append(case)
            for run in m['runs']:require_frozen(models[run['model']],run['model_state_sha256'])
            summary,paired,effects,contrasts=summarize(cases,spec['models'])
            summary.update(clean_reference_checks_passed=80,primary_gradient_arrays_reused=80,
                fresh_backward_passes=0,backend=m['backend'])
            for name,value in [('SUMMARY.json',summary),('DELETION_INDEX.json',control_index),
                ('FROZEN_STATE_CHECKS.json',[dict(model=r['model'],status='unchanged',model_state_sha256=r['model_state_sha256'],
                    all_parameter_gradients_none=True) for r in m['runs']])]:
                write(root/name,value);outputs.append(root/name)
            for name,rows in [('PAIRED_RECORDS.csv',paired),('MODEL_EFFECTS.csv',effects),('GUIDED_RANDOM_CONTRASTS.csv',contrasts)]:
                atomic_csv(root/name,rows);outputs.append(root/name)
            outputs.append(write_figure(root,contrasts,spec['models'],m['pilot_ids']))
            lines=['# Strict-precision matched event-deletion pilot','',
                'SHD*, MT/AMT, seed 42, 40 fixed validation recordings; predetermined natural repeat 0 gradients reused.',
                'All 80 fresh clean references passed before deletion. Weights/state unchanged; no training or new backward.',
                '1% exact event budget, MT absolute-gradient guidance and three uniform exact random controls; shared inputs for both models.',
                'Random effects average within record, not independent seeds. Zero guidance is skipped for both models.',
                '', '| Model | Eligible records | Mean absolute CE change, guided | Mean absolute CE change, random | Mean paired difference |',
                '|---|---:|---:|---:|---:|']
            for g in summary['groups']:
                values=[g[k] for k in ['mean_abs_guided_loss_change','mean_abs_random_loss_change','mean_guided_minus_random_abs_loss_change']]
                formatted=['undefined' if v is None else f'{v:.9g}' for v in values]
                lines.append(f"| {g['model']} | {g['common_guidance_eligible_records']} | {' | '.join(formatted)} |")
            lines += ['',f"Paired condition rows: {len(paired)}; model evaluation rows: {len(effects)}; shared deletion inputs: {len(control_index)}.",
                'Signed finite CE changes and own-gradient linear predictions remain in MODEL_EFFECTS.csv. Their agreement is not a success gate.',
                'MT-guided selection, confidence/loss coupling and recording 3896 limit architectural/causal interpretation. No superiority or significance claim.',
                'Original source_high scores and all prior studies remain unchanged. Stage 7 is incomplete; review before expansion or optimized inputs.','']
            (root/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8');outputs.append(root/'SUMMARY.md')
            write(root/'COMPLETE.json',dict(status='complete',manifest_sha256=identity,
                completed_utc=datetime.now(timezone.utc).isoformat(),attempt_elapsed_seconds=time.monotonic()-started,
                stage7_complete=False,outputs={portable_path(path):pilot.sha256(path) for path in dict.fromkeys(outputs)}))
            progress('complete',registered_outputs=len(verify(root)['outputs']))
            print('Strict matched deletion complete:',portable_path(root/'SUMMARY.md'),flush=True)
        except Exception as error:
            progress('failed',error_type=type(error).__name__,error=str(error));raise
        finally:
            models.clear()
            if torch.cuda.is_available():torch.cuda.empty_cache()

"""Audited continuation preparation and tests; training stays in runner.py."""
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import runpy
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf
from project_paths import ROOT, local_path
from module.study_control import canonical, forward_identity
from extended_seed_study import write_provenance, checkpoint_selection

BASE=Path(__file__).resolve().parent
STUDY='config/studies/dth_complement_extended_100_70_70_v1.yaml'
MODEL='DTH_RDROP_SE_adLIF'
TARGETS={'SHD':100,'ECG':70,'SSC':70}
STARTS={'SHD':30,'ECG':30,'SSC':31}
UPDATES={'SHD':25,'ECG':9,'SSC':294}
SOURCES={'SHD':'b001_rdrop_shd_v1','ECG':'b001_rdrop_ecg_v1','SSC':'b001_rdrop_ssc_v1'}
DATA={'SHD':['data/SHDWrapper/shd_train.h5','data/SHDWrapper/shd_test.h5'],
      'SSC':['data/SSCWrapper/ssc_train.h5','data/SSCWrapper/ssc_valid.h5','data/SSCWrapper/ssc_test.h5'],
      'ECG':['data/QTDB/QTDB_train.mat','data/QTDB/QTDB_test.mat']}
LIB=BASE.parent/'table30_then_ecg_v1/ACTIONS.py'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def rel(path):return Path(path).relative_to(ROOT).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def seed_dir(ds):return BASE/'runs'/MODEL/ds/'seed_42'
def source_dir(ds):return BASE.parent/SOURCES[ds]/'runs'/MODEL/ds/'seed_42'

def tree_hash(value):
    digest=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous()
            digest.update(('tensor:'+str(x.dtype)+':'+str(tuple(x.shape))).encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            digest.update(b'dict')
            for key in sorted(x,key=lambda k:(str(type(k)),str(k))):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple)):
            digest.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:digest.update((str(type(x))+':'+repr(x)).encode())
    visit(value);return digest.hexdigest()

META={'logdir','launcher_logdir','study_file','study_idea','study_stage','study_hypothesis',
      'stage_ceiling','ckpt_path','evaluation_protocol','paper_baseline_deviations',
      'study_continuation_manifest','study_original_schedule_epochs','study_budget_extension_policy'}

def assert_math_equivalence(old,new,ds):
    a=copy.deepcopy(old);b=copy.deepcopy(new)
    if ds=='SSC':
        assert a['n_epochs']==40 and b['n_epochs']==70
        assert b['study_original_schedule_epochs']==40
        assert b['study_budget_extension_policy']=='constant_plateau_budget_extension_v1'
        assert a['scheduler']==b['scheduler']=='plateau' and a['patience']==b['patience']==9999
        a.pop('n_epochs');b.pop('n_epochs')
    for key in META:a.pop(key,None);b.pop(key,None)
    assert a==b, 'Undeclared computation/configuration change: '+str(sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k)))

def strip_callback_paths(callbacks):
    result=copy.deepcopy(callbacks)
    for state in result.values():
        if not isinstance(state,dict):continue
        for key in ('dirpath','best_model_path','last_model_path','kth_best_model_path'):state.pop(key,None)
        if isinstance(state.get('best_k_models'),dict):
            state['best_k_models']={Path(k).name:v for k,v in state['best_k_models'].items()}
    return result

def rebind_state(original,new_cfg,old_run,new_run,ds):
    old_cfg=OmegaConf.to_container(OmegaConf.create(original['hyper_parameters']['cfg']),resolve=True)
    assert_math_equivalence(old_cfg,new_cfg,ds)
    migrated=dict(original,hyper_parameters=copy.deepcopy(original['hyper_parameters']),callbacks=copy.deepcopy(original.get('callbacks',{})))
    migrated['hyper_parameters']['cfg']=copy.deepcopy(new_cfg)
    for state in migrated['callbacks'].values():
        if not isinstance(state,dict):continue
        for key in ('dirpath','best_model_path','last_model_path','kth_best_model_path'):
            if state.get(key):
                path=local_path(state[key])
                if key=='dirpath':
                    assert path==(old_run/'ckpt').resolve();state[key]=rel(new_run/'ckpt')
                else:
                    assert path.parent==(old_run/'ckpt').resolve();state[key]=rel(new_run/'ckpt'/path.name)
        if isinstance(state.get('best_k_models'),dict):
            routes={}
            for path,score in state['best_k_models'].items():
                path=local_path(path);assert path.parent==(old_run/'ckpt').resolve()
                routes[rel(new_run/'ckpt'/path.name)]=score
            state['best_k_models']=routes
    for key in ('state_dict','optimizer_states','lr_schedulers','loops'):
        assert tree_hash(original[key])==tree_hash(migrated[key]),key
    assert migrated['epoch']==original['epoch'] and migrated['global_step']==original['global_step']
    assert tree_hash(strip_callback_paths(original['callbacks']))==tree_hash(strip_callback_paths(migrated['callbacks']))
    return migrated

def snapshot():
    from extended_seed_study import source_identity
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')
    files={}
    for name in sorted(set(names)-{''}):
        path=Path(name)
        if path.parts[0] in ('results','data','docs','reports','archive'):continue
        if path.suffix not in ('.py','.yaml','.yml') and name!='requirements.txt':continue
        if (ROOT/path).is_file():files[name]=sha(ROOT/path)
    return dict(identity=source_identity(ROOT),files_sha256=files)

def check_source():
    from extended_seed_study import source_identity
    frozen=read(BASE/'SOURCE_SNAPSHOT.json')
    assert source_identity(ROOT)['source_files_sha256']==frozen['identity']['source_files_sha256']
    for name,digest in frozen['files_sha256'].items():assert sha(ROOT/name)==digest,name

def prepare():
    import runner
    from unittest.mock import patch
    plan=read(BASE/'CONTINUATION_PLAN.json');assert plan['status']=='preparing'
    records={}
    for ds in ('SHD','ECG','SSC'):
        old=source_dir(ds);new=seed_dir(ds)
        assert not new.exists() or not any(new.iterdir()),'Existing continuation: inspect instead of overwriting'
        source_last=old/'ckpt/last.ckpt';source_sha=sha(source_last)
        state=torch.load(source_last,map_location='cpu',weights_only=False)
        assert state['epoch']+1==STARTS[ds] and state['global_step']==STARTS[ds]*UPDATES[ds]
        assert state['optimizer_states'] and state['lr_schedulers']
        assert canonical(OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']),resolve=True))==canonical(read(old/'FULL_RUN_CONFIG.json'))
        assert read(old/'MODEL_SOURCE_IDENTITY.json')==forward_identity(ds)
        jobs=[];original_build_plan=runner.build_plan
        def record_plan(*args,**kwargs):
            result=original_build_plan(*args,**kwargs);jobs.extend(result);return result
        # Observe the real public dry-run result without changing its guards.
        with patch.object(runner,'build_plan',side_effect=record_plan):
            runner.main(['--models',MODEL,'--datasets',ds,'--seeds','42','--study-file',STUDY,'--stage-ceiling',str(TARGETS[ds]),'--dry-run'])
        assert len(jobs)==1 and jobs[0]['directory'].resolve()==new.resolve()
        cfg=jobs[0]['config'];assert_math_equivalence(OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']),resolve=True),cfg,ds)
        assert cfg['n_epochs']=={'SHD':300,'SSC':70,'ECG':400}[ds]
        (new/'ckpt').mkdir(parents=True);history=new/'origin_history';history.mkdir()
        save(new/'FULL_RUN_CONFIG.json',cfg)
        for filename in ('FULL_RUN_CONFIG.json','LAUNCH_ENVIRONMENTS.json','MODEL_SOURCE_IDENTITY.json','INITIAL_MODEL_STATE.json','RESUME_SOURCE_TRANSITION.json'):
            if (old/filename).is_file():shutil.copy2(old/filename,history/filename)
        paths=list(old.rglob('metrics.csv'));assert len(paths)==1
        rows=list(csv.DictReader(paths[0].open()));validation=[r for r in rows if r.get('val_acc')]
        assert [int(r['epoch']) for r in validation]==list(range(STARTS[ds]))
        target_metrics=new/'logs/mlp_snn/version_0/metrics.csv';target_metrics.parent.mkdir(parents=True)
        shutil.copy2(paths[0],target_metrics);shutil.copy2(paths[0],history/'METRICS_BEFORE_CONTINUATION.csv')
        source_best,score=checkpoint_selection(state,old)
        checkpoint_records=[]
        for kind,source,target in [('best',source_best,new/'ckpt'/source_best.name),('last',source_last,new/'ckpt/last.ckpt')]:
            source_digest=sha(source);original=torch.load(source,map_location='cpu',weights_only=False)
            migrated=rebind_state(original,cfg,old,new,ds)
            torch.save(migrated,target)
            reloaded=torch.load(target,map_location='cpu',weights_only=False)
            signatures={key:tree_hash(original[key]) for key in ('state_dict','optimizer_states','lr_schedulers','loops')}
            assert all(tree_hash(reloaded[key])==digest for key,digest in signatures.items())
            assert sha(source)==source_digest
            checkpoint_records.append(dict(kind=kind,source_checkpoint=rel(source),source_checkpoint_sha256=source_digest,
                derived_checkpoint=rel(target),derived_checkpoint_sha256=sha(target),preserved_state_hashes=signatures,
                preserved_epoch=original['epoch'],preserved_global_step=original['global_step']))
        original_cfg=OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']),resolve=True)
        changes={key:dict(original=original_cfg.get(key),continued=cfg.get(key)) for key in set(original_cfg)|set(cfg) if original_cfg.get(key)!=cfg.get(key)}
        record=dict(mode='full_state_continuation_with_explicit_metadata_routing_migration',source_run=rel(old),new_run=rel(new),
            starting_completed_epochs=STARTS[ds],target_completed_epochs=TARGETS[ds],paper_baseline_schedule_epochs={'SHD':300,'SSC':40,'ECG':400}[ds],
            saved_training_budget=cfg['n_epochs'],model_data_loss_optimizer_schedule_unchanged=True,
            SSC_budget_extension_declared=ds=='SSC',checkpoints=checkpoint_records,configuration_changes=changes,
            old_initial_state=rel(old/'INITIAL_MODEL_STATE.json'),original_metrics_sha256=sha(paths[0]),source_checkpoint_unchanged=sha(source_last)==source_sha,
            restart_random_sequence_not_guaranteed_equivalent_to_uninterrupted=True,prior_tests_exposed=True)
        save(new/'CONTINUATION_ORIGIN.json',record);save(new/'MODEL_SOURCE_IDENTITY.json',forward_identity(ds))
        write_provenance(new,ROOT)
        records[ds]=record
    save(BASE/'CONTINUATION_PLAN.json',dict(status='prepared',training_seed=42,targets=TARGETS,execution_order=['SHD','ECG','SSC'],datasets=records,
        baseline_file_unchanged=True,controls_not_extended=True,no_score_based_decisions=True,prior_tests_exposed=True))
    save(BASE/'SOURCE_SNAPSHOT.json',snapshot())
    save(BASE/'DATA_IDENTITIES.json',{ds:{name:sha(ROOT/name) for name in names} for ds,names in DATA.items()})
    for ds in ('SHD','ECG','SSC'):
        runner.main(['--models',MODEL,'--datasets',ds,'--seeds','42','--study-file',STUDY,'--stage-ceiling',str(TARGETS[ds]),'--dry-run'])
    write_report()
    print('Prepared and verified three full-state continuations; no training launched by preparation.',flush=True)

def make_entry(ds):
    check_source();run=seed_dir(ds);target=TARGETS[ds]
    for name,digest in read(BASE/'DATA_IDENTITIES.json')[ds].items():assert sha(ROOT/name)==digest,name
    receipt_path=run/'stages'/f'{target:03d}'/'RESULT.json';receipt=read(receipt_path)
    assert receipt['status']=='stage_complete' and receipt['completed_epochs']==target
    best=local_path(receipt['best_checkpoint'],must_exist=True);last=local_path(receipt['last_checkpoint'],must_exist=True)
    assert sha(best)==receipt['best_checkpoint_sha256'] and sha(last)==receipt['last_checkpoint_sha256']
    selected=torch.load(best,map_location='cpu',weights_only=False);final=torch.load(last,map_location='cpu',weights_only=False)
    assert final['epoch']+1==target and final['global_step']==target*UPDATES[ds]
    assert final['optimizer_states'] and final['lr_schedulers']
    cfg=OmegaConf.to_container(OmegaConf.create(selected['hyper_parameters']['cfg']),resolve=True)
    assert canonical(cfg)==canonical(read(run/'FULL_RUN_CONFIG.json'))
    assert read(run/'MODEL_SOURCE_IDENTITY.json')==forward_identity(ds)
    paths=list(run.rglob('metrics.csv'));assert len(paths)==1
    history=list(csv.DictReader(paths[0].open()));v=[r for r in history if r.get('val_acc')]
    assert [int(r['epoch']) for r in v]==list(range(target))
    bestrow=max(v,key=lambda r:float(r['val_acc']));epoch=int(bestrow['epoch'])
    assert epoch==selected['epoch']==receipt['selected_epoch_index'] and float(bestrow['val_acc'])==receipt['best_validation_accuracy']
    tr=[r for r in history if r.get('train_acc_epoch') and int(r['epoch'])==epoch]
    ft=[r for r in history if r.get('train_acc_epoch') and int(r['epoch'])==target-1]
    assert len(tr)==len(ft)==1
    entry=dict(dataset=ds,model=MODEL,seed=42,checkpoint=rel(best),checkpoint_sha256=sha(best),last_checkpoint=rel(last),last_checkpoint_sha256=sha(last),
        stage_receipt=rel(receipt_path),stage_receipt_sha256=sha(receipt_path),full_config=rel(run/'FULL_RUN_CONFIG.json'),model_source_identity=forward_identity(ds),
        metrics_files_sha256={rel(p):sha(p) for p in paths},completed_epochs=target,full_budget=cfg['n_epochs'],
        baseline_schedule_epochs={'SHD':300,'SSC':40,'ECG':400}[ds],selected_epoch_index=epoch,selected_completed_epoch=epoch+1,selected_global_step=selected['global_step'],
        training_accuracy_at_selected_epoch=float(tr[0]['train_acc_epoch']),training_accuracy_final_epoch=float(ft[0]['train_acc_epoch']),
        best_validation_accuracy=float(bestrow['val_acc']),validation_loss_at_selected_epoch=float(bestrow['val_loss']),protocol=cfg['evaluation_protocol'],
        continuation_origin=rel(run/'CONTINUATION_ORIGIN.json'),test_data={'SHD':'data/SHDWrapper/shd_test.h5','SSC':'data/SSCWrapper/ssc_test.h5','ECG':'data/QTDB/QTDB_test.mat'}[ds])
    if ds=='SHD':
        audit=read(BASE.parent/'b001_rdrop_shd_v1/SPLIT_AND_SSC_STOP_AUDIT.json')
        assert sha(ROOT/audit['training_file'])==audit['training_file_sha256']
        for key in ('fitting_indices','validation_indices','fitting_speakers','validation_speakers'):entry[key]=audit[key]
    return entry

def freeze(ds):
    out=BASE/'evaluations'/ds;out.mkdir(parents=True,exist_ok=True);assert not (out/'FROZEN_CHECKPOINTS.json').exists()
    e=make_entry(ds)
    old=read(ROOT/'results/experiments/complementary_speaker_dev_v2/evaluations/user_stop_official_test_v1/FROZEN_CHECKPOINTS.json')
    plan=dict(status='frozen',purpose='exploratory_user_requested_extended_training',frozen_at=datetime.now(timezone.utc).isoformat(),checkpoints=[e],
        test_selects_ideas=False,unplanned_training_after_tests=False,remaining_predeclared_dataset_jobs_continue=True,prior_test_exposure=True,
        evaluation_source_sha256=sha(BASE/'ACTIONS.py'),legacy_ECG_evaluator=rel(LIB),legacy_ECG_evaluator_sha256=sha(LIB),
        capture_source=old['capture_source'],capture_source_sha256=old['capture_source_sha256'],
        fitting_indices=read(BASE.parent/'b001_rdrop_ecg_v1/DATA_AUDIT.json')['fitting_indices'] if ds=='ECG' else None,
        validation_indices=read(BASE.parent/'b001_rdrop_ecg_v1/DATA_AUDIT.json')['validation_indices'] if ds=='ECG' else None)
    if ds!='ECG':
        source=BASE.parent/'b001_rdrop_shd_v1/evaluations/user_shd030_test_v1/EVALUATION_SOURCE.txt'
        text=source.read_text(encoding='utf-8').replace("OUT=ROOT/'results/experiments/dth_complement_v1/b001_rdrop_shd_v1/evaluations/user_shd030_test_v1'",f"OUT=ROOT/'{rel(out)}'")
        text=text.replace('exploratory_user_requested_stop_diagnostic','exploratory_user_requested_extended_training').replace('resume_training_after_tests','unplanned_training_after_tests')
        text=text.replace('No training or selection follows.','No unplanned training or score-based selection follows.')
        (out/'EVALUATION_SOURCE.txt').write_text(text,encoding='utf-8')
        plan['evaluation_source_sha256']=sha(out/'EVALUATION_SOURCE.txt')
    save(out/'FROZEN_CHECKPOINTS.json',plan)
    print(json.dumps(dict(dataset=ds,selected_epoch=e['selected_completed_epoch'],checkpoint_sha256=e['checkpoint_sha256'])),flush=True)

def evaluate(ds):
    out=BASE/'evaluations'/ds;plan=read(out/'FROZEN_CHECKPOINTS.json')
    if ds=='ECG':
        assert sha(LIB)==plan['legacy_ECG_evaluator_sha256']
        namespace=runpy.run_path(str(LIB),run_name='audited_ECG_evaluation_library')
        function=namespace['evaluate_ecg'];function.__globals__.update(TASK=BASE,ECG_EVAL=out)
        function()
    else:runpy.run_path(str(out/'EVALUATION_SOURCE.txt'),run_name='__main__')

def write_report():
    old=read(BASE.parent/'table30_then_ecg_v1/FINAL_RESULTS.json')
    table=old['table'];baseline={'SHD':table['SHD_DTH'],'SSC':table['SSC_DTH']}
    results={}
    for ds in TARGETS:
        folder=BASE/'evaluations'/ds
        if (folder/'ALL_RESULTS.json').exists():results[ds]=read(folder/'ALL_RESULTS.json')['results'][0]
        elif (folder/'TEST_RESULTS.json').exists():results[ds]=read(folder/'TEST_RESULTS.json')
    for ds,r in results.items():
        assert r['completed_epochs']==TARGETS[ds] and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        assert r['ordinary_inference'] and not r['test_time_augmentation'] and r['weights_and_rng_unchanged']
        import numpy as np
        assert sha(ROOT/r['prediction_archive'])==r['prediction_archive_sha256']
        z=np.load(ROOT/r['prediction_archive']);accuracy=float((z['labels']==z['predictions']).mean())
        assert accuracy==r['metrics']['test_acc'] and r['inference_forward_calls']==r['batches']
    status='complete' if len(results)==3 else 'pending'
    save(BASE/'UPDATED_RESULTS.json',dict(status=status,latest=results,historical_DTH30=baseline,targets=TARGETS,
        controls_budget_unmatched=True,prior_tests_exposed=True,training_seed=42))
    lines=['# تست DTH + R-Drop پس از ادامهٔ بودجه‌های درخواستی','',
        'بذر ۴۲؛ استنتاج معمولی تک‌گذر و بدون TTA. checkpointها فقط بر اساس بهترین اعتبارسنجی تا سقف هر اجرا انتخاب می‌شوند. خانهٔ pending هنوز تست نشده است؛ عددی از اعتبارسنجی جایگزین آن نیست. نتایج قبلی و جدول هم‌بودجهٔ ۳۰دوره‌ای حفظ شده‌اند.','',
        'Δ paper برابر دقت محلی منهای میانگین مقاله با واحد درصد (pp)، با استفاده از مقادیر گردنشدهٔ محلی است. اعداد محلی یک بذر و مقاله میانگین چند اجرا دارند.','',
        '| دیتاست | مقاله، میانگین ± SD | تست DTH؛ کنترل ۳۰دوره‌ای (Δ paper) | تست DTH + R-Drop؛ ادامهٔ جدید (Δ paper) | سقف جدید | دورهٔ checkpoint |',
        '|---|---:|---:|---:|---:|---:|']
    for ds,paper,sd in [('SHD',93.79,.76),('SSC',80.44,.26),('ECG',88.18,.36)]:
        control=baseline.get(ds);a=(f"{100*control['metrics']['test_acc']:.2f}% ({100*control['metrics']['test_acc']-paper:+.2f} pp)" if control else 'ارزیابی نشده')
        result=results.get(ds);b=(f"{100*result['metrics']['test_acc']:.2f}% ({100*result['metrics']['test_acc']-paper:+.2f} pp)" if result else 'pending')
        epoch=str(result['selected_completed_epoch']) if result else 'pending'
        lines.append(f"| {'SHD*' if ds=='SHD' else ds} | {paper:.2f} ± {sd:.2f}% | {a} | {b} | {TARGETS[ds]} | {epoch} |")
    lines+=['','**مقایسه با کنترل‌های DTH هم‌بودجه نیست:** کنترل‌ها هنوز ۳۰ دوره دارند؛ بهبود نسبت به آن‌ها را نمی‌توان صرفاً به R-Drop نسبت داد. [جدول هم‌بودجهٔ قبلی](../table30_then_ecg_v1/TEST_TABLE_30_FA.md) جدا حفظ شده است.','',
        'هر اجرا از last خود با حفظ وزن‌ها، optimizer، scheduler و شمارندهٔ epoch/step ادامه یافته است. فایل‌های مشتق فقط metadata/routing و مسیرهای callback را با رسید صریح تغییر داده‌اند؛ هش recursive stateها قبل و بعد برابر بود. مسیر تصادفی اجرای متوقف/ادامه‌یافته الزاماً معادل اجرای بدون وقفه نیست.','',
        '**SSC یک انحراف بودجهٔ صریح دارد:** ۷۰ دوره در برابر ۴۰ دورهٔ baseline. نرخ یادگیری، optimizer و plateau با patience9999 تغییر نکرده‌اند؛ این بازتولید هم‌بودجهٔ مقاله نیست. SHD100 از برنامهٔ اصلی300 و ECG70 از برنامهٔ400 استفاده می‌کنند و کوتاه‌تر از بودجهٔ کامل‌اند.','',
        'SHD* تقسیم تصادفی۲۰٪ از آموزش رسمی، splitseed42 است و با speaker-holdout مخلوط نشده است. ECG با ورودی signed و ماسک اصلی، دقت برچسب‌های زمانی را می‌سنجد؛ استقلال بیمار اثبات نشده است. آزمون‌ها قبلاً دیده شده‌اند، پس تمام نتایج اکتشافی‌اند و اثبات برتری پایدار یا معنی‌داری نیستند.','',
        '[مقادیر و شناسه‌های کامل](UPDATED_RESULTS.json) — [رسید ادامه‌ها](CONTINUATION_PLAN.json) — [جدول ۲ مقاله](../../../../reference_paper.pdf#page=11)']
    for ds,r in results.items():
        lines.append(f"\n- {ds}: validation={100*r['best_validation_accuracy']:.6f}%؛ test loss={r['metrics']['test_loss']:.8f}؛ SHA-256: `{r['checkpoint_sha256']}`.")
    (BASE/'UPDATED_TEST_TABLE_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if status=='complete':
        save(BASE/'FINAL_VERIFICATION.json',dict(status='complete',targets=TARGETS,raw_predictions_verified=True,
            no_unplanned_work_scheduled=True,updated_results_sha256=sha(BASE/'UPDATED_RESULTS.json')))
    print(json.dumps(dict(status=status,completed_tests={ds:100*r['metrics']['test_acc'] for ds,r in results.items()})),flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='freeze':freeze(sys.argv[2])
    elif action=='evaluate':evaluate(sys.argv[2])
    elif action=='report':write_report()
    elif action=='memory':
        check_source();ds=sys.argv[2];free,total=torch.cuda.mem_get_info(0);minimum={'SHD':14,'ECG':5,'SSC':18}[ds]
        assert free>=minimum*2**30,f'Insufficient free GPU memory before {ds}: {free/2**30:.2f}GiB'
        print(json.dumps(dict(dataset=ds,free_gib=free/2**30,required_free_gib=minimum)))
    else:raise ValueError('Unknown action')

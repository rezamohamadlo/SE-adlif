"""Explicit distribution-changing full-state migration and evaluation; never trains."""
import contextlib
import copy
import csv
import hashlib
import io
import json
import runpy
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from project_paths import ROOT,local_path
from module.study_control import canonical,forward_identity
from extended_seed_study import write_provenance

BASE=Path(__file__).resolve().parent
PRIOR=BASE.parent/'direction_shd100_v1'
MODEL='DIRECTION_AUG_DTH_RDROP_SE_adLIF'
STUDY='config/studies/dth_direction_aug_shd100_v1.yaml'
RUN=BASE/'runs'/MODEL/'SHD/seed_42'
ORIGIN=PRIOR/'runs/DIRECTION_DTH_RDROP_SE_adLIF/SHD/seed_42'
LIB=BASE.parent/'extended_100_70_70_v1/ACTIONS.py'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def library():return runpy.run_path(str(LIB),run_name='existing_state_audit_library')
def check_source():
    lib=library();lib['check_source'].__globals__['BASE']=BASE;lib['check_source']()
    assert sha(LIB)==read(BASE/'CONTINUATION_PLAN.json')['state_audit_library_sha256']
    assert sha(Path(__file__))==read(BASE/'CONTINUATION_PLAN.json')['audit_helper_sha256']
    for r in read(BASE/'CONTINUATION_PLAN.json')['checkpoints']:
        assert sha(local_path(r['source_checkpoint']))==r['source_checkpoint_sha256']

def assert_change(old,new,ds):
    assert ds=='SHD'
    a=copy.deepcopy(old);b=copy.deepcopy(new)
    allowed={'exp_name','model_variant','logdir','launcher_logdir','evaluation_protocol','paper_baseline_deviations',
        'study_file','study_idea','study_stage','study_hypothesis','stage_ceiling','ckpt_path',
        'study_record_initial_state','study_continuation_manifest'}
    for k in allowed:a.pop(k,None);b.pop(k,None)
    assert a['dataset']['_target_']=='datasets.shd.SHDLDM'
    assert b['dataset'].pop('training_frame_augmentation')==dict(version='shd_frame_augmentation_v1',time_scale=[.9,1.1],channel_shift=2,spike_delete_probability=.05)
    assert b['dataset']['_target_']=='datasets.shd_augmented.AugmentedSHDLDM'
    b['dataset']['_target_']=a['dataset']['_target_']
    assert a==b,'Undeclared model/loss/split/optimizer/schedule change'

def prepare():
    import runner
    assert not RUN.exists(),'Never overwrite prepared or started continuation'
    lib=library();captured=[];original=runner.build_plan
    def capture(*args,**kwargs):
        out=original(*args,**kwargs);captured.extend(out);return out
    with patch.object(runner,'build_plan',capture):
        runner.main(['--models',MODEL,'--datasets','SHD','--seeds','42','--study-file',STUDY,'--stage-ceiling','100','--dry-run'])
    assert len(captured)==1
    cfg=OmegaConf.to_container(OmegaConf.create(captured[0]['config']),resolve=True)
    source_last=ORIGIN/'ckpt/last.ckpt';boundary=read(PRIOR/'STOP_BOUNDARY_READY.json')
    assert sha(source_last)==boundary['last_checkpoint_sha256']
    state=torch.load(source_last,map_location='cpu',weights_only=False)
    assert state['epoch']+1==41 and state['global_step']==1025
    oldcfg=OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']),resolve=True)
    assert_change(oldcfg,cfg,'SHD')
    oldidentity=read(ORIGIN/'MODEL_SOURCE_IDENTITY.json');identity=forward_identity('SHD')
    assert all(identity[k]==v for k,v in oldidentity.items()),'Existing model/data math changed'
    RUN.mkdir(parents=True);(RUN/'ckpt').mkdir()
    shutil.copytree(ORIGIN/'logs',RUN/'logs')
    shutil.copytree(ORIGIN, RUN/'origin_history',ignore=shutil.ignore_patterns('ckpt','logs','stages','*.ckpt'))
    best_paths=list((ORIGIN/'ckpt').glob('epoch=*-step=*.ckpt'));assert len(best_paths)==1
    rebind=lib['rebind_state'];rebind.__globals__['assert_math_equivalence']=assert_change
    records=[]
    for kind,source in [('last',source_last),('best',best_paths[0])]:
        digest=sha(source);old=torch.load(source,map_location='cpu',weights_only=False)
        new=rebind(old,cfg,ORIGIN,RUN,'SHD');target=RUN/'ckpt'/source.name;torch.save(new,target)
        loaded=torch.load(target,map_location='cpu',weights_only=False)
        states={k:lib['tree_hash'](old[k]) for k in ['state_dict','optimizer_states','lr_schedulers','loops']}
        assert all(lib['tree_hash'](loaded[k])==v for k,v in states.items())
        assert old['epoch']==loaded['epoch'] and old['global_step']==loaded['global_step']
        assert sha(source)==digest
        records.append(dict(kind=kind,source_checkpoint=rel(source),source_checkpoint_sha256=digest,
            derived_checkpoint=rel(target),derived_checkpoint_sha256=sha(target),preserved_states=states,
            epoch_index=old['epoch'],global_step=old['global_step']))
    save(RUN/'FULL_RUN_CONFIG.json',cfg);save(RUN/'MODEL_SOURCE_IDENTITY.json',identity)
    changes={k:dict(original=oldcfg.get(k),continued=cfg.get(k)) for k in set(oldcfg)|set(cfg) if oldcfg.get(k)!=cfg.get(k)}
    origin=dict(mode='full_state_continuation_with_explicit_training_augmentation_change',starting_completed_epochs=41,
        target_completed_epochs=100,source_run=rel(ORIGIN),new_run=rel(RUN),checkpoints=records,configuration_changes=changes,
        weights_optimizer_scheduler_loops_preserved=True,pre_augmentation_best_remains_eligible=True,
        original_test_transforms_unchanged=True,fresh_training=False,weights_only=False)
    save(RUN/'CONTINUATION_ORIGIN.json',origin)
    save(BASE/'CONTINUATION_PLAN.json',dict(status='prepared',starting_completed_epochs=41,target_completed_epochs=100,
        training_seed=42,split_seed=42,checkpoints=records,configuration_changes=changes,
        state_audit_library=rel(LIB),state_audit_library_sha256=sha(LIB),audit_helper_sha256=sha(Path(__file__)),prior_test_exposure=True))
    audit=read(BASE.parent/'b001_rdrop_shd_v1/SPLIT_AND_SSC_STOP_AUDIT.json')
    import hydra
    dc=OmegaConf.create(cfg['dataset']);dc.data_path=str(local_path(dc.data_path))
    with contextlib.redirect_stdout(io.StringIO()):data=hydra.utils.instantiate(dc)
    assert list(data.data_train.indices)==audit['fitting_indices'] and list(data.data_val.indices)==audit['validation_indices']
    assert not set(data.data_train.indices)&set(data.data_val.indices)
    assert data.train_dataloader().dataset.full_transform is data.train_augmentation
    assert data.val_dataloader().dataset.full_transform is None
    save(BASE/'DATA_IDENTITIES.json',{'SHD':{audit['training_file']:audit['training_file_sha256']}})
    assert sha(ROOT/audit['training_file'])==audit['training_file_sha256']
    write_provenance(RUN,ROOT)
    save(BASE/'SOURCE_SNAPSHOT.json',lib['snapshot']())
    check_source();report()
    print('Full state41 -> total100 prepared; same split/validation; only declared training augmentation added.',flush=True)

def freeze():
    check_source();lib=library();f=lib['make_entry']
    f.__globals__.update(BASE=BASE,MODEL=MODEL,TARGETS={'SHD':100},check_source=check_source)
    e=f('SHD');e['augmentation_started_after_completed_epoch']=41
    e['selected_checkpoint_trained_with_augmentation']=e['selected_completed_epoch']>41
    out=BASE/'evaluation';out.mkdir(parents=True,exist_ok=True);assert not (out/'FROZEN_CHECKPOINTS.json').exists()
    oldplan=read(BASE.parent/'extended_100_70_70_v1/evaluations/SHD/FROZEN_CHECKPOINTS.json')
    source=BASE.parent/'b001_rdrop_shd_v1/evaluations/user_shd030_test_v1/EVALUATION_SOURCE.txt'
    text=source.read_text(encoding='utf-8').replace("OUT=ROOT/'results/experiments/dth_complement_v1/b001_rdrop_shd_v1/evaluations/user_shd030_test_v1'",f"OUT=ROOT/'{rel(out)}'")
    (out/'EVALUATION_SOURCE.txt').write_text(text,encoding='utf-8')
    save(out/'FROZEN_CHECKPOINTS.json',dict(status='frozen',purpose='exploratory_user_requested_stop_diagnostic',
        frozen_at=datetime.now(timezone.utc).isoformat(),checkpoints=[e],test_selects_ideas=False,resume_training_after_tests=False,
        prior_test_exposure=True,evaluation_source_sha256=sha(out/'EVALUATION_SOURCE.txt'),
        capture_source=oldplan['capture_source'],capture_source_sha256=oldplan['capture_source_sha256']))

def evaluate():
    check_source();runpy.run_path(str(BASE/'evaluation/EVALUATION_SOURCE.txt'),run_name='__main__')

def report():
    status=read(RUN/'STATUS.json') if (RUN/'STATUS.json').exists() else dict(completed_epochs=41,status='prepared')
    p=BASE/'evaluation/ALL_RESULTS.json';test=read(p)['results'][0] if p.exists() else None
    metrics=list(RUN.rglob('metrics.csv'));assert len(metrics)==1
    rows=list(csv.DictReader(metrics[0].open()));v=max((r for r in rows if r.get('val_acc')),key=lambda r:float(r['val_acc']));epoch=int(v['epoch'])
    t=next(r for r in rows if r.get('train_acc_epoch') and int(r['epoch'])==epoch)
    teststr=f"{100*test['metrics']['test_acc']:.2f}% ({100*test['metrics']['test_acc']-93.79:+.2f} pp)" if test else 'pending'
    text=f'''# ادامهٔ Direction-aware DTH + R-Drop با augmentation

| مدل / پروتکل SHD* | ایپک کامل | آموزش در checkpoint منتخب | بهترین اعتبارسنجی | ایپک منتخب | تست رسمی (Δ paper) |
|---|---:|---:|---:|---:|---:|
| مقاله SE-adLIF دو لایه | 300 | — | — | — | 93.79 ± 0.76% |
| Direction-aware DTH + R-Drop؛ augmentation از بعد ایپک41 | {status['completed_epochs']} / 100 | {100*float(t['train_acc_epoch']):.2f}% | {100*float(v['val_acc']):.2f}% | {epoch+1} | {teststr} |

مقیاس زمان0.9–1.1، جابه‌جایی حداکثر±2 کانال ورودی140کاناله، حذف مستقل5% اسپایک‌ها؛ فقط آموزش. اعتبارسنجی و تست بدون augmentation و بدون TTA. از last ایپک41 با حفظ optimizer/scheduler ادامه یافته است. بهترین checkpoint روی کل تاریخچه انتخاب می‌شود؛ ایپک‌های تا41 قبل از augmentation هستند. checkpoint منتخب فعلی متعلق به {'بعد' if epoch+1>41 else 'قبل'} از تغییر augmentation است.

Δ paper برابر محلی منهای میانگین مقاله با واحد درصد است. محلی تک‌بذر42 و مقاله میانگین±SD روی20 اجراست؛ بودجه و سابقه متفاوت‌اند و نتایج اکتشافی‌اند. این ادامهٔ تغییرکرده، آزمایش مستقل از ابتدا یا اثبات اثر augmentation نیست.

[تاریخچه و تنظیمات ادامه](CONTINUATION_PLAN.json) — [راهنما](../../../../docs/experiments/dth-direction-augmentation-continuation-v1.md) — [مقاله](../../../../reports/paper-results.md)
'''
    (BASE/'COMPARISON_FA.md').write_text(text,encoding='utf-8')
    if test:
        import numpy as np
        z=np.load(local_path(test['prediction_archive']))
        assert sha(local_path(test['prediction_archive']))==test['prediction_archive_sha256']
        assert abs(float((z['labels']==z['predictions']).mean())-test['metrics']['test_acc'])<1e-12
        save(BASE/'FINAL_VERIFICATION.json',dict(status='complete',raw_predictions_verified=True,
            selected_checkpoint_trained_with_augmentation=epoch+1>41,no_further_work_scheduled=True))

if __name__=='__main__':
    {'prepare':prepare,'verify':check_source,'freeze':freeze,'evaluate':evaluate,'report':report}[sys.argv[1]]()

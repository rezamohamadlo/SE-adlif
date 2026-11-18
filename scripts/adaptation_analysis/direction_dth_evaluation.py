"""Audit/freeze/evaluate/report only. Training is exclusively runner.py."""
import contextlib
import copy
import hashlib
import io
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf

# Works from another cwd, including Hydra's cwd.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from project_paths import local_path
from module.study_control import canonical, forward_identity
from module.initial_state import state_fingerprint
from models.pl_module import build_training_model

BASE=ROOT/'results/experiments/dth_complement_v1/direction_shd100_v1'
OLD=ROOT/'results/experiments/dth_complement_v1/extended_100_70_70_v1'
MODEL='DIRECTION_DTH_RDROP_SE_adLIF'
STUDY='config/studies/dth_direction_rdrop_shd100_v1.yaml'
RUN=BASE/'runs'/MODEL/'SHD/seed_42'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def save(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def source_paths():
    return sorted(set(list(forward_identity('SHD'))+[STUDY,'config/experiment/SHD_DIRECTION_DTH_RDROP_SE_adLIF.yaml',
        'runner.py','run.py','module/study_control.py','extended_seed_study.py',rel(Path(__file__))]))
def check_source():
    for p,h in read(BASE/'SOURCE_SNAPSHOT.json').items():assert sha(ROOT/p)==h,p
    a=read(BASE/'CONTROL_A.json')
    assert sha(local_path(a['checkpoint']))==a['checkpoint_sha256']
def prepare():
    import runner
    plans=[];original=runner.build_plan
    def capture(*a,**k):
        result=original(*a,**k);plans.extend(result);return result
    with patch.object(runner,'build_plan',capture):
        runner.main(['--models',MODEL,'--datasets','SHD','--seeds','42','--study-file',STUDY,'--stage-ceiling','100','--dry-run'])
    assert len(plans)==1
    cfg=plans[0]['config'];cfg=OmegaConf.to_container(OmegaConf.create(cfg),resolve=True)
    a=read(OLD/'UPDATED_RESULTS.json')['latest']['SHD']
    changed_sources=[p for p,h in a['model_source_identity'].items() if forward_identity('SHD')[p]!=h]
    assert changed_sources==['models/ours/dth_adlif.py'],changed_sources
    import subprocess
    legacy=subprocess.run(['git','show','HEAD:models/ours/dth_adlif.py'],cwd=ROOT,check=True,capture_output=True).stdout
    assert hashlib.sha256(legacy.replace(b'\r\n',b'\n')).hexdigest()==a['model_source_identity']['models/ours/dth_adlif.py']
    oldcfg=read(local_path(a['full_config']))
    meta={'exp_name','model_variant','logdir','launcher_logdir','evaluation_protocol','paper_baseline_deviations',
          'study_file','study_idea','study_stage','study_hypothesis','study_continuation_manifest','ckpt_path'}
    ca=copy.deepcopy(oldcfg);cb=copy.deepcopy(cfg)
    for layer in ('l1','l2'):
        assert cb[layer].pop('direction_aware') is True
    for k in meta:ca.pop(k,None);cb.pop(k,None)
    assert ca==cb,[(k,ca.get(k),cb.get(k)) for k in set(ca)|set(cb) if ca.get(k)!=cb.get(k)]
    assert cfg['stage_ceiling']==100 and cfg['n_epochs']==300 and not cfg.get('ckpt_path')
    torch.set_float32_matmul_precision('high');torch.manual_seed(42)
    with contextlib.redirect_stdout(io.StringIO()):model=build_training_model(OmegaConf.create(cfg))
    fp=state_fingerprint(model)
    original_init=ROOT/'results/experiments/dth_complement_v1/b001_rdrop_shd_v1/runs/DTH_RDROP_SE_adLIF/SHD/seed_42/INITIAL_MODEL_STATE.json'
    initial=read(original_init)
    for k,v in initial['tensors'].items():assert fp['tensors'][k]==v,k
    extra=set(fp['tensors'])-set(initial['tensors']);assert extra=={'l1.offset_gain','l2.offset_gain'}
    assert all(float(model.state_dict()[k])==0 for k in extra)
    assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()==initial['cpu_rng_sha256']
    audit=read(ROOT/'results/experiments/dth_complement_v1/b001_rdrop_shd_v1/SPLIT_AND_SSC_STOP_AUDIT.json')
    assert sha(ROOT/audit['training_file'])==audit['training_file_sha256']
    import hydra
    data_cfg=OmegaConf.create(cfg['dataset']);data_cfg.data_path=str(local_path(data_cfg.data_path))
    with contextlib.redirect_stdout(io.StringIO()):data=hydra.utils.instantiate(data_cfg)
    assert list(data.data_train.indices)==audit['fitting_indices'] and list(data.data_val.indices)==audit['validation_indices']
    assert not set(audit['fitting_indices']) & set(audit['validation_indices'])
    assert len(audit['fitting_indices'])+len(audit['validation_indices'])==8156
    import h5py,numpy as np
    with h5py.File(ROOT/audit['training_file'],'r') as f:labels=np.asarray(f['labels'])
    assert set(labels[audit['fitting_indices']])==set(labels[audit['validation_indices']])==set(range(20))
    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/'LEGACY_DTH_SOURCE.txt').write_bytes(legacy)
    save(BASE/'CONTROL_A.json',a);save(BASE/'PLANNED_CONFIG_B.json',cfg)
    save(BASE/'SOURCE_SNAPSHOT.json',{p:sha(ROOT/p) for p in source_paths()})
    save(BASE/'DATA_IDENTITIES.json',{'SHD':{audit['training_file']:audit['training_file_sha256']}})
    save(BASE/'PRELAUNCH_AUDIT.json',dict(status='verified',shared_configuration_equal_except_declared_metadata_and_direction=True,
        shared_initial_tensors_match=True,shared_cpu_rng_matches=True,extra_parameters=sorted(extra),
        changed_forward_sources=changed_sources,legacy_source_sha256=sha(BASE/'LEGACY_DTH_SOURCE.txt'),
        fitting_indices=audit['fitting_indices'],validation_indices=audit['validation_indices'],
        full_training_coverage=True,mode_B='fresh',budget=100,baseline_schedule=300,
        control_A_interrupted_resume_history=True,stochastic_trajectory_equivalence_claimed=False,
        prior_official_test_exposure=True,original_initial_state=rel(original_init),original_initial_state_sha256=sha(original_init)))
    report()
    print('Verified configuration, all shared initial tensors/RNG, fixed sample indices; no optimizer updates.',flush=True)

def initial_audit():
    expected=read(BASE/'PRELAUNCH_AUDIT.json');assert expected['status']=='verified'
    original=read(ROOT/expected['original_initial_state']);actual=read(RUN/'INITIAL_MODEL_STATE.json')
    for k,v in original['tensors'].items():assert actual['tensors'][k]==v,k
    for k in ['cpu_rng_sha256','cuda_rng_sha256','precision','float32_matmul_precision','cuda_allow_tf32','deterministic_algorithms']:
        assert actual[k]==original[k],k
    return dict(shared_initial_tensors_and_execution_flags_verified=True,initial_B_sha256=sha(RUN/'INITIAL_MODEL_STATE.json'))

def freeze():
    check_source();initial=initial_audit()
    # Reuse the audited receipt/history/epoch/checkpoint validator; no training.
    lib=runpy.run_path(str(OLD/'ACTIONS.py'),run_name='existing_receipt_validation')
    f=lib['make_entry'];f.__globals__.update(BASE=BASE,MODEL=MODEL,TARGETS={'SHD':100},check_source=check_source)
    e=f('SHD');e.pop('continuation_origin',None);e['initial_state_audit']=initial
    out=BASE/'evaluation';assert not (out/'FROZEN_CHECKPOINTS.json').exists()
    oldplan=read(OLD/'evaluations/SHD/FROZEN_CHECKPOINTS.json')
    source=ROOT/'results/experiments/dth_complement_v1/b001_rdrop_shd_v1/evaluations/user_shd030_test_v1/EVALUATION_SOURCE.txt'
    text=source.read_text(encoding='utf-8').replace("OUT=ROOT/'results/experiments/dth_complement_v1/b001_rdrop_shd_v1/evaluations/user_shd030_test_v1'",f"OUT=ROOT/'{rel(out)}'")
    out.mkdir(parents=True,exist_ok=True);(out/'EVALUATION_SOURCE.txt').write_text(text,encoding='utf-8')
    save(out/'FROZEN_CHECKPOINTS.json',dict(status='frozen',purpose='exploratory_user_requested_stop_diagnostic',
        frozen_at=datetime.now(timezone.utc).isoformat(),checkpoints=[e],test_selects_ideas=False,resume_training_after_tests=False,
        prior_test_exposure=True,evaluation_source_sha256=sha(out/'EVALUATION_SOURCE.txt'),
        capture_source=oldplan['capture_source'],capture_source_sha256=oldplan['capture_source_sha256']))

def evaluate():
    check_source();runpy.run_path(str(BASE/'evaluation/EVALUATION_SOURCE.txt'),run_name='__main__')

def report():
    a=read(BASE/'CONTROL_A.json');p=BASE/'evaluation/ALL_RESULTS.json'
    b=read(p)['results'][0] if p.exists() else None
    if b:
        import numpy as np
        z=np.load(local_path(b['prediction_archive']));assert sha(local_path(b['prediction_archive']))==b['prediction_archive_sha256']
        assert abs(float((z['labels']==z['predictions']).mean())-b['metrics']['test_acc'])<1e-12
    rows=[]
    for label,r in [('A: DTH + R-Drop',a),('B: Direction-aware DTH + R-Drop',b)]:
        if r:
            rows.append(f"| {label} | {r['completed_epochs']} | {100*r['training_accuracy_at_selected_epoch']:.2f}% | {100*r['best_validation_accuracy']:.2f}% | {r['selected_completed_epoch']} | {100*r['metrics']['test_acc']:.2f}% |")
        else:rows.append(f'| {label} | pending / 100 | pending | pending | pending | pending |')
    lines=['# مقایسهٔ SHD*: DTH + R-Drop و پاسخ جهت‌دار','',
        '| مدل | ایپک کامل | آموزش در checkpoint منتخب | بهترین اعتبارسنجی | ایپک منتخب | تست رسمی |',
        '|---|---:|---:|---:|---:|---:|',*rows,'',
        'بذر۴۲؛ تقسیم تصادفی۲۰٪ از آموزش رسمی با splitseed42؛ بدون TTA. انتخاب checkpoint فقط با اعتبارسنجی. آزمایش اکتشافی و تک‌بذر است.',
        'A کنترل تاریخی با توقف/ادامه است؛ B از ابتدا با همان مقادیر پارامترهای مشترک و همان تنظیمات آموزش می‌بیند. برابری مسیر تصادفی ادعا نمی‌شود. تنها تغییر محاسباتی پاسخ گذرای آستانه و دو scalar جدید است.']
    if b:lines+=['',f"اختلاف تست B−A: {100*(b['metrics']['test_acc']-a['metrics']['test_acc']):+.4f} واحد درصد. این اختلاف تک‌بذر اثبات برتری پایدار نیست."]
    (BASE/'COMPARISON_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(BASE/'COMPARISON.json',dict(status='complete' if b else 'pending',A=a,B=b,ordinary_inference=True,
        historical_control_resume_caveat=True,test_selected_followups=False))
    if b:save(BASE/'FINAL_VERIFICATION.json',dict(status='complete',raw_predictions_verified=True,initial_state=initial_audit(),
        final_table_sha256=sha(BASE/'COMPARISON_FA.md'),no_further_work_scheduled=True))

if __name__=='__main__':
    action=sys.argv[1]
    {'prepare':prepare,'freeze':freeze,'evaluate':evaluate,'report':report,'verify':check_source}[action]()

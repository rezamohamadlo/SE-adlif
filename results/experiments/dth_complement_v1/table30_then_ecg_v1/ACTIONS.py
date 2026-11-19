"""Bounded study preparation/audits/evaluation. Main training uses runner.py."""
import ast
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from project_paths import ROOT, local_path, portable_config
from module.study_control import canonical, forward_identity
from extended_seed_study import source_identity, read_provenance

TASK=Path(__file__).resolve().parent
BASE=TASK.parent
SSC=BASE/'b001_rdrop_ssc_v1'
SHD=BASE/'b001_rdrop_shd_v1'
ECG=BASE/'b001_rdrop_ecg_v1'
SSC_EVAL=TASK/'ssc_control30_test'
ECG_EVAL=TASK/'ecg30_test'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def rel(path):return Path(path).relative_to(ROOT).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def rows(run):
    paths=list(run.rglob('metrics.csv'));assert len(paths)==1
    return paths,list(csv.DictReader(paths[0].open()))
def source_snapshot():
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')
    files={}
    for name in sorted(set(names)-{''}):
        p=Path(name)
        if p.parts[0] in ('results','data','docs','reports','archive'):continue
        if p.suffix not in ('.py','.yaml','.yml') and name!='requirements.txt':continue
        if (ROOT/p).is_file():files[name]=sha(ROOT/p)
    return dict(identity=source_identity(ROOT),files_sha256=files)

def assert_prepared_source():
    snapshot=read(TASK/'preparation_history/SOURCE_SNAPSHOT_AFTER_SHD.json')
    assert source_identity(ROOT)['source_files_sha256']==snapshot['identity']['source_files_sha256']
    for name,digest in snapshot['files_sha256'].items():assert sha(ROOT/name)==digest,name

def prepare():
    assert read(SHD/'SEQUENCE_STATUS.json')['phase']=='completed_training_tests_report'
    assert len(read(SHD/'evaluations/user_shd030_test_v1/ALL_RESULTS.json')['results'])==2
    history=TASK/'preparation_history';history.mkdir(exist_ok=True)
    assert not (history/'PREPARATION_COMPLETE.json').exists()
    study_path=ROOT/'config/studies/dth_complement_B001_ssc_v1.yaml'
    old_text=study_path.read_text();old=OmegaConf.to_container(OmegaConf.create(old_text))
    assert old['datasets']['SSC']['ceilings']==[10,20,40]
    shutil.copy2(study_path,history/'SSC_STUDY_BEFORE30.yaml')
    new_text=old_text.replace('ceilings: [10, 20, 40]','ceilings: [10, 20, 30, 40]')
    new=OmegaConf.to_container(OmegaConf.create(new_text));check=copy.deepcopy(new)
    assert new['datasets']['SSC']['ceilings']==[10,20,30,40]
    check['datasets']['SSC']['ceilings']=[10,20,40];assert check==old
    study_path.write_text(new_text,encoding='utf-8')
    for src,dst in [('ECG_DTH_RDROP_SE_adLIF_2layer.template.yaml','config/experiment/ECG_DTH_RDROP_SE_adLIF_2layer.yaml'),('dth_complement_B001_ecg30_v1.template.yaml','config/studies/dth_complement_B001_ecg30_v1.yaml')]:
        target=ROOT/dst;assert not target.exists();shutil.copy2(TASK/src,target)
    # Archive original records. Keep every historical launch entry unchanged.
    run=SSC/'runs/DTH_SE_adLIF/SSC/seed_42'
    original=history/'SSC_CONTROL_LAUNCH_ENVIRONMENTS_BEFORE30.json'
    shutil.copy2(run/'LAUNCH_ENVIRONMENTS.json',original)
    if (run/'RESUME_SOURCE_TRANSITION.json').exists():shutil.copy2(run/'RESUME_SOURCE_TRANSITION.json',history/'SSC_CONTROL_TRANSITION_BEFORE30.json')
    assert read(run/'MODEL_SOURCE_IDENTITY.json')==forward_identity('SSC')
    receipt=read(run/'stages/010/RESULT.json')
    assert sha(run/'ckpt/last.ckpt')==receipt['last_checkpoint_sha256']
    state=torch.load(run/'ckpt/last.ckpt',map_location='cpu',weights_only=False)
    assert state['epoch']==9 and state['global_step']==2940 and state['optimizer_states'] and state['lr_schedulers']
    paths,_=rows(run);shutil.copy2(paths[0],history/'SSC_CONTROL_METRICS_BEFORE30.csv')
    # Model/data/loss implementation unchanged; only staged boundary and new
    # independent ECG experiment definitions have been added.
    snapshot=source_snapshot();snapshot_path=history/'SOURCE_SNAPSHOT_AFTER_SHD.json';save(snapshot_path,snapshot)
    transition=dict(version='identical_training_source_resume_v1',created_at=datetime.now(timezone.utc).isoformat(),
        original_provenance=rel(original),original_provenance_sha256=sha(original),
        source_snapshot=rel(snapshot_path),source_snapshot_sha256=sha(snapshot_path),
        guard_only_amendments={'extended_seed_study.py':sha(ROOT/'extended_seed_study.py')},
        resumed_source_identity=snapshot['identity'],
        reason='User requests matched30 table. Resume the existing DTH control10to30. Model/data/loss forward identity exactly unchanged; only allowed ceiling30 and independent ECG definitions added. Original config/provenance/checkpoints preserved.')
    save(run/'RESUME_SOURCE_TRANSITION.json',transition);read_provenance(run,ROOT)
    save(history/'PREPARATION_COMPLETE.json',dict(only_SSC_boundary_added=True,model_source_unchanged=True,
        original_control_last_epoch=10,source_identity=snapshot['identity'],ECG_definitions_prepared=True,ECG_training_started=False))
    print('Prepared own-last SSC control10to30 and independent ECG30 definitions.',flush=True)

def entry(run,dataset,budget):
    assert_prepared_source()
    receipt_path=run/'stages'/f'{budget:03d}'/'RESULT.json';receipt=read(receipt_path)
    assert receipt['status']=='stage_complete' and receipt['completed_epochs']==budget
    best=local_path(receipt['best_checkpoint'],must_exist=True);last=local_path(receipt['last_checkpoint'],must_exist=True)
    assert sha(best)==receipt['best_checkpoint_sha256'] and sha(last)==receipt['last_checkpoint_sha256']
    selected=torch.load(best,map_location='cpu',weights_only=False);final=torch.load(last,map_location='cpu',weights_only=False)
    assert final['epoch']==budget-1 and final['optimizer_states'] and final['lr_schedulers']
    expected_steps={'SSC':294,'ECG':9}[dataset]*budget;assert final['global_step']==expected_steps
    cfg=OmegaConf.to_container(OmegaConf.create(selected['hyper_parameters']['cfg']),resolve=True)
    assert canonical(cfg)==canonical(read(run/'FULL_RUN_CONFIG.json'))
    assert read(run/'MODEL_SOURCE_IDENTITY.json')==forward_identity(dataset)
    assert cfg['paper_baseline_dataset']==dataset and not cfg['run_test'] and cfg['dataset']['validation_only_data']
    paths,history=rows(run);validation=[r for r in history if r.get('val_acc')]
    assert [int(r['epoch']) for r in validation]==list(range(budget))
    bestrow=max(validation,key=lambda r:float(r['val_acc']));index=int(bestrow['epoch'])
    assert index==selected['epoch']==receipt['selected_epoch_index']
    assert float(bestrow['val_acc'])==receipt['best_validation_accuracy']
    training=[r for r in history if r.get('train_acc_epoch') and int(r['epoch'])==index]
    last_training=[r for r in history if r.get('train_acc_epoch') and int(r['epoch'])==budget-1]
    assert len(training)==len(last_training)==1
    return dict(dataset=dataset,model=cfg['model_variant'],seed=42,checkpoint=rel(best),checkpoint_sha256=sha(best),
        last_checkpoint=rel(last),last_checkpoint_sha256=sha(last),stage_receipt=rel(receipt_path),stage_receipt_sha256=sha(receipt_path),
        full_config=rel(run/'FULL_RUN_CONFIG.json'),model_source_identity=forward_identity(dataset),
        metrics_files_sha256={rel(p):sha(p) for p in paths},completed_epochs=budget,full_budget=cfg['n_epochs'],
        selected_epoch_index=index,selected_completed_epoch=index+1,selected_global_step=selected['global_step'],
        training_accuracy_at_selected_epoch=float(training[0]['train_acc_epoch']),training_accuracy_final_epoch=float(last_training[0]['train_acc_epoch']),
        best_validation_accuracy=float(bestrow['val_acc']),validation_loss_at_selected_epoch=float(bestrow['val_loss']),protocol=cfg['evaluation_protocol'],
        test_data={'SSC':'data/SSCWrapper/ssc_test.h5','ECG':'data/QTDB/QTDB_test.mat'}[dataset])

def freeze_ssc():
    SSC_EVAL.mkdir(exist_ok=True);assert not (SSC_EVAL/'FROZEN_CHECKPOINTS.json').exists()
    e=entry(SSC/'runs/DTH_SE_adLIF/SSC/seed_42','SSC',30)
    old=ROOT/'results/experiments/complementary_speaker_dev_v2/evaluations/user_stop_official_test_v1'
    prior=read(old/'FROZEN_CHECKPOINTS.json')
    text=(old/'EVALUATION_SOURCE.txt').read_text().replace("OUT=ROOT/'results/experiments/complementary_speaker_dev_v2/evaluations/user_stop_official_test_v1'",f"OUT=ROOT/'{rel(SSC_EVAL)}'")
    (SSC_EVAL/'EVALUATION_SOURCE.txt').write_text(text,encoding='utf-8')
    save(SSC_EVAL/'FROZEN_CHECKPOINTS.json',dict(status='frozen',purpose='exploratory_user_requested_stop_diagnostic',
        frozen_at=datetime.now(timezone.utc).isoformat(),test_selects_ideas=False,resume_training_after_tests=False,
        checkpoints=[e],evaluation_source_sha256=sha(SSC_EVAL/'EVALUATION_SOURCE.txt'),capture_source=prior['capture_source'],
        capture_source_sha256=prior['capture_source_sha256'],prior_test_exposure=True,selection_window_epochs=30))
    print('Frozen SSC control30 validation-selected checkpoint.',flush=True)

def verify_result(r):
    assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
    assert r['ordinary_inference'] and not r['test_time_augmentation'] and r['weights_and_rng_unchanged']
    assert r['inference_forward_calls']==r['batches']
    assert sha(ROOT/r['prediction_archive'])==r['prediction_archive_sha256']
    z=np.load(ROOT/r['prediction_archive']);correct=z['labels']==z['predictions']
    assert len(correct)==r['metrics']['test_samples'] and int(correct.sum())==r['metrics']['test_correct']
    assert float(correct.mean())==r['metrics']['test_acc']

def build_table():
    shd=read(SHD/'evaluations/user_shd030_test_v1/ALL_RESULTS.json')['results'];assert len(shd)==2
    control=read(SSC_EVAL/'ALL_RESULTS.json')['results'][0]
    candidate=read(SSC/'evaluations/user_epoch019_ssc_test_v1/ALL_RESULTS.json')['results'][0]
    run=SSC/'runs/DTH_RDROP_SE_adLIF/SSC/seed_42';_,history=rows(run)
    validation=[r for r in history if r.get('val_acc') and int(r['epoch'])<30]
    assert len(validation)==30
    chosen=max(validation,key=lambda r:float(r['val_acc']))
    assert int(chosen['epoch'])==candidate['selected_epoch_index'] and float(chosen['val_acc'])==candidate['best_validation_accuracy']
    assert sha(run/'stages/020/best.ckpt')==candidate['checkpoint_sha256']
    for r in [*shd,control,candidate]:verify_result(r)
    assert control['completed_epochs']==30 and all(r['completed_epochs']==30 for r in shd)
    assert shd[0]['model']=='DTH_SE_adLIF' and shd[1]['model']=='DTH_RDROP_SE_adLIF'
    initial=[read(SSC/'runs'/m/'SSC/seed_42/INITIAL_MODEL_STATE.json') for m in ('DTH_SE_adLIF','DTH_RDROP_SE_adLIF')]
    assert initial[0]['model_state_sha256']==initial[1]['model_state_sha256']
    table=dict(status='complete',created_at=datetime.now(timezone.utc).isoformat(),training_seed=42,selection_window_epochs=30,
        SHD_DTH=shd[0],SHD_DTH_RDrop=shd[1],SSC_DTH=control,SSC_DTH_RDrop=candidate,
        SSC_candidate_actual_completed_epochs=31,SSC_candidate_selected_using_first30=True,
        SSC_candidate_test_reused_exact_weights=True,restart_histories_differ=True,exploratory=True)
    save(TASK/'TABLE_RESULTS.json',table)
    lines=['# جدول کامل تست: سقف انتخاب ۳۰ دوره، بذر ۴۲','',
        'همهٔ اعداد، دقت تست رسمی با استنتاج معمولی تک‌گذر و بدون TTA هستند. checkpointها با اعتبارسنجی انتخاب شدند. Δ paper برابر نتیجهٔ محلی منهای میانگین مقاله بر حسب واحد درصد (pp)، با محاسبه از مقادیر گردنشده است. مقایسهٔ تک‌بذر با میانگین مقاله توصیفی است.','',
        '| دیتاست | مقاله، میانگین ± SD | تست DTH (Δ paper) | تست DTH + R-Drop (Δ paper) | اختلاف ترکیب با DTH |','|---|---:|---:|---:|---:|']
    for label,paper,sd,a,b in [('SHD*',93.79,.76,shd[0],shd[1]),('SSC',80.44,.26,control,candidate)]:
        av=100*a['metrics']['test_acc'];bv=100*b['metrics']['test_acc']
        lines.append(f'| {label} | {paper:.2f} ± {sd:.2f}% | {av:.2f}% ({av-paper:+.2f} pp) | {bv:.2f}% ({bv-paper:+.2f} pp) | {bv-av:+.2f} pp |')
    lines+=['','مبنای انتخاب هر چهار خانه، بیشترین دقت اعتبارسنجی در **۳۰ دورهٔ اول** است. ترکیب SSC قبلاً در پایان دورهٔ ۳۱ متوقف شده بود؛ بررسی فقط ۳۰ دورهٔ اول، همان checkpoint دورهٔ ۱۹ را انتخاب می‌کند. هویت وزن‌ها با نتیجهٔ تست قبلی دقیقاً برابر است؛ نتیجهٔ دیگری بر اساس score تست جایگزین نشده است. کنترل SSC از last خود در دورهٔ ۱۰ تا ۳۰ با بازیابی optimizer و scheduler ادامه یافت. تاریخچهٔ توقف و ادامهٔ دو بازو متفاوت است و هم‌ارزی دقیق با آموزش بدون وقفه ادعا نمی‌شود.','',
        'SHD* اعتبارسنجی تصادفی ۲۰٪ از آموزش رسمی با splitseed42 است، نه speaker-holdout. این بودجهٔ ۳۰دوره‌ای کوتاه‌تر از SHD300 و SSC40 کامل است. نتایج اکتشافی و تک‌بذرند؛ برتری پایدار یا معنی‌داری را اثبات نمی‌کنند.','',
        '| دیتاست / مدل | بهترین اعتبارسنجی | آموزش در دورهٔ منتخب | loss تست | دورهٔ checkpoint |','|---|---:|---:|---:|---:|']
    for name,r in [('SHD* / DTH',shd[0]),('SHD* / DTH + R-Drop',shd[1]),('SSC / DTH',control),('SSC / DTH + R-Drop',candidate)]:
        lines.append(f"| {name} | {100*r['best_validation_accuracy']:.2f}% | {100*r['training_accuracy_at_selected_epoch']:.2f}% | {r['metrics']['test_loss']:.6f} | {r['selected_completed_epoch']} |")
    lines+=['','[نتایج کامل و شناسه‌های checkpoint](TABLE_RESULTS.json) — [جدول ۲ مقاله، صفحهٔ ۱۱](../../../../reference_paper.pdf#page=11)','',
        'جدول کامل شده است؛ فقط اکنون مرحلهٔ مستقل ECG طبق درخواست کاربر مجاز است. هیچ خانهٔ این جدول pending نیست.']
    (TASK/'TEST_TABLE_30_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({key:100*table[key]['metrics']['test_acc'] for key in ['SHD_DTH','SHD_DTH_RDrop','SSC_DTH','SSC_DTH_RDrop']}),flush=True)

def ecg_config(seed=117):
    from hydra import compose,initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
        return compose(config_name='main',overrides=['experiment=ECG_DTH_RDROP_SE_adLIF_2layer','+comparison=ECG',f'random_seed={seed}',
            'run_test=false','++dataset.validation_only_data=true','++dataset.validation_batch_size=64','dataset.num_workers=0','++model_variant=DTH_RDROP_SE_adLIF'])

def calibrate_ecg():
    assert read(TASK/'TABLE_RESULTS.json')['status']=='complete'
    assert_prepared_source()
    from hydra.utils import instantiate
    from models.pl_module import build_training_model
    ECG.mkdir(exist_ok=True)
    cfg=ecg_config();torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(117)
    data=instantiate(cfg.dataset,data_path=str(ROOT/'data'));data.setup('fit')
    fit=list(data.data_train.indices);valid=list(data.data_val.indices)
    assert len(fit)==588 and len(valid)==30 and not set(fit)&set(valid)
    raw_x,raw_y=data.train_val_dataset_.ds.tensors
    assert tuple(raw_x.shape)==(618,1301,4) and raw_y.shape==(618,1301)
    assert set(torch.unique(raw_x).tolist())=={-1.,0.,1.}
    assert set(torch.unique(raw_y[valid]).tolist())==set(range(6))
    assert not hasattr(data,'data_test')
    with contextlib.redirect_stdout(io.StringIO()):model=build_training_model(cfg)
    device=torch.device(str(cfg.device));model.to(device).train();optimizer=model.configure_optimizers()['optimizer']
    batch=next(iter(data.train_dataloader()));batch=tuple(v.to(device) for v in batch)
    assert batch[0].shape==(64,1301,4) and batch[1].dtype==torch.int64
    assert batch[1].shape==(64,1302) and torch.all(batch[1][:,0]==-1)
    assert torch.all(batch[2][:,0]==1) and torch.all(batch[2][:,-1]==1301)
    measured=[];terms=[]
    for step in range(4):
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize(device);started=time.perf_counter()
        if step==2:torch.cuda.reset_peak_memory_stats(device)
        _,loss,diagnostics=model.training_terms(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip_val);optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        torch.cuda.synchronize(device)
        if step>=2:measured.append(time.perf_counter()-started)
        terms.append({k:float(v) for k,v in diagnostics.items()})
        assert abs(float(loss)-terms[-1]['train_supervised_loss']-terms[-1]['train_dropout_symmetric_kl'])<1e-5
    assert any(d['train_dropout_symmetric_kl']>0 for d in terms)
    total=torch.cuda.get_device_properties(device).total_memory;peak=torch.cuda.max_memory_reserved(device)
    assert (total-peak)/total>=.15
    model.eval();model.log=lambda *a,**k:None;model.log_dict=lambda *a,**k:None
    calls=[];hook=model.register_forward_hook(lambda *a:calls.append(1));started=time.perf_counter()
    with torch.inference_mode():
        for i,b in enumerate(data.val_dataloader()):
            loss=model.validation_step(tuple(v.to(device) for v in b),i);assert torch.isfinite(loss)
    torch.cuda.synchronize(device);validation_seconds=time.perf_counter()-started;hook.remove()
    assert len(calls)==1
    audit=dict(status='verified',selected_batch=64,disposable_seed=117,optimizer_steps=4,warmup_steps=2,measured_steps=2,
        measured_step_seconds=measured,peak_reserved_gib=peak/2**30,memory_headroom_fraction=(total-peak)/total,
        training_terms=terms,original_signed_values_and_masks_preserved=True,all_gradients_and_parameters_finite=True,
        original_training_file_sha256=sha(ROOT/'data/QTDB/QTDB_train.mat'),original_validation_split_seed=42,
        fitting_indices=fit,validation_indices=valid,validation_count=30,official_test_opened=False,
        model_source_identity=forward_identity('ECG'),calibration_state_discarded=True,main_weights_reused=False,
        batch_selection_reason='Retain inherited paper batch64, verified actual DTH+RDrop memory and throughput; no LR/batch search')
    save(ECG/'ECG_TRAINING_CALIBRATION.json',audit)
    save(ECG/'ECG_VALIDATION_CALIBRATION.json',dict(status='verified',selected_batch=64,original_validation_records=30,
        ordinary_single_forward_calls=1,validation_seconds=validation_seconds,official_test_opened=False))
    save(ECG/'DATA_AUDIT.json',audit)
    print(json.dumps(dict(ECG_preflight='verified',batch=64,step_seconds=statistics.mean(measured),headroom=(total-peak)/total)),flush=True)

def freeze_ecg():
    assert read(TASK/'TABLE_RESULTS.json')['status']=='complete'
    ECG_EVAL.mkdir(exist_ok=True);assert not (ECG_EVAL/'FROZEN_CHECKPOINTS.json').exists()
    e=entry(ECG/'runs/DTH_RDROP_SE_adLIF/ECG/seed_42','ECG',30)
    audit=read(ECG/'DATA_AUDIT.json');assert sha(ROOT/'data/QTDB/QTDB_train.mat')==audit['original_training_file_sha256']
    save(ECG_EVAL/'FROZEN_CHECKPOINTS.json',dict(status='frozen',purpose='exploratory_user_requested_ECG_transfer30',
        frozen_at=datetime.now(timezone.utc).isoformat(),checkpoints=[e],selection_window_epochs=30,
        test_selects_ideas=False,prior_test_exposure=True,main_fresh_from_scratch=True,
        fitting_indices=audit['fitting_indices'],validation_indices=audit['validation_indices'],
        evaluation_source=rel(TASK/'ACTIONS.py'),evaluation_source_sha256=sha(TASK/'ACTIONS.py'),
        table_complete_before_ECG=True))
    print('Frozen ECG30 checkpoint using validation only.',flush=True)

def evaluate_ecg():
    from hydra.utils import instantiate
    from models.pl_module import build_training_model
    plan_path=ECG_EVAL/'FROZEN_CHECKPOINTS.json';plan=read(plan_path);assert plan['status']=='frozen'
    assert sha(TASK/'ACTIONS.py')==plan['evaluation_source_sha256']
    assert not (ECG_EVAL/'TEST_RESULTS.json').exists()
    e=plan['checkpoints'][0]
    assert sha(ROOT/e['checkpoint'])==e['checkpoint_sha256'] and forward_identity('ECG')==e['model_source_identity']
    assert sha(ROOT/e['stage_receipt'])==e['stage_receipt_sha256']
    for name,digest in e['metrics_files_sha256'].items():assert sha(ROOT/name)==digest
    state=torch.load(ROOT/e['checkpoint'],map_location='cpu',weights_only=False)
    saved=OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']),resolve=True)
    assert canonical(saved)==canonical(read(ROOT/e['full_config']))
    assert state['epoch']==e['selected_epoch_index'] and saved['loss_agg']=='summed_membrane_potentials'
    cfg=OmegaConf.create(portable_config(saved));cfg.dataset.data_path=str(ROOT/'data');cfg.dataset.num_workers=0
    test_path=ROOT/e['test_data'];assert test_path.is_file()
    # Test access starts only here, after the complete validation freeze.
    cfg.dataset.validation_only_data=False
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(42)
    data=instantiate(cfg.dataset);data.setup('test')
    assert list(data.data_train.indices)==plan['fitting_indices'] and list(data.data_val.indices)==plan['validation_indices']
    expected=data.data_test.ds.tensors[1].numpy().reshape(-1)
    with contextlib.redirect_stdout(io.StringIO()):model=build_training_model(cfg)
    model.load_state_dict(state['state_dict'],strict=True);device=torch.device(str(cfg.device));model.to(device).eval()
    model.log=lambda *a,**k:None;model.log_dict=lambda *a,**k:None
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(device).clone()
    original=model.update_and_log_metrics;labels=[];predictions=[];record_ids=[];time_ids=[];cursor=0;updates=0
    def observed(outputs,targets,loss,metrics,prefix):
        nonlocal cursor,updates
        assert prefix=='test_' and not model.training and targets.ndim==2
        assert outputs.shape==(targets.numel(),6)
        valid=targets.ne(model.ignore_target_idx);assert torch.all(valid[:,0]==False)
        mask=valid.flatten();labels.extend(targets.flatten()[mask].cpu().tolist())
        predictions.extend(outputs.argmax(-1)[mask].cpu().tolist())
        count,width=targets.shape
        record_ids.extend(torch.arange(cursor,cursor+count,device=device)[:,None].expand(count,width)[valid].cpu().tolist())
        time_ids.extend(torch.arange(-1,width-1,device=device)[None,:].expand(count,width)[valid].cpu().tolist())
        cursor+=count;updates+=1
        return original(outputs,targets,loss,metrics,prefix)
    model.update_and_log_metrics=observed;calls=[];hook=model.register_forward_hook(lambda *a:calls.append(1))
    loader=data.test_dataloader();assert isinstance(loader.sampler,torch.utils.data.SequentialSampler) and not loader.drop_last
    loader.generator=torch.Generator().manual_seed(42)
    total=0.;n=0;started=time.perf_counter()
    print('[ECG_TEST_START] '+e['checkpoint_sha256'],flush=True)
    with torch.inference_mode():
        for i,b in enumerate(loader):
            batch=tuple(v.to(device) for v in b);count=int(batch[1].ne(model.ignore_target_idx).sum())
            loss=model.test_step(batch,i);assert torch.isfinite(loss)
            total+=float(loss)*count;n+=count
    torch.cuda.synchronize(device);elapsed=time.perf_counter()-started;hook.remove();model.update_and_log_metrics=original
    labels=np.asarray(labels,dtype=np.int64);predictions=np.asarray(predictions,dtype=np.int64)
    assert np.array_equal(labels,expected) and cursor==len(data.data_test) and len(calls)==updates==len(loader)
    assert len(labels)==n and np.all(np.asarray(time_ids)>=0)
    accuracy=float((labels==predictions).mean());assert math.isclose(float(model.test_metric.compute()['test_acc']),accuracy,abs_tol=1e-6)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    assert all(torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())
    archive=ECG_EVAL/'PREDICTIONS.npz';np.savez_compressed(archive,record_ids=np.asarray(record_ids),time_indices=np.asarray(time_ids),labels=labels,predictions=predictions)
    result=dict(e,status='evaluated',evaluation_plan_sha256=sha(plan_path),test_data_sha256=sha(test_path),
        metrics=dict(test_acc=accuracy,test_loss=total/n,test_valid_timesteps=n,test_records=cursor,test_correct=int((labels==predictions).sum())),
        ordinary_inference=True,test_time_augmentation=False,weights_and_rng_unchanged=True,inference_forward_calls=len(calls),batches=len(loader),
        prediction_archive=rel(archive),prediction_archive_sha256=sha(archive),evaluation_seconds=elapsed,
        evaluation_execution_deviations=['repository-root path resolution','num_workers0','explicit frozen test access','logging sinks suppressed without changing original test_step'],
        accuracy_unit='Valid time-step labels with original masks; not per-record classification',patient_independence_not_established=True)
    save(ECG_EVAL/'TEST_RESULTS.json',result)
    print(json.dumps(dict(ECG_test_accuracy_percent=100*accuracy,valid_time_steps=n,test_records=cursor)),flush=True)

def final_report():
    table=read(TASK/'TABLE_RESULTS.json');assert table['status']=='complete'
    ecg=read(ECG_EVAL/'TEST_RESULTS.json');assert ecg['status']=='evaluated'
    assert sha(ROOT/ecg['checkpoint'])==ecg['checkpoint_sha256']
    assert sha(ROOT/ecg['prediction_archive'])==ecg['prediction_archive_sha256']
    z=np.load(ROOT/ecg['prediction_archive']);accuracy=float((z['labels']==z['predictions']).mean());assert accuracy==ecg['metrics']['test_acc']
    value=100*accuracy
    lines=[(TASK/'TEST_TABLE_30_FA.md').read_text(encoding='utf-8'),'','## نتیجهٔ مرحلهٔ بعدی ECG','',
        'این مرحله فقط پس از کامل‌شدن جدول بالا اجرا شد. DTH + R-Drop از ابتدا و مستقل از وزن‌های SSC/SHD با بذر ۴۲، batch64 و تنظیمات مخصوص ECG به‌مدت ۳۰ دوره آموزش دید؛ scheduler برنامهٔ اصلی ۴۰۰دوره‌ای حفظ شد. این بررسی زودهنگام است، نه بازتولید کامل مقاله. کنترل DTH روی ECG در این درخواست آموزش داده نشد.','',
        '| مدل / ECG | دقت تست (Δ paper) | loss تست | دورهٔ checkpoint |','|---|---:|---:|---:|',
        '| مقاله، SE-adLIF دولایه | 88.18 ± 0.36% | گزارش نشده | — |',
        f"| DTH + R-Drop؛ ۳۰ دوره | {value:.2f}% ({value-88.18:+.2f} pp) | {ecg['metrics']['test_loss']:.6f} | {ecg['selected_completed_epoch']} |",'',
        f"بهترین اعتبارسنجی ECG: {100*ecg['best_validation_accuracy']:.2f}%. دقت تست روی {ecg['metrics']['test_valid_timesteps']} برچسب زمانی معتبر در {ecg['metrics']['test_records']} رکورد، با ماسک و مقادیر ورودی signed اصلی محاسبه شد؛ نسبت به رکورد نیست. شناسهٔ بیمار در داده موجود نیست و استقلال بیمار ادعا نمی‌شود.",'',
        '[نتیجهٔ کامل ECG](ecg30_test/TEST_RESULTS.json) — [فهرست ثابت قبل از آزمون](ecg30_test/FROZEN_CHECKPOINTS.json)','',
        'تمام مراحل مجاز پایان یافته‌اند؛ هیچ ایده، ادامهٔ آموزش یا انتخاب تنظیم بر اساس این scoreها انجام نمی‌شود.']
    (TASK/'FINAL_REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(TASK/'FINAL_RESULTS.json',dict(table=table,ECG_DTH_RDrop=ecg,status='complete',further_work_scheduled=False))
    p=ROOT/'command.txt';p.write_text('# COMPLETE: requested four30epoch-window test cells plus subsequent ECG DTH+RDrop30 test.\n# Final report: results/experiments/dth_complement_v1/table30_then_ecg_v1/FINAL_REPORT_FA.md\n# No more stages scheduled. Historical commands below.\n\n'+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print('ALL_USER_REQUESTED_TESTS_COMPLETE '+json.dumps(dict(ECG_test_accuracy_percent=value)),flush=True)

if __name__=='__main__':
    actions={'prepare':prepare,'freeze_ssc':freeze_ssc,'table':build_table,'calibrate_ecg':calibrate_ecg,
             'freeze_ecg':freeze_ecg,'evaluate_ecg':evaluate_ecg,'final_report':final_report}
    assert len(sys.argv)==2 and sys.argv[1] in actions
    actions[sys.argv[1]]()

"""Freeze both matched SHD validation selections before official-test access."""
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
from omegaconf import OmegaConf
from extended_seed_study import source_identity
from module.study_control import canonical, forward_identity
from project_paths import ROOT

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def rel(path):
    return Path(path).relative_to(ROOT).as_posix()

assert not (OUT/'FROZEN_CHECKPOINTS.json').exists(), 'Preserve previous evaluation'
audit_path=BASE/'SPLIT_AND_SSC_STOP_AUDIT.json'
audit=json.loads(audit_path.read_text())
assert sha(ROOT/audit['training_file'])==audit['training_file_sha256']
snapshot=json.loads((BASE/'STUDY_SOURCE_SNAPSHOT.json').read_text())
assert source_identity(ROOT)['source_files_sha256']==snapshot['identity']['source_files_sha256']
for name,digest in snapshot['files_sha256'].items():
    assert sha(ROOT/name)==digest,name
entries=[]
initials=[]
configs=[]
for model in ('DTH_SE_adLIF','DTH_RDROP_SE_adLIF'):
    run=BASE/'runs'/model/'SHD/seed_42'
    receipt_path=run/'stages/030/RESULT.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='stage_complete' and receipt['completed_epochs']==30
    assert receipt['full_budget']==300 and not receipt['test_evaluated']
    best=ROOT/receipt['best_checkpoint'];last=ROOT/receipt['last_checkpoint']
    assert sha(best)==receipt['best_checkpoint_sha256']
    assert sha(last)==receipt['last_checkpoint_sha256']
    state=torch.load(best,map_location='cpu',weights_only=False)
    final=torch.load(last,map_location='cpu',weights_only=False)
    assert final['epoch']==29 and final['global_step']==750
    assert final['optimizer_states'] and final['lr_schedulers']
    cfg=OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']),resolve=True)
    assert canonical(cfg)==canonical(json.loads((run/'FULL_RUN_CONFIG.json').read_text()))
    assert cfg['dataset']['validate_on']==.2 and cfg['dataset']['random_seed']==42
    assert cfg['batch_size']==256 and cfg['lr']==.01 and cfg['dropout']==.15
    assert cfg['dataset']['validation_only_data'] and not cfg['run_test']
    assert cfg['l1']['cell']==cfg['l2']['cell']=='dth_se_adlif'
    assert json.loads((run/'MODEL_SOURCE_IDENTITY.json').read_text())==forward_identity('SHD')
    metrics=list(run.rglob('metrics.csv'));assert len(metrics)==1
    rows=list(csv.DictReader(metrics[0].open()));v=[r for r in rows if r.get('val_acc')]
    assert [int(r['epoch']) for r in v]==list(range(30))
    selected=max(v,key=lambda r:float(r['val_acc']));epoch=int(selected['epoch'])
    assert epoch==state['epoch']==receipt['selected_epoch_index']
    assert float(selected['val_acc'])==receipt['best_validation_accuracy']
    tr=[r for r in rows if r.get('train_acc_epoch') and int(r['epoch'])==epoch]
    ft=[r for r in rows if r.get('train_acc_epoch') and int(r['epoch'])==29]
    assert len(tr)==len(ft)==1
    initial=json.loads((run/'INITIAL_MODEL_STATE.json').read_text())
    assert initial['initialization_mode']=='fresh' and initial['optimizer_updates']==0
    initials.append(initial);configs.append(cfg)
    entries.append(dict(dataset='SHD',model=model,arm='DTH+B' if 'RDROP' in model else 'DTH',seed=42,
        checkpoint=rel(best),checkpoint_sha256=sha(best),last_checkpoint=rel(last),last_checkpoint_sha256=sha(last),
        stage_receipt=rel(receipt_path),stage_receipt_sha256=sha(receipt_path),
        full_config=rel(run/'FULL_RUN_CONFIG.json'),model_source_identity=forward_identity('SHD'),
        metrics_files_sha256={rel(p):sha(p) for p in metrics},completed_epochs=30,full_budget=300,
        selected_epoch_index=epoch,selected_completed_epoch=epoch+1,selected_global_step=state['global_step'],
        training_accuracy_at_selected_epoch=float(tr[0]['train_acc_epoch']),
        training_accuracy_final_epoch=float(ft[0]['train_acc_epoch']),
        best_validation_accuracy=float(selected['val_acc']),validation_loss_at_selected_epoch=float(selected['val_loss']),
        protocol='canonical SHD* random20 split42; matched30epoch exploratory screen',
        test_data='data/SHDWrapper/shd_test.h5',fitting_indices=audit['fitting_indices'],
        validation_indices=audit['validation_indices'],fitting_speakers=audit['fitting_speakers'],
        validation_speakers=audit['validation_speakers']))
assert initials[0]['model_state_sha256']==initials[1]['model_state_sha256']
for field in ('cpu_rng_sha256','cuda_rng_sha256','precision','float32_matmul_precision'):
    assert initials[0][field]==initials[1][field],field
# Training/evaluation objective is the only intended computational difference.
for key in ('n_epochs','batch_size','lr','dropout','scheduler','factor','patience','loss_agg','dataset','l1','l2','l_out'):
    assert configs[0][key]==configs[1][key],key
old=ROOT/'results/experiments/complementary_speaker_dev_v2/evaluations/user_stop_official_test_v1/FROZEN_CHECKPOINTS.json'
previous=json.loads(old.read_text())
plan=dict(status='frozen',purpose='exploratory_user_requested_stop_diagnostic',
    authorization='User explicitly requests fresh SHD transfer check and official-test table after stopping SSC',
    frozen_at=datetime.now(timezone.utc).isoformat(),test_selects_ideas=False,resume_training_after_tests=False,
    checkpoints=entries,evaluation_source_sha256=sha(OUT/'EVALUATION_SOURCE.txt'),
    capture_source=previous['capture_source'],capture_source_sha256=previous['capture_source_sha256'],
    split_audit=rel(audit_path),split_audit_sha256=sha(audit_path),
    actual_initial_states_and_rng_match=True,matched_SHD_budget=30,prior_test_exposure=True,
    full300_paper_reproduction=False,canonical_and_speaker_holdout_results_separate=True,
    official_test_contents_opened_before_freeze=False)
assert sha(ROOT/plan['capture_source'])==plan['capture_source_sha256']
(OUT/'FROZEN_CHECKPOINTS.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(frozen_plan_sha256=sha(OUT/'FROZEN_CHECKPOINTS.json'),
    selected_epochs={e['model']:e['selected_completed_epoch'] for e in entries})))

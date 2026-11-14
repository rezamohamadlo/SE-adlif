"""Freeze the completed ten-epoch validation selection before official testing."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf
from module.study_control import canonical, forward_identity
from project_paths import ROOT

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
RUN = BASE / 'runs/DTH_RDROP_SE_adLIF/SSC/seed_42'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rel(path):
    return Path(path).relative_to(ROOT).as_posix()

assert not (OUT / 'FROZEN_CHECKPOINTS.json').exists()
receipt_path = RUN / 'stages/010/RESULT.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['status'] == 'stage_complete' and receipt['completed_epochs'] == 10
assert receipt['full_budget'] == 40 and not receipt['test_evaluated']
paths = list(RUN.rglob('metrics.csv'))
assert len(paths) == 1
rows = list(csv.DictReader(paths[0].open()))
validation = [r for r in rows if r.get('val_acc')]
assert [int(r['epoch']) for r in validation] == list(range(10))
selected = max(validation, key=lambda r: float(r['val_acc']))
epoch = int(selected['epoch'])
assert epoch == receipt['selected_epoch_index']
assert float(selected['val_acc']) == receipt['best_validation_accuracy']
best = ROOT / receipt['best_checkpoint']
last = ROOT / receipt['last_checkpoint']
assert sha(best) == receipt['best_checkpoint_sha256']
assert sha(last) == receipt['last_checkpoint_sha256']
state = torch.load(best, map_location='cpu', weights_only=False)
final = torch.load(last, map_location='cpu', weights_only=False)
assert state['epoch'] == epoch and final['epoch'] == 9 and final['global_step'] == 2940
assert state['optimizer_states'] and final['optimizer_states'] and final['lr_schedulers']
saved = OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']), resolve=True)
assert canonical(saved) == canonical(json.loads((RUN / 'FULL_RUN_CONFIG.json').read_text()))
assert json.loads((RUN / 'MODEL_SOURCE_IDENTITY.json').read_text()) == forward_identity('SSC')
train = [r for r in rows if r.get('train_acc_epoch') and int(r['epoch']) == epoch]
final_train = [r for r in rows if r.get('train_acc_epoch') and int(r['epoch']) == 9]
assert len(train) == len(final_train) == 1
entry = dict(dataset='SSC', model='DTH_RDROP_SE_adLIF', arm='DTH+B', seed=42,
    checkpoint=rel(best), checkpoint_sha256=sha(best), last_checkpoint=rel(last),
    last_checkpoint_sha256=sha(last), stage_receipt=rel(receipt_path),
    stage_receipt_sha256=sha(receipt_path), full_config=rel(RUN / 'FULL_RUN_CONFIG.json'),
    model_source_identity=forward_identity('SSC'),
    metrics_files_sha256={rel(p): sha(p) for p in paths}, completed_epochs=10, full_budget=40,
    selected_epoch_index=epoch, selected_completed_epoch=epoch+1,
    selected_global_step=state['global_step'],
    training_accuracy_at_selected_epoch=float(train[0]['train_acc_epoch']),
    training_accuracy_final_epoch=float(final_train[0]['train_acc_epoch']),
    best_validation_accuracy=float(selected['val_acc']),
    validation_loss_at_selected_epoch=float(selected['val_loss']),
    test_data='data/SSCWrapper/ssc_test.h5', protocol=saved['evaluation_protocol'],
    selection_rule='First maximum recorded clean validation accuracy over exactly ten completed epochs')
old = BASE / 'evaluations/user_requested_ssc_test_v1'
previous_plan = json.loads((old / 'FROZEN_CHECKPOINTS.json').read_text())
# Control reuse is fixed before loading its score. No new control is chosen.
control = previous_plan['checkpoints'][0]
assert control['model'] == 'DTH_SE_adLIF' and control['completed_epochs'] == 10
assert sha(ROOT / control['checkpoint']) == control['checkpoint_sha256']
control_result = old / 'SSC/DTH_SE_adLIF/TEST_RESULTS.json'
plan = dict(status='frozen', purpose='exploratory_user_requested_stop_diagnostic',
    frozen_at=datetime.now(timezone.utc).isoformat(), checkpoints=[entry],
    test_selects_ideas=False, resume_training_after_tests=False,
    evaluation_source_sha256=sha(OUT / 'EVALUATION_SOURCE.txt'),
    capture_source=previous_plan['capture_source'],
    capture_source_sha256=previous_plan['capture_source_sha256'],
    reused_control=dict(checkpoint=control['checkpoint'], sha256=control['checkpoint_sha256'],
        completed_epochs=10, result=rel(control_result), result_sha256=sha(control_result)),
    prior_test_exposure=True, post_test_fixed_budget_completion_authorized_by_user=True,
    training_mode='Full resume own epoch7 last; no weights-only continuation',
    comparison_scope='Same ten-epoch ceiling and shared settings; candidate interrupted/resumed; one seed, exploratory, not full40 confirmation')
assert sha(ROOT / plan['capture_source']) == plan['capture_source_sha256']
(OUT / 'FROZEN_CHECKPOINTS.json').write_text(json.dumps(plan, indent=2)+'\n')
print(json.dumps(dict(frozen_plan_sha256=sha(OUT / 'FROZEN_CHECKPOINTS.json'),
                     selected_completed_epoch=epoch+1, checkpoint_sha256=sha(best))))

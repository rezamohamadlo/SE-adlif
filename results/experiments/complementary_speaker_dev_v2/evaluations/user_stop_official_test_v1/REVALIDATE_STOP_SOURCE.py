"""Revalidate an already completed one-time diagnostic; never train or retest."""
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
B001 = ROOT / 'results/experiments/dth_complement_v1/b001_rdrop_ssc_v1'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, record):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

plan = json.loads((OUT / 'FROZEN_CHECKPOINTS.json').read_text())
assert sha(OUT / 'FROZEN_CHECKPOINTS.json') == '8d7bf541b7b648b8e218c3bea44211bd8550574d3f20795443dcdf8adb8337b5'
# Confirm identities and validation selection before opening saved test metrics.
for entry in plan['checkpoints']:
    for key in ('checkpoint', 'last_checkpoint', 'stage_receipt', 'full_config'):
        assert sha(ROOT / entry[key]) == entry[key + '_sha256']
    for name, digest in entry['metrics_files_sha256'].items():
        assert sha(ROOT / name) == digest
    receipt = json.loads((ROOT / entry['stage_receipt']).read_text())
    for key in ('completed_epochs', 'selected_epoch_index', 'best_validation_accuracy'):
        assert receipt[key] == entry[key]

old_verification = json.loads((OUT / 'FINAL_VERIFICATION.json').read_text())
assert sha(OUT / 'ALL_RESULTS.json') == old_verification['final_aggregate_sha256']
aggregate = json.loads((OUT / 'ALL_RESULTS.json').read_text())
assert aggregate['frozen_plan_sha256'] == sha(OUT / 'FROZEN_CHECKPOINTS.json')
checks = []
for entry, result in zip(plan['checkpoints'], aggregate['results'], strict=True):
    assert (entry['dataset'], entry['model'], entry['checkpoint_sha256']) == (
        result['dataset'], result['model'], result['checkpoint_sha256'])
    assert result['ordinary_inference'] and not result['test_time_augmentation']
    assert result['inference_forward_calls'] == result['batches']
    assert sha(ROOT / result['prediction_archive']) == result['prediction_archive_sha256']
    assert sha(ROOT / result['test_data']) == result['test_data_sha256']
    z = np.load(ROOT / result['prediction_archive'])
    correct = z['labels'] == z['predictions']
    assert np.array_equal(z['row_ids'], np.arange(len(correct)))
    assert int(correct.sum()) == result['metrics']['test_correct']
    assert float(correct.mean()) == result['metrics']['test_acc']
    with h5py.File(ROOT / result['test_data'], 'r') as data:
        assert np.array_equal(z['labels'], data['labels'][:])
        speakers = data['extra']['speaker'][:]
        if result['dataset'] == 'SSC':
            speakers = np.asarray([s.decode() if isinstance(s, bytes) else str(s) for s in speakers])
        assert np.array_equal(z['speaker_ids'], speakers)
    for speaker, values in result['per_speaker'].items():
        mask = z['speaker_ids'] == (int(speaker) if result['dataset'] == 'SHD' else speaker)
        assert int(mask.sum()) == values['samples']
        assert int(correct[mask].sum()) == values['correct']
        assert float(correct[mask].mean()) == values['accuracy']
    checks.append(dict(dataset=result['dataset'], model=result['model'],
                       checkpoint_and_validation_selection_verified=True,
                       original_single_pass_results_and_metadata_verified=True))

# Only checkpoint inspection: no model construction, fit, optimizer step or inference.
preserved = []
for model in ('DTH_SE_adLIF', 'DTH_RDROP_SE_adLIF'):
    directory = B001 / 'runs' / model / 'SSC/seed_42'
    for checkpoint in sorted((directory / 'ckpt').glob('*.ckpt')):
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state.get('optimizer_states') and state.get('lr_schedulers')
        preserved.append(dict(model=model, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                              sha256=sha(checkpoint), completed_epoch=int(state['epoch']) + 1,
                              global_step=state['global_step'], optimizer_preserved=True,
                              scheduler_preserved=True))

raw = subprocess.check_output(['powershell', '-NoProfile', '-Command',
    "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | "
    "Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress"], text=True)
processes = json.loads(raw) if raw.strip() else []
if isinstance(processes, dict):
    processes = [processes]
training = [p for p in processes if re.search(r'(?i)(?:run|runner|pilot|search_shd|tune_shd)\.py(?:\s|$)', p.get('CommandLine') or '')]
assert not training, training
record = dict(verified_at=datetime.now(timezone.utc).isoformat(),
              action='user_requested_stop_and_revalidate_existing_one_time_C005_diagnostic',
              original_freeze_time=plan['frozen_at'], original_test_finished_at=aggregate['finished_at'],
              existing_results_reused_without_repeat_inference=True,
              frozen_plan_sha256=sha(OUT / 'FROZEN_CHECKPOINTS.json'), checks=checks,
              current_training_active=False, current_queue_active=False,
              stop_method='Study-only console Ctrl+C; Lightning acknowledged graceful KeyboardInterrupt; all launcher/training/workers exited.',
              stop_boundary='Interrupted during epoch 8; last completed boundary 7 retained, best completed epoch 6 retained. No new boundary was trained.',
              stopped_study='dth_complement_v1/B001 SSC', stopped_processes=[23192, 8024, 6576, 28308],
              current_study_checkpoint_preservation=preserved,
              unrelated_jobs_terminated=False, goal_status='paused',
              new_training_or_search_started=False, test_scores_trigger_further_work=False)
write(OUT / 'CURRENT_STOP_AND_REVALIDATION.json', record)
write(B001 / 'USER_REQUESTED_STOP_VERIFICATION.json', record)
registry_path = B001 / 'EXPERIMENT_REGISTRY.json'
registry = json.loads(registry_path.read_text())
registry.update(status='paused_by_explicit_user_stop', training_active=False, queue_active=False,
                candidate_completed_epochs=7, control_completed_epochs=10,
                current_training_pid=None, launcher_pid=None,
                automatic_extension_allowed=False, automatic_tests_allowed=False,
                stop_evidence='USER_REQUESTED_STOP_VERIFICATION.json',
                current_pair_budget_status='incomplete_unmatched_10_vs_7_not_tested_not_mixed_with_C005')
write(registry_path, registry)
write(B001 / 'STATUS.json', dict(status='paused_by_explicit_user_stop', training_active=False,
                               queue_active=False, evidence='USER_REQUESTED_STOP_VERIFICATION.json'))

original = (OUT / 'REPORT_FA.md').read_text(encoding='utf-8')
note = '''# تأیید مجدد ارزیابی یک‌باره و توقف فعلی پژوهش

این شش ارزیابی C005 قبلاً پس از ثابت‌شدن فهرست checkpointها انجام شده بودند. در این نوبت آزمون دوباره اجرا نشد: هویت همهٔ checkpointها، معیار انتخاب اعتبارسنجی، فایل‌های metrics/config، checksum دادهٔ آزمون، پیش‌بینی‌های ذخیره‌شده و تعداد/دقت هر گوینده دوباره بررسی شدند. جدول‌های زیر همان نتایج ارزیابی اولیه‌اند؛ هیچ cohort یا کنترل تازه‌ای جایگزین نشد.

**وضعیت فعلی: هیچ آموزش یا صفی فعال نیست؛ Goal واقعاً paused شده است.** مطالعهٔ جدیدتر DTH+B روی SSC با Ctrl+C مخصوص کنسول همان مطالعه متوقف شد؛ Lightning توقف graceful را ثبت کرد و تمام پردازش‌ها خارج شدند. توقف در میانهٔ دورهٔ ۸ بود؛ last پایان دورهٔ ۷ و best دورهٔ ۶ همراه optimizer و scheduler سالم باقی مانده‌اند. کنترل DTH دورهٔ ۱۰ را کامل کرده است. این جفت جدید **ناقص و از نظر بودجه ناهمسان (۱۰ در برابر ۷)** است، آزمون نشده و با نتایج C005 زیر مخلوط نمی‌شود. هیچ آموزش دیگری برای برابرکردن بودجه اجرا نشد.

گزارش اولیهٔ زیر مربوط به cohortهای C005 است: SHD چهار بازو با ۳۰ دورهٔ یکسان و SSC دو بازو با ۱۰ دورهٔ یکسان. هر دو screen کوتاه‌اند، نه آموزش کامل. تاریخ و توضیح توقف طبیعی در متن اولیه فقط به اجرای قبلی C005 مربوط‌اند؛ توقف فعلی در بالا توضیح داده شد. loss آزمون از خروجی اولیهٔ ارزیابی تک‌گذر حفظ و checksum آن تأیید شد؛ از آرشیو صرفاً برچسب‌ها، loss مجدداً محاسبه نشده است.

[شواهد توقف و بازبینی فعلی](CURRENT_STOP_AND_REVALIDATION.json) — [گزارش اولیهٔ بدون تغییر](REPORT_FA.md)

---

'''
(OUT / 'REVALIDATED_REPORT_FA.md').write_text(note + original, encoding='utf-8')
command = ROOT / 'command.txt'
command.write_text('# USER STOP: all study training/queues stopped; Goal paused. Do not run historical commands below.\n'
                   '# Verified one-time C005 SHD/SSC diagnostics; no repeated inference or new training.\n'
                   '# Report: results/experiments/complementary_speaker_dev_v2/evaluations/user_stop_official_test_v1/REVALIDATED_REPORT_FA.md\n'
                   '# New B001 DTH SSC pair stopped at control10/candidate7; incomplete/unmatched and not tested.\n\n'
                   + command.read_text(encoding='utf-8-sig'), encoding='utf-8')
print(json.dumps(dict(status='verified', training_active=False, queue_active=False,
                     goal_status='paused', evaluated_C005_checkpoints=len(checks),
                     repeated_inference=False, report=(OUT / 'REVALIDATED_REPORT_FA.md').relative_to(ROOT).as_posix())))

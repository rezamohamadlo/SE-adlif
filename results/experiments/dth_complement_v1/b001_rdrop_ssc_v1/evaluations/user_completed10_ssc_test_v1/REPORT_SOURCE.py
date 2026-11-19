"""Report the fixed-budget completion; no training or further selection."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from project_paths import ROOT

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

plan = json.loads((OUT / 'FROZEN_CHECKPOINTS.json').read_text())
aggregate = json.loads((OUT / 'ALL_RESULTS.json').read_text())
assert aggregate['frozen_plan_sha256'] == sha(OUT / 'FROZEN_CHECKPOINTS.json')
assert len(aggregate['results']) == 1
candidate = aggregate['results'][0]
reuse = plan['reused_control']
assert sha(ROOT / reuse['result']) == reuse['result_sha256']
control = json.loads((ROOT / reuse['result']).read_text())
assert control['checkpoint_sha256'] == reuse['sha256']
assert control['model'] == 'DTH_SE_adLIF'
assert candidate['model'] == 'DTH_RDROP_SE_adLIF'
for r in (control, candidate):
    assert r['completed_epochs'] == 10 and r['seed'] == 42
    assert sha(ROOT / r['checkpoint']) == r['checkpoint_sha256']
    assert sha(ROOT / r['prediction_archive']) == r['prediction_archive_sha256']
    z = np.load(ROOT / r['prediction_archive'])
    correct = z['labels'] == z['predictions']
    assert len(correct) == 20382 and int(correct.sum()) == r['metrics']['test_correct']
    assert float(correct.mean()) == r['metrics']['test_acc']
    assert r['weights_and_rng_unchanged'] and r['inference_forward_calls'] == r['batches']
    assert r['ordinary_inference'] and not r['test_time_augmentation']
delta = 100*(candidate['metrics']['test_acc'] - control['metrics']['test_acc'])
combined = dict(purpose='user_authorized_fixed_ten_epoch_completion_exploratory',
    created_at=datetime.now(timezone.utc).isoformat(), results=[control, candidate],
    control_reused_without_repeat_inference=True,
    difference_candidate_minus_control_pp=delta, training_ceiling=10,
    candidate_interrupted_and_fully_resumed=True, prior_test_exposure=True,
    independent_untouched_confirmation=False, further_training_allowed=False)
(OUT / 'COMPARISON.json').write_text(json.dumps(combined, indent=2)+'\n')
lines = ['# نتیجهٔ تکمیل محدود DTH + R-Drop تا ۱۰ دوره روی SSC', '',
    'به درخواست کاربر، فقط سه دورهٔ باقی‌ماندهٔ ترکیب از checkpoint آخرِ دورهٔ ۷ تا پایان دورهٔ ۱۰ با بازیابی کامل optimizer و scheduler اجرا شد. کنترل DTH دوباره آموزش ندید. سقف scheduler همان برنامهٔ اصلی ۴۰دوره‌ای بود؛ هیچ تنظیم مدل، loss، split، batch یا preprocessing تغییر نکرد.', '',
    '**این بار سقف آموزش هر دو بازو ۱۰ دوره است.** ترکیب توقف و ادامه داشته است؛ این امر هم‌ارزی دقیق تصادفی با آموزش بدون وقفه را تضمین نمی‌کند. تست ۷دوره‌ای قبلاً دیده شده بود؛ بنابراین تکمیلِ ثابت و از پیش درخواست‌شدهٔ ۱۰دوره‌ای اکتشافی است، نه تأیید مستقل روی آزمون دست‌نخورده.', '',
    'انتخاب checkpoint فقط با بیشترین دقت اعتبارسنجی ثبت‌شدهٔ ده دوره و قبل از این تست انجام شد. برای کنترل، نتیجهٔ تست قبلی با هویت دقیق همان checkpoint بازاستفاده شد. مقایسه با بهترین checkpoint تاریخی یا بهترین score آزمون انجام نشد.', '',
    'Δ paper برابر دقت محلی منهای میانگین مقاله، با واحد درصد (pp) و استفاده از اعداد گردنشدهٔ محلی است. مقاله میانگین پنج اجرا را گزارش می‌کند؛ این آزمایش یک بذر ۴۲ و فقط ۱۰ از ۴۰ دوره را دارد.', '',
    '| مدل | آموزش در دورهٔ منتخب | بهترین اعتبارسنجی | دقت تست (Δ paper) | loss تست | دوره‌های کامل | دورهٔ منتخب |',
    '|---|---:|---:|---:|---:|---:|---:|',
    '| مقاله، SE-adLIF دولایه | گزارش نشده | گزارش نشده | 80.44 ± 0.26%؛ پنج اجرا | گزارش نشده | — | — |']
for name, r in [('DTH', control), ('DTH + R-Drop', candidate)]:
    value = 100*r['metrics']['test_acc']
    lines.append(f"| {name} | {100*r['training_accuracy_at_selected_epoch']:.2f}% | {100*r['best_validation_accuracy']:.2f}% | {value:.2f}% ({value-80.44:+.2f} pp) | {r['metrics']['test_loss']:.6f} | 10 / 40 | {r['selected_completed_epoch']} |")
lines += ['', f'اختلاف ترکیب نسبت به DTH همین مطالعه: **{delta:+.4f} واحد درصد**. این نتیجه مربوط به همین screen کوتاه و یک بذر است و دربارهٔ بهبود عمومی یا برتری بر مقاله ادعایی نمی‌کند.', '',
    'تست روی تمام ۲۰٬۳۸۲ نمونهٔ رسمی SSC، با استنتاج معمولی تک‌گذر و بدون TTA انجام شد. loss آزمون cross-entropy با برچسب سخت است؛ KL فقط در آموزش فعال است. دقت آموزش آنلاینِ دورهٔ منتخب و برای ترکیب مربوط به نمای اول است.', '',
    '## شواهد', '',
    '- [فهرست ثابت checkpoint و کنترل بازاستفاده‌شده](FROZEN_CHECKPOINTS.json)',
    '- [نتیجهٔ آزمون جدید ترکیب](ALL_RESULTS.json)',
    '- [مقایسهٔ کامل همراه نتیجهٔ کنترل](COMPARISON.json)',
    '- [جدول ۲ مقاله، صفحهٔ ۱۱](../../../../../../reference_paper.pdf#page=11)',
    '- [ثبت تکمیل سه دوره و مواجههٔ قبلی با آزمون](../../USER_THREE_EPOCH_COMPLETION_PREREGISTRATION.json)',
    '- [انتقال صریح provenance و حفظ سابقهٔ اولیه](../../runs/DTH_RDROP_SE_adLIF/SSC/seed_42/RESUME_SOURCE_TRANSITION.json)', '']
for name,r in [('DTH',control),('DTH + R-Drop',candidate)]:
    lines += [f"- {name}: `{r['checkpoint']}`؛ SHA-256: `{r['checkpoint_sha256']}`؛ دورهٔ منتخب {r['selected_completed_epoch']}."]
lines += ['', 'آموزش و صف پایان یافته‌اند. هیچ ادامه تا ۴۰ دوره، ایدهٔ جدید یا جست‌وجویی مجاز یا آغاز نشده است؛ پس از گزارش توقف می‌کنیم.']
(OUT / 'REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Retain the exact previously tested epoch6 path even if ModelCheckpoint pruned it.
old = BASE / 'evaluations/user_requested_ssc_test_v1/PRE_RESUME_CHECKPOINTS/epoch=5-step=1764.ckpt'
original = BASE / 'runs/DTH_RDROP_SE_adLIF/SSC/seed_42/ckpt/epoch=5-step=1764.ckpt'
if not original.exists():
    shutil.copy2(old, original)
assert sha(old) == sha(original)
registry_path = BASE / 'EXPERIMENT_REGISTRY.json'
registry = json.loads(registry_path.read_text())
registry.update(status='completed_user_authorized_ten_epoch_test_report_stop',
    training_active=False, queue_active=False, candidate_completed_epochs=10,
    control_completed_epochs=10, current_training_pid=None, launcher_pid=None,
    automatic_extension_allowed=False, current_comparison='evaluations/user_completed10_ssc_test_v1/COMPARISON.json')
registry_path.write_text(json.dumps(registry,indent=2)+'\n')
print(json.dumps(dict(candidate_test_acc=100*candidate['metrics']['test_acc'],
    control_test_acc=100*control['metrics']['test_acc'], delta_pp=delta,
    selected_candidate_epoch=candidate['selected_completed_epoch'])))

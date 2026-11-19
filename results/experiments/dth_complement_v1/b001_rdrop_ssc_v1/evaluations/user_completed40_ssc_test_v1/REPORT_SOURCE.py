"""Verify and report the explicitly authorized full40 test; never train."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from project_paths import ROOT

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

plan = json.loads((OUT/'FROZEN_CHECKPOINTS.json').read_text())
aggregate = json.loads((OUT/'ALL_RESULTS.json').read_text())
assert aggregate['frozen_plan_sha256'] == sha(OUT/'FROZEN_CHECKPOINTS.json')
assert len(aggregate['results']) == 1
r = aggregate['results'][0]
assert r['completed_epochs'] == r['full_budget'] == 40
assert sha(ROOT/r['checkpoint']) == r['checkpoint_sha256']
assert sha(ROOT/r['last_checkpoint']) == r['last_checkpoint_sha256']
assert r['ordinary_inference'] and not r['test_time_augmentation']
assert r['inference_forward_calls'] == r['batches'] and r['weights_and_rng_unchanged']
assert sha(ROOT/r['prediction_archive']) == r['prediction_archive_sha256']
z = np.load(ROOT/r['prediction_archive'])
correct = z['labels'] == z['predictions']
assert len(correct) == 20382 and int(correct.sum()) == r['metrics']['test_correct']
assert float(correct.mean()) == r['metrics']['test_acc']
reuse = plan['reused_control']
assert sha(ROOT/reuse['result']) == reuse['result_sha256']
control = json.loads((ROOT/reuse['result']).read_text())
assert control['checkpoint_sha256'] == reuse['sha256'] and control['completed_epochs'] == 10
value = 100*r['metrics']['test_acc']
delta = value - 100*control['metrics']['test_acc']
comparison = dict(purpose='user_authorized_full40_exploratory_test',candidate=r,
    historical_control=control,comparison_budgets_unmatched=True,
    difference_candidate_minus_control_pp=delta,prior_test_exposure=True,
    independent_confirmation=False,further_training_authorized=False)
(OUT/'COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
lines = ['# تست رسمی DTH + R-Drop پس از تکمیل ۴۰ دورهٔ SSC', '',
    'فقط همان مدل DTH + R-Drop با بذر ۴۲، از checkpoint آخرِ خود، تا پایان دورهٔ ۴۰ با بازیابی کامل optimizer و scheduler ادامه یافت. تنظیمات مدل، loss، batch، preprocessing و برنامهٔ اصلی ۴۰دوره‌ای تغییر نکردند. checkpoint منتخب بر اساس بیشترین دقت اعتبارسنجی و پیش از مشاهدهٔ این نتیجهٔ تست ثابت شد.', '',
    '**نتیجه اکتشافی است:** تست در مراحل قبلی دیده شده بود. این یک بذر با توقف/ادامه است، نه تأیید مستقل روی آزمون دست‌نخورده. کنترل قبلی DTH فقط ۱۰ دوره دارد؛ اختلاف با آن اثر مکمل در یک مقایسهٔ هم‌بودجه را نشان نمی‌دهد. کنترل تازه یا تاریخی دیگری جایگزین نشده است.', '',
    'Δ paper برابر دقت محلی منهای میانگین مقاله با واحد درصد (pp) است و از مقدار گردنشدهٔ محلی محاسبه می‌شود. مقاله میانگین پنج اجرا را گزارش می‌کند؛ نتیجهٔ محلی یک بذر است.', '',
    '| مدل | دقت تست (Δ paper) | loss تست | دوره‌های کامل | دورهٔ checkpoint |',
    '|---|---:|---:|---:|---:|',
    '| مقاله، SE-adLIF دولایه | 80.44 ± 0.26% | گزارش نشده | — | — |',
    f"| DTH؛ کنترل تاریخی همین مطالعه | {100*control['metrics']['test_acc']:.2f}% ({100*control['metrics']['test_acc']-80.44:+.2f} pp) | {control['metrics']['test_loss']:.6f} | 10 | {control['selected_completed_epoch']} |",
    f"| DTH + R-Drop | **{value:.2f}% ({value-80.44:+.2f} pp)** | {r['metrics']['test_loss']:.6f} | **40** | {r['selected_completed_epoch']} |", '',
    f"بهترین اعتبارسنجی: **{100*r['best_validation_accuracy']:.2f}%**؛ دقت آنلاین آموزش در دورهٔ منتخب: **{100*r['training_accuracy_at_selected_epoch']:.2f}%**؛ دقت آنلاین آموزش در آخرین دوره: **{100*r['training_accuracy_final_epoch']:.2f}%**.", '',
    f'اختلاف با کنترل ۱۰دوره‌ای: {delta:+.4f} واحد درصد؛ این اختلاف بودجهٔ متفاوت دارد و مبنای ادعای برتری معماری نیست. هیچ کنترل هم‌بودجهٔ ۴۰دوره‌ای در این درخواست آموزش داده نشد.', '',
    'تست روی تمام ۲۰٬۳۸۲ نمونهٔ رسمی SSC، استنتاج تک‌گذر معمولی و بدون TTA انجام شد. loss آزمون cross-entropy با برچسب سخت است. تعداد forwardها، وزن‌های ثابت و شمارش پیش‌بینی‌های خام تأیید شدند.', '',
    f"هویت checkpoint: `{r['checkpoint']}`؛ SHA-256: `{r['checkpoint_sha256']}`.", '',
    '- [فهرست ثابت پیش از آزمون](FROZEN_CHECKPOINTS.json)',
    '- [نتیجهٔ کامل](ALL_RESULTS.json)',
    '- [مقایسه و کنترل بازاستفاده‌شده](COMPARISON.json)',
    '- [جدول ۲ مقاله، صفحهٔ ۱۱](../../../../../../reference_paper.pdf#page=11)', '',
    'این دنباله پس از گزارش پایان می‌یابد؛ هیچ ایده، تنظیم یا آموزش دیگری بر اساس این تست راه‌اندازی نمی‌شود.']
(OUT/'REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
verification = dict(verified_at=datetime.now(timezone.utc).isoformat(),completed_epochs=40,
    selected_epoch=r['selected_completed_epoch'],checkpoint_sha256=r['checkpoint_sha256'],
    raw_predictions_verified=True,official_test_samples=20382,ordinary_single_pass=True,
    tta=False,control_budget_unmatched=True,result_sha256=sha(OUT/'ALL_RESULTS.json'))
(OUT/'FINAL_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
p=BASE/'EXPERIMENT_REGISTRY.json';registry=json.loads(p.read_text())
registry.update(status='user_authorized_full40_test_complete_stop',training_active=False,
    queue_active=False,candidate_completed_epochs=40,current_training_pid=None,
    launcher_pid=None,automatic_extension_allowed=False,automatic_tests_allowed=False,
    latest_test_checkpoint_epoch=r['selected_completed_epoch'],latest_test_accuracy=r['metrics']['test_acc'],
    current_comparison='evaluations/user_completed40_ssc_test_v1/COMPARISON.json')
p.write_text(json.dumps(registry,indent=2)+'\n')
command=ROOT/'command.txt'
command.write_text(f'# COMPLETE: user-authorized DTH+RDrop SSC40, test={value:.6f}%, selectedepoch={r["selected_completed_epoch"]}.\n'
    '# Report: results/experiments/dth_complement_v1/b001_rdrop_ssc_v1/evaluations/user_completed40_ssc_test_v1/REPORT_FA.md\n'
    '# No further training/stages scheduled by this sequence. Historical commands below.\n\n'
    +command.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(json.dumps(dict(test_accuracy_percent=value,test_loss=r['metrics']['test_loss'],
    selected_epoch=r['selected_completed_epoch'],completed_epochs=40)))

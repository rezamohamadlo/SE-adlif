"""Report the explicitly requested frozen live-checkpoint test; no training."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from project_paths import ROOT

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

plan = json.loads((OUT / 'FROZEN_CHECKPOINTS.json').read_text())
aggregate = json.loads((OUT / 'ALL_RESULTS.json').read_text())
assert sha(OUT / 'FROZEN_CHECKPOINTS.json') == aggregate['frozen_plan_sha256']
current = aggregate['results'][0]
assert len(aggregate['results']) == 1
assert current['checkpoint_sha256'] == plan['checkpoints'][0]['checkpoint_sha256']
assert sha(ROOT / current['checkpoint']) == current['checkpoint_sha256']
assert current['selected_completed_epoch'] == 15
assert current['ordinary_inference'] and not current['test_time_augmentation']
assert current['inference_forward_calls'] == current['batches']
assert current['weights_and_rng_unchanged']
z = np.load(ROOT / current['prediction_archive'])
assert sha(ROOT / current['prediction_archive']) == current['prediction_archive_sha256']
correct = z['labels'] == z['predictions']
assert len(correct) == 20382 and int(correct.sum()) == current['metrics']['test_correct']
assert float(correct.mean()) == current['metrics']['test_acc']
earlier_path = BASE / 'evaluations/user_completed10_ssc_test_v1/COMPARISON.json'
earlier = json.loads(earlier_path.read_text())
control, candidate10 = earlier['results']
assert control['model'] == 'DTH_SE_adLIF' and candidate10['model'] == 'DTH_RDROP_SE_adLIF'
delta_control = 100*(current['metrics']['test_acc'] - control['metrics']['test_acc'])
delta_previous = 100*(current['metrics']['test_acc'] - candidate10['metrics']['test_acc'])
comparison = dict(current_result=current, historical_control=control,
    previous_ten_epoch_candidate=candidate10, historical_source=earlier_path.relative_to(ROOT).as_posix(),
    difference_current_minus_control_pp=delta_control,
    difference_current_minus_previous_candidate_pp=delta_previous,
    comparison_budget_unmatched=True, exploratory=True,
    test_scores_must_not_select_next_checkpoint_or_idea=True)
(OUT / 'COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
lines = ['# تست رسمی بهترین checkpoint موجود در زمان درخواست کاربر', '',
    'مدل DTH + R-Drop روی SSC، بذر ۴۲. در زمان ثابت‌شدن فهرست، ۱۶ دوره کامل شده بود و بیشترین دقت اعتبارسنجی ثبت‌شده متعلق به دورهٔ ۱۵ بود. نسخهٔ دقیق همان checkpoint، همراه SHA-256، پیش از بازکردن نتیجهٔ تست جدا و ثابت شد. هیچ checkpoint بر اساس score تست انتخاب نشد.', '',
    '**آموزش مستقلِ از قبل مجاز تا سقف ۲۰ دوره متوقف یا دوباره شروع نشد.** تست روی نسخهٔ ثابت انجام شد و checkpointهای بعدی آموزش در آن دخالت ندارند. اجرای هم‌زمان استنتاج ممکن است زمان دورهٔ جاری را بیشتر کند. این تست به‌معنای پایان مرحلهٔ ۲۰دوره‌ای نیست.', '',
    'استنتاج معمولی تک‌گذر، بدون TTA، روی تمام ۲۰٬۳۸۲ نمونهٔ رسمی SSC. loss آزمون cross-entropy با برچسب سخت است؛ KL فقط در آموزش فعال است. شمارش پیش‌بینی‌های خام، هش وزن‌ها و تعداد forwardها بررسی شدند.', '',
    'Δ paper یعنی دقت محلی منهای میانگین مقاله، با واحد درصد (pp) و محاسبه از اعداد گردنشده. اعداد محلی یک بذر دارند؛ مقاله میانگین پنج اجراست. مقایسه با کنترل ۱۰دوره‌ای از نظر بودجه همسان نیست.', '',
    '| مدل / مرحله | دقت تست (Δ paper) | loss تست | دورهٔ checkpoint | دوره‌های کامل هنگام انتخاب |',
    '|---|---:|---:|---:|---:|',
    '| مقاله، SE-adLIF دولایه | 80.44 ± 0.26% | گزارش نشده | — | — |']
for label,r in [('DTH؛ کنترل قبلی',control),('DTH + R-Drop؛ مرحلهٔ ۱۰دوره‌ای قبلی',candidate10),('DTH + R-Drop؛ بهترین موجود در این درخواست',current)]:
    acc = 100*r['metrics']['test_acc']
    lines.append(f"| {label} | {acc:.2f}% ({acc-80.44:+.2f} pp) | {r['metrics']['test_loss']:.6f} | {r['selected_completed_epoch']} | {r['completed_epochs']} |")
lines += ['', f"اعتبارسنجی منتخب: **{100*current['best_validation_accuracy']:.2f}%**؛ آموزش آنلاین همان دوره: **{100*current['training_accuracy_at_selected_epoch']:.2f}%**.", '',
    f'اختلاف با تست ترکیب در مرحلهٔ ۱۰دوره‌ای: **{delta_previous:+.4f} واحد درصد**؛ اختلاف با کنترل DTH قبلی: **{delta_control:+.4f} واحد درصد**. اختلاف دوم اثر مستقل مکمل را ثابت نمی‌کند، چون بودجه‌ها متفاوت‌اند. این نتایج اکتشافی‌اند و آزمون قبلاً دیده شده است.', '',
    f"Checkpoint ثابت: `{current['checkpoint']}`؛ SHA-256: `{current['checkpoint_sha256']}`.", '',
    '- [فهرست ثابت قبل از تست](FROZEN_CHECKPOINTS.json)',
    '- [نتیجهٔ کامل تست](ALL_RESULTS.json)',
    '- [مقایسه با شواهد تاریخی مشخص](COMPARISON.json)',
    '- [جدول ۲ مقاله، صفحهٔ ۱۱](../../../../../../reference_paper.pdf#page=11)', '',
    'این ارزیابی هیچ ادامهٔ آموزشی، ایدهٔ جدید، تغییر تنظیمات یا تست خودکار checkpoint بعدی را مجاز نمی‌کند. فقط همان آموزش قبلاً مجازِ محدود به ۲۰ دوره می‌تواند ادامه یابد.']
(OUT / 'REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
verification = dict(verified_at=datetime.now(timezone.utc).isoformat(),
    selected_epoch=15,completed_epochs_at_freeze=16,official_test_samples=20382,
    correct_predictions=int(correct.sum()),checkpoint_sha256=current['checkpoint_sha256'],
    raw_predictions_verified=True,ordinary_single_pass=True,tta=False,
    no_training_started_or_stopped_by_evaluation=True,
    result_sha256=sha(OUT / 'ALL_RESULTS.json'))
(OUT / 'FINAL_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(dict(test_percent=100*current['metrics']['test_acc'],test_loss=current['metrics']['test_loss'],
    difference_control_pp=delta_control,difference_previous_candidate_pp=delta_previous)))

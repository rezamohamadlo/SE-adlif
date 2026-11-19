"""Audit saved ordinary predictions and report; never train or select ideas."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

plan_path = OUT / 'FROZEN_CHECKPOINTS.json'
assert sha(plan_path) == '3e724c071d8f0242ad35564a6bb7d1c4c61e5a21e44260641d132d315732c5aa'
plan = json.loads(plan_path.read_text())
aggregate = json.loads((OUT / 'ALL_RESULTS.json').read_text())
assert aggregate['frozen_plan_sha256'] == sha(plan_path)
results = aggregate['results']
assert len(results) == 2
for frozen, result in zip(plan['checkpoints'], results, strict=True):
    assert result['model'] == frozen['model']
    assert result['checkpoint_sha256'] == frozen['checkpoint_sha256']
    assert sha(ROOT / frozen['checkpoint']) == frozen['checkpoint_sha256']
    assert sha(ROOT / frozen['last_checkpoint']) == frozen['last_checkpoint_sha256']
    assert sha(ROOT / result['prediction_archive']) == result['prediction_archive_sha256']
    z = np.load(ROOT / result['prediction_archive'])
    correct = z['labels'] == z['predictions']
    assert len(correct) == 20382
    assert correct.sum() == result['metrics']['test_correct']
    assert float(correct.mean()) == result['metrics']['test_acc']
    assert result['weights_and_rng_unchanged']
    assert result['ordinary_inference'] and not result['test_time_augmentation']
    assert result['inference_forward_calls'] == result['batches']

delta = 100 * (results[1]['metrics']['test_acc'] - results[0]['metrics']['test_acc'])
names = {'DTH_SE_adLIF': 'DTH', 'DTH_RDROP_SE_adLIF': 'DTH + R-Drop'}
lines = ['# آزمون رسمی اکتشافی DTH با مکمل R-Drop روی SSC', '',
         'تاریخ محلی گزارش: ۲۰۲۶-۱۱-۱۴. بذر آموزش ۴۲، استنتاج تک‌گذر معمولی، بدون TTA و بدون ادامهٔ آموزش. فهرست دو checkpoint با بیشترین دقت اعتبارسنجی ثبت‌شده، پیش از بازکردن داده یا نتایج تست این ارزیابی ثابت شد. هر دو از همین مطالعه‌اند؛ کنترل تاریخی دیگری جایگزین نشده است.', '',
         '**مقایسه ناقص و از نظر بودجه ناهمسان است:** DTH ده دوره و DTH + R-Drop هفت دوره از برنامهٔ ۴۰دوره‌ای را کامل کرده‌اند. از نتیجهٔ آزمون برای انتخاب ایده، تغییر تنظیمات یا ادامهٔ پژوهش استفاده نمی‌شود. Goal همچنان paused است.', '',
         'Δ paper یعنی دقت محلی منهای میانگین منتشرشدهٔ مقاله، با واحد درصد (pp) و محاسبه از مقادیر گردنشدهٔ محلی. نتایج محلی یک بذر و آموزش کوتاه دارند؛ مقاله میانگین چند بذر است، بنابراین این اختلاف‌ها صرفاً توصیفی‌اند.', '',
         '| مدل | آموزش در دورهٔ منتخب | بهترین اعتبارسنجی | دقت تست (Δ paper) | loss تست | دوره‌های کامل | دورهٔ منتخب |',
         '|---|---:|---:|---:|---:|---:|---:|',
         '| مقاله، SE-adLIF دولایه | گزارش نشده | گزارش نشده | 80.44 ± 0.26%؛ پنج اجرا | گزارش نشده | — | — |']
for r in results:
    value = 100 * r['metrics']['test_acc']
    lines.append(f"| {names[r['model']]} | {100*r['training_accuracy_at_selected_epoch']:.2f}% | {100*r['best_validation_accuracy']:.2f}% | {value:.2f}% ({value-80.44:+.2f} pp) | {r['metrics']['test_loss']:.6f} | {r['completed_epochs']} / 40 | {r['selected_completed_epoch']} |")
lines += ['', f'اختلاف دقت تست DTH + R-Drop نسبت به DTH همین مطالعه برابر **{delta:+.4f} واحد درصد** است. به‌دلیل تفاوت بودجه، این اختلاف اثر علّی مکمل یا نتیجهٔ مقایسهٔ هم‌بودجه را اثبات نمی‌کند.', '',
          'دقت آموزش، مقدار آنلاین دورهٔ منتخب با dropout فعال است؛ برای R-Drop فقط نمای اول ثبت شده است. loss تست cross-entropy با برچسب سخت است؛ KL آموزشی در آزمون محاسبه نمی‌شود. هر مدل روی تمام **۲۰٬۳۸۲ نمونهٔ رسمی SSC** آزموده شد.', '',
          'تنظیمات، مدل ثبت‌شده، preprocessing و ماسک‌ها از checkpoint اصلی خوانده شدند. تفاوت‌های اجرایی ارزیابی: دسترسی به مسیر از ریشهٔ مخزن، workers=0 و بارگذاری ترتیبی بدون cache، و غیرفعال‌کردن sinkهای logging بدون تغییر test_step. برای هر batch فقط یک forward انجام شد؛ وزن‌ها و RNG در ارزیابی تغییر نکردند.', '',
          'منبع مقاله: [جدول ۲، صفحهٔ ۱۱](../../../../../../reference_paper.pdf#page=11). نتیجهٔ مقاله برای SHD* برابر 93.79 ± 0.76% و ECG برابر 88.18 ± 0.36% است؛ در این ارزیابی DTH + R-Drop روی این دو مجموعه تست نشده و اختلافی برای آن‌ها گزارش نمی‌شود.', '',
          '## هویت checkpointها و شواهد', '',
          '- [فهرست ثابت پیش از آزمون](FROZEN_CHECKPOINTS.json)',
          '- [نتایج کامل و اطلاعات پیش‌بینی‌ها](ALL_RESULTS.json)']
for r in results:
    lines += [f"- {names[r['model']]}: `{r['checkpoint']}`؛ SHA-256: `{r['checkpoint_sha256']}`؛ [نتیجه](SSC/{r['model']}/TEST_RESULTS.json)."]
(OUT / 'REPORT_FA.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
verification = dict(verified_at=datetime.now(timezone.utc).isoformat(),
    frozen_plan_sha256=sha(plan_path), checkpoints_and_last_states_unchanged=True,
    ordinary_single_pass=True, tta=False, raw_predictions_verified=True,
    test_sample_count=20382, comparison='incomplete_and_unmatched_10_vs_7',
    difference_candidate_minus_control_pp=delta, goal_status='paused', training_started=False,
    results_sha256=sha(OUT / 'ALL_RESULTS.json'))
(OUT / 'FINAL_VERIFICATION.json').write_text(json.dumps(verification, indent=2)+'\n')
print(json.dumps(dict(results=[dict(model=r['model'], accuracy=100*r['metrics']['test_acc'],
    loss=r['metrics']['test_loss']) for r in results], delta_pp=delta)))

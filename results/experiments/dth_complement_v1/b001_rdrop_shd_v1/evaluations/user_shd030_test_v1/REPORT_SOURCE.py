"""Cross-dataset ordinary-test report. No further training or selection."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from project_paths import ROOT

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
plan=json.loads((OUT/'FROZEN_CHECKPOINTS.json').read_text())
aggregate=json.loads((OUT/'ALL_RESULTS.json').read_text())
assert aggregate['frozen_plan_sha256']==sha(OUT/'FROZEN_CHECKPOINTS.json')
assert len(aggregate['results'])==2
results=aggregate['results']
for entry,r in zip(plan['checkpoints'],results,strict=True):
    assert r['model']==entry['model'] and r['checkpoint_sha256']==entry['checkpoint_sha256']
    assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
    assert r['ordinary_inference'] and not r['test_time_augmentation']
    assert r['inference_forward_calls']==r['batches'] and r['weights_and_rng_unchanged']
    assert sha(ROOT/r['prediction_archive'])==r['prediction_archive_sha256']
    z=np.load(ROOT/r['prediction_archive']);correct=z['labels']==z['predictions']
    assert len(correct)==2264 and int(correct.sum())==r['metrics']['test_correct']
    assert float(correct.mean())==r['metrics']['test_acc']
audit=json.loads((BASE/'SPLIT_AND_SSC_STOP_AUDIT.json').read_text())
stop=audit['ssc_source'];ssc=json.loads((ROOT/stop['reused_exact_selected_test']).read_text())['results'][0]
assert ssc['checkpoint_sha256']==stop['best_sha256']
assert sha(ROOT/stop['best_checkpoint'])==stop['best_sha256']
assert not (ROOT/'results/experiments/dth_complement_v1/b001_rdrop_ssc_v1/evaluations/user_completed40_ssc_test_v1/ALL_RESULTS.json').exists(), 'Reconcile any later SSC evaluation before reporting'
ssc_control_path=ROOT/'results/experiments/dth_complement_v1/b001_rdrop_ssc_v1/evaluations/user_completed10_ssc_test_v1/COMPARISON.json'
ssc_control=json.loads(ssc_control_path.read_text())['results'][0]
summary=dict(created_at=datetime.now(timezone.utc).isoformat(),SHD=results,SSC=ssc,
    SSC_stopped_completed_epochs=stop['completed_epochs'],SSC_control=ssc_control,
    SHD_budget_matched=True,SSC_control_budget_matched=False,ordinary_inference=True,
    tta=False,prior_test_exposure=True,exploratory=True,no_more_training_authorized=True)
(BASE/'CROSS_DATASET_TEST_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
shd_base,shd_candidate=results
lines=['# آزمون DTH و DTH + R-Drop روی SSC و SHD*','',
    'تمام نتایج با استنتاج معمولی تک‌گذر، بدون TTA و بذر ۴۲ هستند. تست SHD پس از انتخاب checkpoint فقط با اعتبارسنجی انجام شد. برای SSC نتیجهٔ تست قبلیِ دقیقاً همان checkpoint منتخبِ زمان توقف بازاستفاده شد؛ score تست برای انتخاب checkpoint دیگر استفاده نشد. این مطالعه به‌علت مواجههٔ قبلی با آزمون، اکتشافی است.','',
    'SHD* در این گزارش تقسیم تصادفی ۲۰٪ از فایل آموزش رسمی با splitseed42 است؛ **speaker-holdout نیست**. دو بازوی SHD از وزن تصادفی مستقل از SSC روی یک تقسیم ثابت با ۳۰ دورهٔ یکسان آموزش دیدند. این بررسی کوتاه، بازتولید کامل ۳۰۰دوره‌ای مقاله نیست.','',
    'Δ paper برابر دقت محلی منهای میانگین منتشرشدهٔ مقاله بر حسب واحد درصد (pp)، با استفاده از مقادیر گردنشدهٔ محلی است. نتایج محلی یک بذر و مقاله میانگین چند اجرا دارد.','',
    '| مجموعه / پروتکل | مقاله، میانگین ± SD | تست DTH (Δ paper) | تست DTH + R-Drop (Δ paper) | بودجهٔ محلی |','|---|---:|---:|---:|---|']
for label,paper,sd,a,b,budget in [('SHD*؛ تصادفی۲۰٪',93.79,.76,shd_base,shd_candidate,'هر دو ۳۰ دوره؛ از ۳۰۰'),('SSC؛ اعتبارسنجی رسمی',80.44,.26,ssc_control,ssc,f"کنترل ۱۰؛ ترکیب {stop['completed_epochs']} دوره و متوقف‌شده؛ از ۴۰")]:
    av=100*a['metrics']['test_acc'];bv=100*b['metrics']['test_acc']
    lines.append(f'| {label} | {paper:.2f} ± {sd:.2f}% | {av:.2f}% ({av-paper:+.2f} pp) | {bv:.2f}% ({bv-paper:+.2f} pp) | {budget} |')
lines+=['','**SSC هم‌بودجه نیست:** اختلاف با کنترل ۱۰دوره‌ای، اثر مستقل مکمل را اثبات نمی‌کند. وزن تست‌شدهٔ ترکیب مربوط به دورهٔ ۱۹ و بهترین اعتبارسنجی در زمان توقف SSC بود.','',
    '| مجموعه / مدل | آموزش در دورهٔ منتخب | بهترین اعتبارسنجی | تست | loss تست | دورهٔ checkpoint |','|---|---:|---:|---:|---:|---:|']
for label,r in [('SHD* / DTH',shd_base),('SHD* / DTH + R-Drop',shd_candidate),('SSC / DTH + R-Drop',ssc)]:
    lines.append(f"| {label} | {100*r['training_accuracy_at_selected_epoch']:.2f}% | {100*r['best_validation_accuracy']:.2f}% | {100*r['metrics']['test_acc']:.2f}% | {r['metrics']['test_loss']:.6f} | {r['selected_completed_epoch']} |")
delta=100*(shd_candidate['metrics']['test_acc']-shd_base['metrics']['test_acc'])
lines+=['',f'اختلاف تست ترکیب با DTH روی SHD* در بودجهٔ یکسان: **{delta:+.4f} واحد درصد**. این نتیجهٔ یک بذر، بهبود پایدار یا معنی‌داری آماری را اثبات نمی‌کند.','',
    '- [فهرست ثابت checkpointهای SHD](evaluations/user_shd030_test_v1/FROZEN_CHECKPOINTS.json)',
    '- [نتیجهٔ کامل SHD](evaluations/user_shd030_test_v1/ALL_RESULTS.json)',
    '- [نتایج دو دیتاست و provenance](CROSS_DATASET_TEST_RESULTS.json)',
    '- [جدول ۲ مقاله، صفحهٔ ۱۱](../../../../reference_paper.pdf#page=11)','']
for r in results:lines.append(f"- {r['model']}: `{r['checkpoint']}`؛ SHA-256: `{r['checkpoint_sha256']}`.")
lines+=['','پس از این گزارش صف پایان می‌یابد. هیچ مرحلهٔ تازه، افزایش بودجه، تنظیم یا انتخاب ایده بر اساس scoreهای آزمون آغاز نمی‌شود.']
(BASE/'CROSS_DATASET_TEST_REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'FINAL_VERIFICATION.json').write_text(json.dumps(dict(verified_at=datetime.now(timezone.utc).isoformat(),raw_predictions_verified=True,SHD_samples=2264,matched_SHD_epochs=30,SSC_budget_unmatched=True,SHD_candidate_minus_DTH_pp=delta,tta=False),indent=2)+'\n')
(BASE/'STATUS.json').write_text(json.dumps(dict(status='completed_bounded_SHD_test_report',training_active=False,queue_active=False,report='CROSS_DATASET_TEST_REPORT_FA.md'),indent=2)+'\n')
p=ROOT/'command.txt';p.write_text('# COMPLETE: bounded fresh SHD DTH/RDrop30 plus ordinary tests; SSC stopped.\n# Report: results/experiments/dth_complement_v1/b001_rdrop_shd_v1/CROSS_DATASET_TEST_REPORT_FA.md\n# No further training or experiments scheduled. Historical commands below.\n\n'+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(json.dumps(dict(SHD_DTH_test_percent=100*shd_base['metrics']['test_acc'],SHD_DTH_RDrop_test_percent=100*shd_candidate['metrics']['test_acc'],SSC_DTH_RDrop_test_percent=100*ssc['metrics']['test_acc'],SHD_paired_difference_pp=delta)))

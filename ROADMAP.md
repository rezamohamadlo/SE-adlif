# نقشه راه ساده پروژه DA-LIF

راهنمای اجرایی برای شروع پروژه DA-LIF برای کسی که تازه با شبکه‌های عصبی اسپایکی آشنا می‌شود.

## هدف پروژه

ساخت یک نورون اسپایکی که آستانه شلیک و ثابت زمانی آن با فعالیت شبکه تغییر کند؛ سپس بررسی کنیم که آیا با حفظ دقت، تعداد اسپایک‌ها و هزینه محاسباتی کاهش می‌یابد یا نه.

`Baseline → Dynamic Threshold → Dynamic Time Constant → Dual-Adaptive → Efficiency Tests`

## چهار مفهوم اولیه

- **Spike:** خروجی صفر یا یکِ یک نورون در هر لحظه.
- **Threshold:** اگر پتانسیل نورون از آن عبور کند، نورون اسپایک می‌زند.
- **Time Constant (τ):** تعیین می‌کند اثر اطلاعات قبلی با چه سرعتی فراموش شود.
- **هدف ما:** وابسته‌کردن Threshold و τ به activity شبکه، به‌جای ثابت نگه‌داشتن آن‌ها.

## فاز ۱ — اجرای مقاله پایه

- مدل مرجع **SE-adLIF** را بدون تغییر در ساختار نورون اجرا کن.
- سه دیتاست مرجع این فاز: **SHD، SSC و ECG/QTDB**.
- برای هر دیتاست سه seed برابر با `42`، `123` و `456` اجرا کن.
- دیتاست‌ها باید ترتیبی اجرا شوند: `SHD → SSC → ECG`.
- در این فاز هنوز مدل پیشنهادی DA-LIF پیاده‌سازی نمی‌شود.

**خروجی:** `Mean ± SD` دقت، loss، تعداد پارامترها، GFLOPs، زمان آموزش و تنظیمات دقیق هر اجرا. Spike rate و SynOps در فاز efficiency تکمیل می‌شوند.

### وضعیت فعلی فاز ۱ — ۶ اکتبر ۲۰۲۶

| دیتاست | وضعیت | جزئیات |
| --- | --- | --- |
| SHD | تقریباً کامل | سه seed نتیجه دارند؛ `94.63% ± 0.86%`. اجرای seed 42 در epoch 214 قابل resume است. |
| SSC | کامل | سه seed اصلاح‌شده با تجمیع `summed_membrane_potentials` کامل شده‌اند؛ دقت test برابر `78.21% ± 0.11%` است. |
| ECG/QTDB | آماده اجرا | داده‌های train/test دانلود شده‌اند؛ آموزش هنوز شروع نشده است. |

بهترین دقت‌های فعلی SE-adLIF روی SHD:

| Seed | Best Accuracy | Best Epoch |
| ---: | ---: | ---: |
| 42 | 95.45% | 205 |
| 123 | 94.70% | 128 |
| 456 | 93.73% | 60 |

### Reference-result validity policy

- The original SHD SE-adLIF runs for seeds `42`, `123`, and `456` are valid
  reference runs. Their old `loss_agg: softmax` mode computed
  `sum_t softmax(y_t)`, exactly matching the explicit
  `sum_softmax_over_time` mode.
- Canonical SHD mean/SD calculations use those three original runs exactly
  once. Any future repeat of an existing seed must be labelled verification-only
  and excluded from the independent-run count.
- Earlier SSC runs are different: temporal mean was used instead of temporal
  sum, so those SSC runs are not valid for exact paper reproduction.

> نکته: پیکربندی فعلی SHD از test split برای validation و انتخاب checkpoint استفاده می‌کند؛ بنابراین این اعداد برای بازتولید اولیه مناسب‌اند، اما نتیجه نهایی unbiased مقاله نیستند.

## فاز ۲ — ساخت محیط مقایسه ثابت

- همه مدل‌ها باید با یک seed، batch size، preprocessing و تعداد epoch اجرا شوند.
- یک فایل config برای SHD بساز.
- نتیجه هر اجرا را در CSV یا JSON ذخیره کن.

**خروجی:** یک pipeline تکرارپذیر که بعداً همه مدل‌ها داخل آن تست شوند.

**وضعیت:** runner چنددیتاسته، اجرای چند seed، resume، checkpoint، early stopping، LR scheduler، CSV log و خلاصه خودکار آموزش پیاده‌سازی شده‌اند.

## فاز ۳ — فقط Dynamic Threshold

- Threshold نورون را بر اساس recent activity تغییر بده.
- در این مرحله τ را تغییر نده.
- هدف: کاهش اسپایک‌های غیرضروری، بدون افت محسوس دقت.

**خروجی:** مدل Dynamic-θ به‌همراه مقایسه با LIF و adLIF.

## فاز ۴ — فقط Dynamic Time Constant

- Threshold را ثابت نگه دار.
- τ را بر اساس activity تغییر بده.
- بررسی کن آیا مدل در پردازش زمانی بهتر یا sparse‌تر می‌شود.

**خروجی:** مدل Dynamic-τ و نتایج مستقل آن.

## فاز ۵ — مدل اصلی: Dual-Adaptive

- Dynamic Threshold و Dynamic τ را هم‌زمان فعال کن.
- ابتدا از ساده‌ترین قانون adaptation استفاده کن؛ attention یا شبکه اضافی اضافه نکن.
- اگر آموزش ناپایدار شد، range مربوط به threshold و τ را محدود کن.

**خروجی:** مدل اصلی پروژه، **DA-LIF**.

## فاز ۶ — مقایسه اصلی روی SHD

| مدل | Accuracy | Params | Spike Rate | SynOps |
| --- | --- | --- | --- | --- |
| LIF | — | — | — | — |
| SE-adLIF | 94.63% ± 0.86% | 450,760 | — | — |
| Dynamic-θ | — | — | — | — |
| Dynamic-τ | — | — | — | — |
| DA-LIF | — | — | — | — |

**هدف:** دقت مشابه یا بهتر، با spike کمتر و SynOps کمتر.

## فاز ۷ — اجرای سه seed

- هنگام توسعه فقط از یک seed استفاده کن.
- پس از freeze شدن معماری، seedهای `42`، `123` و `456` را اجرا کن.
- نتیجه را به شکل `Mean ± SD` گزارش کن.

**خروجی:** نتایج نهایی قابل‌استفاده در مقاله.

## فاز ۸ — انتقال به دیتاست‌های بعدی

1. SHD برای توسعه.
2. SSC برای آزمون temporal سخت‌تر.
3. ECG/QTDB برای بررسی تعمیم به سیگنال زمانی غیرگفتاری.

**قانون:** ابتدا baseline هر دیتاست کامل شود؛ سپس همان پروتکل بدون تغییر برای مدل پیشنهادی اجرا شود.

## فاز ۹ — آزمون Energy / Efficiency

- تعداد کل spikeها
- Spike rate
- SynOps
- Latency
- GPU memory
- تعداد پارامترها

**نکته مهم:** مقاله فقط درباره Accuracy نیست؛ باید نشان دهیم adaptive neuron واقعاً محاسبات را کاهش می‌دهد.

## فاز ۱۰ — آزمون‌های تکمیلی

- کاهش timestep
- حذف بخشی از eventها
- اضافه‌کردن temporal noise
- بررسی رفتار threshold و τ در طول زمان

**خروجی:** مشخص شود adaptation فقط Accuracy را تغییر نمی‌دهد، بلکه رفتار زمانی واقعی دارد.

## ترتیب اجرایی کار

1. تکمیل سه seed مدل SE-adLIF روی SHD
2. اجرای سه seed مدل SE-adLIF روی SSC
3. اجرای سه seed مدل SE-adLIF روی ECG/QTDB
4. تجمیع نتایج baseline به شکل `Mean ± SD`
5. پیاده‌سازی Dynamic Threshold
6. آزمون مستقل Dynamic Threshold روی SHD
7. پیاده‌سازی Dynamic Tau
8. آزمون مستقل Dynamic Tau روی SHD
9. ترکیب هر دو در DA-LIF
10. مقایسه DA-LIF و SE-adLIF روی SHD
11. Freeze کردن بهترین معماری
12. اجرای سه seed مدل پیشنهادی روی SHD، SSC و ECG
13. اندازه‌گیری spikeها، SynOps، latency و memory
14. ساخت جدول‌ها و نمودارهای نهایی

## فعلاً چه کاری انجام ندهد

- سه دیتاست را هم‌زمان اجرا نکند؛ آن‌ها را به‌ترتیب کامل کند.
- مدل را پیچیده نکند.
- تا baseline درست نشده، روش جدید پیاده نکند.
- فقط Accuracy را معیار موفقیت قرار ندهد.
- در مرحله توسعه هر تغییر کوچک را روی هر سه seed اجرا نکند.

## تعریف ساده موفقیت پروژه

**Accuracy حفظ شود + Spike کمتر شود + SynOps کمتر شود**

اگر DA-LIF با دقت مشابه، فعالیت کمتری نسبت به LIF/adLIF داشته باشد، جهت اصلی پروژه درست است.

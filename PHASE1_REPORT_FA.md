# گزارش کوتاه پیشرفت فاز اول: baseline و exploration مدل‌های جدید

**تاریخ: ۱۶ مهر ۱۴۰۵ **

## ۱. هدف و وضعیت فعلی

هدف این مرحله، امتحان‌کردن مدل‌های جدید برای رسیدن به عملکرد مشابه یا بهتر از **SE-adLIF baseline** است. تمرکز بر exploration، تحلیل موفقیت یا شکست تغییرات و یافتن ایده‌ای مشخص و قابل دفاع برای ادامه پژوهش است.

baseline روی **SHD، SSC و ECG/QTDB** با seedهای **42، 123 و 456** تهیه شده است. میانگین ± sample SD دقت به‌ترتیب **94.63% ± 0.86، 78.21% ± 0.11 و 88.32% ± 0.35** است؛ واحد SD، واحد درصد است. مدل‌های جدید ابتدا روی **SHD و seed 42** بررسی می‌شوند.

معماری SHD شامل دو لایه recurrent با 360 نورون در هر لایه است. مدل‌ها و نتایج جدید جدا از baseline نگهداری می‌شوند.

## ۲. نقطه شروع: SE-adLIF

ترتیب baseline، محاسبه membrane، سپس spike و reset و در پایان adaptation است:

```text
u_hat_t = alpha*u_(t-1) + (1-alpha)*(I_t-w_(t-1))
s_t = spike(u_hat_t-theta);  u_t = reset(u_hat_t,s_t)
w_t = beta*w_(t-1) + (1-beta)*q*(a*u_t+b*s_t)
```

در این روابط، `alpha=exp(-dt/tau_u)`، `beta=exp(-dt/tau_w)` و `I_t` مجموع جریان ورودی و recurrent است. دقت مرجع **SHD، seed 42 برابر 95.45%** است.

## ۳. مدل‌های امتحان‌شده: منطق و تغییر فرمول

### DTH-SE-adLIF — Dynamic Threshold

**ایده:** تنظیم threshold با دو trace سریع و کند از فعالیت نورون‌ها، برای کنترل فعالیت پایدار و پاسخ به تغییرات سریع:

```text
theta_t = theta_min + (theta_max-theta_min) *
          sigmoid(theta0 + h*(r_slow-target) - g*abs(r_fast-r_slow))
```

threshold و gainها trainable هستند؛ membrane و adaptation تغییر نمی‌کنند. نتیجه **95.19%** بود: نزدیک به baseline، اما بدون بهبود در این seed.

### MT-SE-adLIF — Multi-Timescale Adaptation

**ایده:** استفاده از دو adaptation memory سریع و کند برای الگوهای کوتاه‌مدت و بلندمدت:

```text
w_j,t = beta_j*w_j,(t-1) + (1-beta_j)*q*(a*u_t+b*s_t)
w_effective = m*w_fast + (1-m)*w_slow;  m = sigmoid(mix_logits)
```

membrane از adaptation مؤثرِ گام قبل استفاده می‌کند. نتیجه **93.73%**، در مقابل training accuracy نزدیک **99.99%** در بهترین checkpoint بود؛ حافظه بیشتر generalization را بهتر نکرد. اجرا دستی متوقف و بهترین checkpoint ارزیابی شد.

### MR-SE-adLIF — Multi-Rate Adaptation

**ایده:** به‌روزرسانی adaptation هر K timestep، درحالی‌که membrane و spike در هر گام محاسبه می‌شوند:

```text
beta_K = beta**K
w_new = beta_K*w_old + (1-beta_K)*q*(a*mean(u_block)+b*mean(s_block))
```

بین updateها، adaptation ثابت است. **K=1** همان SE-adLIF است. **K=2** تقریباً دقت را حفظ کرد؛ K=4 افت داشت و K=8 ناموفق بود. کاهش تعداد updateها هنوز به معنای اثبات کاهش latency یا energy نیست.

### FP-SE-adLIF — Frequency Parameterization

**ایده:** یادگیری frequency زیرآستانه‌ای در بازه **1–50 Hz** به‌جای یادگیری مستقیم `a`:

```text
phi = 2*pi*f_hz*(dt_ms/1000)
A = (alpha+beta-2*sqrt(alpha*beta)*cos(phi)) / ((1-alpha)*(1-beta))
a_repository = A/q
```

تقسیم بر q، scaling اصلی را حفظ می‌کند. frequency با firing rate یکی نیست. نتیجه **94.26%** با training accuracy برابر **99.99%** در بهترین checkpoint، از baseline بهتر نشد.

### DA-SE-adLIF — Delay-Aware Input

**ایده فعلی:** استخراج بهتر اطلاعات زمانی با ترکیب جریان feedforward مستقیم و تأخیرهای ثابت **1، 2 و 4 فریم**:

```text
c_t = W*x_t+b
c_filtered,t = (1-g)*c_t + g*sum_d softmax(logits)_d*c_(t-d)
```

gate و mixing weights trainable هستند؛ جای تأخیرها ثابت و recurrent current بدون تغییر است. gate اولیه صفر، رفتار baseline را حفظ می‌کند. مدل **2,880 پارامتر** اضافه دارد. تست‌های صحت و اجرای یک epoch موفق بوده‌اند؛ آموزش اصلی ادامه دارد.

## ۴. خلاصه نتایج SHD — seed 42

| مدل | دقت ثبت‌شده | epoch بهترین checkpoint | وضعیت |
|---|---:|---:|---|
| SE-adLIF baseline | **95.45%** | 205 | نتیجه مرجع موجود |
| DTH-SE-adLIF | 95.19% | 164 | 300 epoch و evaluation انجام شده |
| MT-SE-adLIF | 93.73% | 104 | توقف دستی؛ بهترین checkpoint ارزیابی شده |
| MR، K=1 | 95.01% | 36 | evaluation انجام شده |
| MR، K=2 | 94.92% | 77 | evaluation انجام شده |
| MR، K=4 | 93.29% | 111 | evaluation انجام شده |
| MR، K=8 | 15.50% | 2 | پایان اجرا؛ یادگیری ناموفق |
| FP-SE-adLIF | 94.26% | 62 | evaluation انجام شده |
| DA-SE-adLIF | **93.15%، موقت** | 51 | آموزش ادامه دارد |

در snapshot بررسی‌شده، DA تا epoch **56** پیش رفته بود؛ بهترین checkpoint در epoch 51 بود. epochها از صفر شماره‌گذاری می‌شوند.

**محدودیت مقایسه:** در SHD، test set برای checkpoint selection نیز استفاده شده و ارزیابی مستقل نیست. batch size در MR برابر 256 و در سایر مدل‌ها 512 است؛ early stopping و scheduler نیز کاملاً یکسان نبوده‌اند. این جدول نتیجه exploration است، نه اثبات برتری معماری.

## ۵. برداشت فعلی و ادامه فاز اول

هنوز هیچ مدل جدیدی در این seed از baseline بهتر نشده است. DTH نزدیک‌ترین نتیجه را دارد؛ MT و FP نشان می‌دهند پیچیده‌ترشدن adaptation لزوماً generalization را بهتر نمی‌کند. MR بیشتر برای بررسی trade-off دقت و هزینه مناسب است.

### candidate بعدی برای training: DA-SE-adLIF

**DA-SE-adLIF گزینه منتخب فعلی برای ادامه exploration است و training اولیه آن روی SHD با seed 42 آغاز شده است.** فرضیه این candidate، بهبود استخراج الگوهای زمانی از طریق ترکیب trainable ورودی مستقیم و ورودی‌های delayed، بدون تغییر دینامیک اصلی SE-adLIF است. انتخاب آن برای آزمایش به معنای اثبات برتری نیست؛ نتیجه فعلی موقت است.

گام بعد، **تکمیل training این candidate و مقایسه حالت بدون فیلتر، فیلتر ثابت و فیلتر trainable** است. در صورت مشاهده نتیجه امیدوارکننده، بررسی با seedهای 123 و 456 و سپس SSC و ECG، با پروتکل یکسان و validation مستقل دنبال می‌شود.

هدف، یافتن **ایده‌ای مشخص، قابل آزمون و قابل دفاع** است. نتیجه منفی نیز مسیرهای ضعیف را حذف می‌کند. بهترشدن یک عدد به‌تنهایی اثبات novelty نیست؛ ادعای بهبود accuracy، سرعت یا energy به آزمایش کنترل‌شده نیاز دارد.

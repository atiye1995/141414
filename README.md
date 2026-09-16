# راهنمای بازتولید — ترتیب اجرا

## نصب (یک‌بار)
```
pip install pyscf openfermion openfermionpyscf numpy scipy matplotlib pandas --quiet
```
(در Colab: `!pip install ...`)

## ترتیب اجرای فایل‌ها (دقیقاً همین ترتیب)

همه‌ی فایل‌ها را در یک پوشه بگذارید و به همین ترتیب اجرا کنید:

```
python3 build_H2.py                    # ~۱ ثانیه
python3 build_H4.py                    # ~۱۵-۳۰ ثانیه
python3 build_LiH.py                   # ~۱۰-۱۵ ثانیه
python3 run_comparison.py              # ~۲-۵ دقیقه (۳ سیستم × ۳ بودجه‌ی شات)
python3 make_chart_comparison.py       # نمودار اصلی: فقط-خطی/BMA/Spike-Slab
python3 channel_generality_test.py     # نمودار دوم: عمومیت کانال نویز
```

## چرا این ترتیب؟

- `build_H2.py` / `build_H4.py` / `build_LiH.py` مستقل از هم‌اند اما باید **قبل از** `run_comparison.py` اجرا شوند، چون فایل‌های `{system}_pauli_terms.pkl`، `{system}_psi_opt.npy`، `{system}_q_table.npy` را می‌سازند که `run_comparison.py` و `channel_generality_test.py` به آن‌ها نیاز دارند.
- `run_comparison.py` باید قبل از `make_chart_comparison.py` باشد چون `comparison_summary.csv` را می‌سازد که نمودار از آن می‌خواند.
- `channel_generality_test.py` مستقل است و فقط به فایل‌های سه `build_*.py` نیاز دارد.

## فایل‌های ماژول (import می‌شوند، مستقیم اجرا نکنید)
- `physics_engine.py` — ساخت همیلتونین (OpenFermion+PySCF) و VQE
- `noise_model.py` — مدل نویز (میرایی دامنه+فاز و دپولاریزاسیون)
- `stats_analysis.py` — فقط-خطی، BMA-لاپلاس، Spike-and-Slab

## خروجی نهایی
- `chart_comparison_reproduced.png`
- `chart_channel_generality_reproduced.png`
- `comparison_summary.csv`

**نکته:** اعداد فقط-خطی و r_true تا ۴ رقم اعشار دقیقاً با مقاله یکسان درمی‌آیند. اعداد BMA ممکن است کمی فرق کند (VQE چندپارامتری می‌تواند به نقطه‌ی هم‌ارز اما کمی متفاوت همگرا شود) — اما الگوی کیفی (BMA > Spike-Slab > فقط-خطی) در همه‌جا حفظ می‌شود.

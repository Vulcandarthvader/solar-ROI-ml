# SolarROI — when do rooftop solar panels pay for themselves?

A Python + machine-learning project for a real rooftop solar system in Goa, India.
**Question:** ₹1,70,000 was spent on panels — when does that money come back?

**Answer on the real data:** about **19.9 years** (16.7 years if tariffs rise fast, not within
25 years if they rise slowly). The panels do pay back, but a 7% fixed deposit would have earned more.

---

## The idea in 4 lines

1. **Saving = bill without panels − real bill.** The real bill is known. The bill *without* panels is never seen.
2. So we **estimate the bill without panels** with machine learning — two different ways (Model A and Model B) — and check they agree.
3. Solar generation is only shown as a bar chart in the ShinePhone app, so we **read the numbers off screenshots** with OpenCV + OCR.
4. Past savings are **measured**; only future years are **forecast**, for the 25-year life of the panels.

---

## How the data flows

```
ShinePhone screenshots ──► vision.py ──► kWh per month ─┐
                                                        ├─► cleaning.py ──► clean monthly table
Bill CSV (105 bills) ───────────────────────────────────┘           │
                                                                    ▼
                                         features.py (time, month, tariff slabs)
                                                │                    │
                                                ▼                    ▼
                                   Model A: tariff model   Model B: pre-solar trend
                                                └──────┬─────────────┘
                                                       ▼
                                  finance.py: monthly savings → value of 1 solar unit
                                                       ▼
                                  25-year forecast → payback, NPV, ROI, 3 scenarios
                                                       ▼
                                  app.py (website)   /   analyse.py (terminal)
```

| Step | What happens | Output on the real data |
|---|---|---|
| 1. Read screenshots | Measure each green bar in pixels, convert to kWh | 39 monthly values, **1.01% error** vs the meter |
| 2. Clean bills | Fix duplicates, gaps, copied readings, refunds, arrears | 105 bills → 108 clean months (Dec 2016 – Nov 2025) |
| 3. Find install month | First month with solar output | June 2023 |
| 4. Model A | Learn the price formula, then price *all* units the house used | Ridge (L2) chosen |
| 5. Model B | Learn the bill trend before solar, extend it forward | Lasso (L1) chosen |
| 6. Savings so far | Average of A and B, month by month | ₹18,853 in 2.5 years (₹628/month), A and B agree 98% |
| 7. Value of 1 unit | Last 12 months' savings ÷ solar kWh | ₹3.37 per kWh |
| 8. Forecast | 2,517 kWh/year, −0.6%/yr, tariff +5.5%/yr | Payback 19.9 years |

---

## What each file does

| File | Job |
|---|---|
| `src/vision.py` | Screenshot → kWh per month (OpenCV + Tesseract OCR) |
| `src/cleaning.py` | Load CSV, fix messy bills, add screenshot months, pick trustworthy training rows |
| `src/features.py` | Make model inputs: years since start, month as sin/cos, units above each tariff slab |
| `src/models.py` | The 4 models, TimeSeriesSplit comparison, Model A and Model B |
| `src/finance.py` | Savings so far, 25-year cash flow, payback, NPV, ROI, scenarios |
| `src/pipeline.py` | `run_analysis()` — calls every step in order. Used by both the app and the CLI |
| `src/config.py` | Every assumption (cost, tariff rise, degradation, ...) in one place |
| `app.py` | Streamlit website with 5 tabs |
| `analyse.py` | Same analysis in the terminal |
| `data/` | The bill CSV and 4 sample screenshots |

---

## The 4 machine-learning models

| Model | Idea |
|---|---|
| **Baseline (mean)** | Always predict the average bill. Any real model must beat this. |
| **Linear Regression** | `bill = w1·x1 + w2·x2 + … + b`. Weights chosen to make squared errors smallest. |
| **Ridge (L2)** | Linear Regression + penalty on `weight²`. Keeps weights small → steadier on new months. |
| **Lasso (L1)** | Linear Regression + penalty on `|weight|`. Can set useless weights to exactly 0. |

**How the best model is chosen — TimeSeriesSplit:** train on older months, test on the next newer
block, 5 times. Lowest average error (MAE) wins, then it is retrained on all data.
Months are never shuffled — otherwise the model would learn from the future while being tested on the past.

**Why only linear models?** Under 100 rows of data, they extend trends into future years
(tree models can't), and every prediction can be explained as a weighted sum.

**Why tariff-slab features?** Electricity costs more per unit after 100, 200, 300, 400 units.
With "units above each slab" as columns, a linear model learns a separate price per slab.
Example: 250 units → `kwh_over_100 = 150`, `kwh_over_200 = 50`.

### Model A vs Model B

| | Model A — tariff model | Model B — pre-solar trend |
|---|---|---|
| Learns | bill from units bought + date | bill from date only |
| Trained on | all clean months | only months before solar (61) |
| Bill without solar = | price of (units bought + solar units) | the old trend, continued |
| Uses solar data? | yes | never → independent check |

---

## The money formulas

```
saving (month)      = bill without solar − real bill
value of 1 unit     = savings in last 12 months ÷ solar kWh in them          = ₹3.37
kWh (future year k) = 2,517 × (1 − 0.006)^k
₹ per kWh (year k)  = 3.37 × (1 + 0.055)^k
cash (year k)       = kWh × ₹ per kWh − upkeep − inverter (₹25,000 once, year 12)
payback             = year when (saved so far + future cash) reaches ₹1,70,000
NPV                 = −1,70,000 + saved so far + Σ cash(k) ÷ 1.07^(years from install)
ROI (25 years)      = (total lifetime saving − cost) ÷ cost
```

**NPV in one line:** ₹107 next year is worth ₹100 today if a deposit pays 7%. NPV adds up all
future savings in today's money. Negative NPV → the deposit would have earned more.

**Scenarios** (instead of a complex simulation): change the tariff rise and the sunlight together.

| Scenario | Tariff rise | Sunlight | Payback |
|---|---|---|---|
| Pessimistic | 3.5% | 90% | not within 25 years |
| Expected | 5.5% | 100% | 19.9 years |
| Optimistic | 7.5% | 110% | 16.7 years |

---

## How to run it

1. Install Python 3.10+ and the **Tesseract OCR** program
   (Windows: https://github.com/UB-Mannheim/tesseract/wiki — tick "add to PATH"; macOS: `brew install tesseract`).
2. In the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py            # the website, opens in your browser
python analyse.py               # the same numbers in the terminal
```

On Windows, `src/vision.py` automatically uses Tesseract from `C:\Program Files\Tesseract-OCR\` if it is installed there.

**Your own data:** turn off "Use the sample data" in the app, upload ShinePhone *Month* chart
screenshots and a CSV with columns `year, month, grid_kwh, bill_amount` (and `solar_kwh` if you have it).

---

## Limitations (honest)

- Only 2.5 years of solar data, from one house in Goa.
- Model B only uses the calendar, so it is weak alone (R² below 0). It is there as an independent check.
- Net metering is assumed to balance out over the year.

## Ideas to extend

Weather data as a feature · a real tariff table from the electricity company · a FastAPI endpoint
around `run_analysis()` · automated tests with pytest · deploy on Streamlit Cloud.

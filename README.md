# SolarROI — when do rooftop panels actually pay for themselves?

A machine-learning system that answers one question about a real rooftop solar
installation in Goa, India: **given ₹1.7 lakh spent on panels, when does the
investment break even — and how confident can we be?**

The hard part is not the arithmetic. It is that the quantity you need is never
observed: *the electricity bill you would have received if the panels were not
there.* This project estimates that counterfactual two independent ways,
reconstructs the monthly saving from it, and projects the result across the
system's 25-year life with an explicit uncertainty band.

It also solves a smaller problem that anyone with a Growatt inverter runs into:
the ShinePhone app shows generation as a bar chart and offers no export. A
computer-vision pipeline reads the numbers straight off a phone screenshot,
**within 1.0% of the metered values** on 30 months of ground truth.

```bash
pip install -r requirements.txt
streamlit run app.py          # the interactive app
python analyse.py             # the same pipeline, headless
pytest                        # 38 tests
```

---

## The result on the real data

117 months of utility bills (Dec 2016 – Nov 2025) and 39 months of generation
read off five ShinePhone screenshots, for a system commissioned in June 2023.

| | |
|---|---|
| **Median payback** | **18.9 years** (P10–P90: 16.1 – 22.6) |
| Probability of breaking even inside 25 years | 93% |
| Recovered so far | ₹19,020 — 11.2% of capex, in 2.5 years |
| Average monthly saving | ₹634 |
| Realised value of a generated unit | ₹3.19/kWh (₹3.41 over the last 12 months) |
| Levelised cost of solar (LCOE) | ₹6.77/kWh |
| IRR / NPV @ 7% | 3.6% / −₹57,140 |
| Grid import, before → after | 279 → 129 kWh/month (65% self-sufficient) |
| CO₂ avoided to date | 4.2 tonnes |

**The honest reading of this.** The array will very likely return its cost in
cash, and it is very unlikely to beat a fixed deposit. Both are true at once,
and the app says so rather than picking the flattering half. The reason is
Goa's unusually cheap domestic tariff: each generated unit displaces about
₹3.41 of grid electricity at current rates, so 2,477 kWh/year is worth roughly
₹8,450 against ₹170,000 of capital. The payback case rests almost entirely on
tariff escalation — the sensitivity analysis shows it is the single largest
lever, worth ±2.6 years on its own.

That conclusion is the point. A portfolio project that engineers its way to
"payback in 6 years!" would be easier to sell and would be wrong.

---

## How it works

```
ShinePhone screenshots ─┐
                        ├─► CV extraction ─┐
utility bill CSV ───────┴─► cleaning ──────┼─► tariff surrogate ─┐
                                           │                     ├─► realised savings
                                           └─► synthetic control ┘         │
                                                                           ▼
                          generation model ──► 25-year projection ──► Monte Carlo ──► payback
                                   │
                                   └────────► anomaly detection (panel health)
```

### 1. Reading numbers off a phone screenshot (`src/vision.py`)

Bars are segmented in HSV, the plot band is found by green mass, and each bar's
pixel height is converted to kWh using the axis. The interesting part is that
**neither axis trusts OCR**. At this font size Tesseract reads `80.0` as `30.0`
and `2500.0` as `2900.0` — reliably enough to ruin an extraction, subtly enough
to look plausible.

Both axes are calibrated by consensus instead, exploiting what is structurally
true of any chart axis:

- gridlines are evenly spaced and the bottom one is zero;
- axis steps are "nice" numbers (1, 1.5, 2, 2.5, 3, 4, 5 × 10ᵏ);
- OCR errors on digits are **substitutions**, not insertions — so a candidate
  step is scored with a single-character edit distance, and a label misread by
  one digit still votes for the right answer.

The x-axis is fitted with RANSAC over the OCR'd tick positions, so a misread
tick is simply out-voted by its collinear neighbours.

> An early version knew only the 1/2/2.5/5 mantissas, snapped a true step of 40
> to 50, and read every bar in the 2023 chart 25% too high — with no error
> raised. The mantissa set and the consensus scoring exist because of that bug;
> `tests/test_vision.py` pins it.

**Validation.** The bill CSV independently records metered generation for 30 of
the extracted months, giving real ground truth:

| Metric | Value |
|---|---|
| Mean absolute percentage error | **1.01%** |
| Worst single month | 2.85% |
| Months validated | 30 |

Every extraction carries a confidence score, and anything below 85% is
surfaced for review in the UI with the values editable before they are used.

### 2. Cleaning what a utility actually exports (`src/ingest.py`)

Every repair is recorded as a flag and reported, never applied silently:

| Issue | Found | Treatment |
|---|---|---|
| Two bills in one calendar month | 1 | Money and units summed; generation is a monthly total, so not summed |
| Bi-monthly readings written to both months | 9 | Flagged; excluded from fitting so one meter reading is not counted twice |
| Negative bills (credit adjustments) | 6 | Kept for cash accounting, excluded from tariff fitting |
| Missing months | 4 | Seasonally interpolated and flagged |
| Arrears / true-up bills | 1 | Flagged; headline savings computed robustly around them |

Two of these mattered a lot:

- **Generation is never interpolated.** A seasonal fill once gave a 2020 month
  solar output, which moved the detected commissioning date three years early
  and corrupted every downstream number. Regression-tested.
- **A ₹2,469 arrears bill for 57 units** would otherwise have eaten a fifth of
  the measured return. The headline total is the mean saving over clean months
  scaled to the observed span; the raw as-billed figure is reported next to it.

### 3. The counterfactual — two independent estimators (`src/models.py`)

**A. Tariff surrogate (physical).** Learn the utility's price function from the
bills themselves, `bill = f(units, calendar time)`, then re-query it with
*gross* household demand (billed units + generation). Slab-hinge features
(`units above 100/200/300/400`) turn India's block-rate tariff into something a
linear model can represent exactly, so the linear baseline competes honestly
instead of losing to a feature-engineering artefact.

This also exposes a mistake that routinely understates household solar: because
tariffs are block-rate, generation displaces the *top* slab, so the rate solar
earns is the **marginal** rate, not the average rate printed on the bill. The
model reports both.

**B. Synthetic counterfactual (statistical).** Fit bills on calendar features
using **only the 78 pre-solar months**, then extrapolate that trajectory across
the solar era. It never sees a single solar-affected bill or any generation
figure, so it is a genuine independent check.

The two agree to **77%**, and the app reports their disagreement as a
first-class number rather than hiding behind an average.

### 4. Model selection that shows its work

Every model is scored by rolling-origin cross-validation (`TimeSeriesSplit`,
never `KFold` — shuffling months on a series with a rising tariff trend lets
the model peek at the future and inflates every score). A mean-predicting
baseline is included, because with ~60 monthly observations a boosted forest
genuinely can lose to the mean, and a leaderboard that cannot show that is not
telling the truth.

```
Generation model (monthly kWh)      MAE    RMSE  MAPE_%     R2
  ExtraTrees                       18.80   24.45    9.91   0.23   ← selected
  RandomForest                     19.84   26.38   10.78   0.27
  GradientBoosting                 25.42   29.93   14.05   0.05
  Baseline (mean)                  28.53   36.90   14.54  -0.61
  Ridge                            42.03   48.48   23.43  -1.49
```

**The extrapolation rule.** Tree ensembles predict a constant outside their
training range — a forest asked about 2032 having seen only 2025 returns the
2025 leaf value forever. For the synthetic counterfactual, whose entire job is
extrapolating a trend into unseen years, that is not a small accuracy loss but
a silent failure. Those tasks are restricted to model classes with a global
functional form, and **the leaderboard still shows the excluded trees winning
on CV MAE**, marked `excluded: cannot extrapolate`. The same reasoning shapes
the 25-year generation projection: predictions are made at the last observed
panel age and then declined explicitly at the configured degradation rate,
rather than asking a tree about year 25.

### 5. Panel health (`detect_anomalies`)

Two detectors of deliberately different kinds — a one-sided residual z-score
(interpretable, directional: only shortfalls are faults) and an Isolation
Forest over `(residual, shortfall %, month, panel age)` (unsupervised, catches
odd combinations). A month is flagged only when both agree **and** the
shortfall exceeds 10%, because a statistical flag is not yet an engineering
problem.

On this array it flags **July 2024, 23.8% below its own seasonal norm** — one
month, in a monsoon, worth a look at the panels rather than a service call. The
commissioning month is excluded as partial by construction, and unusually
*good* months are labelled "above normal" rather than alarmed on.

### 6. Money, with error bars (`src/roi.py`)

**Measure the past, model only the future.** Months that have happened
contribute their realised savings; the projection starts the month after the
last bill. Simple and discounted payback, NPV, IRR, LCOE, 25-year ROI and CO₂
all fall out of that blended schedule.

**Monte Carlo (10,000 paths).** Each path draws its own tariff escalation,
degradation rate, per-year weather multiplier, O&M cost, and a 3%/year chance
of a fault costing a quarter of that year's output. The realised era is held
**fixed** — it already happened, and resimulating it would launder measured
savings into simulated ones. Fully vectorised, so the whole simulation re-runs
on every slider change.

**Sensitivity (tornado).** The Monte Carlo says how wide the outcome is; the
tornado says which assumption is responsible:

| Assumption | Swing in payback |
|---|---|
| Tariff escalation 3.5% vs 7.5% | 5.1 years |
| O&M ±50% | 4.1 years |
| Capex ±15% | 3.4 years |
| Annual generation ±10% | 3.2 years |
| Inverter swap ₹40k vs none | 2.7 years |
| Degradation 1.0% vs 0.3%/yr | 1.3 years |

### 7. Explainability (`src/explain.py`)

SHAP for global importance and per-month waterfalls, with the explainer chosen
per model class (`TreeExplainer` for ensembles, permutation-based for scaled
linear pipelines) and a permutation-importance fallback so the UI never breaks
on an unsupported estimator. On the tariff model the top drivers are units
billed, units above the 100-unit slab, and years elapsed — which is exactly
what a block tariff with annual revisions should look like, and a useful
sanity check that the model learned the tariff rather than a coincidence.

---

## Repository layout

```
solar-roi-ml/
├── app.py                  Streamlit UI (7 tabs)
├── analyse.py              CLI running the identical pipeline
├── src/
│   ├── vision.py           screenshot → kWh (OpenCV + OCR consensus calibration)
│   ├── ingest.py           bill cleaning, quality report, generation merge
│   ├── features.py         slab hinges, seasonality, panel age
│   ├── models.py           tariff surrogate, synthetic control, generation, anomalies
│   ├── roi.py              payback, NPV/IRR/LCOE, Monte Carlo, sensitivity
│   ├── explain.py          SHAP + fallbacks
│   ├── pipeline.py         orchestration → AnalysisResult
│   ├── config.py           every economic assumption, sourced
│   └── theme.py            validated colour palette for the charts
├── tests/                  38 tests
├── notebooks/walkthrough.ipynb
└── data/                   billdata.csv + sample screenshots
```

## Using it on your own system

1. **Screenshots** — ShinePhone → Plant → Power|Energy → the *Month* chart for
   each year, plus the *Year* chart.
2. **Bills** — a CSV with `year, month, grid_kwh, bill_amount` (and
   `solar_kwh` if your utility reports it). Most Indian DISCOM portals export
   this.
3. Set your capex and tariff expectations in the sidebar and run.

Where the screenshots and the CSV overlap, the metered CSV value wins and the
disagreement is reported as a cross-validation statistic; where only the
screenshots reach, they extend the record past your last bill — worth 9 extra
months here.

## Design decisions worth defending in an interview

| Decision | Why |
|---|---|
| `TimeSeriesSplit`, never `KFold` | Shuffled folds leak the future into training on a trending series |
| No lagged features anywhere | A lag makes in-sample fit look excellent and the 25-year projection meaningless, because at forecast time there is no lag to feed it |
| Tree models excluded from extrapolation tasks | They predict a constant outside their training range — silent failure, not graceful degradation |
| Two counterfactuals rather than one | Different assumptions landing in the same place is evidence; one confident number is not |
| Realised era held fixed in the simulation | Resimulating measured months manufactures false uncertainty and false precision |
| Marginal, not average, tariff rate | Solar displaces the top slab; using the average rate is the most common way household solar ROI gets understated |
| Robust aggregation around arrears bills | One true-up bill would otherwise consume a fifth of the measured return |
| Confidence score on every extraction | A CV pipeline that cannot say "I am unsure" will be wrong silently |

## Known limitations

- **2.5 years of post-solar bills.** Enough to measure a seasonal cycle twice,
  not enough to measure degradation: the data-fitted degradation term comes out
  slightly *negative*, which is noise, not panels improving with age. The
  projection therefore uses the datasheet assumption and says so.
- **The synthetic counterfactual is weak in isolation** (negative CV R² — no
  better than the mean). Calendar features alone cannot predict a household's
  bill well. It earns its place as an *independent* check, not as a good model,
  and the app shows its cross-validated scores rather than implying otherwise.
- **Net metering is assumed to net out annually.** Gross demand is reconstructed
  as billed units + generation, which is right if export credits are eventually
  consumed and wrong if they expire unused.
- **One household, one climate.** Nothing here has been validated on another
  installation or another DISCOM's tariff structure.

## Data note

The bill history and screenshots are from a single private residence, shared by
the owner for this analysis. No account numbers, addresses or names are
included.

## Licence

MIT.

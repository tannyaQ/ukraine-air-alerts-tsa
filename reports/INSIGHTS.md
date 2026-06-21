# Key findings — Ukraine air-raid alerts time-series analysis

*Target:* national **daily count of region-alerts** (Variant A), Kyiv time, from the
volunteer dataset (2022-02-25 → 2026-06-19, 1 576 days, 101k records). Every claim
below is backed by a reproducible script in `src/`; figures are in
`reports/figures/`.

---

## 1. The series is trend-dominated, not seasonal
STL (weekly period, full history): **strength of trend `F_T = 0.52`**, **strength of
weekly seasonality `F_S = 0.18`** (weak). Residual std (**17.6**) is *larger* than the
whole weekly amplitude (**9.7**) — day-to-day noise dwarfs the day-of-week signal.

The level is strongly non-stationary: the 30-day mean falls to a trough **~26
(Dec 2022)** and then rises **~4.4×** to **~110 (2026)**, with a clear regime change
around **July 2025** (monthly mean steps from ~60 to ~72 and keeps climbing).
*Figures: `01_daily_alert_count.png`, `04_recent_zoom.png`, `05_stl_decomposition.png`.*

## 2. Intensification is depth, not breadth
While the alert count rose ~4.4×, the number of **distinct oblasts under alert per
day stayed range-bound (~13–21, median 17)**. The escalation is *more alerts per
region*, not *more regions* — so the count target reflects real intensification, not
geographic inflation. *Figure: `02_regions_under_alert_per_day.png`.*

## 3. The only stable weekly signal is "Sunday is the quietest day"
Checked across sub-periods (STL seasonal effect by year × day-of-week):

| year | Thu | Sun |
|------|-----|-----|
| 2022 | +10.6 | −2.2 |
| 2023 | +7.2 | −5.8 |
| 2024 | +0.1 | −6.2 |
| 2025 | +2.1 | −4.1 |
| 2026 | +4.7 | −7.2 |

**Sunday is negative every year** — a stable, real signal. The apparent **Thursday
peak is NOT stable**: strong in 2022–23, gone in 2024, and the "peak" day migrates
(Thu→Wed→Sat→Fri). Reporting "Thursday is busiest" would have been an artifact of the
early war period. Honest version: **weekends (esp. Sunday) are quieter; mid-week is
marginally busier, but which mid-week day is not reliable.**

## 4. Forecasting: the predictability is in the trend; weekly seasonality adds nothing
Walk-forward, 12 consecutive weekly folds (h=7), expanding window from 2025-07-01,
**mean across folds**:

| method | MAE | RMSE |
|--------|-----|------|
| seasonal_naive | 31.69 | 37.79 |
| naive_drift | 28.22 | 32.64 |
| **holt_no_seas (trend only)** | **21.84** | **26.67** |
| hw_add (trend + weekly seasonality) | 22.39 | 27.07 |
| hw_mul | 22.60 | 27.26 |

- A **trend-only model (Holt) is the best performer**. Adding weekly seasonality
  (full Holt-Winters) **does not help — it is marginally worse** (+0.55 MAE). This was
  *pre-registered*: we accepted before seeing metrics that seasonality might add
  nothing, and it didn't. No parameters were tuned to make the seasonal model "win".
- Both trend models clearly beat the baselines (naive-drift 28.22, seasonal-naive
  31.69). The edge over naive-drift is **robust level/trend smoothing**, not
  day-of-week structure — `holt_no_seas` ≈ `hw_add` confirms this directly.
- `hw_add` ≈ `hw_mul`, so the growing seasonal amplitude does not justify a
  multiplicative form either.

## 5. Error is event-driven, not "growing with the acceleration"
Per-fold MAE is **not monotonically increasing**; it is dominated by specific volatile
weeks (e.g. the week of **2026-05-09**: ~48–62 MAE for *all* methods). Outside such
event spikes the trend model holds the accelerating regime. naive-drift is the most
erratic — it anchors on a single noisy last day (e.g. 2026-05-30 fold: drift 47 vs
trend models ~9). *Figure: `06_walkforward_mae.png`, table `walkforward_mae.csv`.*

---

## Honesty notes & limitations
- **`naive=True` (4.94%)** rows are a fixed 30-min placeholder for an unknown end time
  (verified: all exactly 30.0 min, std 0). Kept for the count target, excluded from any
  duration analysis (`src/features.py:duration_valid`).
- The snapshot's **last day (2026-06-20) was dropped** as partial collection (9 alerts
  vs ~100 typical) — hygiene, not result-tuning.
- **Univariate, exogenous-driven.** Alerts are caused by attack decisions; no
  univariate model can anticipate those. Forecasts capture *momentum and level*, not
  causes — which is exactly why the honest bar was naive-with-drift.
- **Source caveats:** volunteer data (eTryvoga), oblast level; Luhansk (occupied) is
  effectively absent (3 records over 4+ years) and kept as data reality, not removed.

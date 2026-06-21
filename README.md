# Ukraine Air-Raid Alerts — Time Series Analysis

Time-series analysis of air-raid alerts in Ukraine: data acquisition → EDA →
decomposition → forecasting (with an honest baseline) → insights.

> **Framing.** Air-raid alerts are an **event-driven** series (driven by military
> attacks), not a clean seasonal process. The goal here is therefore twofold:
> (1) characterize the **structure** of the series (daily/weekly seasonality,
> regime shifts), and (2) forecast **honestly** — every model is compared against
> a naive / seasonal-naive baseline on MAE/RMSE. A model that does not beat the
> baseline is reported as such, not hidden.

## Key findings (full detail in [reports/INSIGHTS.md](reports/INSIGHTS.md))

1. **Trend-dominated, not seasonal.** STL: strength of trend `F_T = 0.52` vs weekly
   seasonality `F_S = 0.18`. The level rose ~4.4× (trough ~26 in Dec 2022 → ~110 in
   2026), regime change ~July 2025.
2. **Depth, not breadth.** Counts rose ~4.4× while distinct oblasts under alert/day
   stayed range-bound (~13–21) — more alerts per region, not wider geography.
3. **Only stable weekly signal: Sunday is the quietest day.** The apparent "Thursday
   peak" is a 2022–23 artifact and does not hold across years.
4. **Forecasting — the predictability is in the trend.** A trend-only Holt model
   (MAE 21.84) is the best; adding weekly seasonality does **not** help (Holt-Winters
   22.39). Both beat naive-with-drift (28.22) and seasonal-naive (31.69). Reported
   honestly — no tuning to make the seasonal model win.
5. **Error is event-driven**, not steadily growing with the acceleration.

## Data

- **Source:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT License).
- **Primary file:** `volunteer_data_en.csv` — oblast-level, consistent granularity
  from 2022-02-25 onward. Committed as a snapshot in `data/raw/` (source of truth).
- **Crosscheck file:** `official_data_en.csv` — official data; granularity shifts
  to raion level from Dec 2025. Not committed (~28 MB); fetched on demand.
- **Verified schema** (interval level): `region, started_at, finished_at, naive`.
  Timestamps are ISO 8601 UTC; we convert to `Europe/Kyiv` for analysis.
- **Target series:** national **daily count of alerts**, Kyiv local time. The exact
  deduplication policy is finalized after the EDA "data assumptions" step (see below).
- **Cleaning rules (empirically verified, not assumed):** `naive=True` rows are a
  fixed 30-min placeholder for an unknown end time (4 998 rows, all exactly 30.0 min,
  std = 0) — **kept** for the count target, **excluded** from duration analysis. Rows
  with `finished_at <= started_at` (5) are excluded from duration analysis only. Both
  rules live in `src/features.py:duration_valid`.

## Project structure

```
.
├── data/
│   ├── raw/            # immutable source snapshot (+ DATA_SOURCE.md provenance)
│   ├── interim/        # cleaned intermediate data
│   └── processed/      # final daily series ready for modeling
├── notebooks/          # narrative analysis (numbered)
├── src/                # reusable logic, imported by notebooks
│   └── data_loader.py  # load + validate interval-level data (no aggregation)
├── scripts/
│   └── download_data.py  # fetch raw snapshots (provenance / refresh)
├── reports/figures/    # exported plots
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# 1. Fetch raw data into data/raw/ (volunteer is also committed as a snapshot)
python scripts/download_data.py

# 2. Load + validate the interval-level data (prints an empirical quality report)
python -m src.data_loader

# 3. Build the modeling-ready daily series -> data/processed/daily_alerts.csv
python -m src.build_dataset

# 4. EDA assumption checks, figures, and STL decomposition
python -m src.validation
python -m src.plots
python -m src.decomposition

# 5. Walk-forward forecasting evaluation (baselines vs Holt-Winters)
python -m src.forecasting
```

## Analysis pipeline (planned)

1. **Data assumptions validation** (EDA step 0 — runs *before* any modeling):
   - `naive` flag — measure its share and compare duration distributions for
     `naive=True/False` (confirm/refute "approximate end time" empirically).
   - **Deduplication** — decide how a "unique daily alert" is counted
     (region-alert count vs. merged national episodes); consequences documented.
   - **Regime / break detection** — region count over time, level shifts, zero
     days (real calm vs. missing collection) — found *before* forecasting.
2. **EDA** — trend, by-region, hour-of-day, day-of-week seasonality.
3. **Decomposition** — STL / seasonal_decompose (trend / seasonal / residual).
4. **Forecasting** — **two** baselines (seasonal-naive *and* naive-with-drift)
   vs. a trend-capable model (Holt-Winters / SARIMA). A model that beats only
   seasonal-naive but not drift is capturing trend, not seasonality — reported as
   such. **Walk-forward** validation reports **per-fold** MAE/RMSE (not a single
   aggregate) to expose whether error grows with the series' acceleration. Train
   window starts **2025-07-01** (regime change); seasonality is estimated on the
   full history.
5. **Insights & README** — honest findings.

## License / attribution

Code: see repository license. Data: © contributors of the
[ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset),
MIT License.

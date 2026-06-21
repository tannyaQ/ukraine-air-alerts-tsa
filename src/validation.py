"""EDA step 0 - empirical validation of data assumptions.

Prints RAW numbers only (no interpretation): date coverage and gaps, zero-alert
days, the ``naive`` flag distribution, and region-count stability over time.
These checks are meant to be reviewed BEFORE any modeling (see README pipeline).

Run:  python -m src.validation
"""
from __future__ import annotations

import pandas as pd

from src.data_loader import load_raw_alerts
from src.features import (
    daily_region_alert_count,
    daily_regions_under_alert,
    regions_per_month,
)


def longest_zero_run(counts: pd.Series) -> tuple[int, object, object]:
    """Longest run of consecutive zero-alert days: (length, start_day, end_day)."""
    best_len, best_start, best_end = 0, None, None
    cur_len, cur_start = 0, None
    for day, c in counts.items():
        if c == 0:
            cur_start = day if cur_len == 0 else cur_start
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start, best_end = cur_len, cur_start, day
        else:
            cur_len = 0
    return best_len, best_start, best_end


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    df = load_raw_alerts("volunteer")
    pd.set_option("display.width", 100)
    pd.set_option("display.max_rows", 60)

    # --- 1. Coverage & gaps -------------------------------------------------
    _hr("1. DATE COVERAGE & GAPS")
    counts = daily_region_alert_count(df)
    total_days = len(counts)
    days_with_alerts = int((counts > 0).sum())
    zero_days = int((counts == 0).sum())
    n_nat = int(df["started_at"].isna().sum())
    print(f"rows total: {len(df):,}   (started_at = NaT: {n_nat})")
    print(f"date range (Kyiv): {counts.index.min().date()} -> {counts.index.max().date()}")
    print(f"calendar days in range: {total_days:,}")
    print(f"days with >=1 alert:    {days_with_alerts:,}")
    print(f"zero-alert days:        {zero_days:,}")
    run_len, run_start, run_end = longest_zero_run(counts)
    if run_len:
        print(f"longest zero-alert run: {run_len} days "
              f"({run_start.date()} -> {run_end.date()})")
    print("\ndaily region-alert count (Variant A) - describe:")
    print(counts.describe())

    # --- 2. Zero-alert days detail -----------------------------------------
    _hr("2. ZERO-ALERT DAYS (raw list / monthly clustering)")
    zero_dates = counts.index[counts == 0]
    if zero_days:
        zmonthly = pd.Series(1, index=zero_dates).groupby(
            zero_dates.to_period("M")).sum()
        print("zero-alert days per month:")
        print(zmonthly.to_string())
        print(f"\nall zero-alert days ({zero_days}):")
        print([d.date().isoformat() for d in zero_dates])
    else:
        print("none")

    # --- 3. naive flag (closes assumption #1) ------------------------------
    _hr("3. NAIVE FLAG & DURATION DISTRIBUTION")
    print(f"naive=True share: {df['naive'].mean():.2%}  "
          f"({int(df['naive'].sum()):,} of {len(df):,} rows; "
          f"NaN: {int(df['naive'].isna().sum())})")
    print(f"missing finished_at: {int(df['finished_at'].isna().sum()):,}")
    nonpos = df["duration_min"] <= 0
    print(f"non-positive duration (end <= start): {int(nonpos.sum()):,}")
    print("\nduration_min by naive flag (minutes):")
    print(df.groupby("naive")["duration_min"].describe(
        percentiles=[.25, .5, .75, .9, .99]))

    # --- 4. Region stability (first signal for assumption #3) --------------
    _hr("4. REGION COUNT & STABILITY OVER TIME")
    print(f"unique regions total: {df['region'].nunique()}")
    print("\nrecords per region:")
    print(df["region"].value_counts().to_string())
    print("\ndistinct regions per month:")
    rpm = regions_per_month(df)
    print(rpm.to_string())
    print(f"\nregions/month  min={rpm.min()}  max={rpm.max()}")


if __name__ == "__main__":
    main()

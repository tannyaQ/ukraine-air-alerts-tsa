"""Derived series and documented cleaning rules.

Cleaning rules below were decided EMPIRICALLY in the EDA validation step
(src/validation.py), not assumed:

* ``naive=True`` rows carry a FIXED 30-min placeholder for an unknown end time
  (verified: 4 998 rows, all exactly 30.0 min, std = 0). The start time is real,
  so these rows are KEPT for the count target but EXCLUDED from any duration
  analysis.
* Rows with ``finished_at <= started_at`` are broken interval records (5 rows);
  EXCLUDED from duration analysis only.

Neither rule removes data from the count target.
"""
from __future__ import annotations

import pandas as pd


def duration_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of rows valid for DURATION analysis.

    Drops the naive 30-min placeholders and broken intervals (end <= start).
    Does not affect the count target, which uses the full frame.
    """
    mask = (~df["naive"].astype(bool)) & (df["duration_min"] > 0)
    return df.loc[mask].copy()


def _day_index(df: pd.DataFrame) -> pd.Series:
    """Kyiv calendar day (naive midnight Timestamp) for each alert start."""
    s = df.dropna(subset=["started_at"])["started_at"]
    return s.dt.tz_localize(None).dt.normalize()


def daily_region_alert_count(df: pd.DataFrame) -> pd.Series:
    """Variant A (raw): number of alert records started on each calendar day.

    Reindexed to the full continuous date range so zero-alert days are explicit.
    """
    counts = _day_index(df).value_counts().sort_index()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    return counts.reindex(full, fill_value=0)


def daily_regions_under_alert(df: pd.DataFrame) -> pd.Series:
    """Breadth view: number of DISTINCT oblasts with >=1 alert per calendar day."""
    d = df.dropna(subset=["started_at"]).copy()
    d["day"] = d["started_at"].dt.tz_localize(None).dt.normalize()
    g = d.groupby("day")["region"].nunique().sort_index()
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    return g.reindex(full, fill_value=0)


def regions_per_month(df: pd.DataFrame) -> pd.Series:
    """Distinct region count per month - first signal of granularity/regime shifts."""
    d = df.dropna(subset=["started_at"]).copy()
    month = d["started_at"].dt.tz_localize(None).dt.to_period("M")
    return d.groupby(month)["region"].nunique()


def monthly_stats(daily: pd.Series) -> pd.DataFrame:
    """Monthly mean/median/count of a daily series + month-over-month change in mean.

    Diagnostic aid to locate where the level accelerates when choosing a train
    window (does NOT hardcode any window).
    """
    g = daily.groupby(daily.index.to_period("M"))
    out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "days": g.size()})
    out["mom_delta_mean"] = out["mean"].diff()
    return out

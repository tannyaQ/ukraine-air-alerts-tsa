"""Load and validate raw air-raid alert data (interval level).

Design note
-----------
This module intentionally stops at *clean interval-level data*: one row per
(region, alert) with tz-aware start/end timestamps. It does **not** decide how to
aggregate alerts into a daily count, because that requires a deduplication policy
(region-alert count vs. merged national episodes) which we finalize only after the
EDA "data assumptions validation" step. Aggregation lives in a separate module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

KYIV_TZ = "Europe/Kyiv"
EXPECTED_COLUMNS = ["region", "started_at", "finished_at", "naive"]


def _coerce_bool(s: pd.Series) -> pd.Series:
    """Coerce a True/False column to real booleans regardless of source dtype."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def load_raw_alerts(source: str = "volunteer", path: Path | None = None) -> pd.DataFrame:
    """Read a raw alerts CSV and return tidy, tz-aware interval data.

    Parameters
    ----------
    source : {"volunteer", "official"}
        Which committed snapshot to load (used to build the default path).
    path : Path, optional
        Explicit CSV path; overrides ``source``.

    Returns
    -------
    pandas.DataFrame with columns:
        region, started_at, finished_at (tz-aware Europe/Kyiv), naive (bool),
        duration_min (float; NaN if finished_at is missing).
    """
    if path is None:
        path = RAW_DIR / f"{source}_data_en.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema mismatch in {path.name}: missing {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Timestamps carry a +00:00 offset -> parse as UTC, then present in Kyiv time.
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(KYIV_TZ)

    df["naive"] = _coerce_bool(df["naive"])
    df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0

    return df.sort_values("started_at").reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> None:
    """Print an empirical data-quality summary. Makes no interpretive assumptions."""
    n = len(df)
    print(f"rows: {n:,}")
    print(f"date range (Kyiv): {df['started_at'].min()}  ->  {df['started_at'].max()}")
    print(f"unique regions: {df['region'].nunique()}")
    print(f"  sample: {sorted(df['region'].dropna().unique())[:8]}")

    print(f"naive=True share: {df['naive'].mean():.1%}  ({int(df['naive'].sum()):,} rows)")
    print(f"missing finished_at: {int(df['finished_at'].isna().sum()):,}")
    print(f"non-positive duration (end <= start): {int((df['duration_min'] <= 0).sum()):,}")

    print("\nduration_min by naive flag:")
    print(df.groupby("naive")["duration_min"].describe()[["count", "mean", "50%", "max"]])


if __name__ == "__main__":
    data = load_raw_alerts("volunteer")
    quality_report(data)

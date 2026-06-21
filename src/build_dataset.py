"""Build the modeling-ready daily series and write it to data/processed/.

Target (confirmed empirically in EDA): national **daily count of region-alerts**
(Variant A), Kyiv local time, full continuous history (no gaps, no zero days).
A secondary breadth series (distinct oblasts under alert per day) is included
for context. This is the aggregation step that was deliberately kept out of the
loader until the target was decided.

Run:  python -m src.build_dataset
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_raw_alerts
from src.features import daily_region_alert_count, daily_regions_under_alert

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def build_daily(source: str = "volunteer", drop_last_day: bool = True) -> pd.DataFrame:
    """Return the daily modeling frame: alert_count (target A) + regions_under_alert.

    drop_last_day: the final calendar day of any snapshot is the collection
    boundary and may be partial (in this snapshot 2026-06-20 had 9 alerts vs ~100
    typical). Dropped by default as data hygiene - not as result-tuning.
    """
    df = load_raw_alerts(source)
    a = daily_region_alert_count(df).rename("alert_count")
    b = daily_regions_under_alert(df).rename("regions_under_alert")
    out = pd.concat([a, b], axis=1)
    out.index.name = "date"
    if drop_last_day:
        out = out.iloc[:-1]
    return out


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = build_daily("volunteer")
    path = PROCESSED / "daily_alerts.csv"
    out.to_csv(path, date_format="%Y-%m-%d")
    print(f"saved {path.relative_to(ROOT)}   shape={out.shape}")
    print(f"date range: {out.index.min().date()} -> {out.index.max().date()}")
    print("\nhead:")
    print(out.head().to_string())
    print("\ntail:")
    print(out.tail().to_string())


if __name__ == "__main__":
    main()

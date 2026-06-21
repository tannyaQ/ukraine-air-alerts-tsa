"""Generate EDA figures into reports/figures/.

Raw, descriptive figures for the joint review of target & training window:
  01  daily alert count (Variant A) over time, with 30-day rolling mean + median
  02  distinct oblasts under alert per day
  03  alert duration by ``naive`` flag (boxplot, log scale)

Run:  python -m src.plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_loader import load_raw_alerts  # noqa: E402
from src.features import (  # noqa: E402
    daily_region_alert_count,
    daily_regions_under_alert,
    duration_valid,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"


def fig_daily_count(df):
    s = daily_region_alert_count(df)
    roll = s.rolling(30, center=True).mean()
    median = s.median()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(s.index, s.values, lw=0.5, color="#9aa7b8", label="daily count")
    ax.plot(roll.index, roll.values, lw=2.0, color="#1f4e79",
            label="30-day rolling mean")
    ax.axhline(median, color="#c0392b", ls="--", lw=1.0, label=f"median = {median:.0f}")
    ax.set_title("Daily air-raid alert count (Variant A: region-alert count) - Kyiv time")
    ax.set_xlabel("date")
    ax.set_ylabel("alerts / day")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "01_daily_alert_count.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, s


def fig_regions_per_day(df):
    s = daily_regions_under_alert(df)
    roll = s.rolling(30, center=True).mean()
    median = s.median()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(s.index, s.values, lw=0.5, color="#b0bda0", label="regions / day")
    ax.plot(roll.index, roll.values, lw=2.0, color="#2e7d32",
            label="30-day rolling mean")
    ax.axhline(median, color="#c0392b", ls="--", lw=1.0, label=f"median = {median:.0f}")
    ax.set_ylim(0, 26)
    ax.set_title("Distinct oblasts under alert per day")
    ax.set_xlabel("date")
    ax.set_ylabel("regions / day")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "02_regions_under_alert_per_day.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, s


def fig_duration_boxplot(df):
    pos = df[df["duration_min"] > 0]
    groups = [
        pos.loc[~pos["naive"].astype(bool), "duration_min"].values,
        pos.loc[pos["naive"].astype(bool), "duration_min"].values,
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, showfliers=True, flierprops={"marker": ".", "alpha": 0.3})
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"naive=False (n={len(groups[0]):,})",
                        f"naive=True (n={len(groups[1]):,})"])
    ax.set_yscale("log")
    ax.set_ylabel("duration (min, log scale)")
    ax.set_title("Alert duration by `naive` flag  (duration > 0)")
    fig.tight_layout()
    out = FIG_DIR / "03_duration_naive_boxplot.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_recent_zoom(df, since="2025-01-01", win_ref_days=365):
    """Zoom on the recent regime to locate where the level accelerates.

    Marks the '-12 months' reference line; the precise train boundary is chosen
    jointly from this figure + monthly_stats, not hardcoded.
    """
    s = daily_region_alert_count(df)
    s = s[s.index >= pd.Timestamp(since)]
    roll7 = s.rolling(7, center=True).mean()
    roll30 = s.rolling(30, center=True).mean()
    ref = s.index.max() - pd.Timedelta(days=win_ref_days)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(s.index, s.values, lw=0.6, color="#9aa7b8", label="daily count")
    ax.plot(roll7.index, roll7.values, lw=1.2, color="#b07d2b", label="7-day mean")
    ax.plot(roll30.index, roll30.values, lw=2.2, color="#1f4e79", label="30-day mean")
    ax.axvline(ref, color="#7f8c8d", ls=":", lw=1.3, label=f"-12 months ({ref.date()})")
    ax.set_title("Daily alert count A - recent window (zoom)")
    ax.set_xlabel("date")
    ax.set_ylabel("alerts / day")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "04_recent_zoom.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, s


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_alerts("volunteer")
    p1, s1 = fig_daily_count(df)
    p2, s2 = fig_regions_per_day(df)
    p3 = fig_duration_boxplot(df)

    print("saved figures:")
    for p in (p1, p2, p3):
        print(f"  {p.relative_to(ROOT)}")

    roll = s1.rolling(30, center=True).mean()
    dv = duration_valid(df)
    print("\nRAW reference numbers (no interpretation):")
    print(f"  daily count A:  median={s1.median():.0f}  mean={s1.mean():.1f}  "
          f"min={int(s1.min())}  max={int(s1.max())}")
    print(f"  30-day rolling mean of A:  min={roll.min():.1f} @ {roll.idxmin().date()}  "
          f"->  max={roll.max():.1f} @ {roll.idxmax().date()}")
    print(f"  regions/day:    median={s2.median():.0f}  min={int(s2.min())}  "
          f"max={int(s2.max())}")
    print(f"  duration_valid rows: {len(dv):,} of {len(df):,} "
          f"(excluded naive+broken: {len(df) - len(dv):,})")


if __name__ == "__main__":
    main()

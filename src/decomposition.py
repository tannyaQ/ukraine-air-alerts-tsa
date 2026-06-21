"""STL decomposition of the daily alert-count series (Variant A, full history).

Uses a weekly period (7). Prints raw component numbers, the day-of-week seasonal
effect, and Hyndman's strength-of-trend / strength-of-seasonality measures - the
latter directly inform what a forecaster can realistically add over a baseline.

Run:  python -m src.decomposition
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402

from src.build_dataset import build_daily  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"

_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def run_stl(series, period: int = 7, robust: bool = True):
    """Fit STL with a weekly period on a continuous daily series."""
    return STL(series.asfreq("D"), period=period, robust=robust).fit()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = build_daily("volunteer")["alert_count"].astype(float)
    res = run_stl(s, period=7)

    fig = res.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("STL decomposition - daily alert count A (weekly period, full history)")
    fig.tight_layout()
    out = FIG_DIR / "05_stl_decomposition.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    seasonal, trend, resid = res.seasonal, res.trend, res.resid
    dow = seasonal.groupby(seasonal.index.dayofweek).mean()
    dow.index = _DOW

    # Hyndman strength measures (0..1): how much trend / seasonality vs. residual.
    var_resid = resid.var()
    f_trend = max(0.0, 1.0 - var_resid / (trend + resid).var())
    f_seasonal = max(0.0, 1.0 - var_resid / (seasonal + resid).var())

    print(f"saved {out.relative_to(ROOT)}")
    print("\nRAW decomposition numbers:")
    print(f"  trend: start={trend.iloc[0]:.1f}  end={trend.iloc[-1]:.1f}")
    print(f"  weekly seasonal amplitude (max-min DoW effect): {dow.max() - dow.min():.1f}")
    print(f"  residual std: {resid.std():.2f}")
    print(f"  strength of trend     F_T = {f_trend:.3f}")
    print(f"  strength of seasonal  F_S = {f_seasonal:.3f}")
    print("\n  mean weekly seasonal effect by day-of-week (alerts vs. trend):")
    print(dow.round(2).to_string())

    # Stability check: does the Thu-high / Sun-low profile hold across years,
    # or is it an artifact of the recent (accelerated) window?
    by_year = (seasonal.groupby([seasonal.index.year, seasonal.index.dayofweek])
               .mean().unstack())
    by_year.columns = _DOW
    print("\n  weekly seasonal effect by YEAR x day-of-week (stability check):")
    print(by_year.round(2).to_string())


if __name__ == "__main__":
    main()

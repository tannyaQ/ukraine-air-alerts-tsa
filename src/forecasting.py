"""Walk-forward forecasting evaluation: baselines vs Holt-Winters.

Pre-registered honesty (decided BEFORE seeing metrics): the series is
trend-dominated with weak weekly seasonality (STL F_S=0.18), so Holt-Winters may
fail to beat naive-with-drift. That outcome is a FINDING, not a failure, and is
reported as is. No parameter tuning to 'make HW win'.

Design (confirmed):
  * target = daily alert_count (Variant A)
  * train window starts 2025-07-01 (regime change), expanding
  * 12 consecutive weekly folds, horizon h=7, over the last 84 days
  * metrics reported PER FOLD (method x fold), not just aggregated
  * HW add vs mul seasonality is chosen BY walk-forward MAE, not by eye

Run:  python -m src.forecasting
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402

from src.build_dataset import build_daily  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

TRAIN_START = pd.Timestamp("2025-07-01")
H = 7
N_FOLDS = 12


def mae(actual, fc) -> float:
    return float(np.mean(np.abs(actual - fc)))


def rmse(actual, fc) -> float:
    return float(np.sqrt(np.mean((actual - fc) ** 2)))


# --- methods -------------------------------------------------------------
def seasonal_naive(train: pd.Series, h: int) -> np.ndarray:
    """Forecast = the previous week's values (y_{t-7})."""
    return train.iloc[-h:].to_numpy(dtype=float)


def naive_drift(train: pd.Series, h: int) -> np.ndarray:
    """Random walk with drift: last value + average per-step change over train."""
    y = train.to_numpy(dtype=float)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def hw_forecast(train: pd.Series, h: int, seasonal: str) -> np.ndarray:
    """Holt-Winters: additive trend + (add|mul) weekly seasonality."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                train, trend="add", seasonal=seasonal, seasonal_periods=H,
                initialization_method="estimated",
            ).fit()
        return np.asarray(fit.forecast(h), dtype=float)
    except Exception:
        return np.full(h, np.nan)


def holt_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Holt linear: additive trend, NO seasonality (diagnostic for seasonality value)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                train, trend="add", seasonal=None,
                initialization_method="estimated",
            ).fit()
        return np.asarray(fit.forecast(h), dtype=float)
    except Exception:
        return np.full(h, np.nan)


METHODS = {
    "seasonal_naive": lambda tr: seasonal_naive(tr, H),
    "naive_drift": lambda tr: naive_drift(tr, H),
    "holt_no_seas": lambda tr: holt_forecast(tr, H),
    "hw_add": lambda tr: hw_forecast(tr, H, "add"),
    "hw_mul": lambda tr: hw_forecast(tr, H, "mul"),
}


def _blocks(last: pd.Timestamp):
    """12 consecutive weekly (start, end) blocks ending at `last`, chronological."""
    out = []
    for k in range(N_FOLDS):
        bend = last - pd.Timedelta(days=H * k)
        bstart = bend - pd.Timedelta(days=H - 1)
        out.append((bstart, bend))
    return out[::-1]


def walk_forward(s: pd.Series):
    s = s.asfreq("D")
    rows_mae, rows_rmse = [], []
    for bstart, bend in _blocks(s.index.max()):
        train = s.loc[TRAIN_START:bstart - pd.Timedelta(days=1)]
        actual = s.loc[bstart:bend].to_numpy(dtype=float)
        label = f"{bstart.date()}..{bend.date()}"
        rec_mae = {"fold": label, "n_train": len(train)}
        rec_rmse = {"fold": label, "n_train": len(train)}
        for name, fn in METHODS.items():
            fc = fn(train)
            rec_mae[name] = round(mae(actual, fc), 2)
            rec_rmse[name] = round(rmse(actual, fc), 2)
        rows_mae.append(rec_mae)
        rows_rmse.append(rec_rmse)
    return pd.DataFrame(rows_mae), pd.DataFrame(rows_rmse)


def _plot_per_fold(mae_df: pd.DataFrame, methods_to_plot) -> Path:
    style = {
        "seasonal_naive": ("o--", "#7f8c8d"),
        "naive_drift": ("s--", "#c0392b"),
        "holt_no_seas": ("D-.", "#7d3c98"),
        "hw_add": ("^-", "#1f4e79"),
        "hw_mul": ("v-", "#148f77"),
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(mae_df)))
    for m in methods_to_plot:
        st, col = style.get(m, ("o-", None))
        ax.plot(x, mae_df[m], st, color=col, label=m)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f.split("..")[1] for f in mae_df["fold"]], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("MAE (alerts/day)")
    ax.set_title("Walk-forward MAE per fold (h=7) - does error grow with acceleration?")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "06_walkforward_mae.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = build_daily("volunteer")["alert_count"].astype(float)
    mae_df, rmse_df = walk_forward(s)
    methods = list(METHODS)

    print("MAE per fold (method x fold):")
    print(mae_df.to_string(index=False))
    print("\nRMSE per fold (method x fold):")
    print(rmse_df.to_string(index=False))

    summ = pd.DataFrame({
        "MAE_mean": mae_df[methods].mean(),
        "RMSE_mean": rmse_df[methods].mean(),
    }).round(2)
    print("\nSummary (mean across folds):")
    print(summ.to_string())

    hw_pick = "hw_mul" if summ.loc["hw_mul", "MAE_mean"] < summ.loc["hw_add", "MAE_mean"] else "hw_add"
    print(f"\nHW seasonality chosen by walk-forward MAE: {hw_pick}  "
          f"(hw_add={summ.loc['hw_add', 'MAE_mean']}, hw_mul={summ.loc['hw_mul', 'MAE_mean']})")

    best_base = summ.loc[["seasonal_naive", "naive_drift"], "MAE_mean"].idxmin()
    print(f"best baseline: {best_base} (MAE={summ.loc[best_base, 'MAE_mean']})")
    delta = summ.loc[hw_pick, "MAE_mean"] - summ.loc[best_base, "MAE_mean"]
    verdict = "BEATS" if delta < 0 else "does NOT beat"
    print(f"VERDICT: {hw_pick} {verdict} best baseline "
          f"({summ.loc[hw_pick, 'MAE_mean']} vs {summ.loc[best_base, 'MAE_mean']}, "
          f"delta={delta:+.2f} MAE)")

    diag = summ.loc[hw_pick, "MAE_mean"] - summ.loc["holt_no_seas", "MAE_mean"]
    print(f"\nDIAGNOSTIC (value of seasonality): {hw_pick} vs holt_no_seas "
          f"{summ.loc[hw_pick, 'MAE_mean']} vs {summ.loc['holt_no_seas', 'MAE_mean']} "
          f"(delta={diag:+.2f} MAE)")
    if abs(diag) < 1.0:
        print("  -> weekly seasonality adds ~nothing; HW's edge is trend/level smoothing.")

    mae_df.to_csv(REPORTS / "walkforward_mae.csv", index=False)
    out = _plot_per_fold(mae_df, ["seasonal_naive", "naive_drift", "holt_no_seas", hw_pick])
    print(f"\nsaved {out.relative_to(ROOT)} and reports/walkforward_mae.csv")


if __name__ == "__main__":
    main()

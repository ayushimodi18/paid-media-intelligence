"""
Feature engineering layer.

Pulled out of the forecasting script into its own module so it's testable
in isolation and reusable by both the forecast engine and the EDA report.
Every feature is documented with WHY it's included -- a production
reviewer should never have to guess what a column means or whether it
leaks future information into the training set.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

HOLIDAY_WINDOWS = [
    ("BFCM", lambda d: (d.month == 11 and d.day >= 24) or (d.month == 12 and d.day <= 2)),
    ("Christmas", lambda d: d.month == 12 and 3 <= d.day <= 23),
    ("New_Year_Lull", lambda d: d.month == 1 and d.day <= 7),
    ("Summer_Sale", lambda d: d.month == 7 and d.day <= 7),
    ("Back_to_School", lambda d: d.month == 8 and 15 <= d.day <= 31),
]


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Calendar features. dow/month as both raw ints (for tree models, which
    split on thresholds fine) AND cyclical sin/cos encodings (for linear/
    distance-based models, which would otherwise treat Sunday=6 and
    Monday=0 as maximally far apart)."""
    df = df.copy()
    dow = df[date_col].dt.dayofweek
    month = df[date_col].dt.month
    df["dow"] = dow
    df["month"] = month
    df["is_weekend"] = (dow >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    for name, fn in HOLIDAY_WINDOWS:
        df[f"is_{name}"] = df[date_col].apply(fn).astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, group_cols: list, value_col: str,
                                  lags=(1, 7, 14), windows=(3, 7, 14)) -> pd.DataFrame:
    """Lag and rolling-window features computed WITHIN each group (e.g. per
    channel) and sorted by date first -- critical to avoid leaking future
    values backward. Rolling windows use min_periods=1 so early rows don't
    become all-NaN, but that means early-window rolling stats are noisier
    (computed on fewer days) -- documented here so it's not a silent
    surprise during model debugging."""
    df = df.sort_values(group_cols + ["date"]).copy()
    grp = df.groupby(group_cols)[value_col]
    for lag in lags:
        df[f"{value_col}_lag{lag}"] = grp.shift(lag)
    for window in windows:
        df[f"{value_col}_roll{window}_mean"] = grp.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"{value_col}_roll{window}_std"] = grp.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).std()
        )
    # fill remaining NaNs (first `lag` rows of each group) with the group's
    # overall mean -- flagged via a companion indicator column so downstream
    # models/analysts can tell imputed-cold-start rows from real ones
    lag_cols = [c for c in df.columns if value_col in c and ("_lag" in c or "_roll" in c)]
    for c in lag_cols:
        df[f"{c}_was_imputed"] = df[c].isna().astype(int)
        df[c] = df.groupby(group_cols)[c].transform(lambda s: s.fillna(s.mean()))
    return df


def add_log_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """log1p transform for skewed monetary columns -- spend/revenue
    distributions are right-skewed (a few huge BFCM days), and several
    candidate models (linear/Ridge) assume roughly symmetric residuals."""
    df = df.copy()
    for c in cols:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    return df


def cap_outliers(df: pd.DataFrame, col: str, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Winsorize extreme values (e.g. data-entry-style spikes) at the 1st/
    99th percentile rather than dropping rows -- preserves row count and
    avoids silently deleting real (if extreme) BFCM spend days, which is
    exactly the kind of event a forecasting model needs to learn from."""
    df = df.copy()
    lo, hi = df[col].quantile([lower_q, upper_q])
    n_capped = ((df[col] < lo) | (df[col] > hi)).sum()
    df[f"{col}_capped"] = df[col].clip(lower=lo, upper=hi)
    df[f"{col}_was_outlier"] = ((df[col] < lo) | (df[col] > hi)).astype(int)
    return df, n_capped


def build_feature_table(unified_path: Path = None) -> pd.DataFrame:
    """End-to-end feature table at (channel, date) daily granularity --
    the grain the forecasting models train on."""
    unified_path = unified_path or (DATA / "unified_performance.csv")
    df = pd.read_csv(unified_path, parse_dates=["date"])

    daily = df.groupby(["channel", "date"]).agg(
        spend=("spend", "sum"), revenue=("revenue_attributed", "sum"),
        clicks=("clicks", "sum"), impressions=("impressions", "sum"),
    ).reset_index()

    daily, n_capped_spend = cap_outliers(daily, "spend")
    daily, n_capped_rev = cap_outliers(daily, "revenue")

    daily = add_calendar_features(daily)
    daily = add_lag_and_rolling_features(daily, group_cols=["channel"], value_col="spend")
    daily = add_lag_and_rolling_features(daily, group_cols=["channel"], value_col="revenue")
    daily = add_log_transform(daily, ["spend", "revenue"])

    print(f"Outliers capped (1st/99th pct): spend={n_capped_spend}, revenue={n_capped_rev}")
    print(f"Feature table shape: {daily.shape}")
    return daily


if __name__ == "__main__":
    table = build_feature_table()
    print(table.columns.tolist())
    table.to_csv(DATA / "feature_table.csv", index=False)
    print(f"Saved to {DATA / 'feature_table.csv'}")

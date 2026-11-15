"""
M2 - AI Performance Forecasting Engine.

Goal: forecast daily revenue per channel from spend + lag/trend features,
compare multiple candidate models on a proper time-based holdout, and select
the best model PER CHANNEL by backtest MAPE (channels have different
spend-response shapes, so the winner can differ by channel).

Candidates evaluated:
  1. Prophet           - additive trend+seasonality, spend as a regressor
  2. XGBoost            - gradient boosted trees on engineered features
  3. LightGBM           - gradient boosted trees (faster, leaf-wise)
  4. Ridge (log-log)    - linear baseline on log(spend) -> log(revenue),
                          i.e. a classic diminishing-returns power-law model

We report MAPE, RMSE, and a naive-baseline comparison so "trust the output"
is backed by numbers, not vibes.
"""
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet

# Prophet/LightGBM emit verbose convergence and deprecation chatter during
# repeated CV refits; silence it so pipeline logs stay readable.
warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
HOLDOUT_DAYS = 10


def load_daily_channel_series():
    df = pd.read_csv(DATA / "unified_performance.csv", parse_dates=["date"])
    daily = df.groupby(["channel", "date"]).agg(
        spend=("spend", "sum"),
        revenue=("revenue_attributed", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    ).reset_index()
    return daily


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("date").copy()
    d["dow"] = d["date"].dt.dayofweek
    d["spend_lag1"] = d["spend"].shift(1).fillna(d["spend"].mean())
    d["spend_roll3"] = d["spend"].rolling(3, min_periods=1).mean()
    d["log_spend"] = np.log1p(d["spend"])
    d["t"] = np.arange(len(d))
    return d


def split(d):
    train = d.iloc[: -HOLDOUT_DAYS]
    test = d.iloc[-HOLDOUT_DAYS:]
    return train, test


def eval_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    mape = mean_absolute_percentage_error(np.maximum(y_true, 1e-3), np.maximum(y_pred, 1e-3))
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return round(mape * 100, 2), round(rmse, 1)


def run_prophet(train, test):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    m.add_regressor("spend")
    pdf = train.rename(columns={"date": "ds", "revenue": "y"})[["ds", "y", "spend"]]
    m.fit(pdf)
    future = test.rename(columns={"date": "ds"})[["ds", "spend"]]
    # Prophet's uncertainty interval is simulated with numpy's global RNG;
    # seed it so confidence bands are reproducible run-to-run (the other
    # models already pin random_state=42).
    np.random.seed(42)
    fcst = m.predict(future)
    return fcst["yhat"].values, fcst[["yhat_lower", "yhat_upper"]].values


def run_xgb(train, test, feats):
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                              subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(train[feats], train["revenue"])
    pred = model.predict(test[feats])
    return pred, model


def run_lgb(train, test, feats):
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=-1)
    model.fit(train[feats], train["revenue"])
    pred = model.predict(test[feats])
    return pred, model


def run_ridge_loglog(train, test):
    Xtr = train[["log_spend"]].values
    Xte = test[["log_spend"]].values
    ytr = np.log1p(train["revenue"].values)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    pred = np.expm1(model.predict(Xte))
    return pred, model


FEATS = ["spend", "spend_lag1", "spend_roll3", "dow", "t"]


def rolling_window_cv(d: pd.DataFrame, n_folds=4, fold_size=HOLDOUT_DAYS, min_train_days=180):
    """Proper time-series cross-validation: n_folds consecutive, NON-
    overlapping holdout windows walking forward through the series, each
    with all prior data as training. A single holdout (as in the original
    pilot) can give a lucky or unlucky read depending on which days landed
    in the test window -- rolling-window CV averages over several windows,
    which is what a production model-selection process should actually do
    before committing to a "winning" model.
    Returns a list of (train, test) tuples, oldest-to-newest.
    """
    folds = []
    n = len(d)
    last_test_end = n
    for i in range(n_folds):
        test_end = last_test_end - i * fold_size
        test_start = test_end - fold_size
        if test_start - min_train_days < 0:
            break
        train = d.iloc[:test_start]
        test = d.iloc[test_start:test_end]
        folds.append((train, test))
    return list(reversed(folds))  # oldest fold first, for readable output


def compare_models_cv(n_folds=4):
    """Cross-validated model comparison -- the primary, more rigorous
    selection method. Falls back gracefully to a single holdout (see
    compare_models() below) for quick iteration during development."""
    daily = load_daily_channel_series()
    fold_results = []
    best_models = {}
    forecasts_for_app = {}

    for channel, d in daily.groupby("channel"):
        d = make_features(d)
        folds = rolling_window_cv(d, n_folds=n_folds)
        if not folds:
            continue

        per_model_scores = {"Prophet": [], "XGBoost": [], "LightGBM": [], "Ridge(log-log)": [], "Naive(baseline)": []}

        for fold_idx, (train, test) in enumerate(folds):
            prophet_pred, prophet_ci = run_prophet(train, test)
            xgb_pred, _ = run_xgb(train, test, FEATS)
            lgb_pred, _ = run_lgb(train, test, FEATS)
            ridge_pred, _ = run_ridge_loglog(train, test)
            naive_pred = np.repeat(train["revenue"].values[-1], len(test))

            candidates = {"Prophet": prophet_pred, "XGBoost": xgb_pred, "LightGBM": lgb_pred,
                          "Ridge(log-log)": ridge_pred, "Naive(baseline)": naive_pred}
            for name, pred in candidates.items():
                mape, rmse = eval_metrics(test["revenue"].values, pred)
                per_model_scores[name].append(mape)
                fold_results.append({"channel": channel, "model": name, "fold": fold_idx,
                                      "MAPE_%": mape, "RMSE_$": rmse})

            if fold_idx == len(folds) - 1:  # keep the most recent fold's predictions for the app chart
                last_fold_candidates = candidates
                last_fold_test = test
                last_fold_ci = prophet_ci

        avg_scores = {m: np.mean(scores) for m, scores in per_model_scores.items() if m != "Naive(baseline)"}
        best_name = min(avg_scores, key=avg_scores.get)
        best_models[channel] = best_name

        forecasts_for_app[channel] = {
            "dates": last_fold_test["date"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": last_fold_test["revenue"].round(1).tolist(),
            "best_model": best_name,
            "best_pred": np.maximum(last_fold_candidates[best_name], 0).round(1).tolist(),
            "ci_lower": (last_fold_ci[:, 0].round(1).tolist() if best_name == "Prophet" else None),
            "ci_upper": (last_fold_ci[:, 1].round(1).tolist() if best_name == "Prophet" else None),
        }

    fold_df = pd.DataFrame(fold_results)
    cv_summary = fold_df.groupby(["channel", "model"])["MAPE_%"].agg(["mean", "std", "count"]).reset_index()
    cv_summary.columns = ["channel", "model", "MAPE_mean_%", "MAPE_std_%", "n_folds"]
    cv_summary = cv_summary.sort_values(["channel", "MAPE_mean_%"])

    fold_df.to_csv(DATA / "model_comparison_cv_folds.csv", index=False)
    cv_summary.to_csv(DATA / "model_comparison_results.csv", index=False)  # overwrite single-holdout version

    import json
    with open(DATA / "best_forecasts.json", "w") as f:
        json.dump(forecasts_for_app, f, indent=2)

    return cv_summary, best_models


def compare_models():
    daily = load_daily_channel_series()
    results = []
    best_models = {}
    forecasts_for_app = {}

    for channel, d in daily.groupby("channel"):
        d = make_features(d)
        train, test = split(d)

        prophet_pred, prophet_ci = run_prophet(train, test)
        xgb_pred, xgb_model = run_xgb(train, test, FEATS)
        lgb_pred, lgb_model = run_lgb(train, test, FEATS)
        ridge_pred, ridge_model = run_ridge_loglog(train, test)

        # naive baseline: yesterday's revenue carried forward
        naive_pred = np.repeat(train["revenue"].values[-1], len(test))

        candidates = {
            "Prophet": prophet_pred,
            "XGBoost": xgb_pred,
            "LightGBM": lgb_pred,
            "Ridge(log-log)": ridge_pred,
            "Naive(baseline)": naive_pred,
        }

        for name, pred in candidates.items():
            mape, rmse = eval_metrics(test["revenue"].values, pred)
            results.append({"channel": channel, "model": name, "MAPE_%": mape, "RMSE_$": rmse})

        # pick best real (non-naive) model by MAPE
        real_candidates = {k: v for k, v in candidates.items() if k != "Naive(baseline)"}
        scored = {k: eval_metrics(test["revenue"].values, v)[0] for k, v in real_candidates.items()}
        best_name = min(scored, key=scored.get)
        best_models[channel] = best_name

        forecasts_for_app[channel] = {
            "dates": test["date"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": test["revenue"].round(1).tolist(),
            "best_model": best_name,
            "best_pred": np.maximum(real_candidates[best_name], 0).round(1).tolist(),
            "ci_lower": (prophet_ci[:, 0].round(1).tolist() if best_name == "Prophet" else None),
            "ci_upper": (prophet_ci[:, 1].round(1).tolist() if best_name == "Prophet" else None),
        }

    results_df = pd.DataFrame(results).sort_values(["channel", "MAPE_%"])
    results_df.to_csv(DATA / "model_comparison_results.csv", index=False)

    import json
    with open(DATA / "best_forecasts.json", "w") as f:
        json.dump(forecasts_for_app, f, indent=2)

    return results_df, best_models


if __name__ == "__main__":
    print("Running rolling-window cross-validated model comparison (4 folds)...")
    cv_summary, best_models = compare_models_cv(n_folds=4)
    print(cv_summary.to_string(index=False))
    print("\nSelected model per channel (lowest mean CV MAPE across 4 folds):")
    for ch, m in best_models.items():
        row = cv_summary[(cv_summary.channel == ch) & (cv_summary.model == m)].iloc[0]
        print(f"  {ch}: {m}  (mean MAPE {row['MAPE_mean_%']:.2f}% +/- {row['MAPE_std_%']:.2f}%, n={int(row['n_folds'])} folds)")

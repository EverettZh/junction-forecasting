import pandas as pd
from prophet import Prophet

excel_path = "20251111_JUNCTION_training.xlsx"
forecast_hours = 48
forecast_months = 12

cons = pd.read_excel(excel_path, sheet_name="training_consumption")
prices = pd.read_excel(excel_path, sheet_name="training_prices")

# Parse timestamps but don't worry about tz here
cons["measured_at"] = pd.to_datetime(cons["measured_at"])
prices["measured_at"] = pd.to_datetime(prices["measured_at"])

cons = cons.set_index("measured_at").sort_index()
prices = prices.set_index("measured_at").sort_index()

# Make all timestamps tz-naive (no +00:00)
if isinstance(cons.index, pd.DatetimeIndex) and cons.index.tz is not None:
    cons.index = cons.index.tz_convert(None)

if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)

group_ids = cons.columns.tolist()

forecast_48h = {}
forecast_12m = {}

#for gid in group_ids:
# test, switch this back for all

for gid in group_ids[:3]:
    print(f"Training Prophet for group {gid}...")

    # Prepare df for Prophet
    df = pd.DataFrame({
        "ds": cons.index,
        "y": cons[gid].astype(float)
    })

    # 🔴 CRITICAL: remove any timezone from ds
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive"
    )
    m.fit(df)

    # 48-hour forecast
    future48 = m.make_future_dataframe(periods=forecast_hours, freq="h")
    fc48 = m.predict(future48)
    fc48 = fc48.tail(forecast_hours)[["ds", "yhat"]]
    fc48 = fc48.rename(columns={"ds": "timestamp", "yhat": gid})
    forecast_48h[gid] = fc48.set_index("timestamp")

    # Monthly model for 12-month forecast
    monthly = df.set_index("ds")["y"].resample("ME").sum().reset_index()
    monthly.columns = ["ds", "y"]
    monthly["ds"] = pd.to_datetime(monthly["ds"]).dt.tz_localize(None)

    m_month = Prophet(
        yearly_seasonality=True,
        seasonality_mode="additive"
    )
    m_month.fit(monthly)

    future12 = m_month.make_future_dataframe(periods=forecast_months, freq="ME")
    fc12 = m_month.predict(future12)
    fc12 = fc12.tail(forecast_months)[["ds", "yhat"]]
    fc12 = fc12.rename(columns={"ds": "month", "yhat": gid})
    forecast_12m[gid] = fc12.set_index("month")

    # test
    #print(forecast_48h.keys())
    #print(forecast_12m.keys())

print("Finished all groups!")

# --- 1. Combine 48-hour forecasts into one wide table ---
if not forecast_48h:
    raise RuntimeError("No 48h forecasts were generated. Did the loop run?")

# Concatenate per-group DataFrames side by side
df_48 = pd.concat(forecast_48h.values(), axis=1)

# Make sure index has a nice name
df_48.index.name = "measured_at"

# Turn index into ISO 8601 strings with .000Z
df_48_out = df_48.copy()
df_48_out.insert(
    0,
    "measured_at",
    df_48_out.index.to_series().dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
)

# Save with semicolon separator and comma decimal
df_48_out.to_csv("prophet_48h_forecast.csv", sep=";", decimal=",", index=False)
print("Saved 48h forecast to prophet_48h_forecast.csv")


# --- 2. Combine 12-month forecasts into one wide table ---
if not forecast_12m:
    raise RuntimeError("No 12m forecasts were generated. Did the loop run?")

df_12 = pd.concat(forecast_12m.values(), axis=1)

# Index is month end (because of 'ME'); convert to first-of-month for output
idx_series = df_12.index.to_series()
first_of_month = idx_series.dt.to_period("M").dt.to_timestamp()

df_12.index = first_of_month
df_12.index.name = "measured_at"

df_12_out = df_12.copy()
df_12_out.insert(
    0,
    "measured_at",
    df_12_out.index.to_series().dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
)

df_12_out.to_csv("prophet_12m_forecast.csv", sep=";", decimal=",", index=False)
print("Saved 12m forecast to prophet_12m_forecast.csv")

import numpy as np

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print("\n===========================")
print("BACKTEST: BASELINE vs PROPHET (48h)")
print("===========================\n")

# We'll test only first 3 groups again
test_groups = group_ids[:3]

# Use last 48 hours in the dataset as "test period"
last_ts = cons.index.max()
test_start = last_ts - pd.Timedelta(hours=47)   # inclusive
test_range = pd.date_range(test_start, last_ts, freq="h")

for gid in test_groups:
    print(f"--- Group {gid} ---")

    # 1) Build full hourly df for this group
    df_full = pd.DataFrame({
        "ds": cons.index,
        "y": cons[gid].astype(float)
    })
    df_full["ds"] = pd.to_datetime(df_full["ds"]).dt.tz_localize(None)

    # 2) Train Prophet only on data BEFORE the test period
    cutoff = test_range[0] - pd.Timedelta(hours=1)
    df_train = df_full[df_full["ds"] <= cutoff].copy()

    # Avoid too-short series
    if df_train.shape[0] < 100:
        print("  Not enough history to train Prophet. Skipping.\n")
        continue

    m_bt = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive"
    )
    m_bt.fit(df_train)

    # 3) Forecast next 48 hours after cutoff
    future_bt = m_bt.make_future_dataframe(periods=48, freq="h")
    fc_bt = m_bt.predict(future_bt)
    fc_bt = fc_bt[fc_bt["ds"].isin(test_range)][["ds", "yhat"]]

    # Align Prophet forecast with actuals
    fc_bt = fc_bt.set_index("ds").sort_index()
    prophet_fc = fc_bt["yhat"].values

    # 4) Build baseline: same hour last week
    actual_vals = []
    baseline_vals = []

    for ts in test_range:
        # Actual
        if ts in cons.index:
            actual_vals.append(cons.loc[ts, gid])
        else:
            actual_vals.append(np.nan)

        # Baseline: same hour last week
        ts_last_week = ts - pd.Timedelta(days=7)
        if ts_last_week in cons.index:
            baseline_vals.append(cons.loc[ts_last_week, gid])
        else:
            baseline_vals.append(np.nan)

    actual = np.array(actual_vals, dtype=float)
    baseline = np.array(baseline_vals, dtype=float)

    mask = ~np.isnan(actual) & ~np.isnan(baseline)

    if mask.sum() == 0:
        print("  No overlapping data to compute accuracy for this backtest.\n")
        continue

    mape_baseline = mape(actual[mask], baseline[mask])
    mape_prophet = mape(actual[mask], prophet_fc[mask])

    fva = 100 * (mape_baseline - mape_prophet) / mape_baseline

    print(f"  Baseline MAPE: {mape_baseline:.2f}%")
    print(f"  Prophet  MAPE: {mape_prophet:.2f}%")
    print(f"  FVA % improvement: {fva:.2f}%\n")

    print("\n===========================")
print("BACKTEST: BASELINE vs PROPHET (12 months)")
print("===========================\n")

# Test only first 3 groups for now
test_groups_monthly = group_ids[:3]

for gid in test_groups_monthly:
    print(f"--- Group {gid} ---")

    # 1) Build full monthly series for this group
    df_full = pd.DataFrame({
        "ds": cons.index,
        "y": cons[gid].astype(float)
    })
    df_full["ds"] = pd.to_datetime(df_full["ds"]).dt.tz_localize(None)

    monthly_full = (
        df_full.set_index("ds")["y"]
        .resample("MS")   # month start
        .sum()
        .to_frame("y")
    )
    monthly_full.index = pd.to_datetime(monthly_full.index)

    # Need at least 24 months: 12 for training year + 12 for test year
    if monthly_full.shape[0] < 24:
        print("  Not enough monthly data (need at least 24 months). Skipping.\n")
        continue

    # 2) Split into training (all but last 12 months) and test (last 12 months)
    test_months = monthly_full.index[-12:]
    train_months = monthly_full.index[:-12]

    monthly_train = monthly_full.loc[train_months].reset_index()
    monthly_train.columns = ["ds", "y"]
    monthly_train["ds"] = pd.to_datetime(monthly_train["ds"]).dt.tz_localize(None)

    # 3) Train Prophet on training months only
    m_month_bt = Prophet(
        yearly_seasonality=True,
        seasonality_mode="additive"
    )
    m_month_bt.fit(monthly_train)

    # 4) Forecast next 12 months after training period
    future_12 = m_month_bt.make_future_dataframe(periods=12, freq="MS")
    fc_12 = m_month_bt.predict(future_12)
    fc_12 = fc_12[["ds", "yhat"]]
    fc_12["ds"] = pd.to_datetime(fc_12["ds"]).dt.tz_localize(None)

    # Keep only the forecast months that correspond to test_months
    fc_12 = fc_12.set_index("ds").sort_index()
    fc_12 = fc_12.loc[test_months]
    prophet_fc = fc_12["yhat"].values

    # 5) Build baseline: same month one year earlier
    actual_vals = []
    baseline_vals = []

    for month in test_months:
        # Actual
        actual_vals.append(monthly_full.loc[month, "y"])

        # Baseline = same month last year
        prev_year = month - pd.DateOffset(years=1)
        if prev_year in monthly_full.index:
            baseline_vals.append(monthly_full.loc[prev_year, "y"])
        else:
            baseline_vals.append(np.nan)

    actual = np.array(actual_vals, dtype=float)
    baseline = np.array(baseline_vals, dtype=float)

    mask = ~np.isnan(actual) & ~np.isnan(baseline)

    if mask.sum() == 0:
        print("  No overlapping data to compute monthly accuracy.\n")
        continue

    mape_baseline = mape(actual[mask], baseline[mask])
    mape_prophet = mape(actual[mask], prophet_fc[mask])

    fva = 100 * (mape_baseline - mape_prophet) / mape_baseline

    print(f"  Baseline MAPE: {mape_baseline:.2f}%")
    print(f"  Prophet  MAPE: {mape_prophet:.2f}%")
    print(f"  FVA % improvement: {fva:.2f}%\n")
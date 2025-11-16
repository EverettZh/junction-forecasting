import numpy as np
import pandas as pd
from prophet import Prophet
# ========= 1. LOAD DATA =========
df_raw = pd.read_csv(
    "20251111_JUNCTION_training(training_consumption).csv",
    encoding="latin1" 
)

print("Columns in CSV:", df_raw.columns)

# Make measured_at timezone-naive datetime
df_raw["measured_at"] = (
    pd.to_datetime(df_raw["measured_at"], utc=True)
      .dt.tz_convert(None)
)

# All other columns = separate series to forecast
y_cols = [col for col in df_raw.columns if col != "measured_at"]
print("Series to forecast:", y_cols)

# Identify group columns (all numeric column names except regressors/time)
exclude_cols = ["measured_at"]
group_cols = [c for c in df_raw.columns if c not in exclude_cols]
group_cols = sorted(group_cols, key=int)  # sort by numeric ID

def to_eu_decimal(df):
    """
    Convert numeric columns to strings with comma as decimal separator,
    and keep semicolon-ready DataFrame.
    """
    df_str = df.copy()
    for c in df_str.columns:
        if c == "measured_at":
            continue
        df_str[c] = df_str[c].astype(float).map(
            lambda x: f"{x:.6f}".replace(".", ",")
        )
    return df_str


# ========= 2. TRAIN PROPHET MODELS =========
models = {}
forecasts = {}

for col in y_cols:
    print(f"\n=== Fitting Prophet for series '{col}' ===")

    df = df_raw[["measured_at", col]].copy()
    df = df.dropna(subset=[col])

    # Prophet format
    df = df.rename(columns={
        "measured_at": "ds",
        col: "y"
    })

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",  # good for energy (bigger swings when level is high)
    )

    m.add_seasonality(
    name="yearly_custom",
    period=365.25,
    fourier_order=10  # try e.g. 10–15 if you have multiple full years
    )

    m.fit(df)

    # Adjust 'periods' and 'freq' if needed
    future = m.make_future_dataframe(
        periods=30,
        freq="h"  # 👈 assuming hourly data; use "D" for daily, etc.
    )

    forecast = m.predict(future)

    models[col] = m
    forecasts[col] = forecast

    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    

# ---------- 48-Hour Forecast (hourly) ----------

# 1. Build the 48h time index in UTC
future_48 = pd.DataFrame({
    "ds": pd.date_range("2024-10-01 00:00:00", "2024-10-02 23:00:00", freq="h")
})

# 3. Prepare output frame with correct timestamp format
hourly_out = pd.DataFrame()
hourly_out["measured_at"] = future_48["ds"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

# 4. Predict for each group
for g in group_cols:
    print(f"Making 48h forecast for group {g}")
    m = models[g]  # fitted Prophet
    # Prophet only needs 'ds' for univariate forecasting
    fc = m.predict(future_48[["ds"]])

    # Clip negatives (just in case) and store
    hourly_out[g] = fc["yhat"].clip(lower=0)

# 5. Convert to EU decimal format and save as semicolon-separated CSV
hourly_eu = to_eu_decimal(hourly_out)
hourly_eu.to_csv("fortum_48h_forecast.csv", sep=";", index=False, encoding="utf-8")
print("Saved 48h forecast to fortum_48h_forecast.csv")

# ---------- 12-Month Forecast (monthly totals) ----------

# 1. Build hourly future horizon for 12 months (Oct 2024 - Sep 2025)
future_monthly = pd.DataFrame({
    "ds": pd.date_range("2024-10-01 00:00:00", "2025-09-30 23:00:00", freq="h")
})

# 3. Define the 12 month-start timestamps we need in the output
month_index = pd.date_range("2024-10-01T00:00:00Z", "2025-09-01T00:00:00Z", freq="MS")
monthly_out = pd.DataFrame()
monthly_out["measured_at"] = month_index.strftime("%Y-%m-%dT%H:%M:%S.000Z")

# 4. For each group, forecast hourly and aggregate to monthly totals
for g in group_cols:
    print(f"Making 12-month forecast for group {g}")
    m = models[g]

    fc = m.predict(future_monthly[["ds"]])

    tmp = fc[["ds", "yhat"]].copy()
    tmp["yhat"] = tmp["yhat"].clip(lower=0)

    # Month start for each hourly timestamp
    tmp["month_start"] = tmp["ds"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")

    # Sum yhat per month
    month_sums = tmp.groupby("month_start")["yhat"].sum()

    # Reindex to exactly our 12 months (missing → 0)
    month_sums = month_sums.reindex(month_index, fill_value=0.0)

    monthly_out[g] = month_sums.values

# 5. Convert to EU decimal format and save
monthly_eu = to_eu_decimal(monthly_out)
monthly_eu.to_csv("fortum_12m_forecast.csv", sep=";", index=False, encoding="utf-8")
print("Saved 12-month forecast to fortum_12m_forecast.csv")


# these are FVA calculations

# ========= 3. DEFINE MAPE FUNCTION =========
def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

# ========= 4. BASELINE & FVA =========
# For hourly forecasts: same hour 1 week earlier → lag 7 days
BASELINE_LAG = pd.Timedelta(days=7)

results = []
fva_count = 0
count = 0

for col in y_cols:
    print(f"\n=== Scoring series '{col}' ===")

    # Actuals for this series
    actual = df_raw[["measured_at", col]].copy()
    actual = actual.rename(columns={
        "measured_at": "ds",
        col: "y_true"
    })

    # Prophet forecast
    fc = forecasts[col][["ds", "yhat"]].copy()

    # Only evaluate where we have actuals
    eval_df = fc.merge(actual, on="ds", how="inner").dropna(subset=["y_true"])

    if eval_df.empty:
        print(f"No overlapping actuals for {col}, skipping.")
        continue

    # Build baseline: y_baseline(t) = y_true(t - 7 days)
    eval_df["ds_lag"] = eval_df["ds"] - BASELINE_LAG

    eval_df = eval_df.merge(
        actual[["ds", "y_true"]].rename(columns={
            "ds": "ds_lag",
            "y_true": "y_baseline"
        }),
        on="ds_lag",
        how="left"
    )

    eval_df = eval_df.dropna(subset=["y_baseline"])

    if eval_df.empty:
        print(f"No baseline values for {col}, skipping.")
        continue

    # Compute errors
    mape_model = mape(eval_df["y_true"], eval_df["yhat"])
    mape_baseline = mape(eval_df["y_true"], eval_df["y_baseline"])

    if np.isnan(mape_baseline) or mape_baseline == 0:
        fva = np.nan
    else:
        fva = 100.0 * (mape_baseline - mape_model) / mape_baseline

    results.append({
        "series": col,
        "MAPE_model": mape_model,
        "MAPE_baseline": mape_baseline,
        "FVA_percent": fva
    })

    print(f"  MAPE (model)    : {mape_model:.2f}%")
    print(f"  MAPE (baseline) : {mape_baseline:.2f}%")
    print(f"  FVA%            : {fva:.2f}%")
    fva_count += fva
    count += 1

# ========= 5. SHOW SUMMARY TABLE =========
if results:
    results_df = pd.DataFrame(results)
    print("\n==== SUMMARY (lower MAPE is better, higher FVA% is better) ====")
    print(results_df.sort_values("FVA_percent", ascending=False))

    print(fva_count / count)
else:
    print("\nNo series could be scored (no overlap with actuals / baseline).")
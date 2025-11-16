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

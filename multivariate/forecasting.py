# Python
import pandas as pd
from prophet import Prophet
import numpy as np

#df = pd.read_csv("20251111_JUNCTION_training(groups).csv", encoding="latin1")
df_raw = pd.read_csv("20251111_JUNCTION_training(training_consumption).csv", encoding="latin1")

# 2. Ensure time column is datetime
# Parse as UTC, then drop timezone info → naive datetime
df_raw["measured_at"] = (
    pd.to_datetime(df_raw["measured_at"], utc=True)
      .dt.tz_convert(None)
)

price_df = pd.read_csv("20251111_JUNCTION_training(training_prices).csv", encoding="latin1")
print(df_raw.head())
print(price_df.head())


price_df["measured_at"] = (
    pd.to_datetime(price_df["measured_at"], utc=True)
      .dt.tz_convert(None)
)

# Fit Prophet on price: measured_at vs eur_per_mwh
price_hist = price_df[["measured_at", "eur_per_mwh"]].copy()
price_hist = price_hist.rename(columns={"measured_at": "ds", "eur_per_mwh": "y"})

price_model = Prophet()
price_model.fit(price_hist)

# Choose horizon & frequency for price forecast
HORIZON = 30        # same number of steps as your load forecast
FREQ = "h"          # 'h' for hourly, 'D' for daily, etc.

price_future = price_model.make_future_dataframe(periods=HORIZON, freq=FREQ)
price_fc = price_model.predict(price_future)[["ds", "yhat"]]
price_fc = price_fc.rename(columns={"ds": "measured_at", "yhat": "eur_per_mwh_pred"})

# Merge historical and predicted prices
price_all = price_df.merge(price_fc, on="measured_at", how="outer")
price_all = price_all.sort_values("measured_at")

# Prioritize real price, fallback to predicted for missing future
price_all["price_now"] = price_all["eur_per_mwh"]
price_all["price_now"] = price_all["price_now"].fillna(price_all["eur_per_mwh_pred"])

# Just for sanity:
print("price_all head:")
print(price_all[["measured_at", "eur_per_mwh", "eur_per_mwh_pred", "price_now"]].head())
print("price_all tail:")
print(price_all[["measured_at", "eur_per_mwh", "eur_per_mwh_pred", "price_now"]].tail())


print("price_df columns after loading price CSV:", price_df.columns)

df_raw = df_raw.merge(price_df, on="measured_at", how="left")

print("\n df_raw columns AFTER merge:", df_raw.columns)

# ------- 4. CREATE price_now FROM eur_per_mwh -------
df_raw["price_now"] = df_raw["eur_per_mwh"]

# Suppose 'price_now' is your extra regressor
regressor_cols = ["price_now"]   # you can add more later
exclude_cols = ["measured_at", "eur_per_mwh"] + regressor_cols

y_cols = [c for c in df_raw.columns if c not in exclude_cols]

print("Location series to forecast:", y_cols)
print("Regressors:", regressor_cols)

models = {}
forecasts = {}

# 4. Loop over each y column and fit Prophet
for col in y_cols[:3]:
    print(f"\n=== Fitting Prophet for location '{col}' with price regressor ===")

    df = df_raw[["measured_at", col]].merge(
        price_all[["measured_at", "price_now"]],
        on="measured_at",
        how="left"
    )
    # Drop rows where this series is missing
    df = df.dropna(subset=[col, "price_now"])

    # Rename for Prophet: ds = date, y = value
    df = df.rename(columns={
        "measured_at": "ds",
        col: "y"
    })

    print("Training columns:", df.columns)  # should show: ['ds', 'y', 'price_now']

    # Fit model
    m = Prophet()
    m.add_regressor("price_now", standardize="auto", prior_scale=0.5)

    m.fit(df)

    # Make future dataframe (change periods / freq as you like)
    future = m.make_future_dataframe(periods=HORIZON, freq=FREQ)

    # Add price_now for history + future using price_all
    future = future.merge(
        price_all[["measured_at", "price_now"]].rename(columns={"measured_at": "ds"}),
        on="ds",
        how="left"
    )

    print("Future columns after merge:", future.columns)  # must include 'price_now'


    # use last known price as all future price
    future["price_now"] = future["price_now"].ffill()
    print("Future columns:", future.columns)  # must include price_now


    forecast = m.predict(future)

    # Store results
    models[col] = m
    forecasts[col] = forecast

    # Show last few forecast points
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    
BASELINE_LAG = pd.Timedelta(days=7)

results = []
fva_total_counter = 0
counter = 0

for col in y_cols[:3]:
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
        fva_total_counter += fva
        counter += 1
        print(fva_total_counter)
        print(counter)
    

    results.append({
        "series": col,
        "MAPE_model": mape_model,
        "MAPE_baseline": mape_baseline,
        "FVA_percent": fva
    })

    print(f"  MAPE (model)    : {mape_model:.2f}%")
    print(f"  MAPE (baseline) : {mape_baseline:.2f}%")
    print(f"  FVA%            : {fva:.2f}%")


# ========= 5. SHOW SUMMARY TABLE =========
if results:
    results_df = pd.DataFrame(results)
    print("\n==== SUMMARY (lower MAPE is better, higher FVA% is better) ====")
    print(results_df.sort_values("FVA_percent", ascending=False))
else:
    print("\nNo series could be scored (no overlap with actuals / baseline).")

fva_avg = fva_total_counter / counter
print("\nDone. You now have a Prophet model and forecast for each series in 'models' and 'forecasts with FVA average of'", fva_avg, "%.")
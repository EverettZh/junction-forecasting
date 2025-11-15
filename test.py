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

group_ids = cons.columns.tolist()

forecast_48h = {}
forecast_12m = {}

#for gid in group_ids:
# test, switch this back for all

for gid in group_ids:
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

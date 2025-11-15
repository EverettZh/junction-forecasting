# Junction 2025 – Fortum Forecasting Challenge  
### Electricity Consumption Forecasting (48-Hour & 12-Month Horizons)

This repository contains my full solution for the Fortum challenge at Junction 2025.  
The goal is to forecast electricity consumption for 112 customer groups at two horizons:

- **48-hour short-term forecast** (hourly resolution)
- **12-month long-term forecast** (monthly resolution)

The forecasts are generated using **Prophet** models trained independently on each customer group.

All code, data processing, forecasting, evaluation, and exports required by the challenge
are included here.

Here is the result without any regressor:


=== Scoring series '28' ===
  MAPE (model)    : 12.90%
  MAPE (baseline) : 16.37%
  FVA%            : 21.16%

=== Scoring series '29' ===
  MAPE (model)    : 8.91%
  MAPE (baseline) : 10.87%
  FVA%            : 18.02%

=== Scoring series '30' ===
  MAPE (model)    : 11.04%
  MAPE (baseline) : 15.07%
  FVA%            : 26.74%

==== SUMMARY (lower MAPE is better, higher FVA% is better) ====
  series  MAPE_model  MAPE_baseline  FVA_percent
2     30   11.036318      15.065383    26.743857
0     28   12.904271      16.366659    21.155128
1     29    8.913443      10.872644    18.019543

Adding the price regression however, does not improve our FVA, most likely due to not predicting the price

=== Scoring series '28' ===
  MAPE (model)    : 13.17%
  MAPE (baseline) : 16.38%
  FVA%            : 19.58%

=== Scoring series '29' ===
  MAPE (model)    : 8.95%
  MAPE (baseline) : 10.83%
  FVA%            : 17.42%

=== Scoring series '30' ===
  MAPE (model)    : 11.38%
  MAPE (baseline) : 15.09%
  FVA%            : 24.59%

==== SUMMARY (lower MAPE is better, higher FVA% is better) ====
  series  MAPE_model  MAPE_baseline  FVA_percent
2     30   11.376808      15.086038    24.587172
0     28   13.174392      16.382780    19.583904
1     29    8.947261      10.834303    17.417293

Predicting the price introduces more randomness, does not improve FVA significantly

==== SUMMARY (lower MAPE is better, higher FVA% is better) ====
  series  MAPE_model  MAPE_baseline  FVA_percent
2     30   11.376808      15.086038    24.587172
0     28   13.174392      16.382780    19.583904
1     29    8.947261      10.834303    17.417293

Here is the overall FVA averages in comparison:



For price and consumption prediction:


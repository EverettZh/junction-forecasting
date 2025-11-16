# Junction 2025 – Fortum Forecasting Challenge  
### Electricity Consumption Forecasting (48-Hour & 12-Month Horizons)

This repository contains my full solution for the Fortum challenge at Junction 2025.  
The goal is to forecast electricity consumption for 112 customer groups at two horizons:

- **48-hour short-term forecast** (hourly resolution)
- **12-month long-term forecast** (monthly resolution)

The forecasts are generated using **Prophet** models trained independently on each customer group.

All code, data processing, forecasting, evaluation, and exports required by the challenge
are included here.

See our FVA progress for our FVA for each stage of development.

The FVA results tends to do well in colder areas and large consumption and do poorly on warmer areas with little consumptions. For future ideas on how to proceed, we will definitely focus on the poorly predicted area and maybe make a separate prediction for them as well as scraping temperature and weather condition and consider them into the prediction.
